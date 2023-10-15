import pandas as pd
import torch
import pytorch_lightning as pl
import yaml
import math
import numpy as np
import wandb
import timeit
import econ_layers
from torch.utils.data import DataLoader
from pytorch_lightning.cli import LightningCLI
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import TensorDataset

from typing import List, Optional


class DeterministicSequentialAssetPricing(pl.LightningModule):
    def __init__(
        self,
        # economic parameters
        beta: float,
        y_0: float,
        g: float,
        c: float,
        # parameters for the data generation and use of data
        batch_size: int,
        shuffle_training: bool,
        train_t_min: int,
        train_t_max: int,
        train_t_step: int,
        train_t_extra_points: List[int],
        test_T: int,
        # some general configuration
        verbose: bool,
        hpo_objective_name: str,
        always_log_hpo_objective: bool,
        print_metrics: bool,
        save_metrics: bool,
        save_test_results: bool,
        test_loss_success_threshold: float,
        ml_model: torch.nn.Module,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["ml_model"])
        # default method for residual computation

        self.residuals = self.model_residuals
        # ML model of the p_t : R -> R function (where we are assuming interpolation will work for N to R)
        self.ml_model = ml_model

        self.A = torch.tensor(
            [[1, 0], [self.hparams.c, 1 + self.hparams.g]]
        )  # equation 2
        self.G = torch.tensor([0.0, 1.0])  #
        self.x_0 = torch.tensor([1.0, self.hparams.y_0])
        self.H = self.G @ torch.inverse(
            torch.eye(2, device=self.device, dtype=self.dtype)
            - self.hparams.beta * self.A
        )

    # Forward for the neural net, calculates the value of the NN for an input t, i.e. $\hat{p}(t;\theta)$.
    def forward(self, t):
        return self.ml_model(t)

    def p(self, t):
        return self.forward(t)

    # model residuals given t and y_t
    def model_residuals(self, t, y_t):
        # difference equation: y_t + beta * p(t+1) - p(t)
        return y_t + self.hparams.beta * self.p(t + 1) - self.p(t)

    def training_step(self, batch, batch_idx):
        t, y_t = batch
        residuals = self.residuals(t, y_t)
        loss = (residuals**2).sum() / len(residuals)  # Equation 12
        self.log("train_loss", loss, prog_bar=True)
        return loss

    # Iterates LSS
    # x_{t+1} = A x_t
    # A = [1  0
    #     c 1+g]
    # G = [0 1]
    # x_t = [1 y_t]
    # x_j = A^j x_0
    # p_t = inv(I- beta A) x_t = H x_t

    def iterate_LSS(self, x_0, T):
        y = []
        x = []
        t = torch.arange(0, T)
        p_f = []
        x_t = x_0
        for _ in range(T):
            x.append(x_t)
            y.append(self.G @ x_t)  # observation equation
            p_f.append(self.H @ x_t)  # closed form solution
            x_t = self.A @ x_t  # iterate state equation forward
        return t, torch.stack(x), torch.stack(y), torch.stack(p_f)

    def setup(self, stage):
        # Does the superset of the training and test points given the problem is deterministic
        train_T = (
            max(
                self.hparams.train_t_max,
                max(self.hparams.train_t_extra_points, default=0),
            )
            + 1
        )

        t, x, y, p_f = self.iterate_LSS(self.x_0, train_T)
        t = t.type_as(y).unsqueeze(-1)
        y = y.unsqueeze(-1)
        # unsqueeze makes into a matrix to make broadcasting easier.

        # The indices of the training data are the train_t_min:train_t_step:train_t_max UNION train_t_extra_points (removing duplicates)
        train_t = (
            torch.cat(
                (
                    torch.arange(
                        self.hparams.train_t_min,
                        self.hparams.train_t_max,
                        self.hparams.train_t_step,
                    ),
                    torch.tensor(
                        [self.hparams.train_t_max]
                    ),  # always add the last index
                    torch.tensor(self.hparams.train_t_extra_points),
                )
            )
            .unique()
            .sort()
            .values.to(dtype=torch.int32)
            .unsqueeze(-1)
        )
        self.train_data = TensorDataset(t[train_t], y[train_t])

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.hparams.batch_size
            if self.hparams.batch_size > 0
            else len(self.train_data),
            shuffle=self.hparams.shuffle_training,
        )


@torch.no_grad()
def test_model(model):
    t, x_t, y_t, p_f_t = model.iterate_LSS(model.x_0, model.hparams.test_T)
    y_t = y_t.unsqueeze(-1)
    p_f_t = p_f_t.unsqueeze(-1)
    t = t.type("torch.FloatTensor").unsqueeze(-1)
    p_t = model.p(t)

    # the difference between approximate and the fundamental solution
    p_bubble = p_t - p_f_t
    p_rel_error = p_bubble / p_f_t
    p_abs_rel_error = p_rel_error.abs()
    bubble_mse = torch.nn.functional.mse_loss(p_t, p_f_t)

    residuals = model.residuals(t, y_t)
    loss = (residuals**2).sum() / len(residuals)

    model.test_results = pd.DataFrame(
        {
            "t": t.squeeze().cpu().numpy().tolist(),
            "residuals": residuals.squeeze().cpu().numpy().tolist(),
            "p_t": p_t.squeeze().cpu().numpy().tolist(),
            "p_f_t": p_f_t.squeeze().cpu().numpy().tolist(),
            "p_bubble": p_bubble.squeeze().cpu().numpy().tolist(),
            "p_rel_error": p_rel_error.squeeze().cpu().numpy().tolist(),
            "p_abs_rel_error": p_abs_rel_error.squeeze().cpu().numpy().tolist(),
        }
    )

    model.logger.experiment.log(
        {
            "bubble_mse": bubble_mse,
            "test_loss": loss,
            "p_abs_rel_error": p_abs_rel_error.mean(),
        }
    )

    if hasattr(model.ml_model, "OutputRescalingLayer") and hasattr(
        model.ml_model.OutputRescalingLayer, "weight"
    ):
        model.logger.experiment.log(
            {"rescale_weight": model.ml_model.OutputRescalingLayer.weight[0].item()}
        )
        if model.hparams.g != 0:
            rescale_error = (
                model.ml_model.OutputRescalingLayer.weight[0].item()
                / np.log(1 + model.hparams.g)
                - 1
            )
            model.logger.experiment.log({"rescale_rel_error": rescale_error})


def log_and_save(trainer, model, train_time, train_callback_metrics):
    if type(trainer.logger) is WandbLogger:
        # Valid numeric types
        def not_number_type(value):
            if value is None:
                return True

            if not isinstance(value, (int, float)):
                return True

            if math.isnan(value) or math.isinf(value):
                return True

            return False  # otherwise a valid, non-infinite number

        # If early stopping, evaluate success
        early_stopping_check_failed = math.nan
        early_stopping_monitor = ""
        early_stopping_threshold = math.nan

        for callback in trainer.callbacks:
            if type(callback) == pl.callbacks.early_stopping.EarlyStopping:
                early_stopping_monitor = callback.monitor
                early_stopping_value = (
                    train_callback_metrics[callback.monitor].cpu().numpy().tolist()
                )
                early_stopping_threshold = callback.stopping_threshold
                early_stopping_check_failed = not_number_type(early_stopping_value) or (
                    early_stopping_value > callback.stopping_threshold
                )  # hardcoded to min for now.
                break

        # Check test loss
        if model.hparams.test_loss_success_threshold == 0:
            test_loss_check_failed = math.nan
        elif not_number_type(cli.trainer.logger.experiment.summary["test_loss"]) or (
            cli.trainer.logger.experiment.summary["test_loss"]
            > model.hparams.test_loss_success_threshold
        ):
            test_loss_check_failed = True
        else:
            test_loss_check_failed = False

        if early_stopping_check_failed in [
            False,
            math.nan,
        ] and test_loss_check_failed in [False, math.nan]:
            retcode = 0
            convergence_description = "Success"
        elif early_stopping_check_failed == True:
            retcode = -1
            convergence_description = "Early stopping failure"
        elif test_loss_check_failed == True:
            retcode = -3
            convergence_description = "Test loss failure due to possible overfitting"
        else:
            retcode = -100
            convergence_description = " Unknown failure"

        # Log all calculated results
        trainable_parameters = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        trainer.logger.experiment.log({"train_time": train_time})
        trainer.logger.experiment.log(
            {"early_stopping_monitor": early_stopping_monitor}
        )
        trainer.logger.experiment.log(
            {"early_stopping_threshold": early_stopping_threshold}
        )
        trainer.logger.experiment.log(
            {"early_stopping_check_failed": early_stopping_check_failed}
        )
        trainer.logger.experiment.log(
            {"test_loss_check_failed": test_loss_check_failed}
        )
        trainer.logger.experiment.log({"trainable_parameters": trainable_parameters})
        trainer.logger.experiment.log({"retcode": retcode})
        trainer.logger.experiment.log(
            {"convergence_description": convergence_description}
        )

        # Set objective for hyperparameter optimization
        # Objective value given in the settings, or empty

        if model.hparams.hpo_objective_name is not None:
            hpo_objective_value = dict(cli.trainer.logger.experiment.summary)[
                model.hparams.hpo_objective_name
            ]
        else:
            hpo_objective_value = math.nan

        if model.hparams.always_log_hpo_objective or retcode >= 0:
            trainer.logger.experiment.log({"hpo_objective": hpo_objective_value})
        else:
            trainer.logger.experiment.log({"hpo_objective": math.nan})

        # Save test results
        trainer.logger.log_text(
            key="test_results", dataframe=trainer.model.test_results
        )  # Saves on wandb for querying later

        # save the summary statistics in a file
        if model.hparams.save_metrics and trainer.log_dir is not None:
            metrics_path = Path(trainer.log_dir) / "metrics.yaml"
            with open(metrics_path, "w") as fp:
                yaml.dump(dict(cli.trainer.logger.experiment.summary), fp)

        if model.hparams.print_metrics:
            print(dict(cli.trainer.logger.experiment.summary))
        return

    else:  # almost no features enabled for other loggers. Could refactor later
        if model.hparams.save_test_results and trainer.log_dir is not None:
            model.test_results.to_csv(
                Path(trainer.log_dir) / "test_results.csv", index=False
            )


if __name__ == "__main__":
    cli = LightningCLI(
        DeterministicSequentialAssetPricing,
        seed_everything_default=123,
        run=False,
        save_config_callback=None,  # turn this on to save the full config file rather than just having it uploaded
        parser_kwargs={
            "default_config_files": ["asset_pricing_sequential_defaults.yaml"]
        },
        save_config_kwargs={"save_config_overwrite": True},
    )
    # Fit the model.  Separating training time for plotting, and evaluate generalization
    start = timeit.default_timer()
    cli.trainer.fit(cli.model)
    train_time = timeit.default_timer() - start
    train_callback_metrics = cli.trainer.callback_metrics
    test_model(cli.model)
    # Add additional calculations such as HPO objective to the log and save files
    log_and_save(cli.trainer, cli.model, train_time, train_callback_metrics)
