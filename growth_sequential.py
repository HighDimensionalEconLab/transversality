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
from growth_vfi import solve_growth_model_vfi, VFIParameters
from typing import List, Optional


class DeterministicSequentialGrowthModule(pl.LightningModule):
    def __init__(
        self,
        # economic parameters
        beta: float,
        alpha: float,
        delta: float,
        g: float,
        k_0: float,
        # parameters for the data generation and use of data
        batch_size: int,
        shuffle_training: bool,
        train_t_min: int,
        train_t_max: int,
        train_t_step: int,
        train_t_extra_points: List[int],
        test_T: int,
        # Weights for the residuals
        lambda_1: float,
        lambda_2: float,
        vfi_parameters: VFIParameters,
        verbose: bool,
        hpo_objective_name: str,
        always_log_hpo_objective: bool,
        print_metrics: bool,
        save_metrics: bool,
        save_test_results: bool,
        test_loss_success_threshold: float,
        ml_model: torch.nn.Module,
        a: Optional[float] = None,
        b_1: Optional[float] = None,
        b_2: Optional[float] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["ml_model"])
        # default method for residual computation

        self.residuals = self.model_residuals
        # ML model of the k_t : R -> R function (where we are assuming interpolation will work for N to R)
        self.ml_model = ml_model

        # If in the butterfly production and swap functions as required
        if not (a is None or b_1 is None or b_2 is None):
            # store kink in the production function
            self.k_threshold = (b_2 / (b_1 - 1)) ** (1 / alpha)
            self.f = self.f_butterfly
            self.df_k = self.df_k_butterfly
            self.steady_states = self.steady_states_butterfly

    # k(t) using the ml_model
    def forward(self, t):
        return self.ml_model(t)

    # convenience function for code clarity
    def k(self, t):
        return self.forward(t)

    # Note: production functions swapped to self.f and self.df_k depending on parameters within the init

    # Standard production function
    # f(z, k)
    def f(self, z, k):
        return z ** (1 - self.hparams.alpha) * k**self.hparams.alpha

    # d/dk f(z,k)
    def df_k(self, z, k):
        return (
            self.hparams.alpha
            * z ** (1 - self.hparams.alpha)
            * k ** (self.hparams.alpha - 1)
        )

    def steady_states(self):
        k_ss = torch.tensor(
            ((1 / self.hparams.beta - 1.0 + self.hparams.delta) / self.hparams.alpha)
            ** (1 / (self.hparams.alpha - 1))
        )
        c_ss = self.c(1.0, k_ss, k_ss)
        return [(k_ss, c_ss)]

    # Multiple steady state production function
    # f(z, k)
    def f_butterfly(self, z, k):
        return (
            z ** (1 - self.hparams.alpha)
            * self.hparams.a
            * torch.max(
                k**self.hparams.alpha,
                self.hparams.b_1 * (k**self.hparams.alpha) - self.hparams.b_2,
            )
        )

    # d/dk f(z,k)
    def df_k_butterfly(self, z, k):
        df_k = (
            self.hparams.a
            * self.hparams.alpha
            * z ** (1 - self.hparams.alpha)
            * k ** (self.hparams.alpha - 1)
        )
        # Closed form: for k > k_threshold b_1 * lower branch
        k_threshold_mask = k > self.k_threshold
        df_k[k_threshold_mask] *= self.hparams.b_1  # componentwise
        return df_k

    def steady_states_butterfly(self):
        alpha = self.hparams.alpha
        a = self.hparams.a
        b_1 = self.hparams.b_1
        b_2 = self.hparams.b_2
        delta = self.hparams.delta
        beta = self.hparams.beta

        k_ss_low = torch.tensor(
            ((1 / beta - 1.0 + delta) / (a * alpha)) ** (1 / (alpha - 1))
        )
        k_ss_high = torch.tensor(
            ((1 / beta - 1.0 + delta) / (a * alpha * b_1)) ** (1 / (alpha - 1))
        )

        c_ss_low = self.c(1.0, k_ss_low, k_ss_low)
        c_ss_high = self.c(1.0, k_ss_high, k_ss_high)
        return [(k_ss_low, c_ss_low), (k_ss_high, c_ss_high)]

    # The rest of the functions are for an arbitrary f(z,k) function
    # c(z,k,k')
    def c(self, z, k, k_prime):
        return self.f(z, k) + (1 - self.hparams.delta) * k - k_prime

    # z(t) = (1+g)^t
    def z(self, t):
        return (1.0 + self.hparams.g) ** t

    def model_residuals(self, t):
        k_t = self.k(t)
        k_tp1 = self.k(t + 1)
        k_tp2 = self.k(t + 2)
        z_t = self.z(t)
        z_tp1 = self.z(t + 1)
        c_t = self.c(z_t, k_t, k_tp1)
        c_tp1 = self.c(z_tp1, k_tp1, k_tp2)

        # Euler Residual
        res = c_tp1 / c_t - self.hparams.beta * (
            1 - self.hparams.delta + self.df_k(z_tp1, k_tp1)
        )
        time_zero = torch.zeros([1, 1], device=self.device, dtype=self.dtype)

        # initial condition residual: k(0) - k_0
        ic_residual = self.k(time_zero) - self.hparams.k_0
        return res, ic_residual

    # Loss is the sum of the squares of euler and initial condition residuals
    def training_step(self, batch, batch_idx):
        t = batch
        euler_residual, ic_residual = self.residuals(t)

        loss = self.hparams.lambda_1 * torch.mean(
            euler_residual**2
        ) + self.hparams.lambda_2 * torch.mean(ic_residual**2)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def setup(self, stage):
        # Assumes that test_T > max(train_t, train_t_extra_points...)
        # The indices of the training data are the train_t_min:train_t_step:train_t_max UNION train_t_extra_points (removing duplicates)
        self.train_data = (
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
            .values.to(self.dtype)
            .unsqueeze(-1)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.hparams.batch_size
            if self.hparams.batch_size > 0
            else len(self.train_data),
            shuffle=self.hparams.shuffle_training,
        )


# With larger problems and random test_data use a test_step instead
@torch.no_grad()
def test_model(model):
    def f_scaled(x):
        if not (
            model.hparams.a is None
            or model.hparams.b_1 is None
            or model.hparams.b_2 is None
        ):
            return model.hparams.a * max(
                x**model.hparams.alpha,
                model.hparams.b_1 * (x**model.hparams.alpha) - model.hparams.b_2,
            )
        else:
            return x**model.hparams.alpha

    steady_states = model.steady_states()
    # solve the model using VFI.  Slow, but dependable
    # Uses a multiplier on the minimum or maximum of the k_ss steady states
    vfi_parameters = model.hparams.vfi_parameters
    k_grid = np.linspace(
        vfi_parameters.k_min_multiplier
        * min(model.hparams.k_0, min([t[0] for t in steady_states])),
        vfi_parameters.k_max_multiplier
        * max(model.hparams.k_0, max([t[0] for t in steady_states])),
        vfi_parameters.k_grid_size,
    )

    k_prime_vfi, c_vfi = solve_growth_model_vfi(
        k_grid,
        f_scaled,  # uses scaled f(k)
        model.hparams.beta,
        model.hparams.delta,
        model.hparams.g,
        vfi_tol=vfi_parameters.tol,
        c_solver_tol=vfi_parameters.c_solver_tol,
        max_iter=vfi_parameters.max_iter,
        interpolation_kind=vfi_parameters.interpolation_kind,
    )

    # Use solution
    t = torch.arange(0, model.hparams.test_T).type("torch.FloatTensor").unsqueeze(-1)
    z_t = model.z(t)
    k_t = model.k(t)
    k_tp1 = model.k(t + 1)
    c_t = model.c(z_t, k_t, k_tp1)

    # Solve with VFI
    k_t_vfi = torch.empty_like(z_t)
    c_t_vfi = torch.empty_like(z_t)

    k_t_vfi[0] = model.hparams.k_0
    # Iterating the VFI version forwards
    for t_val in range(model.hparams.test_T):
        if t_val < model.hparams.test_T - 1:
            k_t_vfi[t_val + 1] = k_prime_vfi(z_t[t_val], k_t_vfi[t_val])
        c_t_vfi[t_val] = c_vfi(z_t[t_val], k_t_vfi[t_val])

    # using the same name for simplicity, but could rename later
    k_rel_error = (k_t - k_t_vfi) / k_t_vfi
    c_rel_error = (c_t - c_t_vfi) / c_t_vfi

    euler_residual, ic_residual = model.residuals(t)
    loss = model.hparams.lambda_1 * torch.mean(
        euler_residual**2
    ) + model.hparams.lambda_2 * torch.mean(ic_residual**2)

    model.test_results = pd.DataFrame(
        {
            "t": t.squeeze().cpu().numpy().tolist(),
            "k_t_approx": k_t.squeeze().cpu().numpy().tolist(),
            "c_t_approx": c_t.squeeze().cpu().numpy().tolist(),
            "k_t_sol": k_t_vfi.squeeze().cpu().numpy().tolist(),
            "c_t_sol": c_t_vfi.squeeze().cpu().numpy().tolist(),
            "k_rel_error": k_rel_error.squeeze().cpu().numpy().tolist(),
            "c_rel_error": c_rel_error.squeeze().cpu().numpy().tolist(),
            "euler_residual": euler_residual.squeeze().cpu().numpy().tolist(),
            "ic_residual": ic_residual.squeeze().cpu().numpy().tolist(),
        }
    )
    # the steady-state for consumption and capital, normalized to BGP
    if len(steady_states) == 1:
        k_ss = steady_states[0][0]
        c_ss = steady_states[0][1]
        k_ss_norm = z_t * k_ss
        c_ss_norm = z_t * c_ss

        model.test_results["k_ss_norm"] = k_ss_norm.squeeze().cpu().numpy().tolist()
        model.test_results["c_ss_norm"] = c_ss_norm.squeeze().cpu().numpy().tolist()
    else:
        k_ss_low = steady_states[0][0]
        c_ss_low = steady_states[0][1]
        k_ss_high = steady_states[1][0]
        c_ss_high = steady_states[1][1]
        k_ss_low_norm = z_t * k_ss_low
        c_ss_low_norm = z_t * c_ss_low
        k_ss_high_norm = z_t * k_ss_high
        c_ss_high_norm = z_t * c_ss_high
        model.test_results["k_ss_low_norm"] = (
            k_ss_low_norm.squeeze().cpu().numpy().tolist()
        )
        model.test_results["c_ss_low_norm"] = (
            c_ss_low_norm.squeeze().cpu().numpy().tolist()
        )
        model.test_results["k_ss_high_norm"] = (
            k_ss_high_norm.squeeze().cpu().numpy().tolist()
        )
        model.test_results["c_ss_high_norm"] = (
            c_ss_high_norm.squeeze().cpu().numpy().tolist()
        )

    model.logger.experiment.log(
        {
            "test_loss": loss,
            "k_abs_rel_error": k_rel_error.abs().mean(),
            "c_abs_rel_error": c_rel_error.abs().mean(),
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
        DeterministicSequentialGrowthModule,
        seed_everything_default=123,
        run=False,
        save_config_callback=None,  # turn this on to save the full config file rather than just having it uploaded
        parser_kwargs={"default_config_files": ["growth_sequential_defaults.yaml"]},
        save_config_kwargs={"save_config_overwrite": True},
    )
    # Fit the model.  Separating training time for plotting, and evaluate generalization
    start = timeit.default_timer()
    cli.trainer.fit(cli.model)
    train_time = timeit.default_timer() - start
    train_callback_metrics = cli.trainer.callback_metrics
    cli.model.eval()  # Enter evaluation mode, not training
    test_model(cli.model)

    # Add additional calculations such as HPO objective to the log and save files
    log_and_save(cli.trainer, cli.model, train_time, train_callback_metrics)
