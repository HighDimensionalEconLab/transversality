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
from growth_vfi import solve_growth_model_vfi, VFIParameters
from typing import List, Optional
from pathlib import Path


class DeterministicRecursiveGrowthModule(pl.LightningModule):
    def __init__(
        self,
        beta: float,
        alpha: float,
        delta: float,
        k_0: float,
        batch_size: int,
        shuffle_training: bool,
        k_sim_grid_points: int,
        k_grid_min: float,
        k_grid_max: float,
        max_T_test: int,
        train_grid_test_multiplier: float,
        z_grid_min: float,
        z_grid_max: float,
        z_sim_grid_points: int,
        g: float,
        vfi_parameters: VFIParameters,
        verbose: bool,
        hpo_objective_name: str,
        always_log_hpo_objective: bool,
        print_metrics: bool,
        save_metrics: bool,
        save_test_results: bool,
        test_loss_success_threshold: float,
        TVC_test_treshold: float,
        ml_model: torch.nn.Module,
        val_sim_grid_points: int,
        a: Optional[float] = None,
        b_1: Optional[float] = None,
        b_2: Optional[float] = None,
        k_grid_min_2: Optional[float] = None,
        k_grid_max_2: Optional[float] = None,
        exp_grid_base: Optional[float] = None,
        val_min_1: Optional[float] = None,
        val_min_2: Optional[float] = None,
        val_max_1: Optional[float] = None,
        val_max_2: Optional[float] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["ml_model"])

        # default method for residual computation
        self.residuals = self.model_residuals

        # ML model of the [z_t,k_t] : R^2 -> R function
        self.ml_model = ml_model
        # If in the butterfly production and swap functions as required
        if not (a is None or b_1 is None or b_2 is None):
            # store kink in the production function
            self.k_threshold = (b_2 / (b_1 - 1)) ** (1 / alpha)
            self.f = self.f_butterfly
            self.df_k = self.df_k_butterfly
            self.steady_states = self.steady_states_butterfly

    # k'([z,k]) using the ml_model
    def forward(self, x):
        return self.ml_model(x)

    # convenience function for code clarity
    def k_prime(self, x):
        return self.forward(x).squeeze()  # makes a vector/scalar output

    # f([z, k])
    def f(self, x):
        z = x[:, 0]
        k = x[:, 1]
        return z ** (1 - self.hparams.alpha) * k**self.hparams.alpha

    # d/dk f([z,k])
    def df_k(self, x):
        z = x[:, 0]
        k = x[:, 1]
        return (
            self.hparams.alpha
            * z ** (1 - self.hparams.alpha)
            * k ** (self.hparams.alpha - 1)
        )

    def steady_states(self):
        k_ss = (
            (1 / self.hparams.beta - 1.0 + self.hparams.delta) / self.hparams.alpha
        ) ** (1 / (self.hparams.alpha - 1))

        x = torch.tensor([(1.0), (k_ss)])

        c_ss = self.c(x.resize(1, 2))
        return [(k_ss, c_ss)]

    # Multiple steady state production function
    # f(z, k)
    def f_butterfly(self, x):
        z = x[:, 0]
        k = x[:, 1]
        return (
            z ** (1 - self.hparams.alpha)
            * self.hparams.a
            * torch.max(
                k**self.hparams.alpha,
                self.hparams.b_1 * (k**self.hparams.alpha) - self.hparams.b_2,
            )
        )

    # d/dk f(z,k)
    def df_k_butterfly(self, x):
        z = x[:, 0]
        k = x[:, 1]
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

        k_ss_low = ((1 / beta - 1.0 + delta) / (a * alpha)) ** (1 / (alpha - 1))

        k_ss_high = ((1 / beta - 1.0 + delta) / (a * alpha * b_1)) ** (1 / (alpha - 1))

        x_low = torch.tensor([(1.0), (k_ss_low)])

        x_high = torch.tensor([(1.0), (k_ss_high)])

        c_ss_low = self.c(x_low.resize(1, 2))
        c_ss_high = self.c(x_high.resize(1, 2))
        return [(k_ss_low, c_ss_low), (k_ss_high, c_ss_high)]

    # z'(z)
    def z_prime(self, z):
        return (1.0 + self.hparams.g) * z

    # c([z,k]) using the internal k'([z,k])
    def c(self, x):
        z = x[:, 0]
        k = x[:, 1]
        return self.f(x) + (1 - self.hparams.delta) * k - self.k_prime(x)

    # Euler residuals
    # May have a redundant call to self(x) for code clarity
    # Could embed the c(...) calculations to reuse the k'([z,k]) calculations
    def model_residuals(self, x_t):
        z_t = x_t[:, 0]
        k_t = x_t[:, 1]
        c_t = self.c(x_t)

        # iterate forwards
        k_tp1 = self.k_prime(x_t)
        z_tp1 = self.z_prime(z_t)
        x_tp1 = torch.stack([z_tp1, k_tp1], axis=1)
        c_tp1 = self.c(x_tp1)

        # Euler residual
        res = c_tp1 / c_t - self.hparams.beta * (
            1 - self.hparams.delta + self.df_k(x_tp1)
        )
        return res

    # minimizing the Euler residuals
    def training_step(self, batch, batch_idx):
        res = self.residuals(batch)
        loss = torch.mean(res**2)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        res = self.residuals(batch)
        loss = torch.mean(res**2)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    # Simulates all of the data using the state space model
    def setup(self, stage):
        if stage == "fit" or stage is None:
            # For now this uses the entire grid of points
            if not (
                self.hparams.k_grid_min_2 is None or self.hparams.k_grid_max_2 is None
            ):
                train_1 = torch.cartesian_prod(
                    torch.linspace(
                        self.hparams.z_grid_min,
                        self.hparams.z_grid_max,
                        steps=self.hparams.z_sim_grid_points,
                    ),
                    torch.linspace(
                        self.hparams.k_grid_min_2,
                        self.hparams.k_grid_max_2,
                        steps=self.hparams.k_sim_grid_points,
                    ),
                )
                train_2 = torch.cartesian_prod(
                    torch.linspace(
                        self.hparams.z_grid_min,
                        self.hparams.z_grid_max,
                        steps=self.hparams.z_sim_grid_points,
                    ),
                    torch.linspace(
                        self.hparams.k_grid_min,
                        self.hparams.k_grid_max,
                        steps=self.hparams.k_sim_grid_points,
                    ),
                )
                self.train_data = torch.cat((train_1, train_2))
            else:
                # start_grid = np.log(self.hparams.k_grid_min) / np.log(1.2)
                # end_grid = np.log(self.hparams.k_grid_max) / np.log(1.2)

                self.train_data = torch.cartesian_prod(
                    torch.linspace(
                        self.hparams.z_grid_min,
                        self.hparams.z_grid_max,
                        steps=self.hparams.z_sim_grid_points,
                    ),
                    torch.linspace(
                        self.hparams.k_grid_min,
                        self.hparams.k_grid_max,
                        steps=self.hparams.k_sim_grid_points,
                    ),
                )

            if not (self.hparams.val_sim_grid_points <= 0):
                if self.hparams.val_min_1 is None or self.hparams.val_max_1 is None:
                    self.hparams.val_min_1 = self.hparams.k_grid_min
                    self.hparams.val_max_1 = self.hparams.k_grid_max

                if not (
                    self.hparams.val_min_2 is None or self.hparams.val_max_2 is None
                ):
                    val_1 = torch.cartesian_prod(
                        torch.linspace(
                            self.hparams.z_grid_min,
                            self.hparams.z_grid_max,
                            steps=self.hparams.z_sim_grid_points,
                        ),
                        torch.linspace(
                            self.hparams.val_min_1,
                            self.hparams.val_max_1,
                            steps=self.hparams.val_sim_grid_points,
                        ),
                    )
                    val_2 = torch.cartesian_prod(
                        torch.linspace(
                            self.hparams.z_grid_min,
                            self.hparams.z_grid_max,
                            steps=self.hparams.z_sim_grid_points,
                        ),
                        torch.linspace(
                            self.hparams.val_min_2,
                            self.hparams.val_max_2,
                            steps=self.hparams.val_sim_grid_points,
                        ),
                    )
                    self.val_data = torch.cat((val_1, val_2))
                else:
                    # start_grid = np.log(self.hparams.k_grid_min) / np.log(1.2)
                    # end_grid = np.log(self.hparams.k_grid_max) / np.log(1.2)

                    self.val_data = torch.cartesian_prod(
                        torch.linspace(
                            self.hparams.z_grid_min,
                            self.hparams.z_grid_max,
                            steps=self.hparams.z_sim_grid_points,
                        ),
                        torch.linspace(
                            self.hparams.val_min_1,
                            self.hparams.val_min_1,
                            steps=self.hparams.val_sim_grid_points,
                        ),
                    )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.hparams.batch_size
            if self.hparams.batch_size > 0
            else len(self.train_data),
            shuffle=self.hparams.shuffle_training,
        )

    # set limit_val_batches =0 if val_sim_grid_points<=0
    def val_dataloader(self):
        if self.hparams.val_sim_grid_points > 0:
            return DataLoader(
                self.val_data, batch_size=len(self.val_data), shuffle=False
            )
        else:
            return None


# With larger problems and random test_data use a test_step instead
@torch.no_grad()
def test_model(model):
    alpha, beta, delta, g = (
        model.hparams.alpha,
        model.hparams.beta,
        model.hparams.delta,
        model.hparams.g,
    )

    def f_scaled(k):
        if not (
            model.hparams.a is None
            or model.hparams.b_1 is None
            or model.hparams.b_2 is None
        ):
            return model.hparams.a * max(
                k**model.hparams.alpha,
                model.hparams.b_1 * (k**model.hparams.alpha) - model.hparams.b_2,
            )
        else:
            return k**model.hparams.alpha

    steady_states = model.steady_states()
    # Simple problem sequencing over "t" rather, so skip the test loop

    # solve the model using VFI.  Slow, but dependable
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

    if model.hparams.max_T_test > 0:
        z_t = torch.empty(
            (model.hparams.max_T_test, 1),
            dtype=model.dtype,
            device=model.device,
        )
        k_t = torch.empty_like(z_t)
        k_t_vfi = torch.empty_like(z_t)
        c_t = torch.empty_like(z_t)
        c_t_vfi = torch.empty_like(z_t)

        # start at the initial condition
        z_t[0] = 1.0  # z_0 = 1 is hardcoded
        k_t[0] = model.hparams.k_0
        k_t_vfi[0] = model.hparams.k_0
        # Iterating the z, NN, and VFI version forwards
        for t in range(model.hparams.max_T_test):
            x_t = torch.stack((z_t[t], k_t[t]), axis=1)
            if t < model.hparams.max_T_test - 1:
                k_t[t + 1] = model.k_prime(x_t)
                k_t_vfi[t + 1] = k_prime_vfi(z_t[t], k_t_vfi[t])
                z_t[t + 1] = model.z_prime(z_t[t])

            c_t[t] = model.c(x_t)  # uses ML model
            c_t_vfi[t] = c_vfi(z_t[t], k_t_vfi[t])

        t = torch.arange(
            model.hparams.max_T_test, dtype=model.dtype, device=model.device
        ).unsqueeze(1)

        # Relative errors
        k_rel_error = (k_t - k_t_vfi) / k_t_vfi
        c_rel_error = (c_t - c_t_vfi) / c_t_vfi

        x_t = torch.cat((z_t, k_t), dim=1)
        res_t = model.residuals(x_t)

        # can't use model.log outside of test_step
        model.logger.experiment.log(
            {
                "test_loss": torch.mean(res_t**2),
                "k_abs_rel_error": k_rel_error.abs().mean(),
                "c_abs_rel_error": c_rel_error.abs().mean(),
            }
        )

        model.test_results = pd.DataFrame(
            {
                "t": t.squeeze().cpu().numpy().tolist(),
                "z_t": z_t.squeeze().cpu().numpy().tolist(),
                "k_t_approx": k_t.squeeze().cpu().numpy().tolist(),
                "c_t_approx": c_t.squeeze().cpu().numpy().tolist(),
                "k_t_sol": k_t_vfi.squeeze().cpu().numpy().tolist(),
                "c_t_sol": c_t_vfi.squeeze().cpu().numpy().tolist(),
                "k_rel_error": k_rel_error.squeeze().cpu().numpy().tolist(),
                "c_rel_error": c_rel_error.squeeze().cpu().numpy().tolist(),
                "res_t": res_t.squeeze().cpu().numpy().tolist(),
            }
        )

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
    else:
        # Use the same grid but with train_grid_test_multiplier times as many points
        x_t = torch.cartesian_prod(
            torch.linspace(
                model.hparams.z_grid_min,
                model.hparams.z_grid_max,
                steps=int(
                    model.hparams.z_sim_grid_points
                    * model.hparams.train_grid_test_multiplier
                ),
            ),
            torch.linspace(
                model.hparams.k_grid_min,
                model.hparams.k_grid_max,
                steps=int(
                    model.hparams.k_sim_grid_points
                    * model.hparams.train_grid_test_multiplier
                ),
            ),
        )

        # Calculate residuals and policies
        res_t = model.residuals(x_t)
        z_t = x_t[:, 0]
        k_t = x_t[:, 1]

        # Calculate the policies and relative errors for the NN and VFI solutions
        k_tp1 = model.k_prime(x_t)
        k_tp1_vfi = k_prime_vfi(z_t, k_t)
        c_t = model.c(x_t)
        c_t_vfi = c_vfi(z_t, k_t)

        # using the same name for simplicity, but could rename later
        k_rel_error = (k_tp1 - k_tp1_vfi) / k_tp1_vfi
        c_rel_error = (c_t - c_t_vfi) / c_t_vfi

        # # the steady-state for consumption and capital, normalized to BGP
        # k_ss_norm = z_t * model.k_ss
        # c_ss_norm = z_t * model.c_ss

        # No "t", and the (z_t,k_t) are grid, and the k_rel_error and c_rel_error are
        # the policy errors, not the simulated trajectories.
        model.test_results = pd.DataFrame(
            {
                "z_t": z_t.squeeze().cpu().numpy().tolist(),
                "k_t": k_t.squeeze().cpu().numpy().tolist(),
                "k_tp1": k_tp1.squeeze().cpu().numpy().tolist(),
                "k_tp1_vfi": k_tp1_vfi.squeeze().cpu().numpy().tolist(),
                "k_rel_error": k_rel_error.squeeze().cpu().numpy().tolist(),
                "c_rel_error": c_rel_error.squeeze().cpu().numpy().tolist(),
                "res_t": res_t.squeeze().cpu().numpy().tolist(),
            }
        )
        # can't use model.log outside of test_step
        model.logger.experiment.log(
            {
                "test_loss": torch.mean(res_t**2),
                "k_abs_rel_error": k_rel_error.abs().mean(),
                "c_abs_rel_error": c_rel_error.abs().mean(),
            }
        )


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

        steady_states = trainer.model.steady_states()

        if not (
            trainer.model.hparams.a is None
            or trainer.model.hparams.b_1 is None
            or trainer.model.hparams.b_2 is None
        ):
            k_ss_max = max(steady_states[:][1])
        else:
            k_ss_max = max(steady_states[0])

        if model.hparams.max_T_test > 0:
            k_t_approx_check = trainer.model.test_results.loc[
                trainer.model.test_results["t"] == trainer.model.hparams.max_T_test - 1
            ].k_t_approx.mean()
        else:
            k_t_approx_check = trainer.model.test_results.k_tp1.mean()

        if model.hparams.TVC_test_treshold == 0:
            TVC_check_failed = math.nan
        elif not_number_type(k_t_approx_check) or (
            k_t_approx_check > model.hparams.TVC_test_treshold * k_ss_max
        ):
            TVC_check_failed = True
        else:
            TVC_check_failed = False

        if (
            early_stopping_check_failed
            in [
                False,
                math.nan,
            ]
            and test_loss_check_failed in [False, math.nan]
            and TVC_check_failed in [False, math.nan]
        ):
            retcode = 0
            convergence_description = "Success"
        elif early_stopping_check_failed == True:
            retcode = -1
            convergence_description = "Early stopping failure"
        elif test_loss_check_failed == True:
            retcode = -3
            convergence_description = "Test loss failure due to possible overfitting"
        elif TVC_check_failed == True:
            retcode = -4
            convergence_description = "TVC failure"
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
        if model.hparams.verbose:
            print(model.test_results)
        return
    else:  # almost no features enabled for other loggers. Could refactor later
        if model.hparams.save_test_results and trainer.log_dir is not None:
            model.test_results.to_csv(
                Path(trainer.log_dir) / "test_results.csv", index=False
            )


if __name__ == "__main__":
    cli = LightningCLI(
        DeterministicRecursiveGrowthModule,
        seed_everything_default=123,
        run=False,
        save_config_callback=None,
        parser_kwargs={"default_config_files": ["growth_recursive_defaults.yaml"]},
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
