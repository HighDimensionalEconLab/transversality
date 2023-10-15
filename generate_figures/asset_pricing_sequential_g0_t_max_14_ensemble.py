import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import (
    zoomed_inset_axes,
    mark_inset,
    inset_axes,
)
import wandb
from utilities import get_results_by_tag, plot_params


output_dir = "./figures"
plot_name = "asset_pricing_sequential_g0_t_max_14_ensemble"
output_path = output_dir + "/" + plot_name + ".pdf"
api = wandb.Api()
project = "highdimensionaleconlab/deep_learning_transversality"
params = plot_params((10, 4.5))
tag = "asset_pricing_sequential_g0_t_max_14_ensemble"
quantiles = {"low_quant": 0.1, "mid_quant": 0.5, "high_quant": 0.9}

# artifact and config

results = get_results_by_tag(api, project, tag, get_test_results=True)
assert results.id.nunique() == 100
parameters = get_results_by_tag(api, project, tag, get_config=True, max_runs=1)
max_T_test = int(parameters["test_T"])
max_T = int(parameters["train_t_max"])
t = np.array(range(0, max_T_test))

quant_result_g0 = (
    results.groupby("t")["p_rel_error", "p_t"]
    .quantile(list(quantiles.values()))
    .unstack(level=-1)
    .reset_index()
)
quant_result_g0["p_f_t"] = results.groupby("t")["p_f_t"].mean()

# plotting

plt.rcParams.update(params)

ax_p = plt.subplot(121)
plt.plot(quant_result_g0["p_t"][quantiles["mid_quant"]], "black", label=r"$\hat{p}(t)$")
plt.plot(quant_result_g0["p_f_t"], "black", label=r"$p_f(t)$", linestyle="dashed")
plt.fill_between(
    t,
    quant_result_g0["p_t"][quantiles["low_quant"]],
    quant_result_g0["p_t"][quantiles["high_quant"]],
    facecolor="black",
    alpha=0.15,
)
plt.axvline(x=max_T, color="0.0", linestyle="dashed")
ylim_min = 0.7 * np.amin(
    np.minimum(quant_result_g0["p_t"][quantiles["mid_quant"]], quant_result_g0["p_f_t"])
)
ylim_max = 1.1 * np.amax(
    np.maximum(quant_result_g0["p_t"][quantiles["mid_quant"]], quant_result_g0["p_f_t"])
)
plt.ylim([ylim_min, ylim_max])
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), prop={"size": params["font.size"]})
plt.title(r"Prices: $\hat{p}(t)$")
plt.xlabel(r"Time($t$)")


ax_errors_p = plt.subplot(122, sharex=ax_p)
plt.plot(t, quant_result_g0["p_rel_error"][quantiles["mid_quant"]], "black")
plt.fill_between(
    t,
    quant_result_g0["p_rel_error"][quantiles["low_quant"]],
    quant_result_g0["p_rel_error"][quantiles["high_quant"]],
    facecolor="black",
    alpha=0.4,
)
plt.axvline(x=max_T, color="0.0", linestyle="dashed")
plt.title(r"Relative error: $\varepsilon_p(t)$")
plt.xlabel(r"Time($t$)")
plt.tight_layout()

plt.savefig(output_path)
