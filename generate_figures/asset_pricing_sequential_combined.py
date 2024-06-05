import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import (
    zoomed_inset_axes,
    mark_inset,
    inset_axes,
)
import wandb
from utilities import get_results_by_tag, plot_params

output_dir = "./figures"
plot_name = "asset_pricing_sequential_combined"
output_path = output_dir + "/" + plot_name + ".pdf"
api = wandb.Api()
project = "highdimensionaleconlab/deep_learning_transversality"
params = plot_params((10, 9))

# g = 0, results, for baseline

tag = "asset_pricing_sequential_g0_one_run"
results = get_results_by_tag(api, project, tag, get_test_results=True)
parameters = get_results_by_tag(api, project, tag, get_config=True)
assert results.id.nunique() == 1
max_T_test = parameters["test_T"].item()
max_T = parameters["train_t_max"].item()

p_f_0 = results["p_f_t"]
t = np.array(range(0, max_T_test))

# g= 0, results, for the approximate
quantiles = {"low_quant": 0.1, "mid_quant": 0.5, "high_quant": 0.9}
    ## Original grid artifact
tag_0 = "asset_pricing_sequential_g0_ensemble"
results_0 = get_results_by_tag(api, project, tag_0, get_test_results=True)
assert results_0.id.nunique() == 100
quant_result_g0 = (
    results_0.groupby("t")[["p_rel_error", "p_t"]]
    .quantile(list(quantiles.values()))
    .unstack(level=-1)
    .reset_index()
)
# g>0, results, for approximate
tag_pos = "asset_pricing_sequential_g_positive_ensemble"
results_pos = get_results_by_tag(api, project, tag_pos, get_test_results=True)
assert results_pos.id.nunique() == 100
parameters_pos = get_results_by_tag(api, project, tag_pos, get_config=True, max_runs=1)

quant_result_pos = (
    results_pos.groupby("t")[["p_rel_error", "p_t"]]
    .quantile(list(quantiles.values()))
    .unstack(level=-1)
    .reset_index()
)
quant_result_pos["p_f_t"] = results_pos.groupby("t")["p_f_t"].mean()


# Plottintg
plt.rcParams.update(params)

ax_prices_0 = plt.subplot(221)
plt.plot(t, p_f_0, "black", linestyle="dashed", label=r"$p_f(t)$")
plt.plot(t, quant_result_g0["p_t"][quantiles["mid_quant"]], "black", label=r"$\hat{p}(t)$: median")
plt.fill_between(t, quant_result_g0["p_t"][quantiles["low_quant"]], quant_result_g0["p_t"][quantiles["high_quant"]], facecolor="black",
alpha=0.25,
)
plt.axvline(x=max_T, color="0.0", linestyle="dashed")
ylim_min = 0.9 * np.amin(p_f_0)
ylim_max = 1.1 * np.amax(p_f_0)
plt.ylim([ylim_min, ylim_max])
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), prop={"size": params["font.size"]})
plt.title(r"Prices: $\hat{p}(t)$")
plt.xlabel(r"Time($t$)")
ax_prices_0.legend()
plt.legend(loc="lower left")

ax_errors_0 = plt.subplot(222)
plt.plot(t, quant_result_g0["p_rel_error"][quantiles["mid_quant"]], "black", 
         label=r"$\varepsilon_p(t)$: median")
plt.fill_between(t, quant_result_g0["p_rel_error"][quantiles["low_quant"]],
                  quant_result_g0["p_rel_error"][quantiles["high_quant"]], facecolor="black", alpha=0.25,
)
plt.axvline(x=max_T, color="0.0", linestyle="dashed")
plt.title(r"Relative errors: $\varepsilon_p(t)$")
plt.xlabel(r"Time($t$)")
plt.legend()

ax_prices_pos = plt.subplot(223)
plt.plot(quant_result_pos["p_t"][quantiles["mid_quant"]], "black", label="$\hat{p}(t)$:median")
plt.plot(quant_result_pos["p_f_t"], "black", linestyle="dashed", label="$p_f(t)$")
plt.fill_between(
    t,
    quant_result_pos["p_t"][quantiles["low_quant"]],
    quant_result_pos["p_t"][quantiles["high_quant"]],
    facecolor="gray",
    alpha=0.25,
)
plt.axvline(x=max_T, color="0.0", linestyle="dashed")
ylim_min = 0.9 * np.amin(
    np.amin(quant_result_pos["p_f_t"])
)
ylim_max = 1.1 * np.amax(
    np.amax(quant_result_pos["p_f_t"])
)
plt.ylim([ylim_min, ylim_max])
plt.legend()
plt.title(r"Price: $\hat{p}(t)$")
plt.xlabel(r"Time($t$)")

ax_errors_pos = plt.subplot(224)
plt.plot(t, quant_result_pos["p_rel_error"][quantiles["mid_quant"]], "black", label = r"$\varepsilon_p(t)$: median" )
plt.fill_between(
    t,
    quant_result_pos["p_rel_error"][quantiles["low_quant"]],
    quant_result_pos["p_rel_error"][quantiles["high_quant"]],
    facecolor="gray",
    alpha=0.25,
)
plt.axvline(x=max_T, color="0.0", linestyle="dashed")

plt.title(r"Relative error: $\varepsilon_p(t)$")
plt.xlabel(r"Time($t$)")
plt.legend()
plt.tight_layout()

plt.savefig(output_path)
