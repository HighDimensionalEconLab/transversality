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
plot_name = "asset_pricing_sequential_g0"
output_path = output_dir + "/" + plot_name + ".pdf"
api = wandb.Api()
project = "highdimensionaleconlab/deep_learning_transversality"
params = plot_params((10, 4.5))

# One run with original grid. Comes from Figure 1 artifact and config

tag = "asset_pricing_sequential_g0_one_run"
results = get_results_by_tag(api, project, tag, get_test_results=True)
parameters = get_results_by_tag(api, project, tag, get_config=True)
assert results.id.nunique() == 1
max_T_test = parameters["test_T"].item()
max_T = parameters["train_t_max"].item()

p_f = results["p_f_t"]
t = np.array(range(0, max_T_test))


quantiles = {"low_quant": 0.1, "mid_quant": 0.5, "high_quant": 0.9}

# Original grid artifact
tag = "asset_pricing_sequential_g0_ensemble"
results = get_results_by_tag(api, project, tag, get_test_results=True)
assert results.id.nunique() == 100
quant_result_g0 = (
    results.groupby("t")[["p_rel_error", "p_t"]]
    .quantile(list(quantiles.values()))
    .unstack(level=-1)
    .reset_index()
)




# plotting

plt.rcParams.update(params)

ax_prices = plt.subplot(121)
plt.plot(t, p_f, "black", linestyle="dashed", label=r"$p_f(t)$")
plt.plot(t, quant_result_g0["p_t"][quantiles["mid_quant"]], "black", label=r"$\hat{p}(t)$: median")
plt.fill_between(t, quant_result_g0["p_t"][quantiles["low_quant"]], quant_result_g0["p_t"][quantiles["high_quant"]], facecolor="black",
alpha=0.15,
)
plt.axvline(x=max_T, color="0.0", linestyle="dashed")
ylim_min = 0.9 * np.amin(p_f)
ylim_max = 1.1 * np.amax(p_f)
plt.ylim([ylim_min, ylim_max])
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), prop={"size": params["font.size"]})
plt.title(r"Prices: $\hat{p}(t)$")
plt.xlabel(r"Time($t$)")
ax_prices.legend(fancybox=False, framealpha=1.0)
plt.legend(loc="lower left")

ax_errors = plt.subplot(122, sharex=ax_prices)
plt.plot(t, quant_result_g0["p_rel_error"][quantiles["mid_quant"]], "black", 
         label=r"$\varepsilon_p(t)$: median")
plt.fill_between(t, quant_result_g0["p_rel_error"][quantiles["low_quant"]],
                  quant_result_g0["p_rel_error"][quantiles["high_quant"]], facecolor="black", alpha=0.15,
)
plt.axvline(x=max_T, color="0.0", linestyle="dashed")
plt.title(r"Relative errors: $\varepsilon_p(t)$")
plt.xlabel(r"Time($t$)")
plt.legend()
plt.tight_layout()


plt.savefig(output_path)
