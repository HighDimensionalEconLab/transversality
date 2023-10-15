
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import wandb
from mpl_toolkits.axes_grid1.inset_locator import (
    zoomed_inset_axes,
    mark_inset,
    inset_axes,
)
from utilities import get_results_by_tag, plot_params

output_dir = "./figures"
plot_name = "growth_recursive_g0_multiple_steady_states_capital_rel_error_ensemble"
output_path = output_dir + "/" + plot_name + ".pdf"
tag = "growth_recursive_multiple_ss"
api = wandb.Api()
project = "highdimensionaleconlab/deep_learning_transversality"
params = plot_params((10, 9))


# artifact and config
results = get_results_by_tag(api, project, tag, get_test_results=True) 
assert results.id.nunique() == 400  # number of runs 
results = results[results["retcode"] == 0]
parameters = get_results_by_tag(api, project, tag, get_config=True, max_runs=1)
max_T_test = int(parameters["max_T_test"])
k_min = float(parameters["k_grid_min"])
k_max = float(parameters["k_grid_max"])



t_0 = results[results["t"] == 0]
k_1_0 = t_0[t_0["k_t_approx"] == 0.4000000059604645]["id"].unique()
k_2_0 = t_0[t_0["k_t_approx"] == 1.0]["id"].unique()
k_3_0 = t_0[t_0["k_t_approx"] == 3.299999952316284]["id"].unique()
k_4_0 = t_0[t_0["k_t_approx"] == 4.0]["id"].unique()

k_1_runs = results[results["id"].isin(k_1_0)]
k_2_runs = results[results["id"].isin(k_2_0)]
k_3_runs = results[results["id"].isin(k_3_0)]
k_4_runs = results[results["id"].isin(k_4_0)]


high_steady_state_capital_value = results["k_ss_high_norm"][1]
low_steady_state_capital_value = results["k_ss_low_norm"][1]
high_steady_state_consumption_value = results["c_ss_high_norm"][1]
low_steady_state_consumption_value = results["c_ss_low_norm"][1]

t = np.array(range(0, max_T_test))

quantiles = {"low_quant": 0.1, "mid_quant": 0.5, "high_quant": 0.9}


k_1_quant_result_g0 = (
    k_1_runs.groupby("t")[["c_rel_error", "c_t_approx", "k_rel_error", "k_t_approx"]]
    .quantile(list(quantiles.values()))
    .unstack(level=-1)
    .reset_index()
)
k_1_quant_result_g0["k_t_sol"] = k_1_runs.groupby("t")["k_t_sol"].median()
k_1_quant_result_g0["c_t_sol"] = k_1_runs.groupby("t")["c_t_sol"].median()

k_2_quant_result_g0 = (
    k_2_runs.groupby("t")[["c_rel_error", "c_t_approx", "k_rel_error", "k_t_approx"]]
    .quantile(list(quantiles.values()))
    .unstack(level=-1)
    .reset_index()
)
k_2_quant_result_g0["k_t_sol"] = k_2_runs.groupby("t")["k_t_sol"].median()
k_2_quant_result_g0["c_t_sol"] = k_2_runs.groupby("t")["c_t_sol"].median()


k_3_quant_result_g0 = (
    k_3_runs.groupby("t")[["c_rel_error", "c_t_approx", "k_rel_error", "k_t_approx"]]
    .quantile(list(quantiles.values()))
    .unstack(level=-1)
    .reset_index()
)
k_3_quant_result_g0["k_t_sol"] = k_3_runs.groupby("t")["k_t_sol"].median()
k_3_quant_result_g0["c_t_sol"] = k_3_runs.groupby("t")["c_t_sol"].median()

k_4_quant_result_g0 = (
    k_4_runs.groupby("t")[["c_rel_error", "c_t_approx", "k_rel_error", "k_t_approx"]]
    .quantile(list(quantiles.values()))
    .unstack(level=-1)
    .reset_index()
)
k_4_quant_result_g0["k_t_sol"] = k_3_runs.groupby("t")["k_t_sol"].median()
k_4_quant_result_g0["c_t_sol"] = k_3_runs.groupby("t")["c_t_sol"].median()

# Plotting starts 

plt.rcParams.update(params)

ax_k_err_1 = plt.subplot(221)

plt.plot(
    k_1_quant_result_g0["t"],
    k_1_quant_result_g0["k_rel_error"][quantiles["mid_quant"]],
    "black",
)
plt.fill_between(
    t,
    k_1_quant_result_g0["k_rel_error"][quantiles["low_quant"]],
    k_1_quant_result_g0["k_rel_error"][quantiles["high_quant"]],
    facecolor="black",
    alpha=0.3,
)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
#plt.legend(by_label.values(), by_label.keys(), prop={"size": params["font.size"]})
plt.title(r"Relative error, $\varepsilon_k(t)$ for $k_0 = 0.4$")
plt.xlabel(r"Time($t$)")


ax_k_err_2 = plt.subplot(222, sharey = ax_k_err_1)

plt.plot(
    k_2_quant_result_g0["t"],
    k_2_quant_result_g0["k_rel_error"][quantiles["mid_quant"]],
    "black",
)
plt.fill_between(
    t,
    k_2_quant_result_g0["k_rel_error"][quantiles["low_quant"]],
    k_2_quant_result_g0["k_rel_error"][quantiles["high_quant"]],
    facecolor="black",
    alpha=0.3,
)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
#plt.legend(by_label.values(), by_label.keys(), prop={"size": params["font.size"]})
plt.title(r"Relative error, $\varepsilon_k(t)$ for $k_0 = 1.0$")
plt.xlabel(r"Time($t$)")


ax_k_err_3 = plt.subplot(223)

plt.plot(
    k_3_quant_result_g0["t"],
    k_3_quant_result_g0["k_rel_error"][quantiles["mid_quant"]],
    "black",
)
plt.fill_between(
    t,
    k_3_quant_result_g0["k_rel_error"][quantiles["low_quant"]],
    k_3_quant_result_g0["k_rel_error"][quantiles["high_quant"]],
    facecolor="black",
    alpha=0.3,
)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
#plt.legend(by_label.values(), by_label.keys(), prop={"size": params["font.size"]})
plt.title(r"Relative error, $\varepsilon_k(t)$ for $k_0 = 3.3$")
plt.xlabel(r"Time($t$)")

ax_k_err_4 = plt.subplot(224, sharey = ax_k_err_3 )

plt.plot(
    k_4_quant_result_g0["t"],
    k_4_quant_result_g0["k_rel_error"][quantiles["mid_quant"]],
    "black",
)
plt.fill_between(
    t,
    k_4_quant_result_g0["k_rel_error"][quantiles["low_quant"]],
    k_4_quant_result_g0["k_rel_error"][quantiles["high_quant"]],
    facecolor="black",
    alpha=0.3,
)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
#plt.legend(by_label.values(), by_label.keys(), prop={"size": params["font.size"]})
plt.title(r"Relative error, $\varepsilon_k(t)$ for $k_0 = 4.0$")
plt.xlabel(r"Time($t$)")


plt.tight_layout()

plt.savefig(output_path)
