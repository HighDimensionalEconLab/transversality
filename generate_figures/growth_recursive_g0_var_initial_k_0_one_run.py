import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import wandb
from mpl_toolkits.axes_grid1.inset_locator import (
    zoomed_inset_axes,
    mark_inset,
    inset_axes,
)
import matplotlib as mpl
from utilities import get_results_by_tag, plot_params

output_dir = "./figures"
plot_name = "growth_recursive_g0_var_initial_k_0_one_run"
output_path = output_dir + "/" + plot_name + ".pdf"
tag = "growth_recursive_g0_var_initial_k_0_one_run"
api = wandb.Api()
project = "highdimensionaleconlab/deep_learning_transversality"
params = plot_params((10, 4.5), fontsize=12)
#artifact and config 
results = get_results_by_tag(api, project, tag, get_test_results=True)
assert results.id.nunique() == 3 #number of runs
parameters = get_results_by_tag(api, project, tag, get_config=True, max_runs = 1)
max_T_test = int(parameters["max_T_test"])
k_min = float(parameters["k_grid_min"])
k_max = float(parameters["k_grid_max"])

t = np.array(range(0, max_T_test))

id_results= {}
ids = []

for id in results['id'].unique():
    quant_result_g0 = (
    results[results['id'] == id][['t',"c_rel_error", "c_t_approx", "k_rel_error", "k_t_approx"]]
)
    quant_result_g0["k_t_sol"] = results[results['id'] == id]["k_t_sol"]
    quant_result_g0["c_t_sol"] = results[results['id'] == id]["c_t_sol"]

    id_results[id] = quant_result_g0
    ids.append(id)
    


plt.rcParams.update(params)
mpl.rcParams['text.usetex'] = True


ax_k = plt.subplot(121)
for id in results['id'].unique():
    plt.plot(t, id_results[id]["k_t_approx"], "black", label=r"$\hat{k}(t)$")
    plt.plot(t, id_results[id]["k_t_sol"], "black", label=r"$k(t)$", linestyle="dashed")
 

plt.fill_between(t, k_min, k_max, facecolor="0.9", label=r"$\mathcal{X}_{train}$")
ylim_min = 0.35#0.7 * np.amin(np.minimum(k_min, quant_result_g0["k_t_sol"]))
ylim_max = 4.2 #1.1 * np.amax(np.maximum(k_max, quant_result_g0["k_t_sol"]))
plt.ylim([ylim_min, ylim_max])
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), prop={"size": params['font.size']})
plt.title(r"Capital: $\hat{k}(t)$")
plt.xlabel(r"Time($t$)")


ax_c = plt.subplot(122, sharex=ax_k)
for id in results['id'].unique(): 
    plt.plot(t, id_results[id]["c_t_approx"], "blue", label=r"$\hat{c}(t)$")
    plt.plot(t, id_results[id]["c_t_sol"], "blue", label=r"$c(t)$", linestyle="dashed")

ylim_min = 0.45 #0.7 * np.amin(np.minimum(id_results[id]["c_t_approx"], id_results[id]["c_t_sol"]))
ylim_max = 1.7 #1.1 * np.amax(np.maximum(id_results[id]["c_t_approx"], id_results[id]["c_t_sol"]))
plt.ylim([ylim_min, ylim_max])
handlesc, labelsc = plt.gca().get_legend_handles_labels()
by_labelc = dict(zip(labelsc, handlesc))
plt.legend(by_labelc.values(), by_labelc.keys(), prop={"size":params['font.size']})
plt.title(r"Consumption: $\hat{c}(t)$")
plt.xlabel(r"Time($t$)")

id_0 = ids[0]
id_1 = ids[1]
# Zoom in for top left
time_window = [4, 10]
ave_value = 0.5 * (id_results[id_0]["k_t_sol"][time_window[0]] + id_results[id_0]["k_t_sol"][time_window[1]])
window_width = 0.03 * ave_value
matplotlib.rcParams.update({"ytick.labelsize": 8})

axins = zoomed_inset_axes(ax_k, 4, loc="lower right")
axins.plot(t,id_results[id_0]["k_t_approx"], "black")
axins.plot(t,id_results[id_0]["k_t_sol"], "black", linestyle="dashed")
axins.plot(t,id_results[id_1]["k_t_approx"], "black")
axins.plot(t,id_results[id_1]["k_t_sol"], "black", linestyle="dashed")

x1, x2, y1, y2 = (
    time_window[0],
    time_window[1],
    ave_value - window_width,
    ave_value + window_width,
)
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.xaxis.tick_top()
plt.xticks(fontsize=8, visible=False)
plt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
mark_inset(ax_k, axins, loc1=1, loc2=3, linewidth="0.7", ls="--", ec="0.5")


# Zoom in for bottom left
time_window = [4, 10]
ave_value = 0.5 * (id_results[id_0]["c_t_sol"][time_window[0]] + id_results[id_0]["c_t_sol"][time_window[1]])
window_width = 0.0275 * ave_value
matplotlib.rcParams.update({"ytick.labelsize": 8})

axins = zoomed_inset_axes(ax_c, 4, loc="lower right")
axins.plot(t,id_results[id_0]["c_t_approx"], "blue")
axins.plot(t,id_results[id_0]["c_t_sol"], "blue", linestyle="dashed")
axins.plot(t,id_results[id_1]["c_t_approx"], "blue")
axins.plot(t,id_results[id_1]["c_t_sol"], "blue", linestyle="dashed")
x1, x2, y1, y2 = (
    time_window[0],
    time_window[1],
    ave_value - window_width,
    ave_value + window_width,
)
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.xaxis.tick_top()
plt.xticks(fontsize=8, visible=False)
plt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
plt.yticks(fontsize=8)
mark_inset(ax_c, axins, loc1=1, loc2=3, linewidth="0.7", ls="--", ec="0.5")
plt.savefig(output_path)
