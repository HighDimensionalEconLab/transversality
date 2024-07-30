import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import wandb
from utilities import get_results_by_tag, plot_params

output_dir = "./figures"
plot_name = "growth_sequential_g_histogram"
output_path = output_dir + "/" + plot_name + ".pdf"
tag = "growth_sequential_g_positive_ensemble"
api = wandb.Api()
project = "highdimensionaleconlab/deep_learning_transversality"
params = plot_params((5, 4.5), fontsize=12)

parameters = get_results_by_tag(api, project, tag, get_config=True)
assert parameters.id.nunique() == 100

weights = np.array(parameters["rescale_weight"])

plt.rcParams.update(params)
plt.hist(
    (np.exp(weights) - 1),
    weights=parameters["rescale_weight"], bins = 15, 
    alpha=0.75,
)
plt.vlines(parameters["g"][0], ymin=0.0, ymax=0.4, colors="red", linestyles="dashed")
plt.xlim([weights.min()-0.001, weights.max()+0.001])
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.title(r"$g$ approximation: $\hat{g}$")
plt.xlabel(r"$\hat{g}$")
plt.tight_layout()

plt.savefig(output_path)
