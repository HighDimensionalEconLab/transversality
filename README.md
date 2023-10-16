# Deep learning transversality

Source for "Spooky Boundaries at a Distance: Exploring Transversality and Stability with Deep Learning"

## Installing

### Quick Installation Instructions
Within a python environment, clone this repository with git and execute `pip install -r requirements.txt`.

See more complete instructions below in the [detailed installation](#detailed-installation-instructions) section.

## Deterministic Sequential LSS
This variation of the model solves the LSS model for a `p(t)`. This `p : R \to R` can be evaluated between time-periods but probably shouldn't be.

See the file [asset_pricing_sequential_defaults.yaml](asset_pricing_sequential_defaults.yaml) for the default values.

### Jupyter Notebook for Exploration

You can load the Jupyter notebook [asset_pricing_sequential.ipynb](demo_notebooks/asset_pricing_sequential_demo.ipynb) directly in VS Code or on the command line with `jupyter lab` run in the local directory. This notebook provides a simple example of training the asset pricing sequential model and provides utilities to examine the output without using the command line commands.

### CLI Usage

You can run the baseline parameters with
```bash
python asset_pricing_sequential.py
```

If you want to override the defaults, here are some examples:

```bash
python asset_pricing_sequential.py --seed_everything 101 --trainer.max_epochs=10000
python asset_pricing_sequential.py --trainer.max_time 00:00:01:00
```

To modify the optimizer and/or learning rate schedulers, you can do things like:

```bash
python asset_pricing_sequential.py --trainer.max_epochs=500 --optimizer=torch.optim.Adam --optimizer.lr=0.001 --optimizer.weight_decay=0.00001 --trainer.callbacks.patience=500
python asset_pricing_sequential.py --trainer.max_epochs=1 --optimizer=LBFGS --optimizer.lr=1.0 
```

Modifying the callbacks is similar:
```bash
python asset_pricing_sequential.py  --trainer.callbacks=TQDMProgressBar --trainer.callbacks.refresh_rate=0
python asset_pricing_sequential.py  --trainer.callbacks=LearningRateMonitor --trainer.callbacks.logging_interval=epoch --trainer.callbacks.log_momentum=false
```

To modify the ML model, you can pass options such as:
```bash
python asset_pricing_sequential.py  --model.ml_model.init_args.layers=4
python asset_pricing_sequential.py  --model.ml_model.init_args.hidden_dim=122 --model.ml_model.init_args.layers=6
```

You can also modify the activators, which will use the default parameters of their respective classes:
```bash
python asset_pricing_sequential.py --model.ml_model.init_args.activator torch.nn.Softplus
python asset_pricing_sequential.py --model.ml_model.init_args.last_activator torch.nn.Tanh
```

To change economic variables such as the dividend value c, you can try:
```bash
python asset_pricing_sequential.py --model.c=0.01
```

To see all of the available options, run:
 ```bash
 python asset_pricing_sequential.py --help
 ```

The output of the file will be in something like ./wandb/offline-run-.... You can also view logs online by running:
 ```bash
 wandb sync .\wandb\offline-run-...
 ```

## Neoclassical growth model 
Deterministic Sequential LSS can also be used to solve the neoclassical growth model: solving the LSS model for a k(t). This k : R \to R can be evaluated between time-periods, but probably shouldn't be.

### Jupyter Notebook for Exploration

You can load the Jupyter notebook  [growth_sequential.ipynb](demo_notebooks/growth_sequential_demo.ipynb) directly in VS Code or on the command line with `jupyter lab` run in the local directory. This notebook provides a simple example of training the asset pricing sequential model and provides utilities to examine the output without using the command line commands.

### CLI Usage
You can run with the baseline parameters using:
```bash
python growth_sequential.py
```

You can pass parameters and modify the optimizer or ML model in a similar way as for asset_pricing_sequential.py.

You can also modify the model parameters. For example, the discount factor `beta`:
```bash
python growth_sequential.py --model.beta=0.89 
```

Finally, it's possible to change the starting capital level k_0:

```bash
python growth_sequential.py --model.k_0=0.7
```

Additionally, you can run the model with a "kink" in the production function, which has two steady states. To do so, you need to specify the parameters of the production function with "kink": a, b_1, b_2. We recommend running the model with a larger number of epochs and the ADAM optimizer like this:

```bash
python growth_sequential.py  --model.a=0.5 --model.b_1=3.0 --model.b_2=2.5 --trainer.max_epochs=5000 --optimizer=torch.optim.Adam --optimizer.lr=0.001  
```

## Deterministic Recursive LSS
This instead solves the neoclassical growth model for k'(z,k) where the map k' : R^2 -> R is R^2 because of the stacking z (TFP level) and k (current capital).

See the file [growth_recursive_defaults.yaml](growth_recursive_defaults.yaml) for the default values.

### Jupyter Notebook for Exploration

You can load the Jupyter notebook  [growth_recursive.ipynb](demo_notebooks/growth_recursive_demo.ipynb) directly in VS Code or on the command line with `jupyter lab` run in the local directory. This notebook provides a simple example of training the asset pricing sequential model and provides utilities to examine the output without using the command line commands.


### CLI Usage 

You can run the baseline parameters using:
```bash
python growth_recursive.py
```

All optimizer and ML options are consistent with growth_sequential. Additionally, you can modify the grid structure. For instance, to increase the number of grid points for the capital grid, you can try:
```bash
python growth_recursive.py --model.k_sim_grid_points=24
```

Same as in the sequential case, you can also run the model with two steady states. However, we recommend running this model with special overlapping capital grids and a separate validation set and RADAM optimizer. We recommend running something like:

```bash
python growth_recursive.py --lr_scheduler.class_path=torch.optim.lr_scheduler.StepLR --lr_scheduler.gamma=0.95 --lr_scheduler.step_size=200 --model.a=0.5 --model.b_1=3 --model.b_2=2.5 --model.batch_size=0 --model.k_0=3.3 --model.k_grid_max=25 --model.k_grid_max_2=1.5 --model.k_grid_min=0.4 --model.k_grid_min_2=0.45 --model.k_sim_grid_points=1024 --model.max_T_test=50 --model.ml_model.activator.class_path=torch.nn.ReLU  --model.test_loss_success_threshold=0.0001 --model.val_max_1=4.2 --model.val_max_2=1.2 --model.val_min_1=3.1 --model.val_min_2=0.5 --model.val_sim_grid_points=200 --model.vfi_parameters.interpolation_kind=linear --model.vfi_parameters.k_grid_size=1000 --optimizer.class_path=torch.optim.RAdam --optimizer.lr=0.001 --trainer.callbacks.monitor=val_loss --trainer.callbacks.stopping_threshold=5e-06 --trainer.limit_val_batches=5000 --trainer.max_epochs=5000 
```

For solving the model with two steady states, please give special attention to the `retcode` values. 
# Replication scripts

## Weights and Biases
One tool for testing the methods with different seeds is [Weights and Biases](https://wandb.ai/). This is a free service for academic use. It provides a dashboard to track experiments and a way to run hyperparameter optimization sweeps.

To use it, first create an account with Weights and Biases, then, assuming you have installed the packages above, ensure you have logged in,

```bash
wandb login 
```

### How to run the hyperparameter sweeps and replication scripts
Under `hpo_sweeps`, you can see the hyperparameter sweep files. If you want to start them, run

```bash
wandb sweep replication_scripts/asset_pricing_sequential_g_positive_ensemble.yaml
```
This will create a new sweep on the server. It will give you a URL to the sweep, which you can open in a browser. You can also see the sweep in your W&B dashboard. You will need the returned ID as well.

This doesn't create any "agents". To do that, take the `<sweep_id>` that was returned and run it
```
wandb agent <sweep_id>
```

### Example Results
See [W&B replication script for asset_pricing_sequential ](https://wandb.ai/highdimensionaleconlab/deep_learning_transversality/sweeps/0fqhkwp7?workspace=user-) for an example. You can compare the capital and consumption errors depending on the seed.



# Detailed Installation Instructions
For users with less experience using python, conda, and VS Code, the following provides more details.

1. Ensure you have installed Python. For example, using [Anaconda](https://www.anaconda.com/products/individual)
2. Recommended but not required: Install [VS Code](https://code.visualstudio.com/) along with its [Python Extension](https://code.visualstudio.com/docs/languages/python)
3. Clone this repository
   - Recommended: With VS Code, go `<Shift-Control-P>` to open up the commandbar, then choose `Git Clone`, and use the URL `https://github.com/HighDimensionalEconLab/transversality.git`. That will give you a full environment to work with.
   - Alternatively, you can clone it with git installed `git clone https://github.com/HighDimensionalEconLab/transversality.git`
4. (Optional) create a conda [virtual environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
    ```bash
    conda create -n transversality python=3.9
    conda activate transversality
    ```
    - Python 3.10 is also broadly supported, but PyTorch doesn't fully support Python 3.11 yet. See Troubleshooting below if Python 3.10 has issues.

5. (Optional) In VS Code, you can then do `<Shift-Control-P>` to open up the commandbar, then choose `> Python: Select Interpreter`, and choose the one in the `transversality` environment. Future `> Python: Terminal` commands then automatically activate it.
    - If you are in VS Code, opening a python terminal with  `<Shift-Control-P>` then  `> Python: Terminal` and other terminals should automatically activate the environment and start in the correct location.

6. Install dependencies. With a terminal in that cloned folder (after, optionally, activating an environment as discussed above).
    ```bash
    pip install -r requirements.txt
    ```

**Troubleshooting:**

   - If you are having trouble installing packages on Windows with Python 3.10, then either downgrade to 3.9 or see [here](https://stackoverflow.com/questions/64261546/how-to-solve-error-microsoft-visual-c-14-0-or-greater-is-required-when-inst). To summarize those steps:
     - Download https://visualstudio.microsoft.com/visual-cpp-build-tools/
     - Local to that folder in a terminal, run `vs_buildtools.exe --norestart --passive --downloadThenInstall --includeRecommended --add Microsoft.VisualStudio.Workload.NativeDesktop --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Workload.MSBuildTools
`
   - If PyTorch is not working after the initial installation, consider [installing manually](https://pytorch.org/get-started/locally/#start-locally) with `conda install pytorch cpuonly -c pytorch ` or something similar, and then retrying the dependencies installation. GPUs are not required for these experiments.   If you get compatibility clashes between packages with the `pip install -r requirements.txt` then we recommend using a virtual environment with conda, as described above.

# Hyperparameter Tuning
Deep learning methods use a lot of tuning hyperparameters. A variety of tooling for ML and deep learning is there to help, mostly under the category of "ML DevOps". This includes tools for hyperparameter optimization, model versioning, managing results,  model deployment, and running on clusters/clouds. [Weights and Biases](https://wandb.ai/) can also be used for hyperparameter tuning. It provides useful tools such as visualizations of hyperparameter correlation and evaluations. 
 