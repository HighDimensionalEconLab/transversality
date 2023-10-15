#!/bin/bash
# Function which takes a sweep name, creates the sweep, then creates a single agent before continuing
#!/bin/bash
# Define the project name
PROJECT_NAME="deep_learning_transversality" # swap out globally

# Define the run_sweep_and_agent function
run_sweep_and_agent () {
  # Set the SWEEP_NAME variable
  SWEEP_NAME="$1"
  
  # Run the wandb sweep command and store the output in a temporary file
  wandb sweep --project "$PROJECT_NAME" --name "$SWEEP_NAME" "replication_scripts/$SWEEP_NAME.yaml" >temp_output.txt 2>&1
  
  # Extract the sweep ID using awk
  SWEEP_ID=$(awk '/wandb agent/{ match($0, /wandb agent (.+)/, arr); print arr[1]; }' temp_output.txt)
  
  # Remove the temporary output file
  rm temp_output.txt
  
  # Run the wandb agent command
  wandb agent $SWEEP_ID
}

# Call experiments sequentially.  VERY SLOW given large number of experiments

# run asset pricing 
run_sweep_and_agent "asset_pricing_sequential_g0_one_run"
run_sweep_and_agent "asset_pricing_sequential_g_positive_ensemble"
run_sweep_and_agent "asset_pricing_sequential_g_positive_linear_scaling_ensemble"
run_sweep_and_agent "asset_pricing_sequential_g0_ensemble"
run_sweep_and_agent "asset_pricing_sequential_g0_grid_1_ensemble"
run_sweep_and_agent "asset_pricing_sequential_g0_grid_1_one_run"
run_sweep_and_agent "asset_pricing_sequential_g0_grid_2_ensemble"
run_sweep_and_agent "asset_pricing_sequential_g0_grid_2_one_run"
run_sweep_and_agent "asset_pricing_sequential_g0_t_max_9_ensemble"
run_sweep_and_agent "asset_pricing_sequential_g0_t_max_14_ensemble"

#run growth sequential 

run_sweep_and_agent "growth_sequential_g_positive_ensemble"
run_sweep_and_agent "growth_sequential_g_positive_linear_scaling_ensemble"
run_sweep_and_agent "growth_sequential_g_positive_one_run"
run_sweep_and_agent "growth_sequential_g0_ensemble"
run_sweep_and_agent "growth_sequential_g0_grid_1_ensemble"
run_sweep_and_agent "growth_sequential_g0_grid_1_one_run"
run_sweep_and_agent "growth_sequential_g0_grid_2_ensemble"
run_sweep_and_agent "growth_sequential_g0_grid_2_one_run"
run_sweep_and_agent "growth_sequential_g0_one_run"
run_sweep_and_agent "growth_sequential_g0_small_k_0_one_run"
run_sweep_and_agent "growth_sequential_g0_t_max_4_ensemble"
run_sweep_and_agent "growth_sequential_g0_t_max_4_one_run"
run_sweep_and_agent "growth_sequential_g0_t_max_9_ensemble"
run_sweep_and_agent "growth_sequential_g0_t_max_9_one_run"
run_sweep_and_agent "growth_sequential_multiple_steady_states_four_initial_k_0"
run_sweep_and_agent "growth_sequential_multiple_steady_states_var_initial_k_0"

# run growth recursive

run_sweep_and_agent "growth_recursive_g_positive_ensemble"
run_sweep_and_agent "growth_recursive_g_positive_var_initial_k_0_one_run"
run_sweep_and_agent "growth_recursive_g0_ensemble"
run_sweep_and_agent "growth_recursive_g0_far_from_steady_state_ensemble"
run_sweep_and_agent "growth_recursive_g0_grid_ensemble_ADAM"
run_sweep_and_agent "growth_recursive_g0_grid_ensemble"
run_sweep_and_agent "growth_recursive_g0_using_c_grid_ensemble_ADAM"
run_sweep_and_agent "growth_recursive_g0_using_c_grid_ensemble"
run_sweep_and_agent "growth_recursive_g0_using_c_reg_grid_ensemble_ADAM"
run_sweep_and_agent "growth_recursive_g0_var_initial_k_0_one_run"
run_sweep_and_agent "growth_recursive_multiple_steady_states"



