import time

import numpy as np
from yahpo_gym import local_config, BenchmarkSet
import ConfigSpace as CS
import csv
from datetime import datetime
import os

local_config.init_config()
local_config.set_data_path("C:/Users/IljaKnis/Hyperband-UTS/yahpo_data")
log_file_path = "hyperband_run_log.csv"

# Define Hyperband parameters
R = 81  # Maximum resources (e.g., number of iterations)
eta = 2  # Proportion of configurations discarded in each round

# Load YAHPO Gym benchmark set
bench = BenchmarkSet('rbv2_super')
config_space = bench.get_opt_space()


# Function to get hyperparameter configurations
def get_hyperparameter_configurations(n):
    return [config_space.sample_configuration() for _ in range(n)]

# Dynamically sample configurations for both methods in each stage
def sample_configs(stage, n):
    return get_hyperparameter_configurations(n)

# Function to log the details
def log_run(version, stage, iteration, config, evaluation, status):
    if not os.path.exists(log_file_path):
        with open(log_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Date", "Time", "Version", "Stage", "Iteration", "Status", "Configuration", "Evaluation"])

    current_time = datetime.now()
    date_str = current_time.strftime("%Y-%m-%d")
    time_str = current_time.strftime("%H:%M:%S")

    with open(log_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([date_str, time_str, version, stage, iteration, status, config, evaluation])


def run_then_return_val_loss(configurations, resources):
    losses = []
    for config in configurations:
        if isinstance(config, CS.Configuration):
            config_dict = config.get_dictionary()
        else:
            config_dict = config

        # TODO: Check if 'epoch' is still applicable
        config_dict['epoch'] = resources  # Assuming epochs are still applicable

        try:
            result = bench.objective_function(config_dict, seed=42, logging=False, multithread=True)

            # Print available keys for debugging purposes
            # print(f"Available keys in result: {result[0].keys()}")

            # Use 'logloss' as loss parameter
            if 'logloss' in result[0]:
                loss = result[0]['logloss']
            elif 'accuracy' in result[0]:
                loss = -result[0]['accuracy']  # Assuming you want to minimize loss
            else:
                raise KeyError(f"No suitable key found in result: {result[0]}")

            losses.append(loss)
        except Exception as e:
            print(f"Error in objective function for config {config_dict} with resources {resources}: {e}")
            losses.append(float('inf'))
    return losses


# Function to get the top k configurations
def top_k(configurations, losses, k):
    sorted_configs = [x for _, x in sorted(zip(losses, configurations), key=lambda pair: pair[0])]
    return sorted_configs[:k]


# TODO: expand Logging for benchmarks
# TODO: graphical display of results (plots)
# Hyperband implementation with Universal Thompson Sampling (UTS)
#def hyperband(R, eta):
#    s_max = int(np.floor(np.log(R) / np.log(eta)))
#    B = (s_max + 1) * R
#    best_config = None
#    best_loss = float('inf')
#
#    for s in range(s_max, -1, -1):
#        n = int(np.ceil(B * eta ** (-s) / R))
#        r = R * eta ** (-s)
#        T = get_hyperparameter_configurations(n)
#
#        print(f"Stage {s}: Initial configurations count = {len(T)}")
#
#        # Log all initial configurations before the stage begins
#        for config in T:
#            log_run("UTS", s, 0, config, "N/A", "Initial configuration")
#
#        # TODO: Replace inner loop with proper UTS implementation (probabilistic evaluation)
#        for i in range(s + 1):
#            n_i = int(n * eta ** (-i))
#            r_i = r * eta ** i
#            L = run_then_return_val_loss(T, r_i)
#
#            if not L:
#                print(f"No valid configurations at stage {s}, iteration {i}.")
#                log_run("UTS", s, i, "N/A", "No valid configurations", "Skipped")
#                continue
#
#            k = int(np.floor(n_i / eta))
#            discarded_configs = T[k:]  # Configurations that will be discarded
#            T = top_k(T, L, k)
#
#            # Log discarded configurations
#            for config in discarded_configs:
#                log_run("UTS", s, i, config, "N/A", "Discarded")
#
#            # Recalculate L after filtering T to ensure consistency
#            L = run_then_return_val_loss(T, r_i)
#
#            if not L:
#                print(f"No valid configurations after top_k at stage {s}, iteration {i}.")
#                log_run("UTS", s, i, "N/A", "No valid configurations after top_k", "Skipped")
#                continue
#
#            # Log remaining configurations and the current best loss
#            for idx, config in enumerate(T):
#                log_run("UTS", s, i, config, L[idx], "Remaining")
#
#            # Find the best configuration and update if necessary
#            if min(L) < best_loss:
#                best_loss = min(L)
#                best_config = T[np.argmin(L)]
#
#    return best_config, best_loss

#def hyperband(R, eta):
#    s_max = int(np.floor(np.log(R) / np.log(eta)))
#    B = (s_max + 1) * R
#    best_config = None
#    best_loss = float('inf')
#
#    # Initialize Thompson Sampling parameters
#    K = 5  # Number of arms (configurations)
#    mu = np.zeros(K)  # Means of reward distributions
#    sigma = np.ones(K)  # Standard deviations (initially 1 for all arms)
#    n = np.zeros(K)  # Number of times each arm is pulled
#
#    for s in range(s_max, -1, -1):
#        n_configs = int(np.ceil(B * eta ** (-s) / R))
#        r = R * eta ** (-s)
#        T = get_hyperparameter_configurations(n_configs)
#
#        print(f"Stage {s}: Initial configurations count = {len(T)}")
#
#        # Initialize arm beliefs for UTS
#        mu = np.zeros(len(T))  # Initialize means
#        sigma = np.ones(len(T))  # Initialize standard deviations
#        n_pulls = np.zeros(len(T))  # Number of pulls for each configuration
#
#        for i in range(s + 1):
#            n_i = int(n_configs * eta ** (-i))
#            r_i = r * eta ** i
#
#            # Select arm using Thompson Sampling (UTS)
#            arm_indices = np.argmax(np.random.normal(mu, sigma))  # Select arm with highest reward probability
#            selected_config = T[arm_indices]
#            L = run_then_return_val_loss([selected_config], r_i)  # Evaluate only selected configuration
#
#            if not L:
#                print(f"No valid configurations at stage {s}, iteration {i}.")
#                log_run("UTS", s, i, "N/A", "No valid configurations", "Skipped")
#                continue
#
#            # Update beliefs based on reward (mean and variance)
#            reward = -L[0]  # Use negative loss as reward
#            n_pulls[arm_indices] += 1
#            mu[arm_indices] = (mu[arm_indices] * (n_pulls[arm_indices] - 1) + reward) / n_pulls[arm_indices]
#            sigma[arm_indices] = np.sqrt(1 / n_pulls[arm_indices])  # Decrease uncertainty as arm is pulled more
#
#            # Log selected configurations
#            log_run("UTS", s, i, selected_config, L[0], "Evaluated")
#
#            if min(L) < best_loss:
#                best_loss = min(L)
#                best_config = selected_config
#
#    return best_config, best_loss

# Modified UTS Hyperband function with consistent logging and runtime tracking
def hyperband(R, eta, sample_configs):
    s_max = int(np.floor(np.log(R) / np.log(eta)))
    B = (s_max + 1) * R
    best_config = None
    best_loss = float('inf')

    # Initialize Thompson Sampling parameters
    K = 5  # Number of arms (configurations)
    mu = np.zeros(K)  # Means of reward distributions
    sigma = np.ones(K)  # Standard deviations (initially 1 for all arms)
    n = np.zeros(K)  # Number of times each arm is pulled

    for s in range(s_max, -1, -1):
        n_configs = int(np.ceil(B * eta ** (-s) / R))
        r = R * eta ** (-s)
        T = sample_configs(s,n_configs)

        print(f"Stage {s}: Initial configurations count = {len(T)}")

        # Initialize arm beliefs for UTS
        mu = np.zeros(len(T))  # Initialize means
        sigma = np.ones(len(T))  # Initialize standard deviations
        n_pulls = np.zeros(len(T))  # Number of pulls for each configuration

        for i in range(s + 1):
            n_i = int(n_configs * eta ** (-i))
            r_i = r * eta ** i

            # Start timing here
            start_time = time.time()

            # Select arm using Thompson Sampling (UTS)
            arm_indices = np.argmax(np.random.normal(mu, sigma))  # Select arm with the highest reward probability
            selected_config = T[arm_indices]
            L = run_then_return_val_loss([selected_config], r_i)  # Evaluate only selected configuration

            if not L:
                print(f"No valid configurations at stage {s}, iteration {i}.")
                log_run("UTS", s, i, "N/A", "No valid configurations", "Skipped")
                continue

            # Calculate runtime for the selected configuration
            runtime = time.time() - start_time

            # Update beliefs based on reward (mean and variance)
            reward = -L[0]  # Use negative loss as reward
            n_pulls[arm_indices] += 1
            mu[arm_indices] = (mu[arm_indices] * (n_pulls[arm_indices] - 1) + reward) / n_pulls[arm_indices]
            sigma[arm_indices] = np.sqrt(1 / n_pulls[arm_indices])  # Decrease uncertainty as arm is pulled more

            # Log selected configurations with runtime
            log_run("UTS", s, i, selected_config, L[0], f"Evaluated (Runtime: {runtime:.2f}s)")

            if min(L) < best_loss:
                best_loss = min(L)
                best_config = selected_config

    return best_config, best_loss


# Original Hyperband implementation with Successive Halving
def hyperband_with_successive_halving(R, eta, sample_configs):
    s_max = int(np.floor(np.log(R) / np.log(eta)))
    B = (s_max + 1) * R
    best_config = None
    best_loss = float('inf')

    for s in range(s_max, -1, -1):
        n = int(np.ceil(B * eta ** (-s) / R))
        r = R * eta ** (-s)
        T = sample_configs(s,n)

        print(f"Stage {s}: Initial configurations count = {len(T)}")

        for config in T:
            log_run("Successive Halving", s, 0, config, "N/A", "Initial configuration")

        for i in range(s + 1):
            n_i = int(n * eta ** (-i))
            r_i = r * eta ** i
            L = run_then_return_val_loss(T, r_i)

            if not L:
                print(f"No valid configurations at stage {s}, iteration {i}.")
                log_run("Successive Halving", s, i, "N/A", "No valid configurations", "Skipped")
                continue

            k = int(np.floor(n_i / eta))
            discarded_configs = T[k:]  # Configurations that will be discarded
            T = top_k(T, L, k)

            for config in discarded_configs:
                log_run("Successive Halving", s, i, config, "N/A", "Discarded")

            L = run_then_return_val_loss(T, r_i)

            if not L:
                print(f"No valid configurations after top_k at stage {s}, iteration {i}.")
                log_run("Successive Halving", s, i, "N/A", "No valid configurations after top_k", "Skipped")
                continue

            for idx, config in enumerate(T):
                log_run("Successive Halving", s, i, config, L[idx], "Remaining")

            if min(L) < best_loss:
                best_loss = min(L)
                best_config = T[np.argmin(L)]

    return best_config, best_loss


# Compare performance of Hyperband with UTS vs. Successive Halving
best_overall_config = None
best_overall_loss = float('inf')
best_version = ""


for i in range(10):
    print(f"Run {i + 1} of 10 with Successive Halving")
    best_config_sh, best_loss_sh = hyperband_with_successive_halving(R, eta, sample_configs)
    print("Best configuration found with Successive Halving: ", best_config_sh)
    print("Best loss with Successive Halving: ", best_loss_sh)
    log_run("Successive Halving", "Final", 0, best_config_sh, best_loss_sh, "Best configuration")

    if best_loss_sh < best_overall_loss:
        best_overall_loss = best_loss_sh
        best_overall_config = best_config_sh
        best_version = "Successive Halving"

    print(f"Run {i + 1} of 10 with UTS")
    best_config_uts, best_loss_uts = hyperband(R, eta, sample_configs)
    print("Best configuration found with UTS: ", best_config_uts)
    print("Best loss with UTS: ", best_loss_uts)
    log_run("UTS", "Final", 0, best_config_uts, best_loss_uts, "Best configuration")

    if best_loss_uts < best_overall_loss:
        best_overall_loss = best_loss_uts
        best_overall_config = best_config_uts
        best_version = "UTS"

# Log the best overall configuration and version
print(f"\nBest overall configuration: {best_overall_config}")
print(f"Best overall loss: {best_overall_loss}")
print(f"Best performing version: {best_version}")
log_run(best_version, "Overall Best", 0, best_overall_config, best_overall_loss, "Overall Best Configuration")


# TODO: Plot with Pandas and Matplotlib
# TODO: Extract Sample size
# TODO: Expand different BenchmarkSets (optional) - requires additional configuration