import numpy as np
from yahpo_gym import local_config, BenchmarkSet
import ConfigSpace as CS
import csv
from datetime import datetime
import os
import matplotlib.pyplot as plt
import pandas as pd

local_config.init_config()
local_config.set_data_path("C:/Users/IljaKnis/Hyperband-UTS/yahpo_data")
log_file_path = "hyperband_run_log.csv"

R = 81  # Maximum resources (e.g., number of iterations)
eta = 2  # Proportion of configurations discarded in each round
s_max = int(np.floor(np.log(R) / np.log(eta)))
n_max = int(np.ceil((s_max + 1) * R / (eta ** s_max)))

bench = BenchmarkSet('rbv2_super')
config_space = bench.get_opt_space()

losses_sh = []
losses_uts = []
cumulative_sh_wins = 0
cumulative_uts_wins = 0
cumulative_ties = 0
total_runs = 0
sh_win_percentages = []
uts_win_percentages = []
win_data_file = "cumulative_wins.csv"
sh_wins = 0
uts_wins = 0
ties = 0

runs = 50 # Number of runs to compare the two methods

sh_losses_per_iter = []  # Collect losses per iteration for SH
uts_losses_per_iter = []  # Collect losses per iteration for UTS

def get_hyperparameter_configurations(n):
    return [config_space.sample_configuration() for _ in range(n)]

def load_previous_win_data():
    global cumulative_sh_wins, cumulative_uts_wins, total_runs, cumulative_ties
    if os.path.exists(win_data_file):
        data = pd.read_csv(win_data_file)
        cumulative_sh_wins = data['SH Wins'].iloc[-1]
        cumulative_uts_wins = data['UTS Wins'].iloc[-1]
        cumulative_ties = data['Ties'].iloc[-1]
        total_runs = data['Total Runs'].iloc[-1]
        print(f"Loaded previous win data: SH Wins={cumulative_sh_wins}, UTS Wins={cumulative_uts_wins}, Ties={cumulative_ties}, Total Runs={total_runs}")

def save_win_data():
    data = {
        "SH Wins": [cumulative_sh_wins],
        "UTS Wins": [cumulative_uts_wins],
        "Ties": [cumulative_ties],
        "Total Runs": [total_runs]
    }
    df = pd.DataFrame(data)
    if os.path.exists(win_data_file):
        df.to_csv(win_data_file, mode='a', header=False, index=False)
    else:
        df.to_csv(win_data_file, mode='w', header=True, index=False)

# Method to log the details
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

# Function to get the top k configurations
def top_k(configurations, losses, k):
    sorted_configs = [x for _, x in sorted(zip(losses, configurations), key=lambda pair: pair[0])]
    return sorted_configs[:k]


def run_then_return_val_loss(configurations, resources):
    losses = []
    for config in configurations:
        if isinstance(config, CS.Configuration):
            config_dict = config.get_dictionary()
        else:
            config_dict = config

        config_dict['epoch'] = resources

        try:
            result = bench.objective_function(config_dict, seed=42, logging=False, multithread=True)

            # Check result for key metrics
            if 'logloss' in result[0]:
                loss = result[0]['logloss']
            elif 'accuracy' in result[0]:
                loss = -result[0]['accuracy']
            else:
                raise KeyError(f"No suitable key found in result: {result[0]}")

            losses.append(loss)
        except Exception as e:
            print(f"Error in objective function for config {config_dict} with resources {resources}: {e}")
            losses.append(float('inf'))

    return losses

sample_configs = get_hyperparameter_configurations(n_max)

# Modified UTS Hyperband function with consistent logging
def hyperband_with_uts(R, eta, sample_configs):
    s_max = int(np.floor(np.log(R) / np.log(eta)))
    B = (s_max + 1) * R
    best_config = None
    best_loss = float('inf')

    for s in range(s_max, -1, -1):
        n_configs = int(np.ceil(B * eta ** (-s) / R))
        r = R * eta ** (-s)
        T = sample_configs[:n_configs]

        print(f"Stage {s}: Initial configurations count = {len(T)}")

        mu = np.zeros(len(T))
        sigma = np.ones(len(T))
        n_pulls = np.zeros(len(T))

        for i in range(s + 1):
            r_i = r * eta ** i

            # Select arm using Thompson Sampling (UTS)
            arm_indices = np.argmax(np.random.normal(mu, sigma))
            selected_config = T[arm_indices]
            L = run_then_return_val_loss([selected_config], r_i)

            if not L:
                print(f"No valid configurations at stage {s}, iteration {i}.")
                log_run("UTS", s, i, "N/A", "No valid configurations", "Skipped")
                continue

            uts_losses_per_iter.append(min(L))  # Store the loss per iteration

            reward = -L[0]
            n_pulls[arm_indices] += 1
            mu[arm_indices] = (mu[arm_indices] * (n_pulls[arm_indices] - 1) + reward) / n_pulls[arm_indices]
            sigma[arm_indices] = np.sqrt(1 / n_pulls[arm_indices])

            log_run("UTS", s, i, selected_config, L[0], f"Evaluated")

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
        T = sample_configs[:n]

        print(f"Stage {s}: Initial configurations count = {len(T)}")

        for config in T:
            log_run("Successive Halving", s, 0, config, "N/A", "Initial configuration")

        for i in range(s + 1):
            n_i = int(n * eta ** (-i))
            r_i = r * eta ** i

            # Unpack losses
            L = run_then_return_val_loss(T, r_i)

            if not L:
                print(f"No valid configurations at stage {s}, iteration {i}.")
                log_run("Successive Halving", s, i, "N/A", "No valid configurations", "Skipped")
                continue

            sh_losses_per_iter.append(min(L))  # Store the loss per iteration

            k = int(np.floor(n_i / eta))
            discarded_configs = T[k:]  # Configurations that will be discarded
            T = top_k(T, L, k)

            for config in discarded_configs:
                log_run("Successive Halving", s, i, config, "N/A", "Discarded")

            # Recalculate L after filtering T to ensure consistency
            L = run_then_return_val_loss(T, r_i)

            if not L:
                print(f"No valid configurations after top_k at stage {s}, iteration {i}.")
                log_run("Successive Halving", s, i, "N/A", "No valid configurations after top_k", "Skipped")
                continue

            for idx, config in enumerate(T):
                log_run("Successive Halving", s, i, config, L[idx], f"Remaining")

            # Compare using the minimum loss value from the list
            if min(L) < best_loss:
                best_loss = min(L)
                best_config = T[np.argmin(L)]

    return best_config, best_loss


# Compare performance of Hyperband with UTS vs. Successive Halving
best_overall_config = None
best_overall_loss = float('inf')
best_version = ""

for i in range(runs):
    load_previous_win_data()
    print(f"Run {i + 1} of {runs} with Successive Halving")
    best_config_sh, best_loss_sh = hyperband_with_successive_halving(R, eta, sample_configs)
    losses_sh.append(best_loss_sh)
    print("Best configuration found with Successive Halving: ", best_config_sh)
    print("Best loss with Successive Halving: ", best_loss_sh)
    log_run("Successive Halving", "Final", 0, best_config_sh, best_loss_sh, "Best configuration")

    if best_loss_sh < best_overall_loss:
        best_overall_loss = best_loss_sh
        best_overall_config = best_config_sh
        best_version = "Successive Halving"

    print(f"Run {i + 1} of {runs} with UTS")
    best_config_uts, best_loss_uts = hyperband_with_uts(R, eta, sample_configs)
    losses_uts.append(best_loss_uts)
    print("Best configuration found with UTS: ", best_config_uts)
    print("Best loss with UTS: ", best_loss_uts)
    log_run("UTS", "Final", 0, best_config_uts, best_loss_uts, "Best configuration")

    if best_loss_uts < best_overall_loss:
        best_overall_loss = best_loss_uts
        best_overall_config = best_config_uts
        best_version = "UTS"

    if best_loss_sh == best_loss_uts:
        best_overall_loss = best_loss_sh  # or best_loss_uts, as they are equal
        best_overall_config = best_config_sh  # or best_config_uts, as they are identical
        best_version = "Tie (SH and UTS)"

    best_loss_sh = losses_sh[i]
    best_loss_uts = losses_uts[i]

    # Determine which method wins and update cumulative wins
    if best_loss_sh < best_loss_uts:
        sh_wins += 1
        cumulative_sh_wins += 1
    elif best_loss_uts < best_loss_sh:
        uts_wins += 1
        cumulative_uts_wins += 1
    else:  # It's a tie
        ties += 1
        cumulative_ties += 1
        cumulative_sh_wins += 0.5
        cumulative_uts_wins += 0.5

    total_runs += 1

    # Calculate the cumulative win percentages
    sh_win_percentage = (cumulative_sh_wins / total_runs) * 100
    uts_win_percentage = (cumulative_uts_wins / total_runs) * 100

    # Store the percentages for plotting
    sh_win_percentages.append(sh_win_percentage)
    uts_win_percentages.append(uts_win_percentage)

    print(f"Run {total_runs}: SH Win Percentage = {sh_win_percentage:.2f}%, UTS Win Percentage = {uts_win_percentage:.2f}%")

    save_win_data()

print(f"\nBest overall configuration: {best_overall_config}")
print(f"Best overall loss: {best_overall_loss}")
print(f"Best performing version: {best_version}")
log_run(best_version, "Overall Best", 0, best_overall_config, best_overall_loss, "Overall Best Configuration")

#Calculate mean losses
mean_loss_sh = np.mean(losses_sh)
mean_loss_uts = np.mean(losses_uts)
print(f"Mean Loss for SH: {mean_loss_sh}")
print(f"Mean Loss for UTS: {mean_loss_uts}")

sh_win_percentage = (sh_wins / runs) * 100
uts_win_percentage = (uts_wins / runs) * 100
print(f"Successive Halving Wins: {sh_win_percentage}%")
print(f"Successive Halving absolute Wins: {sh_wins}")
print(f"UTS Wins: {uts_win_percentage}%")
print(f"UTS absolute Wins: {uts_wins}")
print(f"Absolute Ties: {ties}")

delta_losses = [losses_sh[i] - losses_uts[i] for i in range(runs)]
mean_delta_loss = np.mean(delta_losses)
print(f"Mean Delta Loss (SH - UTS): {mean_delta_loss}")

std_loss_sh = np.std(sh_losses_per_iter)
std_loss_uts = np.std(uts_losses_per_iter)
print(f"Standard Deviation of Loss for SH: {std_loss_sh}")
print(f"Standard Deviation of Loss for UTS: {std_loss_uts}")


plt.plot(range(1, len(sh_win_percentages) + 1), sh_win_percentages, label='SH Win Percentage', marker='o')
plt.plot(range(1, len(uts_win_percentages) + 1), uts_win_percentages, label='UTS Win Percentage', marker='o')
plt.xlabel('Run')
plt.ylabel('Win Percentage (%)')
plt.title(f'Cumulative Win Percentage Over {total_runs} Runs for SH and UTS')
plt.legend()
plt.grid(True)
plt.show()
