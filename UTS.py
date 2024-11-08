import numpy as np

# Number of arms or configurations
K = 5

# Initialize the priors (mean and variance for each arm)
mu = np.zeros(K)  # Initialize mean to 0 for each arm
sigma = np.ones(K)  # Initialize variance to 1 for each arm
n_pulls = np.zeros(K)  # Number of pulls for each arm

def initialize_arms(K):
    # Initialize the mean, variance, and count of pulls for each arm
    mu = np.zeros(K)
    sigma = np.ones(K)
    n_pulls = np.zeros(K)
    return mu, sigma, n_pulls

def pull_arm(arm):
    # Simulate pulling an arm by sampling from the Gaussian posterior distribution
    sample = np.random.normal(mu[arm], sigma[arm])
    return sample

def update_posterior(arm, reward):
    # Update the mean and variance of the arm based on observed reward
    n_pulls[arm] += 1
    # Using a simple Bayesian update rule here (normal-inverse-gamma conjugate prior)
    # More complex update rules may apply depending on the reward distribution
    mu[arm] = (mu[arm] * (n_pulls[arm] - 1) + reward) / n_pulls[arm]
    sigma[arm] = 1 / n_pulls[arm]  # Simplistic update for demonstration


def run_uts(K, iterations):
    mu, sigma, n_pulls = initialize_arms(K)

    for iteration in range(iterations):
        # Sample from the posteriors to choose an arm
        arm_samples = [np.random.normal(mu[i], sigma[i]) for i in range(K)]
        chosen_arm = np.argmax(arm_samples)

        # Simulate pulling the arm and getting a reward
        reward = pull_arm(chosen_arm)  # Replace this with your actual reward function
        print(f"Iteration {iteration + 1}: Pulled arm {chosen_arm}, Reward: {reward}")

        # Update the posterior distribution for the chosen arm
        update_posterior(chosen_arm, reward)

    return mu, sigma, n_pulls

# Run the UTS algorithm for a specified number of iterations
# Set number of iterations
iterations = 100

# Run UTS
final_mu, final_sigma, final_n_pulls = run_uts(K=5, iterations=iterations)
