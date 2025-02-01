import numpy as np
import matplotlib.pyplot as plt

class BernoulliBandit:
    def __init__(self, k=10):
        self.k = k
        # Sample probabilities uniformly from [0,1]
        self.probabilities = np.random.uniform(0, 1, k)
        
    def pull(self, arm):
        # Return 1 with probability p, 0 with probability 1-p
        return np.random.binomial(1, self.probabilities[arm])

class EpsilonGreedy:
    def __init__(self, k=10, epsilon=0.1):
        self.k = k
        self.epsilon = epsilon
        self.reset()
        
    def reset(self):
        self.action_counts = np.zeros(self.k)
        self.value_estimates = np.zeros(self.k)
        
    def select_arm(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.k)
        return np.argmax(self.value_estimates)
        
    def update(self, arm, reward):
        self.action_counts[arm] += 1
        n = self.action_counts[arm]
        value = self.value_estimates[arm]
        self.value_estimates[arm] = ((n - 1) * value + reward) / n

class UCB1:
    def __init__(self, k=10):
        self.k = k
        self.reset()
        
    def reset(self):
        self.action_counts = np.zeros(self.k)
        self.value_estimates = np.zeros(self.k)
        self.t = 0
        
    def select_arm(self):
        # Try each arm once
        for arm in range(self.k):
            if self.action_counts[arm] == 0:
                return arm
                
        # Calculate UCB values
        ucb_values = self.value_estimates + np.sqrt(
            2 * np.log(self.t) / self.action_counts
        )
        return np.argmax(ucb_values)
        
    def update(self, arm, reward):
        self.t += 1
        self.action_counts[arm] += 1
        n = self.action_counts[arm]
        value = self.value_estimates[arm]
        self.value_estimates[arm] = ((n - 1) * value + reward) / n

def run_experiment(n_rounds=1000, n_experiments=100):
    algorithms = {
        'ε-greedy (ε=0)': EpsilonGreedy(epsilon=0),
        'ε-greedy (ε=0.01)': EpsilonGreedy(epsilon=0.01),
        'ε-greedy (ε=0.1)': EpsilonGreedy(epsilon=0.1),
        'UCB1': UCB1()
    }
    
    # Store results for each algorithm
    results = {name: np.zeros(n_rounds) for name in algorithms.keys()}
    optimal_actions = {name: np.zeros(n_rounds) for name in algorithms.keys()}
    
    for _ in range(n_experiments):
        # Create a new bandit problem
        bandit = BernoulliBandit()
        optimal_arm = np.argmax(bandit.probabilities)
        
        # Run each algorithm on this problem
        for name, algorithm in algorithms.items():
            algorithm.reset()
            
            for t in range(n_rounds):
                # Select arm and get reward
                arm = algorithm.select_arm()
                reward = bandit.pull(arm)
                
                # Update algorithm
                algorithm.update(arm, reward)
                
                # Store results
                results[name][t] += reward
                optimal_actions[name][t] += (arm == optimal_arm)
    
    # Average results
    for name in algorithms.keys():
        results[name] /= n_experiments
        optimal_actions[name] /= n_experiments
        
    return results, optimal_actions

# Run experiment
n_rounds = 1000
results, optimal_actions = run_experiment(n_rounds=n_rounds)

# Plot results
plt.figure(figsize=(15, 6))

# Plot average rewards
plt.subplot(1, 2, 1)
for name, rewards in results.items():
    plt.plot(rewards, label=name)
plt.xlabel('Rounds')
plt.ylabel('Average Reward')
plt.title('Average Reward over Time')
plt.legend()

# Plot percentage of optimal actions
plt.subplot(1, 2, 2)
for name, optimal in optimal_actions.items():
    plt.plot(optimal * 100, label=name)
plt.xlabel('Rounds')
plt.ylabel('% Optimal Action')
plt.title('Percentage of Optimal Actions over Time')
plt.legend()

plt.tight_layout()
plt.show()

# Print final statistics
print("\nFinal Statistics (averaged over last 100 rounds):")
for name in results.keys():
    avg_reward = np.mean(results[name][-100:])
    avg_optimal = np.mean(optimal_actions[name][-100:]) * 100
    print(f"\n{name}:")
    print(f"Average Reward: {avg_reward:.3f}")
    print(f"Optimal Action %: {avg_optimal:.1f}%")