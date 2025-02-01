import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, k=10):
        self.k = k  # Number of arms
        self.q_true = np.random.normal(0, 1, k)  # True action values
        self.q_est = np.zeros(k)  # Estimated values
        self.action_counts = np.zeros(k)  # Count of actions taken
    
    def step(self, action):
        return np.random.normal(self.q_true[action], 1)  # Gaussian reward
    
    def update_estimates(self, action, reward):
        self.action_counts[action] += 1
        self.q_est[action] += (reward - self.q_est[action]) / self.action_counts[action]

    def select_action(self, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(self.k)  # Exploration
        return np.argmax(self.q_est)  # Exploitation


def run_experiment(bandit_count=100, steps=1000, epsilon=0.1):
    rewards = np.zeros(steps)
    
    for _ in range(bandit_count):
        bandit = Bandit()
        
        for step in range(steps):
            action = bandit.select_action(epsilon)
            reward = bandit.step(action)
            bandit.update_estimates(action, reward)
            rewards[step] += reward
    
    return rewards / bandit_count  # Average reward over bandit tasks


# Run experiments for epsilon values 0, 0.01, and 0.1
steps = 1000
epsilons = [0, 0.01, 0.1]
results = {epsilon: run_experiment(100, steps, epsilon) for epsilon in epsilons}

# Plot results
plt.figure(figsize=(10, 6))
for epsilon, rewards in results.items():
    plt.plot(rewards, label=f'$\epsilon$ = {epsilon}')

plt.xlabel('Plays')
plt.ylabel('Average Reward')
plt.title('Average Reward of $\epsilon$-Greedy Methods')
plt.legend()
plt.show()
