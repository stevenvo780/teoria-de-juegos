import numpy as np
import matplotlib.pyplot as plt


class PoliticalGameSimulation:
    def __init__(self, iterations, num_countries, num_strategies, num_resources, political_systems):
        self.num_countries = num_countries
        self.num_strategies = num_strategies
        self.num_resources = num_resources
        self.political_systems = political_systems
        self.resources = np.random.randint(10, 100, (num_countries, num_resources)).astype(float)
        self.strategy_history = np.zeros((iterations, num_countries))
        self.payoff_history = np.zeros((iterations, num_countries))
        self.stability_history = np.zeros((iterations,))
        self.diplomatic_relations = np.random.randint(-1, 2, (num_countries, num_countries))

    def calculate_payoffs(self, q_values):
        return np.array([np.sum(self.resources[i]) * ((np.argmax(q_values[i]) + 1) if self.political_systems[i] == 'dictatorship' else (11 - np.argmax(q_values[i]))) / 10 for i in range(self.num_countries)])

    def update_strategies(self, q_values, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.2):
        payoffs = self.calculate_payoffs(q_values)
        for i in range(self.num_countries):
            # Introduce exploration
            if np.random.rand() < exploration_rate:
                strategy = np.random.randint(0, self.num_strategies)
            else:
                strategy = np.argmax(q_values[i])

            # Change strategy if a key resource is low
            strategy_resource_idx = strategy % self.num_resources
            if self.resources[i, strategy_resource_idx] < 20:
                strategy = np.argmin(self.resources[i])

            q_values[i, strategy] = (1 - learning_rate) * q_values[i, strategy] + learning_rate * (payoffs[i] + discount_factor * np.max(q_values[i]))

            # Introduce some random fluctuations in resources
            self.resources[i] += np.random.randint(-5, 5, self.num_resources)

    def run_simulation(self, iterations, q_values):
        for t in range(iterations):
            payoffs = self.calculate_payoffs(q_values)
            self.update_strategies(q_values)
            self.strategy_history[t] = np.argmax(q_values, axis=1)
            self.payoff_history[t] = payoffs
            self.stability_history[t] = np.std(payoffs)

# Initialization
iterations = 1000
political_systems = ['dictatorship', 'democracy', 'democracy', 'dictatorship', 'democracy']
countries = ['Country A (Dictatorship)', 'Country B (Democracy)', 'Country C (Democracy)', 'Country D (Dictatorship)', 'Country E (Democracy)']
markers = ['o', 'x', 's', 'd', '^']
titles = ['Strategies Over Time', 'Payoffs Over Time', 'Stability Over Time', 'Total Strategy Changes Over Time']
labels = ['Strategy', 'Payoff', 'Stability', 'Total Changes']

# Re-running the simulation with the updated class
game = PoliticalGameSimulation(iterations, 5, 10, 4, political_systems)
q_values = np.random.rand(5, 10)
game.run_simulation(iterations, q_values)

# Plotting
fig, axs = plt.subplots(4, 1, figsize=(12, 24))
for i, country in enumerate(countries):
    for ax, history, marker, title, label in zip(axs[:-1], [game.strategy_history, game.payoff_history, game.stability_history], markers, titles, labels):
        ax.plot(range(iterations), history if len(history.shape) > 1 else history, label=f"{label} {country}" if len(history.shape) > 1 else label, marker=marker)
        ax.set_title(title)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(label)
        ax.legend()

# Adding a specialized plot for strategy changes
strategy_changes = np.diff(game.strategy_history, axis=0)
axs[3].plot(range(1, iterations), np.sum(np.abs(strategy_changes), axis=1), label='Total Strategy Changes', marker='d')
axs[3].set_title('Total Strategy Changes Over Time')
axs[3].set_xlabel('Iteration')
axs[3].set_ylabel('Total Changes')
axs[3].legend()

plt.tight_layout()
plt.show()