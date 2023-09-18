import numpy as np
import matplotlib.pyplot as plt

iterations = 1000
resources_max = 10000
num_countries = 5
num_strategies = 10
event_chance_value = 0.6
exploration_rate = 0.2
crisis_chance_value = 0.1
alliance_chance_value = 0.4


class PoliticalGameSimulation:
    def __init__(self, iterations, num_countries, num_strategies, num_resources, political_systems):
        self.iterations = iterations
        self.num_countries = num_countries
        self.num_strategies = num_strategies
        self.num_resources = num_resources
        self.political_systems = political_systems
        self.resources = np.random.randint(
            10, resources_max, (num_countries, num_resources)).astype(float)
        self.strategy_history = np.zeros((iterations, num_countries))
        self.payoff_history = np.zeros((iterations, num_countries))
        self.stability_history = np.zeros((iterations, num_countries))
        self.total_resources_history = np.zeros((iterations, num_countries))
        self.diplomatic_relations = np.random.randint(
            -1, 2, (num_countries, num_countries))
        self.inflation_rates = np.random.uniform(1, 3, num_countries)
        self.population = np.random.randint(10, 50, num_countries)

    def calculate_payoffs(self, q_values):
        payoffs = np.zeros(self.num_countries)
        for i in range(self.num_countries):
            strategy = np.argmax(q_values[i])
            if self.political_systems[i] == 'dictatorship':
                payoffs[i] = np.sum(self.resources[i]) * (strategy + 1) / 10
            else:  # democracy
                payoffs[i] = np.sum(self.resources[i]) * (11 - strategy) / 10
        return payoffs

    def update_resources(self):
        for i in range(self.num_countries):
            self.resources[i] += np.random.uniform(-5, 5, self.num_resources)
            self.resources[i] = np.clip(self.resources[i], 0, resources_max)

    def update_strategies(self, q_values, learning_rate=0.1, discount_factor=0.9):
        payoffs = self.calculate_payoffs(q_values)
        for i in range(self.num_countries):
            strategy = np.argmax(q_values[i])
            reward = payoffs[i]
            max_future_value = np.max(q_values[i])
            new_value = (1 - learning_rate) * q_values[i, strategy] + learning_rate * (
                reward + discount_factor * max_future_value)
            q_values[i, strategy] = new_value

            # Strategy change if a key resource is low
            low_resource_idx = np.argmin(self.resources[i])
            if self.resources[i, low_resource_idx] < 20:
                new_strategy = low_resource_idx % self.num_strategies
                q_values[i] = np.zeros(self.num_strategies)
                q_values[i, new_strategy] = 1

            self.resources[i] += np.random.uniform(-5, 5, self.num_resources)
            event_chance = np.random.rand()
            if event_chance > event_chance_value:
                event_country = np.random.randint(0, self.num_countries)
                self.resources[event_country] += np.random.randint(
                    5, 15, self.num_resources)
            elif event_chance < 0.3:
                event_country = np.random.randint(0, self.num_countries)
                self.resources[event_country] -= np.random.randint(
                    5, 15, self.num_resources)

            crisis_chance = np.random.rand()
            if crisis_chance > crisis_chance_value:
                crisis_country = np.random.randint(0, self.num_countries)
                self.resources[crisis_country] -= np.random.randint(
                    5, 15, self.num_resources)

            for j in range(self.num_countries):
                if self.diplomatic_relations[i, j] > 0:
                    self.resources[i] += np.random.randint(
                        0, 3, self.num_resources)
                elif self.diplomatic_relations[i, j] < 0:
                    self.resources[i] -= np.random.randint(
                        0, 3, self.num_resources)

            for j in range(i+1, self.num_countries):
                alliance_chance = np.random.rand()
                if alliance_chance > alliance_chance_value:
                    self.resources[i] += np.random.randint(
                        1, 4, self.num_resources)
                    self.resources[j] += np.random.randint(
                        1, 4, self.num_resources)

    def run_simulation(self, iterations, q_values):
        for t in range(iterations):
            payoffs = self.calculate_payoffs(q_values)
            self.update_resources()
            self.update_strategies(q_values)
            self.strategy_history[t] = np.argmax(q_values, axis=1)
            self.payoff_history[t] = payoffs
            self.stability_history[t] = np.std(payoffs)
            self.total_resources_history[t] = np.sum(self.resources, axis=1)


# Re-running the simulation with the updated class
political_systems = ['dictatorship', 'democracy',
                     'democracy', 'dictatorship', 'democracy']
q_values = np.random.rand(num_countries, num_strategies)
game = PoliticalGameSimulation(
    iterations, num_countries, num_strategies, 4, political_systems)
game.run_simulation(iterations, q_values)

# Plotting
fig, axs = plt.subplots(5, 1, figsize=(12, 30))
titles = ['Strategies Over Time', 'Payoffs Over Time', 'Stability Over Time',
          'Total Strategy Changes Over Time', 'Total Resources Over Time']
labels = ['Strategy', 'Payoff', 'Stability',
          'Total Changes', 'Total Resources']

for i, (history, title, label) in enumerate(zip([game.strategy_history, game.payoff_history, game.stability_history, np.sum(np.abs(np.diff(game.strategy_history, axis=0)), axis=1), np.sum(game.total_resources_history, axis=1)], titles, labels)):
    axs[i].plot(range(iterations) if i != 3 else range(1, iterations), history if len(
        history.shape) > 1 else history, label=label if len(history.shape) > 1 else label)
    axs[i].set_title(title)
    axs[i].set_xlabel('Iteration')
    axs[i].set_ylabel(label)
    if len(history.shape) > 1:
        axs[i].legend()

plt.tight_layout()
plt.show()
