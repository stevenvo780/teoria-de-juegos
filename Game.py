import numpy as np
import matplotlib.pyplot as plt

iterations = 1000
resources_max = 10000
num_countries = 5
num_strategies = 10
num_resources = 4
event_chance_value = 0.6
crisis_chance_value = 0.1
alliance_chance_value = 0.4
learning_rate = 0.1
discount_factor = 0.9
political_systems = ['dictatorship', 'democracy',
                     'democracy', 'dictatorship', 'democracy']
q_values = np.random.rand(num_countries, num_strategies)
titles = ['Strategies Over Time', 'Payoffs Over Time',
          'Stability Over Time', 'Total Resources Over Time', 'Nash Equilibria']
labels = ['Strategy', 'Payoff', 'Stability', 'Total Resources', 'Equilibrium']


class GameSimulation:
    def __init__(self):
        self.resources = np.random.randint(
            10, resources_max, (num_countries, num_resources)).astype(float)
        self.strategy_history = np.zeros((iterations, num_countries))
        self.payoff_history = np.zeros((iterations, num_countries))
        self.stability_history = np.zeros((iterations, num_countries))
        self.total_resources_history = np.zeros((iterations, num_countries))
        self.diplomatic_relations = np.random.randint(
            -1, 2, (num_countries, num_countries))
        self.population = np.random.randint(10, 50, num_countries)

    def calculate_payoffs(self, q_values):
        payoffs = np.zeros(num_countries)
        for i in range(num_countries):
            strategy = np.argmax(q_values[i])
            if political_systems[i] == 'dictatorship':
                payoffs[i] = (np.sum(self.resources[i]) +
                              self.population[i]) * (strategy + 1) / 10
            else:  # democracy
                payoffs[i] = (np.sum(self.resources[i]) +
                              self.population[i]) * (11 - strategy) / 10
        return payoffs

    def update_resources(self):
        for i in range(num_countries):
            self.resources[i] += np.random.uniform(-5, 5, num_resources)
            self.resources[i] = np.clip(self.resources[i], 0, resources_max)

    def update_events(self):
        for i in range(num_countries):
            event_chance = np.random.rand()
            if event_chance > event_chance_value:
                event_country = np.random.randint(0, num_countries)
                self.resources[event_country] += np.random.randint(
                    5, 15, num_resources)
            elif event_chance < 0.3:
                event_country = np.random.randint(0, num_countries)
                self.resources[event_country] -= np.random.randint(
                    5, 15, num_resources)

            crisis_chance = np.random.rand()
            if crisis_chance > crisis_chance_value:
                crisis_country = np.random.randint(0, num_countries)
                self.resources[crisis_country] -= np.random.randint(
                    5, 15, num_resources)

            for j in range(num_countries):
                if self.diplomatic_relations[i, j] > 0:
                    self.resources[i] += np.random.randint(0, 3, num_resources)
                elif self.diplomatic_relations[i, j] < 0:
                    self.resources[i] -= np.random.randint(0, 3, num_resources)

            for j in range(i+1, num_countries):
                alliance_chance = np.random.rand()
                if alliance_chance > alliance_chance_value:
                    self.resources[i] += np.random.randint(1, 4, num_resources)
                    self.resources[j] += np.random.randint(1, 4, num_resources)

    def update_strategies(self, q_values):
        payoffs = self.calculate_payoffs(q_values)
        for i in range(num_countries):
            strategy = np.argmax(q_values[i])
            reward = payoffs[i]
            max_future_value = np.max(q_values[i])
            new_value = (1 - learning_rate) * q_values[i, strategy] + learning_rate * (
                reward + discount_factor * max_future_value)
            q_values[i, strategy] = new_value

            low_resource_idx = np.argmin(self.resources[i])
            if self.resources[i, low_resource_idx] < 20:
                new_strategy = low_resource_idx % num_strategies
                q_values[i] = np.zeros(num_strategies)
                q_values[i, new_strategy] = 1

    def check_equilibrium(self):
        equilibria = []
        for t in range(1, iterations):
            if np.all(self.strategy_history[t] == self.strategy_history[t-1]):
                equilibria.append(t)
        return equilibria

    def run_simulation(self):
        for t in range(iterations):
            self.update_resources()
            self.update_strategies(q_values)
            self.update_events()
            payoffs = self.calculate_payoffs(q_values)
            self.strategy_history[t] = np.argmax(q_values, axis=1)
            self.payoff_history[t] = payoffs
            self.stability_history[t] = np.std(payoffs)
            self.total_resources_history[t] = np.sum(self.resources, axis=1)


theory_game = GameSimulation()
theory_game.run_simulation()
equilibrio_nash = theory_game.check_equilibrium()

fig, axs = plt.subplots(5, 1, figsize=(12, 30))

for i, (history, title, label) in enumerate(zip([theory_game.strategy_history, theory_game.payoff_history, theory_game.stability_history, theory_game.total_resources_history], titles, labels)):
    axs[i].plot(history)
    axs[i].set_title(title)
    axs[i].set_xlabel('Iteration')
    axs[i].set_ylabel(label)

axs[4].plot(equilibrio_nash, [1]*len(equilibrio_nash), 'ro')
axs[4].set_title(titles[-1])
axs[4].set_xlabel('Iteration')
axs[4].set_ylabel(labels[-1])

plt.tight_layout()
plt.show()
