
import numpy as np
import matplotlib.pyplot as plt

iterations = 10000
resources_max = 10000
num_countries = 5
num_strategies = 10
num_resources = 4
event_chance_value = 0.7
crisis_chance_value = 0.9
alliance_chance_value = 0.4
learning_rate = 0.1
discount_factor = 0.9

resource_min_initial = 10
resource_move_min = -200
resource_move_max = 100
event_resource_change = np.array([5, resources_max * 0.03])
crisis_resource_change = np.array([5, resources_max * 0.05])
alliance_resource_bonus = np.array([1, 4])
population_min = 10
population_max = 5000
diplomatic_min = -1
diplomatic_max = 2
strategy_stability_threshold = 5
resource_stability_threshold = resources_max * 0.4

political_systems = ['dictatorship', 'democracy',
                     'democracy', 'dictatorship', 'democracy']
q_values = np.random.rand(num_countries, num_strategies)
titles = ['Strategies Over Time', 'Payoffs Over Time',
          'Stability Over Time', 'Total Resources Over Time', 'Nash Equilibria']
labels = ['Strategy', 'Payoff', 'Stability', 'Total Resources', 'Equilibrium']


class GameSimulation:
    def __init__(self):
        self.resources = np.random.randint(
            resource_min_initial, resources_max, (num_countries, num_resources)).astype(float)
        self.strategy_history = np.zeros((iterations, num_countries))
        self.payoff_history = np.zeros((iterations, num_countries))
        self.stability_history = np.zeros((iterations, num_countries))
        self.total_resources_history = np.zeros((iterations, num_countries))
        self.diplomatic_relations = np.random.randint(
            diplomatic_min, diplomatic_max, (num_countries, num_countries))
        self.population = np.random.randint(
            population_min, population_max, num_countries)

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
            self.resources[i] += np.random.uniform(
                resource_move_min, resource_move_max, num_resources)
            self.resources[i] = np.clip(self.resources[i], 0, resources_max)

    def update_events(self):
        for i in range(num_countries):
            event_chance = np.random.rand()
            if event_chance > event_chance_value:
                event_country = np.random.randint(0, num_countries)
                self.resources[event_country] += np.random.randint(
                    event_resource_change[0], event_resource_change[1], num_resources)
            elif event_chance < event_chance_value:
                event_country = np.random.randint(0, num_countries)
                self.resources[event_country] -= np.random.randint(
                    event_resource_change[0], event_resource_change[1], num_resources)

            crisis_chance = np.random.rand()
            if crisis_chance > crisis_chance_value:
                crisis_country = np.random.randint(0, num_countries)
                self.resources[crisis_country] -= np.random.randint(
                    crisis_resource_change[0], crisis_resource_change[1], num_resources)

            for j in range(num_countries):
                if self.diplomatic_relations[i, j] > 0:
                    self.resources[i] += np.random.randint(0, 3, num_resources)
                elif self.diplomatic_relations[i, j] < 0:
                    self.resources[i] -= np.random.randint(0, 3, num_resources)

            for j in range(i+1, num_countries):
                alliance_chance = np.random.rand()
                if alliance_chance > alliance_chance_value:
                    self.resources[i] += np.random.randint(
                        alliance_resource_bonus[0], alliance_resource_bonus[1], num_resources)
                    self.resources[j] += np.random.randint(
                        alliance_resource_bonus[0], alliance_resource_bonus[1], num_resources)

    def update_strategies(self, q_values):
        payoffs = self.calculate_payoffs(q_values)
        for i in range(num_countries):
            strategy = np.argmax(q_values[i])
            reward = payoffs[i]
            max_future_value = np.max(q_values[i])
            new_value = (1 - learning_rate) * q_values[i, strategy] + learning_rate * (
                reward + discount_factor * max_future_value)
            q_values[i, strategy] = new_value

            if np.std(self.payoff_history[-10:, i]) < strategy_stability_threshold and self.resources[i].mean() < resource_stability_threshold:
                new_strategy = np.argmin(q_values[i])
                q_values[i] = np.zeros(num_strategies)
                q_values[i, new_strategy] = 1

    def check_equilibrium(self):
        equilibria = []
        stable_range = 5000
        for t in range(1, iterations - 100):
            window = self.payoff_history[t:t+100]
            for wt in window:
                max_payoff = np.max(wt)
                min_payoff = np.min(wt)
                if max_payoff - min_payoff <= stable_range:
                    equilibria.append(t)
                    break
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
    if title == 'Strategies Over Time':

        axs[i].axhline(9, color='b', linestyle='--',
                       label='Total Dictatorship')
        axs[i].axhline(0, color='r', linestyle='--', label='Total Democracy')
        axs[i].legend()

if len(equilibrio_nash) > 0:
    axs[4].plot(equilibrio_nash, [1]*len(equilibrio_nash), 'ro')
else:
    axs[4].text(0.5, 0.5, 'No Nash Equilibria Found', horizontalalignment='center',
                verticalalignment='center', fontsize=15, color='red')

axs[4].set_title('Nash Equilibria')
axs[4].set_xlabel('Iteration')
axs[4].set_ylabel('Equilibrium')

plt.tight_layout()
plt.show()
