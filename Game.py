
import numpy as np
import matplotlib.pyplot as plt
import threading

iterations = 10000
resources_max = 10000
num_countries = 5
num_strategies = 10
num_resources = 4
event_chance_value = 0.4
crisis_chance_value = 0.7
alliance_chance_value = 0.6
learning_rate = 0.7
discount_factor = 0.8

resource_min_initial = 10
resource_move_min = -110
resource_move_max = 100
event_resource_change = np.array([5, resources_max * 0.03])
crisis_resource_change = np.array([5, resources_max * 0.05])
alliance_resource_bonus = np.array([1, 4])
population_min = 10
population_max = 5000
diplomatic_min = -1
diplomatic_max = 2
strategy_stability_threshold = 10
resource_stability_threshold = resources_max * 0.6

political_systems = [
    'Authoritarian', 'Totalitarian', 'Monarchy', 'Theocracy',
    'Oligarchy', 'Military Dictatorship', 'One-Party State',
    'Direct Democracy', 'Representative Democracy', 'Constitutional Monarchy',
    'Parliamentary Democracy', 'Presidential Democracy', 'Federal Democracy',
    'Social Democracy', 'Anarchism', 'Libertarianism'
]
q_values = np.random.rand(num_countries, num_strategies)
titles = ['Strategies Over Time', 'Payoffs Over Time',
          'Stability Over Time', 'Total Resources Over Time', 'Nash Equilibria']
labels = ['Strategy', 'Payoff', 'Stability', 'Total Resources', 'Equilibrium']


class GameSimulation:
    def __init__(self, history_length=1000):
        self.history_length = history_length
        self.resources = np.random.randint(
            resource_min_initial, resources_max, (num_countries, num_resources)).astype(float)
        self.strategy_history = np.zeros((self.history_length, num_countries))
        self.payoff_history = np.zeros((self.history_length, num_countries))
        self.stability_history = np.zeros((self.history_length, num_countries))
        self.total_resources_history = np.zeros(
            (self.history_length, num_countries))
        self.diplomatic_relations = np.random.randint(
            diplomatic_min, diplomatic_max, (num_countries, num_countries))
        self.population = np.random.randint(
            population_min, population_max, num_countries)
        self.technology_level = np.random.randint(1, 10, num_countries)
        self.environment_health = np.random.randint(50, 100, num_countries)

    def calculate_payoffs(self, q_values):
        payoffs = np.zeros(num_countries)
        for i in range(num_countries):
            strategy = np.argmax(q_values[i])
            base_payoff = (
                np.sum(self.resources[i]) + self.population[i]) * (strategy + 1) / 10
            tech_influence = self.technology_level[i] / 10
            env_influence = self.environment_health[i] / 100
            payoffs[i] = base_payoff * (1 + tech_influence) * env_influence

            political_index = political_systems.index(political_systems[i])
            payoffs[i] *= (1 + political_index / len(political_systems) * 0.2)

        return payoffs

    def update_resources(self):
        for i in range(num_countries):
            self.resources[i] += np.random.uniform(
                resource_move_min, resource_move_max, num_resources)
            self.resources[i] = np.clip(self.resources[i], 0, resources_max)
        self.resources += (self.technology_level[:, np.newaxis] - 5) * 10
        self.resources -= (100 - self.environment_health[:, np.newaxis]) * 10

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

            if np.random.rand() < 0.1:
                self.technology_level[i] += 1
            if np.random.rand() < 0.1:
                self.environment_health[i] -= 1

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

            if np.std(self.payoff_history[-10:, i]) > strategy_stability_threshold * 2:
                current_index = political_systems.index(political_systems[i])
                new_index = (current_index + 1) % len(political_systems)
                political_systems[i] = political_systems[new_index]

    def check_equilibrium(self):
        equilibria = []
        stable_range = 4000
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
        fig1, axs1 = plt.subplots(4, 1, figsize=(12, 24))
        fig2, ax2 = plt.subplots(1, 1, figsize=(12, 6))
        figs_countries = [plt.figure(figsize=(12, 8)) for _ in range(num_countries)]
        axs_countries = [fig.subplots(2, 2).flatten() for fig in figs_countries] # type: ignore

        country_stats_history = {
            'resources': np.zeros((self.history_length, num_countries, num_resources)),
            'population': np.zeros((self.history_length, num_countries)),
            'tech': np.zeros((self.history_length, num_countries)),
            'env_health': np.zeros((self.history_length, num_countries))
        }

        while True:
            self.update_resources()
            self.update_strategies(q_values)
            self.update_events()
            payoffs = self.calculate_payoffs(q_values)

            # Roll and update histories
            history_items = [
                ('strategy', np.argmax(q_values, axis=1)),
                ('payoff', payoffs),
                ('stability', np.std(payoffs)),
                ('total_resources', np.sum(self.resources, axis=1))
            ]
            for name, new_data in history_items:
                setattr(self, f"{name}_history", np.roll(getattr(self, f"{name}_history"), -1, axis=0))
                getattr(self, f"{name}_history")[-1] = new_data

            # Roll and update country-specific histories
            for name, attr in zip(['resources', 'population', 'tech', 'env_health'], 
                                [self.resources, self.population, self.technology_level, self.environment_health]):
                country_stats_history[name] = np.roll(country_stats_history[name], -1, axis=0)
                country_stats_history[name][-1] = attr

            # Update general graphs
            for i, ax in enumerate(axs1):
                ax.clear()
                ax.plot(getattr(self, f"{['strategy', 'payoff', 'stability', 'total_resources'][i]}_history"))
                ax.set_title(titles[i])
                ax.set_xlabel('Iteration')
                ax.set_ylabel(labels[i])
                if i == 0:
                    ax.legend(['Total Dictatorship', 'Total Democracy'])

            # Update Nash Equilibrium graph
            equilibria = self.check_equilibrium()
            ax2.clear()
            ax2.plot(equilibria, [1]*len(equilibria), 'ro') if len(
                equilibria) > 0 else ax2.text(0.5, 0.5, 'No Nash Equilibria Found', fontsize=15, color='red')
            ax2.set_title('Nash Equilibria')

            # Update country-specific graphs
            for i, axs_set in enumerate(axs_countries):
                for j, (ax, name) in enumerate(zip(axs_set, ['resources', 'population', 'tech', 'env_health'])):
                    ax.clear()
                    ax.plot(country_stats_history[name][:, i] if j == 0 else country_stats_history[name][:, i])
                    ax.set_title(f'{name.capitalize()} for Country {i + 1}')

            # Update plots
            fig1.canvas.draw()
            fig2.canvas.draw()
            [fig.canvas.draw() for fig in figs_countries]
            plt.pause(0.0000001)


# Inicializar y correr la simulación
theory_game = GameSimulation()
theory_game.run_simulation()
