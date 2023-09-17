
import numpy as np

class PoliticalGameSimulation:
    def __init__(self,iterations, num_countries, num_strategies, num_resources, political_systems):
        self.num_countries = num_countries
        self.num_strategies = num_strategies
        self.num_resources = num_resources
        self.political_systems = political_systems
        self.resources = np.random.randint(10, 100, (num_countries, num_resources)).astype(float)
        self.strategy_history = np.zeros((iterations, num_countries))
        self.payoff_history = np.zeros((iterations, num_countries))
        self.stability_history = np.zeros((iterations, num_countries))
        self.diplomatic_relations = np.random.randint(-1, 2, (num_countries, num_countries))
        self.alliances = np.zeros((num_countries, num_countries))
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
    
    def update_resources(self, payoffs):
        for i in range(self.num_countries):
            self.resources[i] += payoffs[i] / 10
            self.resources[i] = np.clip(self.resources[i], 0, 100)
            
    def update_strategies(self, q_values, learning_rate=0.1, discount_factor=0.9):
        payoffs = self.calculate_payoffs(q_values)
        for i in range(self.num_countries):
            strategy = np.argmax(q_values[i])
            reward = payoffs[i]
            max_future_value = np.max(q_values[i])
            new_value = (1 - learning_rate) * q_values[i, strategy] + learning_rate * (reward + discount_factor * max_future_value)
            q_values[i, strategy] = new_value
            
            if self.political_systems[i] == 'dictatorship':
                q_values[i] = q_values[i] * 0.95
            
            # Geopolitical events affecting resources
            event_chance = np.random.rand()
            if event_chance > 0.9:
                event_country = np.random.randint(0, self.num_countries)
                self.resources[event_country] += np.random.randint(5, 15, self.num_resources)
            elif event_chance < 0.1:
                event_country = np.random.randint(0, self.num_countries)
                self.resources[event_country] -= np.random.randint(5, 15, self.num_resources)
            
            # Adding geopolitical events or crises
            crisis_chance = np.random.rand()
            if crisis_chance > 0.9:
                crisis_country = np.random.randint(0, self.num_countries)
                self.resources[crisis_country] -= np.random.randint(5, 15, self.num_resources)
            
            # Adjust diplomatic relations and its impact
            for i in range(self.num_countries):
                for j in range(self.num_countries):
                    if self.diplomatic_relations[i, j] > 0:
                        self.resources[i] += np.random.randint(0, 3, self.num_resources)
                    elif self.diplomatic_relations[i, j] < 0:
                        self.resources[i] -= np.random.randint(0, 3, self.num_resources)
            
            # Diversify resources based on demand and supply
            resource_focus = np.argmax(self.resources[i])
            self.resources[i][resource_focus] += np.random.randint(1, 4)
            
            # Form strategic alliances
            for i in range(self.num_countries):
                for j in range(i+1, self.num_countries):
                    alliance_chance = np.random.rand()
                    if alliance_chance > 0.9:
                        self.alliances[i, j] = 1
                        self.alliances[j, i] = 1
                        self.resources[i] += np.random.randint(1, 4, self.num_resources)
                        self.resources[j] += np.random.randint(1, 4, self.num_resources)
            
            # Introduce economic model factors like inflation
            inflation_factor = self.inflation_rates[i] / 100
            self.resources[i] = self.resources[i] * (1 + inflation_factor)
            
            # Introduce population dynamics
            population_growth = np.random.uniform(0.5, 1.5)
            self.population[i] = int(self.population[i] * (1 + population_growth / 100))
            self.resources[i] -= self.population[i] // 10

    def run_simulation(self, iterations, q_values):
        for t in range(iterations):
            payoffs = self.calculate_payoffs(q_values)
            self.update_resources(payoffs)
            self.update_strategies(q_values)
            self.strategy_history[t] = np.argmax(q_values, axis=1)
            self.payoff_history[t] = payoffs
            self.stability_history[t] = np.std(payoffs)