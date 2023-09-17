
import numpy as np
import matplotlib.pyplot as plt

# Base class for the political game simulation
class PoliticalGameBase:
    def __init__(self, num_countries, num_strategies, num_resources, political_systems):
        self.num_countries = num_countries
        self.num_strategies = num_strategies
        self.num_resources = num_resources
        self.political_systems = political_systems
        self.resources = np.random.randint(10, 100, (num_countries, num_resources))
        self.strategy_history = np.zeros((100, num_countries))
        self.payoff_history = np.zeros((100, num_countries))
        self.stability_history = np.zeros((100, num_countries))
        
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
    
    def run_simulation(self, iterations, q_values):
        for t in range(iterations):
            payoffs = self.calculate_payoffs(q_values)
            self.update_resources(payoffs)
            self.update_strategies(q_values)
            self.strategy_history[t] = np.argmax(q_values, axis=1)
            self.payoff_history[t] = payoffs
            self.stability_history[t] = np.std(payoffs)

# Final class for the political game simulation
class PoliticalGameFinal(PoliticalGameBase):
    def update_strategies(self, q_values, learning_rate=0.1, discount_factor=0.9):
        super().update_strategies(q_values, learning_rate, discount_factor)
        for i in range(self.num_countries):
            if self.political_systems[i] == 'dictatorship':
                q_values[i] = q_values[i] * 0.95  # Dictatorships are resistant to change

# Improved class for the political game simulation
class PoliticalGameImproved(PoliticalGameFinal):
    def update_strategies(self, q_values, learning_rate=0.1, discount_factor=0.9):
        super().update_strategies(q_values, learning_rate, discount_factor)
        # Geopolitical events affecting resources
        event_chance = np.random.rand()
        if event_chance > 0.9:
            event_country = np.random.randint(0, self.num_countries)
            self.resources[event_country] += np.random.randint(5, 15, self.num_resources)
        elif event_chance < 0.1:
            event_country = np.random.randint(0, self.num_countries)
            self.resources[event_country] -= np.random.randint(5, 15, self.num_resources)

# Further Improved class for the political game simulation
class PoliticalGameFurtherImproved(PoliticalGameImproved):
    def __init__(self, num_countries, num_strategies, num_resources, political_systems):
        super().__init__(num_countries, num_strategies, num_resources, political_systems)
        self.diplomatic_relations = np.random.randint(-1, 2, (num_countries, num_countries))
        
    def update_strategies(self, q_values, learning_rate=0.1, discount_factor=0.9):
        super().update_strategies(q_values, learning_rate, discount_factor)
        
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
