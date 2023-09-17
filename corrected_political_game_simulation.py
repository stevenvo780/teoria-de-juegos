
import numpy as np
import matplotlib.pyplot as plt

class PoliticalGameBase:
    def __init__(self, num_countries, num_strategies, num_resources, political_systems):
        self.num_countries = num_countries
        self.num_strategies = num_strategies
        self.num_resources = num_resources
        self.political_systems = political_systems
        self.resources = np.random.randint(10, 100, (num_countries, num_resources)).astype('float64')
        self.strategy_history = np.zeros((100, num_countries))
        self.payoff_history = np.zeros((100, num_countries))
        self.stability_history = np.zeros((100, num_countries))
        
    def calculate_payoffs(self, q_values):
        payoffs = np.zeros(self.num_countries)
        for i in range(self.num_countries):
            strategy = np.argmax(q_values[i])
            if self.political_systems[i] == 'dictatorship':
                payoffs[i] = np.sum(self.resources[i]) * (strategy + 1) / 10
            else:
                payoffs[i] = np.sum(self.resources[i]) * (11 - strategy) / 10
        return payoffs
    
    def update_resources(self, payoffs):
        for i in range(self.num_countries):
            self.resources[i] += payoffs[i] / 10.0
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

political_systems = ['dictatorship', 'democracy', 'democracy', 'dictatorship', 'democracy']
game = PoliticalGameBase(num_countries=5, num_strategies=10, num_resources=4, political_systems=political_systems)
q_values = np.random.rand(5, 10)

game.run_simulation(iterations=100, q_values=q_values)

fig, axs = plt.subplots(3, 1, figsize=(12, 18))
for i, country in enumerate(['Country A (Dictatorship)', 'Country B (Democracy)', 'Country C (Democracy)', 'Country D (Dictatorship)', 'Country E (Democracy)']):
    axs[0].plot(range(100), game.strategy_history[:, i], label=f'Strategy {country}', marker='o')
    axs[1].plot(range(100), game.payoff_history[:, i], label=f'Payoff {country}', marker='x')
    axs[2].plot(range(100), game.stability_history[:, i], label=f'Stability {country}', marker='s')

axs[0].set_title('Strategies Over Time')
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel('Strategy')
axs[0].legend()

axs[1].set_title('Payoffs Over Time')
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel('Payoff')
axs[1].legend()

axs[2].set_title('Stability Over Time')
axs[2].set_xlabel('Iteration')
axs[2].set_ylabel('Stability (Standard Deviation of Payoffs)')
axs[2].legend()

plt.tight_layout()
plt.show()
