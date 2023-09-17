
import numpy as np
import matplotlib.pyplot as plt

class PoliticalGameBase:
    # ... (Same as before)

class PoliticalGameFinal(PoliticalGameBase):
    # ... (Same as before)

class PoliticalGameFinalIteration(PoliticalGameFinal):
    # ... (Same as before)

class PoliticalGameFinalPlus(PoliticalGameFinalIteration):
    def update_strategies(self, q_values, learning_rate=0.1, discount_factor=0.9):
        super().update_strategies(q_values, learning_rate, discount_factor)
        public_sentiment = np.random.randn(self.num_countries)
        q_values += public_sentiment[:, np.newaxis] * 0.2

# Initialization and simulation code
political_systems = ['dictatorship', 'democracy', 'democracy', 'dictatorship', 'democracy']
game = PoliticalGameFinalPlus(num_countries=5, num_strategies=10, num_resources=4, political_systems=political_systems)
q_values = np.random.rand(5, 10)
game.run_simulation(iterations=100, q_values=q_values)

# Plotting code
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
