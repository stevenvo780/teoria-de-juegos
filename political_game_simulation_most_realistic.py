
import numpy as np
import matplotlib.pyplot as plt

class PoliticalGameBase:
    # ... (Same as before)

class PoliticalGameFinal(PoliticalGameBase):
    # ... (Same as before)

class PoliticalGameImproved(PoliticalGameFinal):
    # ... (Same as before)

class PoliticalGameFurtherImproved(PoliticalGameImproved):
    # ... (Same as before)

class PoliticalGameMoreRealistic(PoliticalGameFurtherImproved):
    # ... (Same as before)

class PoliticalGameMostRealistic(PoliticalGameMoreRealistic):
    def update_strategies(self, q_values, learning_rate=0.1, discount_factor=0.9):
        super().update_strategies(q_values, learning_rate, discount_factor)
        # Change strategy based on resource availability and recent payoffs
        for i in range(self.num_countries):
            least_resource = np.argmin(self.resources[i])
            most_resource = np.argmax(self.resources[i])
            if self.resources[i, least_resource] < 10:
                best_strategy_for_resource = np.argmin(q_values[i] * most_resource)
                q_values[i, best_strategy_for_resource] *= 1.2
            elif self.resources[i, most_resource] > 90:
                best_strategy_for_resource = np.argmax(q_values[i] * least_resource)
                q_values[i, best_strategy_for_resource] *= 1.2

# Initialization and simulation code
political_systems = ['dictatorship', 'democracy', 'democracy', 'dictatorship', 'democracy']
game = PoliticalGameMostRealistic(num_countries=5, num_strategies=10, num_resources=4, political_systems=political_systems)
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
