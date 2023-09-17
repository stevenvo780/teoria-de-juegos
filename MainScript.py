
import numpy as np
import matplotlib.pyplot as plt
from PoliticalGameBase import PoliticalGameBase
from PoliticalGameFinal import PoliticalGameFinal
from PoliticalGameImproved import PoliticalGameImproved
from PoliticalGameFurtherImproved import PoliticalGameFurtherImproved
from PoliticalGameFurtherImprovedV2 import PoliticalGameFurtherImprovedV2
from PoliticalGameFurtherImprovedV3 import PoliticalGameFurtherImprovedV3

# Initialization and simulation code
political_systems = ['dictatorship', 'democracy', 'democracy', 'dictatorship', 'democracy']
game = PoliticalGameFurtherImprovedV3(num_countries=5, num_strategies=10, num_resources=4, political_systems=political_systems)
q_values = np.random.rand(5, 10)

game.run_simulation(iterations=500, q_values=q_values)

# Plotting code
fig, axs = plt.subplots(3, 1, figsize=(12, 18))
for i, country in enumerate(['Country A (Dictatorship)', 'Country B (Democracy)', 'Country C (Democracy)', 'Country D (Dictatorship)', 'Country E (Democracy)']):
    axs[0].plot(range(500), game.strategy_history[:, i], label=f'Strategy {country}', marker='o')
    axs[1].plot(range(500), game.payoff_history[:, i], label=f'Payoff {country}', marker='x')
    axs[2].plot(range(500), game.stability_history[:, i], label=f'Stability {country}', marker='s')

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
