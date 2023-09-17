
from PoliticalGameBase import PoliticalGameBase
import numpy as np

# Initialize game object
num_countries = 2
num_strategies = 5
num_resources = 3
political_systems = ['democracy', 'dictatorship']
game = PoliticalGameBase(num_countries, num_strategies, num_resources, political_systems)

# Generate random Q-values for testing
q_values = np.random.rand(num_countries, num_strategies)

# Run a single update to test functionality
game.update_strategies(q_values)

print("Base class test executed successfully.")
