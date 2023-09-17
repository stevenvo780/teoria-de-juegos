
from PoliticalGameFurtherImproved import PoliticalGameFurtherImproved
import numpy as np

class PoliticalGameFurtherImprovedV2(PoliticalGameFurtherImproved):
    def __init__(self, num_countries, num_strategies, num_resources, political_systems):
        super().__init__(num_countries, num_strategies, num_resources, political_systems)
        self.alliances = np.zeros((num_countries, num_countries))
        
    def update_strategies(self, q_values, learning_rate=0.1, discount_factor=0.9):
        super().update_strategies(q_values, learning_rate, discount_factor)
        # Diversify resources based on demand and supply
        for i in range(self.num_countries):
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
