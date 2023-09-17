
from PoliticalGameImproved import PoliticalGameImproved
import numpy as np

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
