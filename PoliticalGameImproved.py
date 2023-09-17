
from PoliticalGameFinal import PoliticalGameFinal
import numpy as np

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
