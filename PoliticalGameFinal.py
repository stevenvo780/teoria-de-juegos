
from PoliticalGameBase import PoliticalGameBase
import numpy as np

class PoliticalGameFinal(PoliticalGameBase):
    def update_strategies(self, q_values, learning_rate=0.1, discount_factor=0.9):
        super().update_strategies(q_values, learning_rate, discount_factor)
        for i in range(self.num_countries):
            if self.political_systems[i] == 'dictatorship':
                q_values[i] = q_values[i] * 0.95  # Dictatorships are resistant to change
