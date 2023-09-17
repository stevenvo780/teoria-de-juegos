
import numpy as np
import matplotlib.pyplot as plt

class PoliticalGameBase:
    def __init__(self, num_countries, num_strategies, num_resources, political_systems):
        self.num_countries = num_countries
        self.num_strategies = num_strategies
        self.num_resources = num_resources
        self.political_systems = political_systems
        
        self.resources = np.random.randint(50, 100, (num_countries, num_resources))
        self.resource_values = np.random.randint(1, 5, num_resources)
        
        self.strategies = np.random.randint(0, num_strategies, num_countries)
        self.strategy_history = []
        self.payoff_history = []
        
    def calculate_payoffs(self):
        payoffs = np.zeros(self.num_countries)
        for i in range(self.num_countries):
            payoffs[i] = np.dot(self.resources[i], self.resource_values)
        return payoffs
    
    def generate_new_resources(self):
        new_resources_prob = np.random.rand(self.num_countries, self.num_resources)
        self.resources += (new_resources_prob < 0.2).astype(int)

class PoliticalGameFinal(PoliticalGameBase):
    def __init__(self, num_countries, num_strategies, num_resources, political_systems):
        super().__init__(num_countries, num_strategies, num_resources, political_systems)
        self.relations = np.random.randint(-1, 2, (num_countries, num_countries))
        self.global_demand = np.random.randint(50, 100, num_resources)
        self.stability_history = []
    
    def calculate_stability(self):
        if len(self.payoff_history) < 10:
            return np.zeros(self.num_countries)
        last_10_payoffs = np.array(self.payoff_history[-10:])
        return np.std(last_10_payoffs, axis=0)
        
    def update_strategies(self, q_values, learning_rate=0.1, discount_factor=0.9):
        current_payoffs = self.calculate_payoffs()
        self.stability_history.append(self.calculate_stability())
        
        self.resources -= np.random.randint(0, 2, (self.num_countries, self.num_resources))
        self.generate_new_resources()
        
        economic_cycle = np.random.choice([-20, -10, 0, 10, 20])
        
        for i in range(self.num_countries):
            future_best_value = np.max(q_values[i])
            economic_shock = np.random.choice([-50, 0, 0, 0, 50])
            
            q_values[i, self.strategies[i]] = (1 - learning_rate) * q_values[i, self.strategies[i]] + learning_rate * (current_payoffs[i] + discount_factor * future_best_value + economic_cycle + economic_shock)
            
            if self.political_systems[i] == 'dictatorship':
                if np.sum(self.resources[i]) < 20 or np.random.rand() < 0.4:
                    self.strategies[i] = np.random.randint(0, self.num_strategies)
            else:
                if np.random.rand() < 0.5:
                    self.strategies[i] = np.argmax(q_values[i])
            
            self.global_demand += np.random.randint(-1, 2, self.num_resources)
            if np.dot(self.global_demand, self.resources[i]) < np.sum(self.global_demand) / 2:
                self.strategies[i] = np.random.randint(0, self.num_strategies)
        
        self.strategy_history.append(self.strategies.copy())
        self.payoff_history.append(current_payoffs)
        
    def trade_resources(self):
        for i in range(self.num_countries):
            for j in range(self.num_countries):
                if i != j:
                    trade_value = np.dot(self.resources[i], self.resource_values) - np.dot(self.resources[j], self.resource_values)
                    if trade_value > 0:
                        self.resources[i] -= 1
                        self.resources[j] += 1

    def run_simulation(self, iterations, q_values, learning_rate=0.1, discount_factor=0.9):
        for _ in range(iterations):
            self.update_strategies(q_values, learning_rate, discount_factor)
            self.trade_resources()
        
        self.strategy_history = np.array(self.strategy_history)
        self.payoff_history = np.array(self.payoff_history)
        self.stability_history = np.array(self.stability_history)

# Inicializar variables y correr la simulación
political_systems = ['dictatorship', 'democracy', 'democracy', 'dictatorship', 'democracy']
game = PoliticalGameFinal(num_countries=5, num_strategies=10, num_resources=4, political_systems=political_systems)
q_values = np.random.rand(5, 10)

game.run_simulation(iterations=100, q_values=q_values)

# Gráficos
fig, axs = plt.subplots(3, 1, figsize=(12, 18))
for i, country in enumerate(['País A (Dictadura)', 'País B (Democracia)', 'País C (Democracia)', 'País D (Dictadura)', 'País E (Democracia)']):
    axs[0].plot(range(100), game.strategy_history[:, i], label=f'Estrategia {country}', marker='o')
    axs[1].plot(range(100), game.payoff_history[:, i], label=f'Pago {country}', marker='x')
    axs[2].plot(range(100), game.stability_history[:, i], label=f'Estabilidad {country}', marker='s')

axs[0].set_title('Estrategias a lo largo del tiempo')
axs[0].set_xlabel('Iteración')
axs[0].set_ylabel('Estrategia')
axs[0].legend()

axs[1].set_title('Pagos a lo largo del tiempo')
axs[1].set_xlabel('Iteración')
axs[1].set_ylabel('Pago')
axs[1].legend()

axs[2].set_title('Estabilidad a lo largo del tiempo')
axs[2].set_xlabel('Iteración')
axs[2].set_ylabel('Estabilidad (Desviación Estándar de Pagos)')
axs[2].legend()

plt.tight_layout()
plt.show()