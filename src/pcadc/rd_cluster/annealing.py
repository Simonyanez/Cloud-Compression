import numpy as np

class SimulatedAnnealing:
    def __init__(self, initial_temperature: float, final_temperature: float, cooling_rate: float):
        self.temperature = initial_temperature
        self.final_temperature = final_temperature
        self.cooling_rate = cooling_rate

    def cool_down(self):
        """Updates the temperature according to the cooling schedule."""
        self.temperature = max(self.final_temperature, self.temperature * self.cooling_rate)

    def choose(self, costs: np.ndarray, num_choices: int) -> int:
        """
        Probabilistically chooses an item based on a list of costs, using the
        current temperature. Lower costs have a higher probability of being chosen.
        """
        costs = np.array(costs)

        # If temperature is very low, or all costs are effectively equal,
        # fall back to greedy choice to avoid numerical instability.
        if self.temperature < np.finfo(float).eps or np.all(np.isclose(costs, costs[0])):
            return np.argmin(costs)

        # Normalize costs to prevent overflow in exp.
        # Subtracting min cost makes the smallest exponent 0, preventing overflow.
        costs_norm = costs - np.min(costs)
        
        # Calculate exponentials.
        exponentials = np.exp(-costs_norm / self.temperature)
        
        sum_exp = np.sum(exponentials)

        # If all exponentials are zero (due to underflow) or sum is zero, fall back to greedy.
        if sum_exp == 0:
            return np.argmin(costs)
        
        probabilities = exponentials / sum_exp
        
        # Handle potential NaN if sum is 0 or other numerical issues.
        if np.isnan(probabilities).any():
            return np.argmin(costs)
        
        # Re-normalize to ensure sum is exactly 1, correcting for floating-point inaccuracies
        probabilities /= probabilities.sum()
        
        return np.random.choice(num_choices, p=probabilities)
