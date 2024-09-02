import numpy as np
from scipy.optimize import minimize

def optimize_proportions(initial_proportions, budget, max_iter=10000):
    n = len(initial_proportions)

    # Objective function (negative sum of squares)
    def objective(alpha):
        return -np.sum(alpha**2)

    # Equality constraint: sum of proportions must equal 1
    def equality_constraint(alpha):
        return np.sum(alpha) - 1

    # Inequality constraint: sum of negative changes must be within budget
    def inequality_constraint(alpha):
        return budget - np.sum(np.maximum(0, initial_proportions - alpha))

    # Non-negativity constraint
    bounds = [(0, None) for _ in range(n)]

    # Initial guess (starting point)
    initial_guess = np.array(initial_proportions)
    initial_guess = initial_guess / np.sum(initial_guess)  # Normalize to sum to 1

    # Define constraints in the form required by `scipy.optimize.minimize`
    constraints = [
        {"type": "eq", "fun": equality_constraint},
        {"type": "ineq", "fun": inequality_constraint},
    ]

    # Perform the optimization with increased iteration limit
    result = minimize(
        objective,
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": max_iter, "disp": True},
    )

    if result.success:
        optimized_proportions = result.x
        optimized_sum_of_squares = -result.fun  # Since we minimized the negative sum of squares
        return optimized_proportions, optimized_sum_of_squares
    else:
        print("Optimization Result:", result)
        return initial_proportions, None  # Return initial proportions and None for failure to optimize

def main():
    initial_proportions = np.array([0.03170448, 0.15575916, 0.10282141, 0.03941245, 0.37507272,
                                    0.02239674, 0.03541303, 0.05948226, 0.1567772, 0.02116056])
    budget = 0.1  # Example budget

    optimized_proportions, optimized_sum_of_squares = optimize_proportions(initial_proportions, budget)
    if optimized_sum_of_squares is not None:
        print("Optimized Proportions:", optimized_proportions)
        print("Optimized Sum of Squares:", optimized_sum_of_squares)
    else:
        print("Optimization failed.")

if __name__ == "__main__":
    main()
