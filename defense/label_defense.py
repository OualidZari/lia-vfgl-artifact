import numpy as np
import random
from scipy.optimize import minimize
import torch


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

    # Print debugging information
    print("Optimization Result:", result)

    if result.success:
        optimized_proportions = result.x
        optimized_sum_of_squares = (
            -result.fun
        )  # Since we minimized the negative sum of squares
        return optimized_proportions, optimized_sum_of_squares
    else:
        raise ValueError("Optimization failed: " + result.message)


from collections import Counter


def balance_labels(current_labels, optimized_proportions):
    """
    Implement the BalanceLabels algorithm to redistribute labels based on optimized proportions.

    Args:
    current_labels (np.array): Current labels of nodes
    optimized_proportions (np.array): Optimized proportions for each label

    Returns:
    np.array: New labels after redistribution
    """
    total_nodes = len(current_labels)

    # Convert proportions to rounded counts
    optimized_counts = proportions_to_rounded_counts(optimized_proportions, total_nodes)

    # Count current labels using NumPy
    unique_labels, label_counts = np.unique(current_labels, return_counts=True)
    label_counts = dict(zip(unique_labels, label_counts))

    # Calculate in and out for each label
    label_changes = {}
    for i, label in enumerate(unique_labels):
        current_count = label_counts.get(label, 0)
        optimized_count = optimized_counts[i]
        difference = optimized_count - current_count
        label_changes[label] = {"in": max(0, difference), "out": max(0, -difference)}

    # Create a list of labels sorted by 'in' count (descending)
    labels_needing_nodes = sorted(
        label_changes.keys(), key=lambda x: label_changes[x]["in"], reverse=True
    )

    # Keep track of flipped nodes
    flipped_nodes = set()
    new_labels = current_labels.copy()

    # Iterate over labels needing nodes
    for target_label in labels_needing_nodes:
        nodes_needed = label_changes[target_label]["in"]
        if nodes_needed == 0:
            continue

        # Iterate over labels with excess nodes
        for source_label in label_changes.keys():
            if source_label == target_label:
                continue
            nodes_available = label_changes[source_label]["out"]
            if nodes_available == 0:
                continue

            # Determine number of nodes to flip
            nodes_to_flip = min(nodes_needed, nodes_available)

            # Find eligible nodes to flip
            eligible_nodes = [
                i
                for i, label in enumerate(new_labels)
                if label == source_label and i not in flipped_nodes
            ]
            nodes_to_flip = min(nodes_to_flip, len(eligible_nodes))

            # Flip the labels
            for node in random.sample(eligible_nodes, nodes_to_flip):
                new_labels[node] = target_label
                flipped_nodes.add(node)

            # Update counts
            label_changes[target_label]["in"] -= nodes_to_flip
            label_changes[source_label]["out"] -= nodes_to_flip

            nodes_needed -= nodes_to_flip
            if nodes_needed == 0:
                break

    return new_labels


def proportions_to_rounded_counts(proportions, total_nodes):
    """
    Convert proportions to rounded counts while ensuring the total sum equals total_nodes.

    Args:
    proportions (np.array): A NumPy array of label proportions
    total_nodes (int): The total number of nodes in the graph

    Returns:
    np.array: An array of integer counts that sum to total_nodes
    """
    raw_counts = proportions * total_nodes
    rounded_counts = np.floor(raw_counts).astype(int)
    difference = total_nodes - np.sum(rounded_counts)
    sorted_indices = np.argsort(raw_counts - np.floor(raw_counts))[::-1]

    for i in range(int(difference)):
        rounded_counts[sorted_indices[i % len(sorted_indices)]] += 1

    return rounded_counts


def labels_defense(labels, budget):
    # clalculate the class proportions
    labels = labels.numpy()
    unique, counts = np.unique(labels, return_counts=True)
    N = labels.size
    alpha_c = counts / N
    # optimize the proportions
    try:
        optimized_proportions, optimized_sum_of_squares = optimize_proportions(
            alpha_c, budget
        )
        print("Optimized Proportions:", optimized_proportions)
        print("Optimized Sum of Squares:", optimized_sum_of_squares)
    except ValueError as e:
        print(e)
    # balance the labels
    new_labels = balance_labels(labels, optimized_proportions)
    return torch.tensor(new_labels)
