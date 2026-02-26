import pandas as pd
import numpy as np
import random
import math

# ============================================================
# LOAD GENERATOR FILES (NO HEADERS)
# ============================================================

order_itemtypes = pd.read_csv("order_itemtypes.csv", header=None)
order_quantities = pd.read_csv("order_quantities.csv", header=None)
orders_totes = pd.read_csv("orders_totes.csv", header=None)

n_orders = len(order_itemtypes)

# ============================================================
# RECONSTRUCT DATA STRUCTURE
# ============================================================

# Dictionary: tote -> list of (order, quantity)
tote_data = {}

for order_id in range(n_orders):
    for col in range(order_itemtypes.shape[1]):

        if pd.isna(order_itemtypes.iloc[order_id, col]):
            continue

        tote = int(orders_totes.iloc[order_id, col])
        qty = int(order_quantities.iloc[order_id, col])

        if tote not in tote_data:
            tote_data[tote] = []

        tote_data[tote].append((order_id, qty))

# All totes
all_totes = list(tote_data.keys())

# ============================================================
# SIMULATION FUNCTION
# ============================================================

def evaluate_sequence(sequence):

    current_time = 0
    active_orders = set()
    completion_time = {o: None for o in range(n_orders)}

    # Remaining quantities per order
    remaining = {o: 0 for o in range(n_orders)}
    for tote in tote_data:
        for (order, qty) in tote_data[tote]:
            remaining[order] += qty

    for tote in sequence:

        # Activate orders if space
        for (order, qty) in tote_data[tote]:
            if order not in active_orders:
                if len(active_orders) < 4:
                    active_orders.add(order)

        # Process tote (processing time = 1)
        current_time += 1

        # Remove quantities
        for (order, qty) in tote_data[tote]:
            remaining[order] -= qty

            if remaining[order] <= 0 and completion_time[order] is None:
                completion_time[order] = current_time
                if order in active_orders:
                    active_orders.remove(order)

    return sum(completion_time.values())

# ============================================================
# DETERMINISTIC SIMULATED ANNEALING
# ============================================================

def simulated_annealing(initial_seq,
                        T0=1000,
                        cooling=0.995,
                        iterations=5000):

    current_seq = initial_seq[:]
    best_seq = initial_seq[:]

    current_cost = evaluate_sequence(current_seq)
    best_cost = current_cost

    T = T0

    for _ in range(iterations):

        # Swap move
        new_seq = current_seq[:]
        i, j = random.sample(range(len(new_seq)), 2)
        new_seq[i], new_seq[j] = new_seq[j], new_seq[i]

        new_cost = evaluate_sequence(new_seq)
        delta = new_cost - current_cost

        if delta < 0 or random.random() < math.exp(-delta / T):
            current_seq = new_seq
            current_cost = new_cost

            if new_cost < best_cost:
                best_seq = new_seq
                best_cost = new_cost

        T *= cooling
        if T < 1e-6:
            break

    return best_seq, best_cost

# ============================================================
# RUN OPTIMIZATION
# ============================================================

random.shuffle(all_totes)

best_sequence, best_cost = simulated_annealing(all_totes)

print("Best Total Order Completion Time:", best_cost)

# ============================================================
# SAVE OUTPUT (MATCHING GENERATOR STYLE: NO HEADER)
# ============================================================

pd.DataFrame(best_sequence).to_csv("tote_sequence.csv",
                                   index=False,
                                   header=False)

print("Optimized tote sequence saved to tote_sequence.csv")