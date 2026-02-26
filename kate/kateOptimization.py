import pandas as pd
import numpy as np
import random
import math
import copy

# ============================================================
# LOAD GENERATOR FILES
# ============================================================

order_itemtypes = pd.read_csv("order_itemtypes.csv", header=None)
order_quantities = pd.read_csv("order_quantities.csv", header=None)
orders_totes = pd.read_csv("orders_totes.csv", header=None)

n_orders = len(order_itemtypes)

# ============================================================
# BUILD DATA STRUCTURE
# ============================================================

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

all_totes = list(tote_data.keys())
all_orders = list(range(n_orders))

# ============================================================
# SIMULATION
# ============================================================

def evaluate_solution(tote_seq, order_priority, alpha=0.5):

    current_time = 0
    circulation = 0

    active_orders = set(order_priority[:4])
    next_order_index = 4

    remaining = {o: 0 for o in all_orders}
    completion = {o: None for o in all_orders}

    # Count total required quantities
    for tote in tote_data:
        for (order, qty) in tote_data[tote]:
            remaining[order] += qty

    for tote in tote_seq:

        # Sort items by priority
        items = sorted(tote_data[tote],
                       key=lambda x: order_priority.index(x[0]))

        for (order, qty) in items:

            # Count circulation penalty if inactive
            if order not in active_orders:
                circulation += qty

            # ALWAYS process the item
            remaining[order] -= qty

            if remaining[order] <= 0 and completion[order] is None:
                completion[order] = current_time + 1  # completes after tote processed

                if order in active_orders:
                    active_orders.remove(order)

                if next_order_index < len(order_priority):
                    active_orders.add(order_priority[next_order_index])
                    next_order_index += 1

        current_time += 1

    # Safety: ensure no None values
    for o in completion:
        if completion[o] is None:
            completion[o] = current_time

    total_completion = sum(completion.values())

    return total_completion + alpha * circulation

# ============================================================
# SIMULATED ANNEALING (JOINT OPTIMIZATION)
# ============================================================

def simulated_annealing(iterations=8000, T0=1000, cooling=0.995):

    tote_seq = all_totes[:]
    order_priority = all_orders[:]

    random.shuffle(tote_seq)
    random.shuffle(order_priority)

    best_tote = tote_seq[:]
    best_order = order_priority[:]

    best_cost = evaluate_solution(best_tote, best_order)
    current_cost = best_cost

    T = T0

    for _ in range(iterations):

        new_tote = tote_seq[:]
        new_order = order_priority[:]

        move_type = random.choice(["tote", "order"])

        if move_type == "tote":
            i, j = random.sample(range(len(new_tote)), 2)
            new_tote[i], new_tote[j] = new_tote[j], new_tote[i]

        else:
            i, j = random.sample(range(len(new_order)), 2)
            new_order[i], new_order[j] = new_order[j], new_order[i]

        new_cost = evaluate_solution(new_tote, new_order)
        delta = new_cost - current_cost

        if delta < 0 or random.random() < math.exp(-delta / T):
            tote_seq = new_tote
            order_priority = new_order
            current_cost = new_cost

            if new_cost < best_cost:
                best_tote = new_tote[:]
                best_order = new_order[:]
                best_cost = new_cost

        T *= cooling
        if T < 1e-6:
            break

    return best_tote, best_order, best_cost

# ============================================================
# RUN
# ============================================================

best_tote_seq, best_order_seq, best_cost = simulated_annealing()

print("Best Objective:", best_cost)

# ============================================================
# GENERATE OUTPUT FILES
# ============================================================

# 1. Tote Sequence
pd.DataFrame(best_tote_seq).to_csv("tote_sequence.csv",
                                   index=False,
                                   header=False)

# 2. Order Activation Priority
pd.DataFrame(best_order_seq).to_csv("order_sequence.csv",
                                    index=False,
                                    header=False)

# 3. Item Release Plan
rows = []
for tote in best_tote_seq:
    items = tote_data[tote]
    items_sorted = sorted(items,
                          key=lambda x: best_order_seq.index(x[0]))
    for (order, qty) in items_sorted:
        rows.append([tote, order, qty])

pd.DataFrame(rows,
             columns=["tote", "order", "quantity"]
             ).to_csv("item_release_sequence.csv",
                      index=False)

print("All outputs generated.")