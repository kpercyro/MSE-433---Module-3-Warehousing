"""
compare_solutions.py
====================
Apples-to-apples comparison of Jeevan's and Kate's SA-based optimizers
for the 4-belt conveyor problem (seed=100, 11 orders, 36 items).

Runs both optimizers, converts each solution into the other's format,
and evaluates all solutions through BOTH simulators.
"""

import sys
import os
import random
import math
import csv
from collections import defaultdict

# ---------------------------------------------------------------------------
# Import Jeevan's module
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'jeevan'))
from conveyor_optimizer import (
    generate_data, get_demands, get_tote_contents, ototal, sname,
    ConveyorSim, SA, optimize_loading_order, NUM_BELTS,
)


# ============================================================
# KATE'S EVALUATOR (adapted to use shared data, no pandas)
# ============================================================

def build_tote_data(data):
    """Build Kate-style tote_data from Jeevan's data structure.

    Returns {tote_id: [(order_id, qty), ...]}  -- matches Kate's format exactly.
    """
    tote_data = {}
    for order_id in range(data['n_orders']):
        for j in range(len(data['order_itemtypes'][order_id])):
            tote = data['orders_totes'][order_id][j]
            qty = data['order_quantities'][order_id][j]
            tote_data.setdefault(tote, []).append((order_id, qty))
    return tote_data


def build_tote_data_enriched(data):
    """Enriched version: {tote_id: [(order_id, item_type, qty), ...]}"""
    tote_data = {}
    for order_id in range(data['n_orders']):
        for j, item_type in enumerate(data['order_itemtypes'][order_id]):
            tote = data['orders_totes'][order_id][j]
            qty = data['order_quantities'][order_id][j]
            tote_data.setdefault(tote, []).append((order_id, item_type, qty))
    return tote_data


def kate_evaluate(tote_seq, order_priority, tote_data, n_orders, alpha=0.5):
    """Kate's evaluate_solution, parameterized (no globals).

    Returns (objective, total_completion, circulation).
    """
    current_time = 0
    circulation = 0

    active_orders = set(order_priority[:4])
    next_order_index = 4

    remaining = {o: 0 for o in range(n_orders)}
    completion = {o: None for o in range(n_orders)}

    for tote in tote_data:
        for (order, qty) in tote_data[tote]:
            remaining[order] += qty

    for tote in tote_seq:
        if tote not in tote_data:
            continue
        items = sorted(tote_data[tote],
                       key=lambda x: order_priority.index(x[0]))
        for (order, qty) in items:
            if order not in active_orders:
                circulation += qty
            remaining[order] -= qty
            if remaining[order] <= 0 and completion[order] is None:
                completion[order] = current_time + 1
                active_orders.discard(order)
                if next_order_index < len(order_priority):
                    active_orders.add(order_priority[next_order_index])
                    next_order_index += 1
        current_time += 1

    for o in completion:
        if completion[o] is None:
            completion[o] = current_time

    total_completion = sum(completion.values())
    objective = total_completion + alpha * circulation
    return objective, total_completion, circulation


def kate_sa(tote_data, all_totes, all_orders,
            iterations=8000, T0=1000, cooling=0.995):
    """Kate's SA, parameterized (no globals)."""
    n_orders = len(all_orders)

    tote_seq = all_totes[:]
    order_priority = all_orders[:]
    random.shuffle(tote_seq)
    random.shuffle(order_priority)

    best_tote = tote_seq[:]
    best_order = order_priority[:]
    best_cost = kate_evaluate(best_tote, best_order, tote_data, n_orders)[0]
    current_cost = best_cost

    T = T0
    for it in range(iterations):
        new_tote = tote_seq[:]
        new_order = order_priority[:]

        if random.choice(["tote", "order"]) == "tote":
            i, j = random.sample(range(len(new_tote)), 2)
            new_tote[i], new_tote[j] = new_tote[j], new_tote[i]
        else:
            i, j = random.sample(range(len(new_order)), 2)
            new_order[i], new_order[j] = new_order[j], new_order[i]

        new_cost = kate_evaluate(new_tote, new_order, tote_data, n_orders)[0]
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

        if (it + 1) % 2000 == 0:
            print(f"    Iter {it+1}: best={best_cost:.2f}, T={T:.2f}")

    return best_tote, best_order, best_cost


# ============================================================
# CONVERSION FUNCTIONS
# ============================================================

def kate_to_belt_queues(order_priority, tote_data, tote_seq):
    """Convert Kate's solution -> Jeevan's belt_queues.

    Simulates Kate's activation pattern to determine which belt
    each subsequent order is assigned to.
    """
    n_orders = len(order_priority)
    belt_queues = [[] for _ in range(4)]

    # Total items per order (from all totes)
    remaining = defaultdict(int)
    for tote in tote_data:
        for (order, qty) in tote_data[tote]:
            remaining[order] += qty

    # First 4 orders -> belts 0-3
    active = {}  # order -> belt
    next_idx = 0
    for b in range(min(4, n_orders)):
        o = order_priority[next_idx]
        belt_queues[b].append(o)
        active[o] = b
        next_idx += 1

    # Simulate tote processing to find when orders complete
    for tote in tote_seq:
        if tote not in tote_data:
            continue
        items = sorted(tote_data[tote],
                       key=lambda x: order_priority.index(x[0]))
        for (order, qty) in items:
            remaining[order] -= qty
            if remaining[order] <= 0 and order in active:
                belt = active[order]
                del active[order]
                if next_idx < n_orders:
                    o = order_priority[next_idx]
                    belt_queues[belt].append(o)
                    active[o] = belt
                    next_idx += 1

    return belt_queues


def kate_to_loading_order(tote_seq, order_priority, belt_queues,
                          tote_data_enriched):
    """Convert Kate's tote sequence -> Jeevan's loading_order.

    Only includes items for the first order on each belt
    (the simulator adds items for subsequent orders dynamically).
    Items are ordered by Kate's optimized tote sequence.
    """
    first_orders = {}  # order_id -> belt
    for b in range(4):
        if belt_queues[b]:
            first_orders[belt_queues[b][0]] = b

    loading_order = []
    for tote in tote_seq:
        if tote not in tote_data_enriched:
            continue
        items = sorted(tote_data_enriched[tote],
                       key=lambda x: order_priority.index(x[0]))
        for (order_id, item_type, qty) in items:
            if order_id in first_orders:
                belt = first_orders[order_id]
                for _ in range(qty):
                    loading_order.append((item_type, belt))

    return loading_order


def jeevan_to_order_priority(belt_queues):
    """Flatten belt_queues -> order_priority (interleaved by depth).

    depth 0: belt0[0], belt1[0], belt2[0], belt3[0]
    depth 1: belt0[1], belt1[1], ...
    """
    order_priority = []
    max_depth = max((len(q) for q in belt_queues), default=0)
    for depth in range(max_depth):
        for b in range(4):
            if depth < len(belt_queues[b]):
                order_priority.append(belt_queues[b][depth])
    return order_priority


def jeevan_to_tote_sequence(belt_queues, data):
    """Derive tote sequence from Jeevan's belt assignment.

    Totes are ordered by when their associated orders appear in the
    interleaved priority, preserving Jeevan's scheduling insight.
    """
    order_priority = jeevan_to_order_priority(belt_queues)

    # order -> its totes (unique, in data order)
    order_totes = defaultdict(list)
    for order_id in range(data['n_orders']):
        for tote in data['orders_totes'][order_id]:
            if tote not in order_totes[order_id]:
                order_totes[order_id].append(tote)

    tote_sequence = []
    seen = set()
    for order_id in order_priority:
        for tote in order_totes[order_id]:
            if tote not in seen:
                tote_sequence.append(tote)
                seen.add(tote)

    # Safety: add any remaining totes
    all_tote_ids = set()
    for order_id in range(data['n_orders']):
        for tote in data['orders_totes'][order_id]:
            all_tote_ids.add(tote)
    for tote in sorted(all_tote_ids):
        if tote not in seen:
            tote_sequence.append(tote)
            seen.add(tote)

    return tote_sequence


# ============================================================
# MAIN
# ============================================================

def main():
    SEED = 100

    print("=" * 70)
    print("CROSS-EVALUATION: Jeevan vs Kate Conveyor Optimizers")
    print("=" * 70)

    # -- 1. Shared data --------------------------------------------------
    data = generate_data(SEED)
    demands = get_demands(data)
    n_orders = data['n_orders']
    total_items = sum(ototal(d) for d in demands)
    tote_data = build_tote_data(data)
    tote_data_rich = build_tote_data_enriched(data)
    all_totes = sorted(tote_data.keys())
    all_orders = list(range(n_orders))

    print(f"\nProblem: {n_orders} orders, {total_items} items, "
          f"{len(all_totes)} totes, {data['n_itemtypes']} item types\n")

    for i in range(n_orders):
        items = ", ".join(f"{q}x {sname(t)}" for t, q in sorted(demands[i].items()))
        print(f"  O{i:2d}: {ototal(demands[i]):2d} items  ({items})")

    sim = ConveyorSim(demands)

    # -- 2. Run Jeevan's optimizer ----------------------------------------
    print(f"\n{'-' * 70}")
    print("Running Jeevan's SA (belt assignment + loading order)...")
    print("-" * 70)

    random.seed(42)
    sa1 = SA(sim, demands, 'total_completion_time', iters=20000)
    j_belts1, _ = sa1.run(verbose=True)

    random.seed(43)
    sa2 = SA(sim, demands, 'makespan', iters=20000)
    j_belts2, _ = sa2.run(verbose=True)

    res1 = sim.simulate(j_belts1)
    res2 = sim.simulate(j_belts2)

    if res1['makespan'] <= res2['makespan']:
        j_belts = j_belts1
        print(f"\n  Selected: total_completion_time optimizer "
              f"(makespan={res1['makespan']:.0f}s)")
    else:
        j_belts = j_belts2
        print(f"\n  Selected: makespan optimizer "
              f"(makespan={res2['makespan']:.0f}s)")

    _, strategies = optimize_loading_order(sim, j_belts, demands)
    best_strat = min(strategies, key=lambda k: strategies[k][1]['makespan'])
    j_loading = strategies[best_strat][0]

    print(f"  Loading strategy: {best_strat}")
    for b in range(4):
        ords = " -> ".join(f"O{o}" for o in j_belts[b])
        ld = sum(ototal(demands[o]) for o in j_belts[b])
        print(f"  Belt {b} [{ld:2d} items]: {ords}")

    # -- 3. Run Kate's optimizer ------------------------------------------
    print(f"\n{'-' * 70}")
    print("Running Kate's SA (tote sequence + order priority)...")
    print("-" * 70)

    random.seed(42)
    k_totes, k_orders, k_cost = kate_sa(tote_data, all_totes, all_orders)

    print(f"  Best objective: {k_cost:.2f}")
    print(f"  Order priority: {k_orders}")
    print(f"  Tote sequence:  {k_totes}")

    # -- 4. Convert solutions ---------------------------------------------
    print(f"\n{'-' * 70}")
    print("Converting solutions between formats...")
    print("-" * 70)

    # Kate -> Jeevan format
    k2j_belts = kate_to_belt_queues(k_orders, tote_data, k_totes)
    k2j_loading = kate_to_loading_order(k_totes, k_orders, k2j_belts,
                                         tote_data_rich)

    print("  Kate -> Jeevan belt_queues:")
    for b in range(4):
        ords = " -> ".join(f"O{o}" for o in k2j_belts[b])
        print(f"    Belt {b}: {ords}")
    print(f"  Kate -> Jeevan loading_order: {len(k2j_loading)} items")

    # Jeevan -> Kate format
    j2k_orders = jeevan_to_order_priority(j_belts)
    j2k_totes = jeevan_to_tote_sequence(j_belts, data)

    print(f"  Jeevan -> Kate order_priority: {j2k_orders}")
    print(f"  Jeevan -> Kate tote_sequence:  {j2k_totes}")

    # -- 5. Naive baselines -----------------------------------------------
    naive_belts = [[] for _ in range(4)]
    for i in range(n_orders):
        naive_belts[i % 4].append(i)
    naive_loading = sim.smart_loading_order(naive_belts)

    naive_totes = sorted(all_totes)
    naive_orders = list(range(n_orders))

    # -- 6. Cross-evaluate ------------------------------------------------
    print(f"\n{'=' * 70}")
    print("RESULTS: All solutions through both simulators")
    print("=" * 70)

    solutions = {
        "Jeevan's SA": {
            'belts': j_belts,
            'loading': j_loading,
            'totes': j2k_totes,
            'orders': j2k_orders,
        },
        "Kate's SA": {
            'belts': k2j_belts,
            'loading': k2j_loading,
            'totes': k_totes,
            'orders': k_orders,
        },
        "Naive baseline": {
            'belts': naive_belts,
            'loading': naive_loading,
            'totes': naive_totes,
            'orders': naive_orders,
        },
    }

    # Compute results
    results = {}
    for name, sol in solutions.items():
        j_res = sim.simulate(sol['belts'], sol['loading'])
        k_obj, k_ct, k_circ = kate_evaluate(
            sol['totes'], sol['orders'], tote_data, n_orders)
        results[name] = {'jeevan': j_res, 'kate_obj': k_obj,
                         'kate_ct': k_ct, 'kate_circ': k_circ}

    # Table header
    hdr = (f"{'Solution':<20s} | {'Makespan':>10s} | {'TotalCT':>10s} | "
           f"{'AvgCT':>8s} | {'Recirc':>6s} | "
           f"{'Kate Obj':>8s} | {'Kate CT':>7s} | {'Circ':>5s}")
    units = (f"{'':20s} | {'(sec)':>10s} | {'(sec)':>10s} | "
             f"{'(sec)':>8s} | {'':>6s} | "
             f"{'':>8s} | {'(steps)':>7s} | {'':>5s}")
    print(f"\n{hdr}")
    print(f"{units}")
    print("-" * len(hdr))

    for name in solutions:
        r = results[name]
        j = r['jeevan']
        print(f"{name:<20s} | {j['makespan']:>10.1f} | "
              f"{j['total_completion_time']:>10.1f} | "
              f"{j['avg_completion_time']:>8.1f} | "
              f"{j['recirculation_events']:>6d} | "
              f"{r['kate_obj']:>8.1f} | "
              f"{r['kate_ct']:>7.0f} | "
              f"{r['kate_circ']:>5.0f}")

    # -- 7. Normalized comparison -----------------------------------------
    print(f"\n{'=' * 70}")
    print("NORMALIZED: % improvement over naive baseline")
    print("=" * 70)

    naive_r = results["Naive baseline"]
    naive_ms = naive_r['jeevan']['makespan']
    naive_tct = naive_r['jeevan']['total_completion_time']
    naive_ko = naive_r['kate_obj']

    hdr2 = (f"{'Solution':<20s} | {'Makespan':>10s} | "
            f"{'TotalCT':>10s} | {'Kate Obj':>10s}")
    print(f"\n{hdr2}")
    print("-" * len(hdr2))

    for name in solutions:
        r = results[name]
        j = r['jeevan']
        ms_imp = (naive_ms - j['makespan']) / naive_ms * 100
        ct_imp = (naive_tct - j['total_completion_time']) / naive_tct * 100
        ko_imp = (naive_ko - r['kate_obj']) / naive_ko * 100
        print(f"{name:<20s} | {ms_imp:>+9.1f}% | "
              f"{ct_imp:>+9.1f}% | {ko_imp:>+9.1f}%")

    # -- 8. Detail: order completion times --------------------------------
    print(f"\n{'=' * 70}")
    print("DETAIL: Order completion times (Jeevan's simulator, seconds)")
    print("=" * 70)

    for name in ["Jeevan's SA", "Kate's SA"]:
        r = results[name]
        cts = sorted(r['jeevan']['order_completion_times'].items(),
                     key=lambda x: x[1])
        print(f"\n  {name}:")
        for o, ct in cts:
            items = ", ".join(f"{q}x {sname(t)}"
                              for t, q in sorted(demands[o].items()))
            print(f"    {ct:6.0f}s : O{o} ({ototal(demands[o])} items: {items})")

    # -- 9. Export CSV ----------------------------------------------------
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'comparison_results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['solution',
                     'makespan_sec', 'total_completion_time_sec',
                     'avg_completion_time_sec', 'recirculation_events',
                     'kate_objective', 'kate_total_completion_steps',
                     'kate_circulation',
                     'makespan_improvement_pct',
                     'total_ct_improvement_pct',
                     'kate_obj_improvement_pct'])
        for name in solutions:
            r = results[name]
            j = r['jeevan']
            ms_imp = (naive_ms - j['makespan']) / naive_ms * 100
            ct_imp = (naive_tct - j['total_completion_time']) / naive_tct * 100
            ko_imp = (naive_ko - r['kate_obj']) / naive_ko * 100
            w.writerow([name,
                        f"{j['makespan']:.1f}",
                        f"{j['total_completion_time']:.1f}",
                        f"{j['avg_completion_time']:.1f}",
                        j['recirculation_events'],
                        f"{r['kate_obj']:.1f}",
                        r['kate_ct'],
                        r['kate_circ'],
                        f"{ms_imp:.1f}",
                        f"{ct_imp:.1f}",
                        f"{ko_imp:.1f}"])

    print(f"\n  CSV saved to {csv_path}")
    print()


if __name__ == '__main__':
    main()
