"""
stochastic_comparison.py
========================
Runs 100 stochastic simulations (different random seeds) for each SA-based
solver and compares against deterministic baselines.

All output is written to results/stochastic/:
  - stochastic_results.csv        (raw per-run data)
  - stochastic_summary.csv        (statistics: mean, std, min, max, median)
  - stochastic_boxplot.png         (box plots of makespan by solver)
  - stochastic_violin.png          (violin plots of makespan + composite)
  - stochastic_heatmap.png         (metric heatmap across solvers)
  - stochastic_distributions.png   (histograms of makespan distributions)
  - stochastic_radar.png           (radar chart of normalized metrics)
  - stochastic_improvement.png     (% improvement over naive baseline)
  - stochastic_seeds.png           (per-seed line plots)
"""

import sys
import os
import random
import math
import csv
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Imports (same as compare_solutions.py)
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_script_dir, 'jeevan'))
from conveyor_optimizer import (
    generate_data, get_demands, get_tote_contents, ototal, sname,
    ConveyorSim, SA, optimize_loading_order, NUM_BELTS,
)

import importlib.util

def _load_event_solver(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_arkhan_mod = _load_event_solver("arkhan_solver_stoch", os.path.join(_script_dir, 'arkhan', 'solver.py'))
arkhan_load_problem = _arkhan_mod.load_problem
arkhan_run_sa = _arkhan_mod.run_simulated_annealing
ArkhanParams = _arkhan_mod.SolverParams

_arkhan_sof_mod = _load_event_solver("arkhan_sof_solver_stoch", os.path.join(_script_dir, 'arkhan', 'sof_solver.py'))
arkhan_sof_load_problem = _arkhan_sof_mod.load_problem
arkhan_sof_build = _arkhan_sof_mod.build_sof_solution
arkhan_sof_simulate = _arkhan_sof_mod.simulate
ArkhanSofConveyorParams = _arkhan_sof_mod.ConveyorParams

_liam_mod = _load_event_solver("liam_solver_stoch", os.path.join(_script_dir, 'liam', 'solver.py'))
liam_load_problem = _liam_mod.load_problem
liam_run_sa = _liam_mod.run_simulated_annealing
LiamParams = _liam_mod.SolverParams


# ---------------------------------------------------------------------------
# Kate's evaluator + SA (from compare_solutions.py)
# ---------------------------------------------------------------------------
def build_tote_data(data):
    tote_data = {}
    for order_id in range(data['n_orders']):
        for j in range(len(data['order_itemtypes'][order_id])):
            tote = data['orders_totes'][order_id][j]
            qty = data['order_quantities'][order_id][j]
            tote_data.setdefault(tote, []).append((order_id, qty))
    return tote_data

def build_tote_data_enriched(data):
    tote_data = {}
    for order_id in range(data['n_orders']):
        for j, item_type in enumerate(data['order_itemtypes'][order_id]):
            tote = data['orders_totes'][order_id][j]
            qty = data['order_quantities'][order_id][j]
            tote_data.setdefault(tote, []).append((order_id, item_type, qty))
    return tote_data

def kate_evaluate(tote_seq, order_priority, tote_data, n_orders, alpha=0.5):
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
        items = sorted(tote_data[tote], key=lambda x: order_priority.index(x[0]))
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

def kate_sa(tote_data, all_totes, all_orders, iterations=8000, T0=1000, cooling=0.995):
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
    return best_tote, best_order, best_cost

# ---------------------------------------------------------------------------
# Conversion helpers (from compare_solutions.py)
# ---------------------------------------------------------------------------
def kate_to_belt_queues(order_priority, tote_data, tote_seq):
    n_orders = len(order_priority)
    belt_queues = [[] for _ in range(4)]
    remaining = defaultdict(int)
    for tote in tote_data:
        for (order, qty) in tote_data[tote]:
            remaining[order] += qty
    active = {}
    next_idx = 0
    for b in range(min(4, n_orders)):
        o = order_priority[next_idx]
        belt_queues[b].append(o)
        active[o] = b
        next_idx += 1
    for tote in tote_seq:
        if tote not in tote_data:
            continue
        items = sorted(tote_data[tote], key=lambda x: order_priority.index(x[0]))
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

def kate_to_loading_order(tote_seq, order_priority, belt_queues, tote_data_enriched):
    first_orders = {}
    for b in range(4):
        if belt_queues[b]:
            first_orders[belt_queues[b][0]] = b
    loading_order = []
    for tote in tote_seq:
        if tote not in tote_data_enriched:
            continue
        items = sorted(tote_data_enriched[tote], key=lambda x: order_priority.index(x[0]))
        for (order_id, item_type, qty) in items:
            if order_id in first_orders:
                belt = first_orders[order_id]
                for _ in range(qty):
                    loading_order.append((item_type, belt))
    return loading_order

def belt_queues_to_order_priority(belt_queues):
    order_priority = []
    max_depth = max((len(q) for q in belt_queues), default=0)
    for depth in range(max_depth):
        for b in range(4):
            if depth < len(belt_queues[b]):
                order_priority.append(belt_queues[b][depth])
    return order_priority

def belt_queues_to_tote_sequence(belt_queues, data):
    order_priority = belt_queues_to_order_priority(belt_queues)
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
    all_tote_ids = set()
    for order_id in range(data['n_orders']):
        for tote in data['orders_totes'][order_id]:
            all_tote_ids.add(tote)
    for tote in sorted(all_tote_ids):
        if tote not in seen:
            tote_sequence.append(tote)
            seen.add(tote)
    return tote_sequence

def manjary_lpt(demands, num_belts=4):
    sizes = sorted([(ototal(d), i) for i, d in enumerate(demands)], reverse=True)
    qs = [[] for _ in range(num_belts)]
    load = [0] * num_belts
    for sz, oi in sizes:
        b = min(range(num_belts), key=lambda x: load[x])
        qs[b].append(oi)
        load[b] += sz
    return qs, load


# ---------------------------------------------------------------------------
# Compute composite score (same formula as compare_solutions.py)
# ---------------------------------------------------------------------------
def compute_composite(j_res, naive_ms, naive_tct, naive_avg, naive_spread):
    cts = j_res['order_completion_times']
    spread = max(cts.values()) - min(cts.values()) if cts else 0
    composite = (
        0.35 * (j_res['makespan'] / naive_ms) +
        0.35 * (j_res['avg_completion_time'] / naive_avg) +
        0.15 * (j_res['total_completion_time'] / naive_tct) +
        0.15 * (spread / naive_spread if naive_spread > 0 else 1.0)
    ) * 100
    return composite, spread


# ---------------------------------------------------------------------------
# Run a single trial for each stochastic solver
# ---------------------------------------------------------------------------
def run_arkhan_sa_trial(csv_dir, seed_val):
    problem = arkhan_load_problem(
        os.path.join(csv_dir, 'order_itemtypes.csv'),
        os.path.join(csv_dir, 'order_quantities.csv'),
        os.path.join(csv_dir, 'orders_totes.csv'),
    )
    params = ArkhanParams(iterations=5000, seed=seed_val, t0=1.0, alpha=0.99)
    best_sol, best_res, _ = arkhan_run_sa(problem, params, verbose=False)
    return best_sol.belt_queues, best_sol.tote_sequence, best_res

def run_liam_sa_trial(csv_dir, seed_val):
    problem = liam_load_problem(
        os.path.join(csv_dir, 'order_itemtypes.csv'),
        os.path.join(csv_dir, 'order_quantities.csv'),
        os.path.join(csv_dir, 'orders_totes.csv'),
    )
    params = LiamParams(iterations=5000, seed=seed_val, t0=1.0, alpha=0.99)
    best_sol, best_res, _ = liam_run_sa(problem, params, verbose=False)
    return best_sol.belt_queues, best_sol.tote_sequence, best_res

def run_jeevan_sa_trial(sim, demands, seed_val):
    """Jeevan's multi-restart SA with a specific seed base."""
    n_restarts = 3
    candidates = []
    for restart in range(n_restarts):
        for obj_name in ('total_completion_time', 'makespan'):
            s = seed_val + restart * 10 + (0 if obj_name == 'total_completion_time' else 1)
            random.seed(s)
            sa = SA(sim, demands, obj_name, iters=40000, T0=200, alpha=0.99985)
            sol, _ = sa.run(verbose=False)
            res = sim.simulate(sol)
            candidates.append((res['makespan'], sol, res))
    _, best_sol, best_res = min(candidates, key=lambda x: x[0])
    return best_sol

def run_kate_sa_trial(tote_data, tote_data_rich, all_totes, all_orders, data, seed_val):
    """Kate's SA with a specific seed."""
    random.seed(seed_val)
    k_totes, k_orders, k_cost = kate_sa(tote_data, all_totes, all_orders)
    k2j_belts = kate_to_belt_queues(k_orders, tote_data, k_totes)
    k2j_loading = kate_to_loading_order(k_totes, k_orders, k2j_belts, tote_data_rich)
    return k2j_belts, k2j_loading, k_totes, k_orders


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    N_SIMS = 100
    DATA_SEED = 100

    print("=" * 70)
    print(f"STOCHASTIC COMPARISON: {N_SIMS} simulations per SA solver")
    print("=" * 70)

    # -- 1. Shared data --
    data = generate_data(DATA_SEED)
    demands = get_demands(data)
    n_orders = data['n_orders']
    total_items = sum(ototal(d) for d in demands)
    tote_data = build_tote_data(data)
    tote_data_rich = build_tote_data_enriched(data)
    all_totes = sorted(tote_data.keys())
    all_orders = list(range(n_orders))
    sim = ConveyorSim(demands)
    csv_dir = os.path.join(_script_dir, 'data', 'seed100')
    _results_dir = os.path.join(_script_dir, 'results', 'stochastic')
    os.makedirs(_results_dir, exist_ok=True)

    print(f"Problem: {n_orders} orders, {total_items} items, "
          f"{len(all_totes)} totes, {data['n_itemtypes']} item types\n")

    # -- 2. Deterministic baselines (run once) --
    print("Running deterministic baselines...")

    # Naive
    naive_belts = [[] for _ in range(4)]
    for i in range(n_orders):
        naive_belts[i % 4].append(i)
    naive_loading = sim.smart_loading_order(naive_belts)
    naive_res = sim.simulate(naive_belts, naive_loading)
    naive_ms = naive_res['makespan']
    naive_tct = naive_res['total_completion_time']
    naive_avg = naive_res['avg_completion_time']
    naive_cts = naive_res['order_completion_times']
    naive_spread = max(naive_cts.values()) - min(naive_cts.values())

    # Manjary LPT
    m_belts, _ = manjary_lpt(demands)
    m_loading = sim.smart_loading_order(m_belts)
    m_res = sim.simulate(m_belts, m_loading)
    m_composite, m_spread = compute_composite(m_res, naive_ms, naive_tct, naive_avg, naive_spread)

    # Arkhan SOF
    prob = arkhan_sof_load_problem(
        os.path.join(csv_dir, 'order_itemtypes.csv'),
        os.path.join(csv_dir, 'order_quantities.csv'),
        os.path.join(csv_dir, 'orders_totes.csv'),
    )
    asof_queues, asof_tote_seq = arkhan_sof_build(prob)
    asof_loading = sim.smart_loading_order(asof_queues)
    asof_res = sim.simulate(asof_queues, asof_loading)
    asof_composite, asof_spread = compute_composite(asof_res, naive_ms, naive_tct, naive_avg, naive_spread)

    print(f"  Naive baseline:  makespan={naive_ms:.1f}s")
    print(f"  Manjary (LPT):   makespan={m_res['makespan']:.1f}s, composite={m_composite:.1f}")
    print(f"  Arkhan (SOF):    makespan={asof_res['makespan']:.1f}s, composite={asof_composite:.1f}")

    # -- 3. Stochastic trials --
    # Each solver gets seeds: 1, 2, 3, ..., 100
    solvers = ["Arkhan (SA)", "Jeevan (SA)", "Kate (SA)", "Liam (SA)"]
    all_results = {s: [] for s in solvers}

    # Also store deterministic results for unified output
    all_results["Manjary (LPT)"] = []
    all_results["Arkhan (SOF)"] = []
    all_results["Naive baseline"] = []

    for trial in range(N_SIMS):
        seed_val = trial + 1
        if (trial + 1) % 10 == 0 or trial == 0:
            print(f"\n--- Trial {trial + 1}/{N_SIMS} (seed={seed_val}) ---")

        # Arkhan SA
        try:
            a_belts, a_tote_seq, a_own_res = run_arkhan_sa_trial(csv_dir, seed_val)
            a_loading = sim.smart_loading_order(a_belts)
            a_res = sim.simulate(a_belts, a_loading)
            a_totes = a_tote_seq
            a_orders = belt_queues_to_order_priority(a_belts)
            k_obj_a, _, _ = kate_evaluate(a_totes, a_orders, tote_data, n_orders)
            a_composite, a_spread = compute_composite(a_res, naive_ms, naive_tct, naive_avg, naive_spread)
            all_results["Arkhan (SA)"].append({
                'seed': seed_val, 'makespan': a_res['makespan'],
                'total_ct': a_res['total_completion_time'],
                'avg_ct': a_res['avg_completion_time'],
                'recirculation': a_res['recirculation_events'],
                'kate_obj': k_obj_a, 'composite': a_composite, 'spread': a_spread,
                'items_delivered': a_res['items_sorted'],
            })
        except Exception as e:
            print(f"  Arkhan SA trial {trial+1} failed: {e}")

        # Liam SA
        try:
            l_belts, l_tote_seq, l_own_res = run_liam_sa_trial(csv_dir, seed_val)
            l_loading = sim.smart_loading_order(l_belts)
            l_res = sim.simulate(l_belts, l_loading)
            l_totes = l_tote_seq
            l_orders = belt_queues_to_order_priority(l_belts)
            k_obj_l, _, _ = kate_evaluate(l_totes, l_orders, tote_data, n_orders)
            l_composite, l_spread = compute_composite(l_res, naive_ms, naive_tct, naive_avg, naive_spread)
            all_results["Liam (SA)"].append({
                'seed': seed_val, 'makespan': l_res['makespan'],
                'total_ct': l_res['total_completion_time'],
                'avg_ct': l_res['avg_completion_time'],
                'recirculation': l_res['recirculation_events'],
                'kate_obj': k_obj_l, 'composite': l_composite, 'spread': l_spread,
                'items_delivered': l_res['items_sorted'],
            })
        except Exception as e:
            print(f"  Liam SA trial {trial+1} failed: {e}")

        # Jeevan SA
        try:
            j_belts = run_jeevan_sa_trial(sim, demands, seed_val * 100)
            j_loading = sim.smart_loading_order(j_belts)
            j_res = sim.simulate(j_belts, j_loading)
            j_totes = belt_queues_to_tote_sequence(j_belts, data)
            j_orders = belt_queues_to_order_priority(j_belts)
            k_obj_j, _, _ = kate_evaluate(j_totes, j_orders, tote_data, n_orders)
            j_composite, j_spread = compute_composite(j_res, naive_ms, naive_tct, naive_avg, naive_spread)
            all_results["Jeevan (SA)"].append({
                'seed': seed_val, 'makespan': j_res['makespan'],
                'total_ct': j_res['total_completion_time'],
                'avg_ct': j_res['avg_completion_time'],
                'recirculation': j_res['recirculation_events'],
                'kate_obj': k_obj_j, 'composite': j_composite, 'spread': j_spread,
                'items_delivered': j_res['items_sorted'],
            })
        except Exception as e:
            print(f"  Jeevan SA trial {trial+1} failed: {e}")

        # Kate SA
        try:
            k_belts, k_loading, k_totes, k_orders = run_kate_sa_trial(
                tote_data, tote_data_rich, all_totes, all_orders, data, seed_val * 1000)
            k_res = sim.simulate(k_belts, k_loading)
            k_obj_k, _, _ = kate_evaluate(k_totes, k_orders, tote_data, n_orders)
            k_composite, k_spread = compute_composite(k_res, naive_ms, naive_tct, naive_avg, naive_spread)
            all_results["Kate (SA)"].append({
                'seed': seed_val, 'makespan': k_res['makespan'],
                'total_ct': k_res['total_completion_time'],
                'avg_ct': k_res['avg_completion_time'],
                'recirculation': k_res['recirculation_events'],
                'kate_obj': k_obj_k, 'composite': k_composite, 'spread': k_spread,
                'items_delivered': k_res['items_sorted'],
            })
        except Exception as e:
            print(f"  Kate SA trial {trial+1} failed: {e}")

    # Add deterministic results (repeated N_SIMS times for consistent comparison)
    naive_composite, naive_spread_val = compute_composite(naive_res, naive_ms, naive_tct, naive_avg, naive_spread)
    naive_ko, _, _ = kate_evaluate(sorted(all_totes), list(range(n_orders)), tote_data, n_orders)
    for _ in range(N_SIMS):
        all_results["Naive baseline"].append({
            'seed': 0, 'makespan': naive_ms,
            'total_ct': naive_tct, 'avg_ct': naive_avg,
            'recirculation': naive_res['recirculation_events'],
            'kate_obj': naive_ko, 'composite': naive_composite, 'spread': naive_spread,
            'items_delivered': naive_res['items_sorted'],
        })
        m_ko, _, _ = kate_evaluate(
            belt_queues_to_tote_sequence(m_belts, data),
            belt_queues_to_order_priority(m_belts), tote_data, n_orders)
        all_results["Manjary (LPT)"].append({
            'seed': 0, 'makespan': m_res['makespan'],
            'total_ct': m_res['total_completion_time'], 'avg_ct': m_res['avg_completion_time'],
            'recirculation': m_res['recirculation_events'],
            'kate_obj': m_ko, 'composite': m_composite, 'spread': m_spread,
            'items_delivered': m_res['items_sorted'],
        })
        asof_ko, _, _ = kate_evaluate(asof_tote_seq, belt_queues_to_order_priority(asof_queues),
                                       tote_data, n_orders)
        all_results["Arkhan (SOF)"].append({
            'seed': 0, 'makespan': asof_res['makespan'],
            'total_ct': asof_res['total_completion_time'], 'avg_ct': asof_res['avg_completion_time'],
            'recirculation': asof_res['recirculation_events'],
            'kate_obj': asof_ko, 'composite': asof_composite, 'spread': asof_spread,
            'items_delivered': asof_res['items_sorted'],
        })

    # -- 4. Compute statistics --
    print(f"\n{'=' * 70}")
    print("STOCHASTIC RESULTS SUMMARY")
    print("=" * 70)

    display_order = ["Arkhan (SA)", "Liam (SA)", "Jeevan (SA)", "Kate (SA)",
                     "Arkhan (SOF)", "Manjary (LPT)", "Naive baseline"]

    summary = {}
    for name in display_order:
        runs = all_results[name]
        if not runs:
            continue
        metrics = {}
        for key in ['makespan', 'total_ct', 'avg_ct', 'recirculation', 'kate_obj', 'composite', 'spread']:
            vals = [r[key] for r in runs]
            metrics[key] = {
                'mean': statistics.mean(vals),
                'std': statistics.stdev(vals) if len(vals) > 1 else 0,
                'min': min(vals),
                'max': max(vals),
                'median': statistics.median(vals),
                'q25': sorted(vals)[len(vals) // 4],
                'q75': sorted(vals)[3 * len(vals) // 4],
            }
        summary[name] = metrics

    # Print table
    print(f"\n{'Solver':<20s} | {'Makespan (s)':>30s} | {'Composite':>30s}")
    print(f"{'':20s} | {'mean ± std (min-max)':>30s} | {'mean ± std (min-max)':>30s}")
    print("-" * 85)
    for name in display_order:
        if name not in summary:
            continue
        ms = summary[name]['makespan']
        co = summary[name]['composite']
        print(f"{name:<20s} | {ms['mean']:>6.1f} ± {ms['std']:>5.1f} ({ms['min']:>5.1f}-{ms['max']:>6.1f}) | "
              f"{co['mean']:>6.1f} ± {co['std']:>5.1f} ({co['min']:>5.1f}-{co['max']:>6.1f})")

    print(f"\n  Detailed metrics (mean ± std):")
    print(f"  {'Solver':<20s} | {'AvgCT':>14s} | {'TotalCT':>14s} | {'Spread':>14s} | {'Recirc':>10s} | {'Kate Obj':>14s}")
    print(f"  {'-'*95}")
    for name in display_order:
        if name not in summary:
            continue
        s = summary[name]
        print(f"  {name:<20s} | "
              f"{s['avg_ct']['mean']:>6.1f}±{s['avg_ct']['std']:>5.1f} | "
              f"{s['total_ct']['mean']:>6.1f}±{s['total_ct']['std']:>5.1f} | "
              f"{s['spread']['mean']:>6.1f}±{s['spread']['std']:>5.1f} | "
              f"{s['recirculation']['mean']:>4.1f}±{s['recirculation']['std']:>3.1f} | "
              f"{s['kate_obj']['mean']:>6.1f}±{s['kate_obj']['std']:>5.1f}")

    # % improvement over naive
    print(f"\n  % improvement over naive (mean):")
    for name in display_order:
        if name == "Naive baseline" or name not in summary:
            continue
        ms_imp = (naive_ms - summary[name]['makespan']['mean']) / naive_ms * 100
        comp_imp = 100 - summary[name]['composite']['mean']
        print(f"    {name:<20s}  composite: {comp_imp:>+6.1f}%  makespan: {ms_imp:>+6.1f}%")

    # -- 5. Export raw CSV --
    raw_csv = os.path.join(_results_dir, 'stochastic_results.csv')
    with open(raw_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['solver', 'seed', 'makespan_sec', 'total_ct_sec', 'avg_ct_sec',
                     'spread_sec', 'recirculation', 'kate_obj', 'composite_score',
                     'items_delivered'])
        for name in display_order:
            for r in all_results[name]:
                w.writerow([name, r['seed'], f"{r['makespan']:.1f}",
                            f"{r['total_ct']:.1f}", f"{r['avg_ct']:.1f}",
                            f"{r['spread']:.0f}", r['recirculation'],
                            f"{r['kate_obj']:.1f}", f"{r['composite']:.1f}",
                            r['items_delivered']])
    print(f"\n  Raw results: {raw_csv}")

    # -- 6. Export summary CSV --
    sum_csv = os.path.join(_results_dir, 'stochastic_summary.csv')
    with open(sum_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['solver', 'metric', 'mean', 'std', 'min', 'q25', 'median', 'q75', 'max', 'n_trials'])
        for name in display_order:
            if name not in summary:
                continue
            for metric in ['makespan', 'total_ct', 'avg_ct', 'spread', 'recirculation', 'kate_obj', 'composite']:
                s = summary[name][metric]
                w.writerow([name, metric, f"{s['mean']:.2f}", f"{s['std']:.2f}",
                            f"{s['min']:.2f}", f"{s['q25']:.2f}", f"{s['median']:.2f}",
                            f"{s['q75']:.2f}", f"{s['max']:.2f}", len(all_results[name])])
    print(f"  Summary stats: {sum_csv}")

    # -- 7. Generate plots --
    print(f"\n  Generating plots...")
    generate_plots(all_results, summary, display_order, naive_ms, _results_dir)

    print(f"\n{'=' * 70}")
    print("DONE")
    print("=" * 70)


# ---------------------------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------------------------
def generate_plots(all_results, summary, display_order, naive_ms, outdir):
    # Color scheme
    colors = {
        "Arkhan (SA)": "#2196F3",
        "Liam (SA)": "#4CAF50",
        "Jeevan (SA)": "#FF9800",
        "Kate (SA)": "#E91E63",
        "Arkhan (SOF)": "#9C27B0",
        "Manjary (LPT)": "#795548",
        "Naive baseline": "#9E9E9E",
    }

    sa_solvers = ["Arkhan (SA)", "Liam (SA)", "Jeevan (SA)", "Kate (SA)"]
    all_solvers = [s for s in display_order if s in all_results and all_results[s]]

    # -------------------------------------------------------------------------
    # 1. Box plot: Makespan distribution
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    data_box = []
    labels_box = []
    box_colors = []
    for name in all_solvers:
        vals = [r['makespan'] for r in all_results[name]]
        data_box.append(vals)
        labels_box.append(name.replace(" (", "\n("))
        box_colors.append(colors.get(name, '#666'))

    bp = ax.boxplot(data_box, labels=labels_box, patch_artist=True, widths=0.6,
                    medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(y=naive_ms, color='gray', linestyle='--', linewidth=1.5, label=f'Naive baseline ({naive_ms:.0f}s)')
    ax.set_ylabel('Makespan (seconds)', fontsize=12)
    ax.set_title(f'Makespan Distribution Across {len(all_results[sa_solvers[0]])} Stochastic Trials', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'stochastic_boxplot.png'), dpi=150)
    plt.close()
    print(f"    stochastic_boxplot.png")

    # -------------------------------------------------------------------------
    # 2. Violin plot: Makespan + Composite side by side
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for ax, metric, title in [(ax1, 'makespan', 'Makespan (seconds)'),
                                (ax2, 'composite', 'Composite Score (lower=better)')]:
        data_v = []
        positions = []
        for i, name in enumerate(all_solvers):
            vals = [r[metric] for r in all_results[name]]
            data_v.append(vals)
            positions.append(i)

        parts = ax.violinplot(data_v, positions=positions, showmeans=True,
                              showmedians=True, showextrema=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors.get(all_solvers[i], '#666'))
            pc.set_alpha(0.6)
        parts['cmeans'].set_color('red')
        parts['cmedians'].set_color('black')

        ax.set_xticks(positions)
        ax.set_xticklabels([s.replace(" (", "\n(") for s in all_solvers], fontsize=9)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.grid(axis='y', alpha=0.3)

        if metric == 'makespan':
            ax.axhline(y=naive_ms, color='gray', linestyle='--', linewidth=1.5)
        elif metric == 'composite':
            ax.axhline(y=100, color='gray', linestyle='--', linewidth=1.5, label='Naive=100')
            ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'stochastic_violin.png'), dpi=150)
    plt.close()
    print(f"    stochastic_violin.png")

    # -------------------------------------------------------------------------
    # 3. Histogram: Makespan distributions per SA solver
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, name in enumerate(sa_solvers):
        ax = axes[idx // 2][idx % 2]
        vals = [r['makespan'] for r in all_results[name]]
        n_unique = len(set(vals))
        n_bins = min(max(n_unique, 5), 25)
        ax.hist(vals, bins=n_bins, color=colors[name], alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axvline(x=statistics.mean(vals), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {statistics.mean(vals):.1f}s')
        ax.axvline(x=min(vals), color='green', linestyle=':', linewidth=1.5,
                   label=f'Best: {min(vals):.1f}s')
        ax.axvline(x=naive_ms, color='gray', linestyle='--', linewidth=1.5,
                   label=f'Naive: {naive_ms:.0f}s')
        ax.set_xlabel('Makespan (seconds)', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle(f'Makespan Distributions ({len(all_results[sa_solvers[0]])} trials each)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'stochastic_distributions.png'), dpi=150)
    plt.close()
    print(f"    stochastic_distributions.png")

    # -------------------------------------------------------------------------
    # 4. Heatmap: Mean metrics across all solvers
    # -------------------------------------------------------------------------
    metrics_to_show = ['makespan', 'avg_ct', 'total_ct', 'spread', 'recirculation', 'kate_obj', 'composite']
    metric_labels = ['Makespan\n(s)', 'Avg CT\n(s)', 'Total CT\n(s)', 'Spread\n(s)',
                     'Recirc', 'Kate\nObj', 'Composite']

    heatmap_data = []
    heatmap_labels = []
    for name in all_solvers:
        if name not in summary:
            continue
        row = [summary[name][m]['mean'] for m in metrics_to_show]
        heatmap_data.append(row)
        heatmap_labels.append(name)

    heatmap_arr = np.array(heatmap_data)
    # Normalize each column to [0, 1] for color mapping
    col_min = heatmap_arr.min(axis=0)
    col_max = heatmap_arr.max(axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1
    heatmap_norm = (heatmap_arr - col_min) / col_range

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(heatmap_norm, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_yticks(range(len(heatmap_labels)))
    ax.set_yticklabels(heatmap_labels, fontsize=10)

    # Annotate with actual values
    for i in range(len(heatmap_labels)):
        for j in range(len(metrics_to_show)):
            val = heatmap_arr[i, j]
            fmt = f"{val:.0f}" if val > 10 else f"{val:.1f}"
            text_color = 'white' if heatmap_norm[i, j] > 0.65 else 'black'
            ax.text(j, i, fmt, ha='center', va='center', fontsize=9, color=text_color, fontweight='bold')

    ax.set_title('Mean Metric Values Across Solvers (green=better, red=worse)', fontsize=13)
    plt.colorbar(im, ax=ax, label='Normalized (0=best, 1=worst)')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'stochastic_heatmap.png'), dpi=150)
    plt.close()
    print(f"    stochastic_heatmap.png")

    # -------------------------------------------------------------------------
    # 5. Radar chart: Normalized mean metrics for SA solvers
    # -------------------------------------------------------------------------
    radar_metrics = ['makespan', 'avg_ct', 'composite', 'kate_obj', 'spread']
    radar_labels = ['Makespan', 'Avg CT', 'Composite', 'Kate Obj', 'Spread']
    n_metrics = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Normalize: lower is better, so invert to make bigger = better for radar
    radar_min = {m: min(summary[s][m]['mean'] for s in sa_solvers if s in summary) for m in radar_metrics}
    radar_max = {m: max(summary[s][m]['mean'] for s in sa_solvers if s in summary) for m in radar_metrics}

    for name in sa_solvers:
        if name not in summary:
            continue
        vals = []
        for m in radar_metrics:
            r = radar_max[m] - radar_min[m]
            if r == 0:
                vals.append(1.0)
            else:
                # Invert: lower metric = higher on radar = better
                vals.append(1.0 - (summary[name][m]['mean'] - radar_min[m]) / r)
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', linewidth=2, label=name, color=colors[name])
        ax.fill(angles, vals, alpha=0.1, color=colors[name])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title('SA Solver Performance Comparison\n(larger area = better)', fontsize=13, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'stochastic_radar.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    stochastic_radar.png")

    # -------------------------------------------------------------------------
    # 6. Improvement bar chart
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Makespan improvement
    imp_names = [s for s in all_solvers if s != "Naive baseline"]
    ms_imps = [(naive_ms - summary[s]['makespan']['mean']) / naive_ms * 100 for s in imp_names]
    bar_colors = [colors.get(s, '#666') for s in imp_names]

    bars = ax1.barh(range(len(imp_names)), ms_imps, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_yticks(range(len(imp_names)))
    ax1.set_yticklabels(imp_names, fontsize=10)
    ax1.set_xlabel('% Improvement over Naive', fontsize=11)
    ax1.set_title('Mean Makespan Improvement', fontsize=13)
    ax1.axvline(x=0, color='black', linewidth=1)
    ax1.grid(axis='x', alpha=0.3)
    for i, v in enumerate(ms_imps):
        ax1.text(v + 0.5 if v >= 0 else v - 3, i, f'{v:.1f}%', va='center', fontsize=9, fontweight='bold')

    # Composite improvement
    comp_imps = [100 - summary[s]['composite']['mean'] for s in imp_names]

    bars = ax2.barh(range(len(imp_names)), comp_imps, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_yticks(range(len(imp_names)))
    ax2.set_yticklabels(imp_names, fontsize=10)
    ax2.set_xlabel('% Improvement over Naive (composite)', fontsize=11)
    ax2.set_title('Mean Composite Score Improvement', fontsize=13)
    ax2.axvline(x=0, color='black', linewidth=1)
    ax2.grid(axis='x', alpha=0.3)
    for i, v in enumerate(comp_imps):
        ax2.text(v + 0.5 if v >= 0 else v - 3, i, f'{v:.1f}%', va='center', fontsize=9, fontweight='bold')

    plt.suptitle(f'Solver Performance vs Naive Baseline (mean over {len(all_results[sa_solvers[0]])} trials)',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'stochastic_improvement.png'), dpi=150)
    plt.close()
    print(f"    stochastic_improvement.png")

    # -------------------------------------------------------------------------
    # 7. Seed-by-seed comparison (SA solvers only)
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    for name in sa_solvers:
        runs = all_results[name]
        if not runs:
            continue
        seeds = [r['seed'] for r in runs]
        makespans = [r['makespan'] for r in runs]
        composites = [r['composite'] for r in runs]

        ax1.plot(seeds, makespans, 'o-', label=name, color=colors[name], markersize=3, alpha=0.7, linewidth=1)
        ax2.plot(seeds, composites, 'o-', label=name, color=colors[name], markersize=3, alpha=0.7, linewidth=1)

    ax1.axhline(y=naive_ms, color='gray', linestyle='--', linewidth=1.5, label='Naive baseline')
    ax1.set_xlabel('Seed', fontsize=11)
    ax1.set_ylabel('Makespan (seconds)', fontsize=11)
    ax1.set_title('Makespan by Seed', fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.axhline(y=100, color='gray', linestyle='--', linewidth=1.5, label='Naive=100')
    ax2.set_xlabel('Seed', fontsize=11)
    ax2.set_ylabel('Composite Score', fontsize=11)
    ax2.set_title('Composite Score by Seed', fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.suptitle('Per-Seed Results for SA Solvers', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'stochastic_seeds.png'), dpi=150)
    plt.close()
    print(f"    stochastic_seeds.png")


if __name__ == '__main__':
    main()
