"""
stochastic_data_seeds.py
========================
Runs 100 simulations across DIFFERENT problem instances (data seeds 1-100).
Each data seed generates a different set of orders, items, and totes.
All solvers (SA + deterministic) are tested on every instance.

This tests ROBUSTNESS: how well does each solver generalize to new problems,
not just different random restarts on the same problem.

All output is written to results/dataseed/:
  - dataseed_results.csv           (raw per-run data)
  - dataseed_summary.csv           (statistics: mean, std, min, max, median)
  - dataseed_boxplot.png           (box plots of makespan by solver)
  - dataseed_violin.png            (violin plots: makespan + composite)
  - dataseed_distributions.png     (histograms per solver)
  - dataseed_heatmap.png           (metric heatmap)
  - dataseed_improvement.png       (% improvement bar chart)
  - dataseed_scatter.png           (problem size vs makespan)
  - dataseed_pairwise.png          (head-to-head pairwise wins)
  - dataseed_lines.png             (per-seed makespan line chart)
"""

import sys
import os
import random
import math
import csv
import statistics
import tempfile
import shutil
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Imports
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

_arkhan_mod = _load_event_solver("arkhan_solver_ds", os.path.join(_script_dir, 'arkhan', 'solver.py'))
arkhan_load_problem = _arkhan_mod.load_problem
arkhan_run_sa = _arkhan_mod.run_simulated_annealing
ArkhanParams = _arkhan_mod.SolverParams

_arkhan_sof_mod = _load_event_solver("arkhan_sof_solver_ds", os.path.join(_script_dir, 'arkhan', 'sof_solver.py'))
arkhan_sof_load_problem = _arkhan_sof_mod.load_problem
arkhan_sof_build = _arkhan_sof_mod.build_sof_solution
arkhan_sof_simulate = _arkhan_sof_mod.simulate
ArkhanSofConveyorParams = _arkhan_sof_mod.ConveyorParams

_liam_mod = _load_event_solver("liam_solver_ds", os.path.join(_script_dir, 'liam', 'solver.py'))
liam_load_problem = _liam_mod.load_problem
liam_run_sa = _liam_mod.run_simulated_annealing
LiamParams = _liam_mod.SolverParams


# ---------------------------------------------------------------------------
# Helpers from compare_solutions.py
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
# Write temp CSVs for event-driven solvers
# ---------------------------------------------------------------------------
def write_temp_csvs(data, tmpdir):
    """Write order_itemtypes.csv, order_quantities.csv, orders_totes.csv to tmpdir."""
    # order_itemtypes.csv
    with open(os.path.join(tmpdir, 'order_itemtypes.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        for row in data['order_itemtypes']:
            w.writerow(row)

    # order_quantities.csv
    with open(os.path.join(tmpdir, 'order_quantities.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        for row in data['order_quantities']:
            w.writerow(row)

    # orders_totes.csv
    with open(os.path.join(tmpdir, 'orders_totes.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        for row in data['orders_totes']:
            w.writerow(row)


# ---------------------------------------------------------------------------
# Composite score
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
# Run all solvers on a single data instance
# ---------------------------------------------------------------------------
def run_all_solvers(data_seed, sa_seed=42):
    """Run all solvers on the problem generated with data_seed.
    Returns dict: {solver_name: {metrics...}} and problem_info dict."""

    data = generate_data(data_seed)
    demands = get_demands(data)
    n_orders = data['n_orders']
    total_items = sum(ototal(d) for d in demands)
    tote_data = build_tote_data(data)
    tote_data_rich = build_tote_data_enriched(data)
    all_totes = sorted(tote_data.keys())
    all_orders = list(range(n_orders))
    sim = ConveyorSim(demands)

    problem_info = {
        'data_seed': data_seed,
        'n_orders': n_orders,
        'total_items': total_items,
        'n_totes': len(all_totes),
        'n_itemtypes': data['n_itemtypes'],
    }

    # Write temp CSVs for event-driven solvers
    tmpdir = tempfile.mkdtemp(prefix=f'conveyor_ds{data_seed}_')
    write_temp_csvs(data, tmpdir)

    results = {}

    # --- Naive baseline ---
    naive_belts = [[] for _ in range(4)]
    for i in range(n_orders):
        naive_belts[i % 4].append(i)
    naive_loading = sim.smart_loading_order(naive_belts)
    naive_res = sim.simulate(naive_belts, naive_loading)
    naive_ms = naive_res['makespan']
    naive_tct = naive_res['total_completion_time']
    naive_avg = naive_res['avg_completion_time']
    naive_cts = naive_res['order_completion_times']
    naive_spread = max(naive_cts.values()) - min(naive_cts.values()) if naive_cts else 1
    if naive_spread == 0:
        naive_spread = 1  # avoid division by zero

    naive_comp, naive_sp = compute_composite(naive_res, naive_ms, naive_tct, naive_avg, naive_spread)
    naive_ko, _, _ = kate_evaluate(sorted(all_totes), list(range(n_orders)), tote_data, n_orders)
    results["Naive baseline"] = {
        'makespan': naive_ms, 'total_ct': naive_tct, 'avg_ct': naive_avg,
        'recirculation': naive_res['recirculation_events'],
        'kate_obj': naive_ko, 'composite': naive_comp, 'spread': naive_sp,
        'items_delivered': naive_res['items_sorted'],
    }

    # --- Manjary LPT ---
    try:
        m_belts, _ = manjary_lpt(demands)
        m_loading = sim.smart_loading_order(m_belts)
        m_res = sim.simulate(m_belts, m_loading)
        m_comp, m_sp = compute_composite(m_res, naive_ms, naive_tct, naive_avg, naive_spread)
        m_totes = belt_queues_to_tote_sequence(m_belts, data)
        m_orders = belt_queues_to_order_priority(m_belts)
        m_ko, _, _ = kate_evaluate(m_totes, m_orders, tote_data, n_orders)
        results["Manjary (LPT)"] = {
            'makespan': m_res['makespan'], 'total_ct': m_res['total_completion_time'],
            'avg_ct': m_res['avg_completion_time'],
            'recirculation': m_res['recirculation_events'],
            'kate_obj': m_ko, 'composite': m_comp, 'spread': m_sp,
            'items_delivered': m_res['items_sorted'],
        }
    except Exception as e:
        results["Manjary (LPT)"] = None

    # --- Arkhan SOF ---
    try:
        prob = arkhan_sof_load_problem(
            os.path.join(tmpdir, 'order_itemtypes.csv'),
            os.path.join(tmpdir, 'order_quantities.csv'),
            os.path.join(tmpdir, 'orders_totes.csv'),
        )
        asof_queues, asof_tote_seq = arkhan_sof_build(prob)
        asof_loading = sim.smart_loading_order(asof_queues)
        asof_res = sim.simulate(asof_queues, asof_loading)
        asof_comp, asof_sp = compute_composite(asof_res, naive_ms, naive_tct, naive_avg, naive_spread)
        asof_orders = belt_queues_to_order_priority(asof_queues)
        asof_ko, _, _ = kate_evaluate(asof_tote_seq, asof_orders, tote_data, n_orders)
        results["Arkhan (SOF)"] = {
            'makespan': asof_res['makespan'], 'total_ct': asof_res['total_completion_time'],
            'avg_ct': asof_res['avg_completion_time'],
            'recirculation': asof_res['recirculation_events'],
            'kate_obj': asof_ko, 'composite': asof_comp, 'spread': asof_sp,
            'items_delivered': asof_res['items_sorted'],
        }
    except Exception as e:
        results["Arkhan (SOF)"] = None

    # --- Arkhan SA ---
    try:
        problem = arkhan_load_problem(
            os.path.join(tmpdir, 'order_itemtypes.csv'),
            os.path.join(tmpdir, 'order_quantities.csv'),
            os.path.join(tmpdir, 'orders_totes.csv'),
        )
        params = ArkhanParams(iterations=5000, seed=sa_seed, t0=1.0, alpha=0.99)
        best_sol, best_res, _ = arkhan_run_sa(problem, params, verbose=False)
        a_loading = sim.smart_loading_order(best_sol.belt_queues)
        a_res = sim.simulate(best_sol.belt_queues, a_loading)
        a_comp, a_sp = compute_composite(a_res, naive_ms, naive_tct, naive_avg, naive_spread)
        a_orders = belt_queues_to_order_priority(best_sol.belt_queues)
        a_ko, _, _ = kate_evaluate(best_sol.tote_sequence, a_orders, tote_data, n_orders)
        results["Arkhan (SA)"] = {
            'makespan': a_res['makespan'], 'total_ct': a_res['total_completion_time'],
            'avg_ct': a_res['avg_completion_time'],
            'recirculation': a_res['recirculation_events'],
            'kate_obj': a_ko, 'composite': a_comp, 'spread': a_sp,
            'items_delivered': a_res['items_sorted'],
        }
    except Exception as e:
        results["Arkhan (SA)"] = None

    # --- Liam SA ---
    try:
        problem = liam_load_problem(
            os.path.join(tmpdir, 'order_itemtypes.csv'),
            os.path.join(tmpdir, 'order_quantities.csv'),
            os.path.join(tmpdir, 'orders_totes.csv'),
        )
        params = LiamParams(iterations=5000, seed=sa_seed, t0=1.0, alpha=0.99)
        best_sol, best_res, _ = liam_run_sa(problem, params, verbose=False)
        l_loading = sim.smart_loading_order(best_sol.belt_queues)
        l_res = sim.simulate(best_sol.belt_queues, l_loading)
        l_comp, l_sp = compute_composite(l_res, naive_ms, naive_tct, naive_avg, naive_spread)
        l_orders = belt_queues_to_order_priority(best_sol.belt_queues)
        l_ko, _, _ = kate_evaluate(best_sol.tote_sequence, l_orders, tote_data, n_orders)
        results["Liam (SA)"] = {
            'makespan': l_res['makespan'], 'total_ct': l_res['total_completion_time'],
            'avg_ct': l_res['avg_completion_time'],
            'recirculation': l_res['recirculation_events'],
            'kate_obj': l_ko, 'composite': l_comp, 'spread': l_sp,
            'items_delivered': l_res['items_sorted'],
        }
    except Exception as e:
        results["Liam (SA)"] = None

    # --- Jeevan SA (3 restarts x 2 objectives) ---
    try:
        candidates = []
        for restart in range(3):
            for obj_name in ('total_completion_time', 'makespan'):
                s = sa_seed + restart * 10 + (0 if obj_name == 'total_completion_time' else 1)
                random.seed(s)
                sa = SA(sim, demands, obj_name, iters=40000, T0=200, alpha=0.99985)
                sol, _ = sa.run(verbose=False)
                res = sim.simulate(sol)
                candidates.append((res['makespan'], sol, res))
        _, j_belts, j_res = min(candidates, key=lambda x: x[0])
        j_loading = sim.smart_loading_order(j_belts)
        j_res = sim.simulate(j_belts, j_loading)
        j_comp, j_sp = compute_composite(j_res, naive_ms, naive_tct, naive_avg, naive_spread)
        j_totes = belt_queues_to_tote_sequence(j_belts, data)
        j_orders = belt_queues_to_order_priority(j_belts)
        j_ko, _, _ = kate_evaluate(j_totes, j_orders, tote_data, n_orders)
        results["Jeevan (SA)"] = {
            'makespan': j_res['makespan'], 'total_ct': j_res['total_completion_time'],
            'avg_ct': j_res['avg_completion_time'],
            'recirculation': j_res['recirculation_events'],
            'kate_obj': j_ko, 'composite': j_comp, 'spread': j_sp,
            'items_delivered': j_res['items_sorted'],
        }
    except Exception as e:
        results["Jeevan (SA)"] = None

    # --- Kate SA ---
    try:
        random.seed(sa_seed)
        k_totes, k_orders, k_cost = kate_sa(tote_data, all_totes, all_orders)
        k2j_belts = kate_to_belt_queues(k_orders, tote_data, k_totes)
        k2j_loading = kate_to_loading_order(k_totes, k_orders, k2j_belts, tote_data_rich)
        k_res = sim.simulate(k2j_belts, k2j_loading)
        k_comp, k_sp = compute_composite(k_res, naive_ms, naive_tct, naive_avg, naive_spread)
        k_ko, _, _ = kate_evaluate(k_totes, k_orders, tote_data, n_orders)
        results["Kate (SA)"] = {
            'makespan': k_res['makespan'], 'total_ct': k_res['total_completion_time'],
            'avg_ct': k_res['avg_completion_time'],
            'recirculation': k_res['recirculation_events'],
            'kate_obj': k_ko, 'composite': k_comp, 'spread': k_sp,
            'items_delivered': k_res['items_sorted'],
        }
    except Exception as e:
        results["Kate (SA)"] = None

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)

    return results, problem_info


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    N_SIMS = 100
    SA_SEED = 42  # Fixed optimizer seed — only DATA varies

    print("=" * 70)
    print(f"DATA-SEED ROBUSTNESS TEST: {N_SIMS} different problem instances")
    print(f"  Optimizer seed fixed at {SA_SEED} (isolating data variation)")
    print("=" * 70)

    solver_names = ["Arkhan (SA)", "Liam (SA)", "Jeevan (SA)", "Kate (SA)",
                    "Arkhan (SOF)", "Manjary (LPT)", "Naive baseline"]

    all_results = {s: [] for s in solver_names}
    problem_infos = []

    for trial in range(N_SIMS):
        data_seed = trial + 1
        if (trial + 1) % 10 == 0 or trial == 0:
            print(f"\n--- Data seed {data_seed}/{N_SIMS} ---")

        try:
            results, pinfo = run_all_solvers(data_seed, SA_SEED)
            problem_infos.append(pinfo)

            if (trial + 1) % 10 == 0 or trial == 0:
                print(f"    Problem: {pinfo['n_orders']} orders, {pinfo['total_items']} items, "
                      f"{pinfo['n_totes']} totes")

            for name in solver_names:
                r = results.get(name)
                if r is not None:
                    r['data_seed'] = data_seed
                    r['n_orders'] = pinfo['n_orders']
                    r['total_items'] = pinfo['total_items']
                    all_results[name].append(r)
                else:
                    if (trial + 1) % 10 == 0:
                        print(f"    {name}: FAILED")
        except Exception as e:
            print(f"  Data seed {data_seed} completely failed: {e}")

    # -- Statistics --
    print(f"\n{'=' * 70}")
    print("DATA-SEED ROBUSTNESS RESULTS")
    print("=" * 70)

    # Problem size summary
    n_orders_list = [p['n_orders'] for p in problem_infos]
    total_items_list = [p['total_items'] for p in problem_infos]
    print(f"\n  Problem sizes across {len(problem_infos)} instances:")
    print(f"    Orders: {min(n_orders_list)}-{max(n_orders_list)} "
          f"(mean={statistics.mean(n_orders_list):.1f})")
    print(f"    Items:  {min(total_items_list)}-{max(total_items_list)} "
          f"(mean={statistics.mean(total_items_list):.1f})")

    summary = {}
    for name in solver_names:
        runs = all_results[name]
        if not runs:
            print(f"\n  {name}: no successful runs")
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
        metrics['n_runs'] = len(runs)
        summary[name] = metrics

    # Print table
    print(f"\n{'Solver':<20s} | {'N':>3s} | {'Makespan (s)':>30s} | {'Composite':>30s}")
    print(f"{'':20s} | {'':>3s} | {'mean ± std (min-max)':>30s} | {'mean ± std (min-max)':>30s}")
    print("-" * 95)
    for name in solver_names:
        if name not in summary:
            continue
        ms = summary[name]['makespan']
        co = summary[name]['composite']
        n = summary[name]['n_runs']
        print(f"{name:<20s} | {n:>3d} | {ms['mean']:>6.1f} ± {ms['std']:>5.1f} ({ms['min']:>5.1f}-{ms['max']:>6.1f}) | "
              f"{co['mean']:>6.1f} ± {co['std']:>5.1f} ({co['min']:>5.1f}-{co['max']:>6.1f})")

    # % improvement over naive (per-instance, then averaged)
    print(f"\n  Mean % improvement over naive (computed per-instance):")
    for name in solver_names:
        if name == "Naive baseline" or name not in summary:
            continue
        runs = all_results[name]
        naive_runs = all_results["Naive baseline"]
        # Match by data_seed
        naive_by_seed = {r['data_seed']: r for r in naive_runs}
        ms_imps = []
        comp_imps = []
        for r in runs:
            nr = naive_by_seed.get(r['data_seed'])
            if nr:
                ms_imps.append((nr['makespan'] - r['makespan']) / nr['makespan'] * 100)
                comp_imps.append(100 - r['composite'])
        if ms_imps:
            print(f"    {name:<20s}  composite: {statistics.mean(comp_imps):>+6.1f}% ± {statistics.stdev(comp_imps):>4.1f}  "
                  f"makespan: {statistics.mean(ms_imps):>+6.1f}% ± {statistics.stdev(ms_imps):>4.1f}")

    # Pairwise wins
    print(f"\n  Pairwise head-to-head wins (makespan, across {len(problem_infos)} instances):")
    sa_plus = ["Arkhan (SA)", "Liam (SA)", "Jeevan (SA)", "Kate (SA)", "Arkhan (SOF)", "Manjary (LPT)"]
    wins = {s: {s2: 0 for s2 in sa_plus if s2 != s} for s in sa_plus}
    ties = {s: {s2: 0 for s2 in sa_plus if s2 != s} for s in sa_plus}

    by_seed = {s: {} for s in sa_plus}
    for s in sa_plus:
        for r in all_results[s]:
            by_seed[s][r['data_seed']] = r

    for ds in range(1, N_SIMS + 1):
        for i, s1 in enumerate(sa_plus):
            for s2 in sa_plus[i+1:]:
                r1 = by_seed[s1].get(ds)
                r2 = by_seed[s2].get(ds)
                if r1 and r2:
                    if r1['makespan'] < r2['makespan']:
                        wins[s1][s2] = wins[s1].get(s2, 0) + 1
                    elif r2['makespan'] < r1['makespan']:
                        wins[s2][s1] = wins[s2].get(s1, 0) + 1
                    else:
                        ties[s1][s2] = ties[s1].get(s2, 0) + 1
                        ties[s2][s1] = ties[s2].get(s1, 0) + 1

    # Print wins matrix
    short = {"Arkhan (SA)": "Ark-SA", "Liam (SA)": "Liam", "Jeevan (SA)": "Jeev",
             "Kate (SA)": "Kate", "Arkhan (SOF)": "SOF", "Manjary (LPT)": "LPT"}
    header = "    " + "".join(f"{short[s]:>8s}" for s in sa_plus)
    print(header)
    for s1 in sa_plus:
        row = f"    {short[s1]:<8s}"
        for s2 in sa_plus:
            if s1 == s2:
                row += f"{'---':>8s}"
            else:
                row += f"{wins[s1].get(s2, 0):>8d}"
        print(row)

    # -- Export CSVs --
    _results_dir = os.path.join(_script_dir, 'results', 'dataseed')
    os.makedirs(_results_dir, exist_ok=True)
    raw_csv = os.path.join(_results_dir, 'dataseed_results.csv')
    with open(raw_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['solver', 'data_seed', 'n_orders', 'total_items',
                     'makespan_sec', 'total_ct_sec', 'avg_ct_sec',
                     'spread_sec', 'recirculation', 'kate_obj', 'composite_score',
                     'items_delivered'])
        for name in solver_names:
            for r in all_results[name]:
                w.writerow([name, r['data_seed'], r.get('n_orders', ''), r.get('total_items', ''),
                            f"{r['makespan']:.1f}", f"{r['total_ct']:.1f}", f"{r['avg_ct']:.1f}",
                            f"{r['spread']:.0f}", r['recirculation'],
                            f"{r['kate_obj']:.1f}", f"{r['composite']:.1f}",
                            r['items_delivered']])
    print(f"\n  Raw results: {raw_csv}")

    sum_csv = os.path.join(_results_dir, 'dataseed_summary.csv')
    with open(sum_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['solver', 'metric', 'mean', 'std', 'min', 'q25', 'median', 'q75', 'max', 'n_trials'])
        for name in solver_names:
            if name not in summary:
                continue
            for metric in ['makespan', 'total_ct', 'avg_ct', 'spread', 'recirculation', 'kate_obj', 'composite']:
                s = summary[name][metric]
                w.writerow([name, metric, f"{s['mean']:.2f}", f"{s['std']:.2f}",
                            f"{s['min']:.2f}", f"{s['q25']:.2f}", f"{s['median']:.2f}",
                            f"{s['q75']:.2f}", f"{s['max']:.2f}", summary[name]['n_runs']])
    print(f"  Summary stats: {sum_csv}")

    # -- Plots --
    print(f"\n  Generating plots...")
    generate_plots(all_results, summary, solver_names, problem_infos, wins, sa_plus, short, _results_dir)

    print(f"\n{'=' * 70}")
    print("DONE")
    print("=" * 70)


# ---------------------------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------------------------
def generate_plots(all_results, summary, solver_names, problem_infos, wins, sa_plus, short, outdir):
    colors = {
        "Arkhan (SA)": "#2196F3",
        "Liam (SA)": "#4CAF50",
        "Jeevan (SA)": "#FF9800",
        "Kate (SA)": "#E91E63",
        "Arkhan (SOF)": "#9C27B0",
        "Manjary (LPT)": "#795548",
        "Naive baseline": "#9E9E9E",
    }

    active_solvers = [s for s in solver_names if s in summary and summary[s]['n_runs'] > 0]

    # -------------------------------------------------------------------------
    # 1. Box plot: Makespan across data seeds
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 6))
    data_box = []
    labels_box = []
    box_colors = []
    for name in active_solvers:
        vals = [r['makespan'] for r in all_results[name]]
        data_box.append(vals)
        labels_box.append(name.replace(" (", "\n("))
        box_colors.append(colors.get(name, '#666'))

    bp = ax.boxplot(data_box, labels=labels_box, patch_artist=True, widths=0.6,
                    medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Makespan (seconds)', fontsize=12)
    ax.set_title(f'Makespan Across {len(problem_infos)} Different Problem Instances (Data Seeds)',
                 fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'dataseed_boxplot.png'), dpi=150)
    plt.close()
    print(f"    dataseed_boxplot.png")

    # -------------------------------------------------------------------------
    # 2. Violin plot: Makespan + Composite
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    for ax, metric, title in [(ax1, 'makespan', 'Makespan (seconds)'),
                                (ax2, 'composite', 'Composite Score')]:
        data_v = []
        positions = []
        for i, name in enumerate(active_solvers):
            vals = [r[metric] for r in all_results[name]]
            data_v.append(vals)
            positions.append(i)

        parts = ax.violinplot(data_v, positions=positions, showmeans=True,
                              showmedians=True, showextrema=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors.get(active_solvers[i], '#666'))
            pc.set_alpha(0.6)
        parts['cmeans'].set_color('red')
        parts['cmedians'].set_color('black')

        ax.set_xticks(positions)
        ax.set_xticklabels([s.replace(" (", "\n(") for s in active_solvers], fontsize=9)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.grid(axis='y', alpha=0.3)

        if metric == 'composite':
            ax.axhline(y=100, color='gray', linestyle='--', linewidth=1.5, label='Naive=100')
            ax.legend()

    plt.suptitle(f'Robustness Test: {len(problem_infos)} Different Problem Instances', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'dataseed_violin.png'), dpi=150)
    plt.close()
    print(f"    dataseed_violin.png")

    # -------------------------------------------------------------------------
    # 3. Histograms: Makespan per solver
    # -------------------------------------------------------------------------
    sa_solvers = ["Arkhan (SA)", "Liam (SA)", "Jeevan (SA)", "Kate (SA)", "Arkhan (SOF)", "Manjary (LPT)"]
    plot_solvers = [s for s in sa_solvers if s in summary]
    n_plots = len(plot_solvers)
    ncols = 3
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows))
    if nrows == 1:
        axes = [axes]
    axes_flat = [ax for row in axes for ax in (row if hasattr(row, '__len__') else [row])]

    for idx, name in enumerate(plot_solvers):
        ax = axes_flat[idx]
        vals = [r['makespan'] for r in all_results[name]]
        n_bins = min(max(len(set(vals)), 5), 30)
        ax.hist(vals, bins=n_bins, color=colors.get(name, '#666'), alpha=0.7,
                edgecolor='black', linewidth=0.5)
        ax.axvline(x=statistics.mean(vals), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {statistics.mean(vals):.1f}s')
        ax.axvline(x=min(vals), color='green', linestyle=':', linewidth=1.5,
                   label=f'Best: {min(vals):.1f}s')
        ax.set_xlabel('Makespan (seconds)', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    # Hide extra axes
    for idx in range(len(plot_solvers), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.suptitle(f'Makespan Distributions Across {len(problem_infos)} Problem Instances', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'dataseed_distributions.png'), dpi=150)
    plt.close()
    print(f"    dataseed_distributions.png")

    # -------------------------------------------------------------------------
    # 4. Heatmap
    # -------------------------------------------------------------------------
    metrics_to_show = ['makespan', 'avg_ct', 'total_ct', 'spread', 'recirculation', 'kate_obj', 'composite']
    metric_labels = ['Makespan\n(s)', 'Avg CT\n(s)', 'Total CT\n(s)', 'Spread\n(s)',
                     'Recirc', 'Kate\nObj', 'Composite']

    heatmap_data = []
    heatmap_labels = []
    for name in active_solvers:
        row = [summary[name][m]['mean'] for m in metrics_to_show]
        heatmap_data.append(row)
        heatmap_labels.append(name)

    heatmap_arr = np.array(heatmap_data)
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
    for i in range(len(heatmap_labels)):
        for j in range(len(metrics_to_show)):
            val = heatmap_arr[i, j]
            fmt = f"{val:.0f}" if val > 10 else f"{val:.1f}"
            text_color = 'white' if heatmap_norm[i, j] > 0.65 else 'black'
            ax.text(j, i, fmt, ha='center', va='center', fontsize=9, color=text_color, fontweight='bold')
    ax.set_title('Mean Metrics Across Problem Instances (green=better, red=worse)', fontsize=13)
    plt.colorbar(im, ax=ax, label='Normalized (0=best, 1=worst)')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'dataseed_heatmap.png'), dpi=150)
    plt.close()
    print(f"    dataseed_heatmap.png")

    # -------------------------------------------------------------------------
    # 5. Improvement bar chart
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Compute per-instance improvement, then average
    naive_by_seed = {r['data_seed']: r for r in all_results["Naive baseline"]}
    imp_names = [s for s in active_solvers if s != "Naive baseline"]

    ms_imps_mean = []
    comp_imps_mean = []
    ms_imps_std = []
    comp_imps_std = []
    for name in imp_names:
        runs = all_results[name]
        ms_imps = []
        comp_imps = []
        for r in runs:
            nr = naive_by_seed.get(r['data_seed'])
            if nr:
                ms_imps.append((nr['makespan'] - r['makespan']) / nr['makespan'] * 100)
                comp_imps.append(100 - r['composite'])
        ms_imps_mean.append(statistics.mean(ms_imps) if ms_imps else 0)
        ms_imps_std.append(statistics.stdev(ms_imps) if len(ms_imps) > 1 else 0)
        comp_imps_mean.append(statistics.mean(comp_imps) if comp_imps else 0)
        comp_imps_std.append(statistics.stdev(comp_imps) if len(comp_imps) > 1 else 0)

    bar_colors = [colors.get(s, '#666') for s in imp_names]
    y_pos = range(len(imp_names))

    ax1.barh(y_pos, ms_imps_mean, xerr=ms_imps_std, color=bar_colors, alpha=0.8,
             edgecolor='black', linewidth=0.5, capsize=3)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(imp_names, fontsize=10)
    ax1.set_xlabel('% Improvement over Naive (mean ± std)', fontsize=11)
    ax1.set_title('Makespan Improvement', fontsize=13)
    ax1.axvline(x=0, color='black', linewidth=1)
    ax1.grid(axis='x', alpha=0.3)
    for i, v in enumerate(ms_imps_mean):
        ax1.text(max(v, 0) + ms_imps_std[i] + 1, i, f'{v:.1f}%', va='center', fontsize=9, fontweight='bold')

    ax2.barh(y_pos, comp_imps_mean, xerr=comp_imps_std, color=bar_colors, alpha=0.8,
             edgecolor='black', linewidth=0.5, capsize=3)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(imp_names, fontsize=10)
    ax2.set_xlabel('% Improvement over Naive (composite, mean ± std)', fontsize=11)
    ax2.set_title('Composite Score Improvement', fontsize=13)
    ax2.axvline(x=0, color='black', linewidth=1)
    ax2.grid(axis='x', alpha=0.3)
    for i, v in enumerate(comp_imps_mean):
        ax2.text(max(v, 0) + comp_imps_std[i] + 1, i, f'{v:.1f}%', va='center', fontsize=9, fontweight='bold')

    plt.suptitle(f'Solver Robustness: Mean Improvement Over Naive ({len(problem_infos)} problem instances)',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'dataseed_improvement.png'), dpi=150)
    plt.close()
    print(f"    dataseed_improvement.png")

    # -------------------------------------------------------------------------
    # 6. Scatter: Problem size (total_items) vs makespan
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for name in ["Arkhan (SA)", "Kate (SA)", "Arkhan (SOF)", "Manjary (LPT)", "Naive baseline"]:
        if name not in all_results:
            continue
        runs = all_results[name]
        items = [r.get('total_items', 0) for r in runs]
        makespans = [r['makespan'] for r in runs]
        ax1.scatter(items, makespans, label=name, color=colors.get(name, '#666'),
                    alpha=0.6, s=30, edgecolors='black', linewidth=0.3)

    ax1.set_xlabel('Total Items in Problem', fontsize=11)
    ax1.set_ylabel('Makespan (seconds)', fontsize=11)
    ax1.set_title('Problem Size vs Makespan', fontsize=13)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    for name in ["Arkhan (SA)", "Kate (SA)", "Arkhan (SOF)", "Manjary (LPT)", "Naive baseline"]:
        if name not in all_results:
            continue
        runs = all_results[name]
        orders = [r.get('n_orders', 0) for r in runs]
        makespans = [r['makespan'] for r in runs]
        ax2.scatter(orders, makespans, label=name, color=colors.get(name, '#666'),
                    alpha=0.6, s=30, edgecolors='black', linewidth=0.3)

    ax2.set_xlabel('Number of Orders', fontsize=11)
    ax2.set_ylabel('Makespan (seconds)', fontsize=11)
    ax2.set_title('Number of Orders vs Makespan', fontsize=13)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.suptitle('Solver Scaling with Problem Size', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'dataseed_scatter.png'), dpi=150)
    plt.close()
    print(f"    dataseed_scatter.png")

    # -------------------------------------------------------------------------
    # 7. Pairwise wins heatmap
    # -------------------------------------------------------------------------
    n_solvers = len(sa_plus)
    win_matrix = np.zeros((n_solvers, n_solvers))
    for i, s1 in enumerate(sa_plus):
        for j, s2 in enumerate(sa_plus):
            if i != j:
                win_matrix[i, j] = wins[s1].get(s2, 0)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(win_matrix, cmap='YlOrRd', aspect='auto')
    short_names = [short[s] for s in sa_plus]
    ax.set_xticks(range(n_solvers))
    ax.set_xticklabels(short_names, fontsize=10, rotation=45, ha='right')
    ax.set_yticks(range(n_solvers))
    ax.set_yticklabels(short_names, fontsize=10)
    ax.set_xlabel('Opponent (column)', fontsize=11)
    ax.set_ylabel('Winner (row)', fontsize=11)

    for i in range(n_solvers):
        for j in range(n_solvers):
            if i == j:
                ax.text(j, i, '—', ha='center', va='center', fontsize=10, color='gray')
            else:
                val = int(win_matrix[i, j])
                text_color = 'white' if val > len(problem_infos) * 0.6 else 'black'
                ax.text(j, i, str(val), ha='center', va='center', fontsize=10,
                        color=text_color, fontweight='bold')

    ax.set_title(f'Pairwise Wins (Makespan) Across {len(problem_infos)} Problem Instances\n'
                 f'Row beats Column N times', fontsize=13)
    plt.colorbar(im, ax=ax, label='Number of wins')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'dataseed_pairwise.png'), dpi=150)
    plt.close()
    print(f"    dataseed_pairwise.png")

    # -------------------------------------------------------------------------
    # 8. Per-seed line chart (makespan by data seed)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(16, 7))
    for name in ["Arkhan (SA)", "Kate (SA)", "Arkhan (SOF)", "Manjary (LPT)", "Naive baseline"]:
        if name not in all_results:
            continue
        runs = sorted(all_results[name], key=lambda r: r['data_seed'])
        seeds = [r['data_seed'] for r in runs]
        makespans = [r['makespan'] for r in runs]
        style = '-' if 'SA' in name or 'SOF' in name else '--'
        ax.plot(seeds, makespans, style, label=name, color=colors.get(name, '#666'),
                alpha=0.8, linewidth=1.5, markersize=3)

    ax.set_xlabel('Data Seed (different problem instance)', fontsize=11)
    ax.set_ylabel('Makespan (seconds)', fontsize=11)
    ax.set_title(f'Makespan Across {len(problem_infos)} Problem Instances', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'dataseed_lines.png'), dpi=150)
    plt.close()
    print(f"    dataseed_lines.png")


if __name__ == '__main__':
    main()
