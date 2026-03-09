"""
compare_solutions.py
====================
Apples-to-apples comparison of all 5 team members' optimizers
for the 4-belt conveyor problem (seed=100, 11 orders, 36 items).

Members:
- Arkhan:  Multi-restart SA (belt assignment), varied configs + SOF heuristic
- Jeevan:  Multi-restart SA (belt assignment + loading order)
- Kate:    SA optimizing tote sequence + order priority
- Liam:    Event-driven SA (joint belt + tote optimization)
- Manjary: LPT (Longest Processing Time) heuristic

Runs each approach, evaluates through BOTH Jeevan's ConveyorSim
and Kate's evaluator, then prints a unified comparison.
"""

import sys
import os
import random
import math
import csv
import statistics
from collections import defaultdict

# ---------------------------------------------------------------------------
# Import Jeevan's module (shared simulator / helpers)
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_script_dir, 'jeevan'))
from conveyor_optimizer import (
    generate_data, get_demands, get_tote_contents, ototal, sname,
    ConveyorSim, SA, optimize_loading_order, NUM_BELTS,
)

# ---------------------------------------------------------------------------
# Import event-driven solvers (Arkhan + Liam) via importlib to avoid conflicts
# ---------------------------------------------------------------------------
import importlib.util

def _load_event_solver(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_arkhan_mod = _load_event_solver("arkhan_solver", os.path.join(_script_dir, 'arkhan', 'solver.py'))
arkhan_load_problem = _arkhan_mod.load_problem
arkhan_run_sa = _arkhan_mod.run_simulated_annealing
ArkhanParams = _arkhan_mod.SolverParams

_arkhan_sof_mod = _load_event_solver("arkhan_sof_solver", os.path.join(_script_dir, 'arkhan', 'sof_solver.py'))
arkhan_sof_load_problem = _arkhan_sof_mod.load_problem
arkhan_sof_build = _arkhan_sof_mod.build_sof_solution
arkhan_sof_simulate = _arkhan_sof_mod.simulate
ArkhanSofConveyorParams = _arkhan_sof_mod.ConveyorParams

_liam_mod = _load_event_solver("liam_solver", os.path.join(_script_dir, 'liam', 'solver.py'))
liam_load_problem = _liam_mod.load_problem
liam_run_sa = _liam_mod.run_simulated_annealing
LiamParams = _liam_mod.SolverParams


# ============================================================
# KATE'S EVALUATOR (adapted to use shared data, no pandas)
# ============================================================

def build_tote_data(data):
    """Build Kate-style tote_data: {tote_id: [(order_id, qty), ...]}."""
    tote_data = {}
    for order_id in range(data['n_orders']):
        for j in range(len(data['order_itemtypes'][order_id])):
            tote = data['orders_totes'][order_id][j]
            qty = data['order_quantities'][order_id][j]
            tote_data.setdefault(tote, []).append((order_id, qty))
    return tote_data


def build_tote_data_enriched(data):
    """Enriched: {tote_id: [(order_id, item_type, qty), ...]}."""
    tote_data = {}
    for order_id in range(data['n_orders']):
        for j, item_type in enumerate(data['order_itemtypes'][order_id]):
            tote = data['orders_totes'][order_id][j]
            qty = data['order_quantities'][order_id][j]
            tote_data.setdefault(tote, []).append((order_id, item_type, qty))
    return tote_data


def kate_evaluate(tote_seq, order_priority, tote_data, n_orders, alpha=0.5):
    """Kate's evaluate_solution. Returns (objective, total_completion, circulation)."""
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
# CONVERSION HELPERS
# ============================================================

def kate_to_belt_queues(order_priority, tote_data, tote_seq):
    """Convert Kate's solution -> belt_queues."""
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
    """Convert Kate's tote sequence -> loading_order for ConveyorSim."""
    first_orders = {}
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


def belt_queues_to_order_priority(belt_queues):
    """Flatten belt_queues -> order_priority (interleaved by depth)."""
    order_priority = []
    max_depth = max((len(q) for q in belt_queues), default=0)
    for depth in range(max_depth):
        for b in range(4):
            if depth < len(belt_queues[b]):
                order_priority.append(belt_queues[b][depth])
    return order_priority


def belt_queues_to_tote_sequence(belt_queues, data):
    """Derive tote sequence from belt assignment."""
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


# ============================================================
# MANJARY'S LPT POLICY
# ============================================================

def manjary_lpt(demands, num_belts=4):
    """LPT assignment (Manjary's approach from policies.py)."""
    sizes = sorted([(ototal(d), i) for i, d in enumerate(demands)], reverse=True)
    qs = [[] for _ in range(num_belts)]
    load = [0] * num_belts
    for sz, oi in sizes:
        b = min(range(num_belts), key=lambda x: load[x])
        qs[b].append(oi)
        load[b] += sz
    return qs, load


# ============================================================
# ARKHAN'S EVENT-DRIVEN SA
# ============================================================

def arkhan_sa(data_dir):
    """Run Arkhan's event-driven SA. Returns belt_queues and tote_sequence."""
    problem = arkhan_load_problem(
        os.path.join(data_dir, 'order_itemtypes.csv'),
        os.path.join(data_dir, 'order_quantities.csv'),
        os.path.join(data_dir, 'orders_totes.csv'),
    )
    params = ArkhanParams(iterations=5000, seed=42, t0=1.0, alpha=0.99)
    best_sol, best_res, _ = arkhan_run_sa(problem, params, verbose=False)
    return best_sol.belt_queues, best_sol.tote_sequence, best_res


# ============================================================
# ARKHAN'S SOF (Shortest-Order-First) HEURISTIC
# ============================================================

def arkhan_sof(data_dir):
    """Run Arkhan's SOF heuristic. Returns belt_queues, tote_sequence, and sim results."""
    prob = arkhan_sof_load_problem(
        os.path.join(data_dir, 'order_itemtypes.csv'),
        os.path.join(data_dir, 'order_quantities.csv'),
        os.path.join(data_dir, 'orders_totes.csv'),
    )
    queues, tote_seq = arkhan_sof_build(prob)
    params = ArkhanSofConveyorParams()
    feasible, makespan, avg_ct, dropped, steps = arkhan_sof_simulate(prob, queues, tote_seq, params)
    return queues, tote_seq, feasible, makespan, avg_ct, dropped


# ============================================================
# JEEVAN'S MULTI-RESTART SA
# ============================================================

def jeevan_sa(sim, demands):
    """Jeevan's multi-restart SA (3 restarts x 2 objectives, 40k iters)."""
    n_restarts = 3
    candidates = []
    for restart in range(n_restarts):
        for obj_name in ('total_completion_time', 'makespan'):
            seed_val = 42 + restart * 10 + (0 if obj_name == 'total_completion_time' else 1)
            random.seed(seed_val)
            sa = SA(sim, demands, obj_name, iters=40000, T0=200, alpha=0.99985)
            sol, _ = sa.run(verbose=False)
            res = sim.simulate(sol)
            candidates.append((res['makespan'], sol, res))

    _, best_sol, best_res = min(candidates, key=lambda x: x[0])
    return best_sol


# ============================================================
# LIAM'S EVENT-DRIVEN SA
# ============================================================

def liam_sa(data_dir):
    """Run Liam's event-driven SA. Returns belt_queues and tote_sequence."""
    problem = liam_load_problem(
        os.path.join(data_dir, 'order_itemtypes.csv'),
        os.path.join(data_dir, 'order_quantities.csv'),
        os.path.join(data_dir, 'orders_totes.csv'),
    )
    params = LiamParams(iterations=5000, seed=42, t0=1.0, alpha=0.99)
    best_sol, best_res, _ = liam_run_sa(problem, params, verbose=False)
    return best_sol.belt_queues, best_sol.tote_sequence, best_res


# ============================================================
# MAIN
# ============================================================

def main():
    SEED = 100

    print("=" * 70)
    print("CROSS-EVALUATION: Arkhan (SA+SOF) vs Jeevan vs Kate vs Liam vs Manjary")
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

    # -- 2. Run each optimizer -------------------------------------------

    # --- Arkhan ---
    print(f"\n{'-' * 70}")
    print("Running Arkhan's SA (event-driven simulation)...")
    print("-" * 70)
    csv_dir = os.path.join(_script_dir, 'jeevan')
    a_belts, a_tote_seq, a_own_res = arkhan_sa(csv_dir)
    a_loading = sim.smart_loading_order(a_belts)
    a_res = sim.simulate(a_belts, a_loading)
    print(f"  Makespan (Jeevan sim): {a_res['makespan']:.0f}s")
    print(f"  Makespan (Arkhan sim): {a_own_res.makespan:.1f}")
    for b in range(4):
        ords = " -> ".join(f"O{o}" for o in a_belts[b])
        ld = sum(ototal(demands[o]) for o in a_belts[b])
        print(f"  Belt {b+1} [{ld:2d} items]: {ords}")

    # --- Arkhan SOF ---
    print(f"\n{'-' * 70}")
    print("Running Arkhan's SOF (Shortest-Order-First heuristic)...")
    print("-" * 70)
    asof_belts, asof_tote_seq, asof_feasible, asof_mk, asof_avg, asof_dropped = arkhan_sof(csv_dir)
    asof_loading = sim.smart_loading_order(asof_belts)
    asof_res = sim.simulate(asof_belts, asof_loading)
    print(f"  Makespan (Jeevan sim): {asof_res['makespan']:.0f}s")
    print(f"  Makespan (SOF sim):    {asof_mk:.1f}s")
    print(f"  Feasible: {asof_feasible}, Dropped: {asof_dropped}")
    for b in range(4):
        ords = " -> ".join(f"O{o}" for o in asof_belts[b])
        ld = sum(ototal(demands[o]) for o in asof_belts[b])
        print(f"  Belt {b+1} [{ld:2d} items]: {ords}")

    # --- Jeevan ---
    print(f"\n{'-' * 70}")
    print("Running Jeevan's SA (3 restarts x 2 objectives, 40k iters)...")
    print("-" * 70)
    j_belts = jeevan_sa(sim, demands)
    _, j_strategies = optimize_loading_order(sim, j_belts, demands)
    j_best_strat = min(j_strategies, key=lambda k: j_strategies[k][1]['makespan'])
    j_loading = j_strategies[j_best_strat][0]
    j_res = sim.simulate(j_belts, j_loading)
    print(f"  Makespan: {j_res['makespan']:.0f}s, strategy: {j_best_strat}")
    for b in range(4):
        ords = " -> ".join(f"O{o}" for o in j_belts[b])
        ld = sum(ototal(demands[o]) for o in j_belts[b])
        print(f"  Belt {b+1} [{ld:2d} items]: {ords}")

    # --- Kate ---
    print(f"\n{'-' * 70}")
    print("Running Kate's SA (tote sequence + order priority)...")
    print("-" * 70)
    random.seed(42)
    k_totes, k_orders, k_cost = kate_sa(tote_data, all_totes, all_orders)
    k2j_belts = kate_to_belt_queues(k_orders, tote_data, k_totes)
    k2j_loading = kate_to_loading_order(k_totes, k_orders, k2j_belts, tote_data_rich)
    print(f"  Kate objective: {k_cost:.2f}")
    print(f"  Order priority: {k_orders}")

    # --- Liam ---
    print(f"\n{'-' * 70}")
    print("Running Liam's SA (event-driven simulation)...")
    print("-" * 70)
    # Liam reads CSVs directly; use jeevan's dir since it has the CSV files
    csv_dir = os.path.join(_script_dir, 'jeevan')
    l_belts, l_tote_seq, l_own_res = liam_sa(csv_dir)
    l_loading = sim.smart_loading_order(l_belts)
    l_res = sim.simulate(l_belts, l_loading)
    print(f"  Makespan (Jeevan sim): {l_res['makespan']:.0f}s")
    print(f"  Makespan (Liam sim):   {l_own_res.makespan:.1f}")
    for b in range(4):
        ords = " -> ".join(f"O{o}" for o in l_belts[b])
        ld = sum(ototal(demands[o]) for o in l_belts[b])
        print(f"  Belt {b+1} [{ld:2d} items]: {ords}")

    # --- Manjary ---
    print(f"\n{'-' * 70}")
    print("Running Manjary's LPT heuristic...")
    print("-" * 70)
    m_belts, m_loads = manjary_lpt(demands)
    m_loading = sim.smart_loading_order(m_belts)
    m_res = sim.simulate(m_belts, m_loading)
    print(f"  Makespan: {m_res['makespan']:.0f}s")
    print(f"  Belt loads: {m_loads}")
    for b in range(4):
        ords = " -> ".join(f"O{o}" for o in m_belts[b])
        ld = sum(ototal(demands[o]) for o in m_belts[b])
        print(f"  Belt {b+1} [{ld:2d} items]: {ords}")

    # --- Naive baseline ---
    naive_belts = [[] for _ in range(4)]
    for i in range(n_orders):
        naive_belts[i % 4].append(i)
    naive_loading = sim.smart_loading_order(naive_belts)
    naive_totes = sorted(all_totes)
    naive_orders = list(range(n_orders))

    # -- 3. Build solutions dict -----------------------------------------
    solutions = {
        "Arkhan (SA)": {
            'belts': a_belts,
            'loading': a_loading,
            'totes': a_tote_seq,
            'orders': belt_queues_to_order_priority(a_belts),
        },
        "Arkhan (SOF)": {
            'belts': asof_belts,
            'loading': asof_loading,
            'totes': asof_tote_seq,
            'orders': belt_queues_to_order_priority(asof_belts),
        },
        "Jeevan (SA)": {
            'belts': j_belts,
            'loading': j_loading,
            'totes': belt_queues_to_tote_sequence(j_belts, data),
            'orders': belt_queues_to_order_priority(j_belts),
        },
        "Kate (SA)": {
            'belts': k2j_belts,
            'loading': k2j_loading,
            'totes': k_totes,
            'orders': k_orders,
        },
        "Liam (SA)": {
            'belts': l_belts,
            'loading': l_loading,
            'totes': l_tote_seq,
            'orders': belt_queues_to_order_priority(l_belts),
        },
        "Manjary (LPT)": {
            'belts': m_belts,
            'loading': m_loading,
            'totes': belt_queues_to_tote_sequence(m_belts, data),
            'orders': belt_queues_to_order_priority(m_belts),
        },
        "Naive baseline": {
            'belts': naive_belts,
            'loading': naive_loading,
            'totes': naive_totes,
            'orders': naive_orders,
        },
    }

    # -- 4. Cross-evaluate -----------------------------------------------
    print(f"\n{'=' * 70}")
    print("RESULTS: All solutions through both simulators")
    print("=" * 70)

    results = {}
    for name, sol in solutions.items():
        j_res = sim.simulate(sol['belts'], sol['loading'])
        k_obj, k_ct, k_circ = kate_evaluate(
            sol['totes'], sol['orders'], tote_data, n_orders)
        results[name] = {
            'jeevan': j_res,
            'kate_obj': k_obj, 'kate_ct': k_ct, 'kate_circ': k_circ
        }

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

    # -- 5. Neutral composite objective ------------------------------------
    # Uses only Jeevan's physical simulator (most realistic timing model).
    # Composite = 0.35*makespan + 0.35*avg_completion + 0.15*total_CT + 0.15*spread
    # All normalized to naive baseline = 100 so lower is better.
    print(f"\n{'=' * 70}")
    print("COMPOSITE SCORE (neutral, from physical simulator)")
    print("  35% makespan + 35% avg completion + 15% total CT + 15% wait spread")
    print("  Normalized: naive baseline = 100, lower = better")
    print("=" * 70)

    naive_r = results["Naive baseline"]
    naive_ms = naive_r['jeevan']['makespan']
    naive_tct = naive_r['jeevan']['total_completion_time']
    naive_avg = naive_r['jeevan']['avg_completion_time']
    naive_ko = naive_r['kate_obj']

    cts_naive = naive_r['jeevan']['order_completion_times']
    naive_spread = max(cts_naive.values()) - min(cts_naive.values())

    for name in solutions:
        r = results[name]
        j = r['jeevan']
        cts = j['order_completion_times']
        spread = max(cts.values()) - min(cts.values()) if cts else 0
        r['spread'] = spread
        # Composite: normalized so naive = 100
        r['composite'] = (
            0.35 * (j['makespan'] / naive_ms) +
            0.35 * (j['avg_completion_time'] / naive_avg) +
            0.15 * (j['total_completion_time'] / naive_tct) +
            0.15 * (spread / naive_spread if naive_spread > 0 else 1.0)
        ) * 100

    ranked = sorted(
        [(name, results[name]) for name in solutions if name != "Naive baseline"],
        key=lambda x: x[1]['composite'])

    hdr2 = (f"{'Rank':<5s} {'Solution':<20s} | {'Composite':>10s} | "
            f"{'Makespan':>10s} | {'AvgCT':>10s} | {'Spread':>10s} | "
            f"{'Kate Obj':>10s}")
    print(f"\n{hdr2}")
    print("-" * len(hdr2))

    for rank, (name, r) in enumerate(ranked, 1):
        j = r['jeevan']
        print(f"  {rank:<3d} {name:<20s} | {r['composite']:>9.1f} | "
              f"{j['makespan']:>9.0f}s | "
              f"{j['avg_completion_time']:>9.1f}s | "
              f"{r['spread']:>9.0f}s | "
              f"{r['kate_obj']:>9.1f}")

    # Naive baseline reference
    print(f"  --- {'Naive baseline':<20s} | {100.0:>9.1f} | "
          f"{naive_ms:>9.0f}s | "
          f"{naive_avg:>9.1f}s | "
          f"{naive_spread:>9.0f}s | "
          f"{naive_ko:>9.1f}")

    # Improvement summary
    print(f"\n  % improvement over naive:")
    for rank, (name, r) in enumerate(ranked, 1):
        j = r['jeevan']
        ms_imp = (naive_ms - j['makespan']) / naive_ms * 100
        avg_imp = (naive_avg - j['avg_completion_time']) / naive_avg * 100
        comp_imp = (100 - r['composite'])
        print(f"    {rank}. {name:<20s}  composite: {comp_imp:>+5.1f}%  "
              f"makespan: {ms_imp:>+5.1f}%  avg_ct: {avg_imp:>+5.1f}%")

    # -- 6. Detail: order completion times -------------------------------
    print(f"\n{'=' * 70}")
    print("DETAIL: Order completion times (Jeevan's simulator, seconds)")
    print("=" * 70)

    for name in ["Arkhan (SA)", "Arkhan (SOF)", "Jeevan (SA)", "Kate (SA)", "Liam (SA)", "Manjary (LPT)"]:
        r = results[name]
        cts = sorted(r['jeevan']['order_completion_times'].items(),
                     key=lambda x: x[1])
        print(f"\n  {name}:")
        for o, ct in cts:
            items = ", ".join(f"{q}x {sname(t)}"
                              for t, q in sorted(demands[o].items()))
            print(f"    {ct:6.0f}s : O{o} ({ototal(demands[o])} items: {items})")

    # -- 7. Actual physical run comparison --------------------------------
    # Day 1: Ran two SA variants (Arkhan/Jeevan SA identical, Kate/Liam SA identical)
    # Day 2: Narrowed to SA vs SOF final showdown
    print(f"\n{'=' * 70}")
    print("ACTUAL PHYSICAL RUNS vs SIMULATOR (4 runs across 2 days)")
    print("=" * 70)

    actual_runs = [
        # (label, file, algorithm, day)
        ("Day1 Run A (SA)",  os.path.join(_script_dir, 'grp_3_run_1_a_Liam_SA.csv'),    "SA",  1),
        ("Day1 Run B (SA)",  os.path.join(_script_dir, 'grp3_run_1_b_Arkhan_SA.csv'),    "SA",  1),
        ("Day2 Run A (SA)",  os.path.join(_script_dir, 'grp_3_run_2_a_Liam_SA.csv'),     "SA",  2),
        ("Day2 Run B (SOF)", os.path.join(_script_dir, 'grp_3_run_2_b_Arkhan_SOF.csv'),  "SOF", 2),
    ]

    actual_results = {}
    for label, path, algo, day in actual_runs:
        if not os.path.exists(path):
            print(f"  {label}: file not found ({path})")
            continue
        rows = []
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                rows.append({
                    'belt': int(row['conv_num']),
                    'shape': row['shape_name'],
                    'time': float(row['time']),
                })
        rows.sort(key=lambda r: r['time'])
        makespan = rows[-1]['time'] if rows else 0
        first_item = rows[0]['time'] if rows else 0
        n_items = len(rows)

        # Per-belt item counts
        belt_counts = defaultdict(int)
        shape_counts = defaultdict(int)
        for r in rows:
            belt_counts[r['belt']] += 1
            shape_counts[r['shape']] += 1

        # Inter-item gaps for throughput analysis
        gaps = [rows[i+1]['time'] - rows[i]['time'] for i in range(len(rows)-1)]
        avg_gap = sum(gaps) / len(gaps) if gaps else 0
        median_gap = sorted(gaps)[len(gaps)//2] if gaps else 0

        # Same-belt vs diff-belt gap analysis
        same_gaps, diff_gaps = [], []
        for i in range(len(rows)-1):
            gap = rows[i+1]['time'] - rows[i]['time']
            if rows[i+1]['belt'] == rows[i]['belt']:
                same_gaps.append(gap)
            else:
                diff_gaps.append(gap)

        actual_results[label] = {
            'makespan': makespan,
            'first_item': first_item,
            'n_items': n_items,
            'belt_counts': dict(sorted(belt_counts.items())),
            'shape_counts': dict(sorted(shape_counts.items())),
            'avg_gap': avg_gap,
            'median_gap': median_gap,
            'same_belt_avg_gap': sum(same_gaps)/len(same_gaps) if same_gaps else 0,
            'diff_belt_avg_gap': sum(diff_gaps)/len(diff_gaps) if diff_gaps else 0,
            'throughput': n_items / makespan if makespan > 0 else 0,
            'algo': algo,
            'day': day,
        }

    # Map algorithms to simulated counterparts
    sim_map = {
        "SA":  ["Arkhan (SA)", "Liam (SA)", "Jeevan (SA)", "Kate (SA)"],
        "SOF": ["Arkhan (SOF)"],
    }

    print(f"\n  Day 1: Arkhan SA = Jeevan SA (same optimizer), Kate SA = Liam SA (same optimizer)")
    print(f"  Day 2: Narrowed to SA vs SOF for final comparison\n")

    hdr_act = (f"{'Run':<20s} | {'Items':>5s} | {'Makespan':>10s} | "
               f"{'Throughput':>10s} | {'Avg Gap':>8s} | {'SameBelt':>8s} | "
               f"{'DiffBelt':>8s} | {'Belt Distribution'}")
    print(f"{hdr_act}")
    print("-" * (len(hdr_act) + 10))

    for label, path, algo, day in actual_runs:
        if label not in actual_results:
            continue
        ar = actual_results[label]
        belt_str = ", ".join(f"B{b}:{c}" for b, c in ar['belt_counts'].items())
        print(f"{label:<20s} | {ar['n_items']:>5d} | {ar['makespan']:>9.1f}s | "
              f"{ar['throughput']:>8.3f}/s | {ar['avg_gap']:>7.1f}s | "
              f"{ar['same_belt_avg_gap']:>7.1f}s | {ar['diff_belt_avg_gap']:>7.1f}s | "
              f"{belt_str}")

    # Compare against simulated predictions
    print(f"\n  Simulated vs Actual Makespan Comparison:")
    print(f"  {'Run':<20s} | {'Actual':>10s} | {'Sim Best':>10s} | {'Error':>8s}")
    print(f"  {'-'*60}")
    for label, path, algo, day in actual_runs:
        if label not in actual_results:
            continue
        ar = actual_results[label]
        sim_names = sim_map.get(algo, [])
        best_sim_ms = None
        best_sim_name = ""
        for sn in sim_names:
            if sn in results:
                ms = results[sn]['jeevan']['makespan']
                if best_sim_ms is None or ms < best_sim_ms:
                    best_sim_ms = ms
                    best_sim_name = sn
        if best_sim_ms is not None:
            error_pct = abs(best_sim_ms - ar['makespan']) / ar['makespan'] * 100
            print(f"  {label:<20s} | {ar['makespan']:>9.1f}s | {best_sim_ms:>9.1f}s | "
                  f"{error_pct:>6.1f}%")

    print(f"\n  Note: {total_items} items in problem; physical runs delivered fewer"
          f" (no circles in physical totes)")

    # -- 8. Export CSV (simulation results) --------------------------------
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'comparison_results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['solution', 'source',
                     'composite_score',
                     'makespan_sec', 'total_completion_time_sec',
                     'avg_completion_time_sec', 'wait_spread_sec',
                     'recirculation_events',
                     'kate_objective',
                     'items_delivered', 'throughput_items_per_s',
                     'avg_gap_sec', 'same_belt_avg_gap_sec', 'diff_belt_avg_gap_sec',
                     'belt_balance_std',
                     'makespan_improvement_pct',
                     'composite_improvement_pct'])

        # Simulated results
        for name in solutions:
            r = results[name]
            j = r['jeevan']
            ms_imp = (naive_ms - j['makespan']) / naive_ms * 100
            comp_imp = 100 - r['composite']
            w.writerow([name, 'simulation',
                        f"{r['composite']:.1f}",
                        f"{j['makespan']:.1f}",
                        f"{j['total_completion_time']:.1f}",
                        f"{j['avg_completion_time']:.1f}",
                        f"{r['spread']:.0f}",
                        j['recirculation_events'],
                        f"{r['kate_obj']:.1f}",
                        total_items, '', '', '', '', '',
                        f"{ms_imp:.1f}",
                        f"{comp_imp:.1f}"])

        # Physical run results
        for label, path, algo, day in actual_runs:
            if label not in actual_results:
                continue
            ar = actual_results[label]
            counts = list(ar['belt_counts'].values())
            belt_std = statistics.stdev(counts) if len(counts) > 1 else 0
            w.writerow([f"{label} [{algo}]", 'physical',
                        '', '',
                        f"{ar['makespan']:.1f}",
                        '', '', '', '',
                        ar['n_items'],
                        f"{ar['throughput']:.4f}",
                        f"{ar['avg_gap']:.2f}",
                        f"{ar['same_belt_avg_gap']:.2f}",
                        f"{ar['diff_belt_avg_gap']:.2f}",
                        f"{belt_std:.2f}",
                        '', ''])

    print(f"\n  CSV saved to {csv_path}")
    print()


if __name__ == '__main__':
    main()
