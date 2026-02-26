"""
MSE433 Module 3 - Conveyor Belt Order Consolidation Optimizer v3
================================================================
Complete solution with:
1. Data generation (matching notebook exactly)
2. Order-to-belt assignment optimization (SA, Zhou 2017)
3. Tote loading sequence optimization (which items to load first)
4. Small test instance generation for physical conveyor demo
5. Full reporting and CSV output

System (from slides):
- 4 conveyor belts forming a loop
- Each belt has 1 scanner + 1 lane = 1 order at a time
- Items loaded via ramp onto conveyor, circulate until claimed
- Pneumatic arms push items off when scanner detects a match
- Items of same type are INTERCHANGEABLE across totes
- Once on conveyor, you cannot touch items
- When order completes on a belt, next order in queue starts
"""

import random
import math
import csv
import sys
import json
from collections import defaultdict
from itertools import permutations

# ============================================================
# CONSTANTS
# ============================================================
SHAPES = ['circle', 'pentagon', 'trapezoid', 'triangle', 'star', 'moon', 'heart', 'cross']
NUM_BELTS = 4

# Timing calibrated from example output: 15 items sorted in 213s
# Gaps between consecutive sorts range from 1.7s to 40s, avg ~14s
# Items on same belt back-to-back: ~8s
# Items switching belts: ~14-20s
# First item delay (startup): ~20s
T_STARTUP = 20.0
T_SAME_BELT = 8.0      # consecutive items claimed by same belt
T_DIFF_BELT = 14.0     # item claimed by different belt than previous
T_RECIRCULATE = 25.0   # item goes around loop unclaimed


# ============================================================
# DATA GENERATION (exact reproduction of notebook)
# ============================================================
def generate_data(seed=100):
    """Reproduce MSE433_M3_data_generator.ipynb exactly."""
    def gr(s, a, b):
        random.seed(s)
        return random.randint(a, b)

    n_orders = gr(seed, 10, 15)
    n_itemtypes = gr(seed, 7, 10)
    n_totes = gr(seed, 15, 20)

    # After last gr() call, random state = seed(seed) + 1 randint consumed
    order_itemtypes = []
    order_quantities = []
    for i in range(n_orders):
        sz = random.randint(1, 3)
        tt = random.sample(range(0, n_itemtypes - 1), sz)
        qq = [random.randint(1, 3) for _ in range(sz)]
        order_itemtypes.append(tt)
        order_quantities.append(qq)

    orders_totes = [[] for _ in range(n_orders)]
    for i in range(n_orders):
        for j in range(len(order_itemtypes[i])):
            if j == 0:
                orders_totes[i].append(random.randint(0, n_totes - 1))
            else:
                if random.randint(0, 1) == 0:
                    orders_totes[i].append(orders_totes[i][0])
                else:
                    orders_totes[i].append(random.randint(0, n_totes - 1))

    return {
        'n_orders': n_orders, 'n_itemtypes': n_itemtypes, 'n_totes': n_totes,
        'order_itemtypes': order_itemtypes, 'order_quantities': order_quantities,
        'orders_totes': orders_totes, 'seed': seed
    }


def get_demands(data):
    """For each order, return {item_type: quantity}."""
    demands = []
    for i in range(data['n_orders']):
        d = defaultdict(int)
        for j, t in enumerate(data['order_itemtypes'][i]):
            d[t] += data['order_quantities'][i][j]
        demands.append(dict(d))
    return demands


def get_tote_contents(data):
    """What's physically in each tote: {tote_id: {item_type: qty}}."""
    totes = defaultdict(lambda: defaultdict(int))
    for i in range(data['n_orders']):
        for j, itype in enumerate(data['order_itemtypes'][i]):
            tote = data['orders_totes'][i][j]
            qty = data['order_quantities'][i][j]
            totes[tote][itype] += qty
    return totes


def ototal(d):
    return sum(d.values())


def sname(t):
    return SHAPES[t] if 0 <= t < 8 else f"type{t}"


# ============================================================
# CONVEYOR SIMULATION (calibrated to physical system)
# ============================================================
class ConveyorSim:
    """
    Simulates the physical conveyor belt system.

    The key insight: items are loaded onto the ramp one at a time.
    They circulate on the loop. Each belt's scanner checks if the
    passing item matches its current order. If yes, pneumatic arm
    pushes it off. If no, item continues circulating.

    What WE control:
    1. belt_queues: which orders go on which belt, in what sequence
    2. loading_order: the order we physically place items on the ramp

    The loading order is critical because:
    - Items loaded first reach belts first
    - Loading items for the SAME belt consecutively reduces idle time
    - Loading items for orders that aren't yet active wastes circulation time
    """

    def __init__(self, demands):
        self.demands = demands
        self.n = len(demands)

    def simulate(self, belt_queues, loading_order=None):
        """
        Full simulation.

        belt_queues: [[order_idx, ...], ...] x4 belts
        loading_order: [(item_type, target_belt), ...] sequence of items to load

        Returns detailed timing results.
        """
        if loading_order is None:
            loading_order = self.smart_loading_order(belt_queues)

        # State
        belt_current_order = [None] * NUM_BELTS
        belt_remaining = [None] * NUM_BELTS
        belt_queue = [list(q) for q in belt_queues]
        order_completion = {}
        order_first_item = {}

        # Assign initial orders
        for b in range(NUM_BELTS):
            if belt_queue[b]:
                oi = belt_queue[b].pop(0)
                belt_current_order[b] = oi
                belt_remaining[b] = dict(self.demands[oi])

        time = T_STARTUP
        last_belt_claimed = -1
        items_sorted = 0
        total_items = sum(ototal(d) for d in self.demands)

        item_list = list(loading_order)
        recirculating = []
        safety = 0

        while (item_list or recirculating) and items_sorted < total_items:
            safety += 1
            if safety > total_items * 8:
                break

            # If main queue empty, recirculating items come back
            if not item_list and recirculating:
                time += T_RECIRCULATE
                item_list = recirculating
                recirculating = []
                last_belt_claimed = -1

            if not item_list:
                break

            itype, target_belt = item_list.pop(0)

            # Try each belt starting from target (item reaches target first)
            claimed = False
            for offset in range(NUM_BELTS):
                b = (target_belt + offset) % NUM_BELTS

                if belt_remaining[b] is not None and belt_remaining[b].get(itype, 0) > 0:
                    # Belt claims item
                    if b == last_belt_claimed:
                        time += T_SAME_BELT
                    else:
                        time += T_DIFF_BELT

                    belt_remaining[b][itype] -= 1
                    items_sorted += 1
                    last_belt_claimed = b

                    oi = belt_current_order[b]
                    if oi not in order_first_item:
                        order_first_item[oi] = time

                    # Check if order complete
                    if all(v <= 0 for v in belt_remaining[b].values()):
                        order_completion[oi] = time

                        # Next order on this belt
                        if belt_queue[b]:
                            next_oi = belt_queue[b].pop(0)
                            belt_current_order[b] = next_oi
                            belt_remaining[b] = dict(self.demands[next_oi])
                            # Load items for new order
                            for it, q in self.demands[next_oi].items():
                                for _ in range(q):
                                    item_list.append((it, b))
                        else:
                            belt_current_order[b] = None
                            belt_remaining[b] = None

                    claimed = True
                    break

            if not claimed:
                recirculating.append((itype, target_belt))

        # Handle incomplete
        for b in range(NUM_BELTS):
            oi = belt_current_order[b]
            if oi is not None and oi not in order_completion:
                rem = sum(max(0, v) for v in (belt_remaining[b] or {}).values())
                order_completion[oi] = time + rem * T_DIFF_BELT * 2

        makespan = max(order_completion.values()) if order_completion else 9999
        total_ct = sum(order_completion.values()) if order_completion else 9999

        return {
            'order_completion_times': order_completion,
            'order_first_item_times': order_first_item,
            'makespan': makespan,
            'total_completion_time': total_ct,
            'avg_completion_time': total_ct / max(len(order_completion), 1),
            'items_sorted': items_sorted,
            'total_items': total_items,
            'recirculation_events': len(recirculating),
        }

    def smart_loading_order(self, belt_queues):
        """
        Generate an optimized loading sequence.

        Strategy (from Zhou thesis insight):
        1. Load items for ACTIVE orders first (orders currently on belts)
        2. Group items by target belt to minimize belt-switching delays
        3. Within each belt's items, load smaller orders first (SPT)
        4. Interleave belts to keep all belts fed

        This is the "tote loading sequence" optimization.
        """
        items = []

        # Phase 1: Items for the first order on each belt
        first_orders = []
        for b in range(NUM_BELTS):
            if belt_queues[b]:
                oi = belt_queues[b][0]
                first_orders.append((b, oi))

        # Interleave: load a few items per belt at a time (round-robin)
        belt_items = defaultdict(list)
        for b, oi in first_orders:
            for itype, qty in sorted(self.demands[oi].items()):
                for _ in range(qty):
                    belt_items[b].append((itype, b))

        # Round-robin interleave across belts
        max_items = max(len(v) for v in belt_items.values()) if belt_items else 0
        for idx in range(max_items):
            for b in range(NUM_BELTS):
                if idx < len(belt_items[b]):
                    items.append(belt_items[b][idx])

        return items

    def naive_loading_order(self, belt_queues):
        """Load all items for belt 0 first, then belt 1, etc."""
        items = []
        for b in range(NUM_BELTS):
            if belt_queues[b]:
                oi = belt_queues[b][0]
                for itype, qty in self.demands[oi].items():
                    for _ in range(qty):
                        items.append((itype, b))
        return items

    def fast_eval(self, belt_queues, objective='total_completion_time'):
        """Fast heuristic for SA (no full sim)."""
        total_ct = 0
        max_ct = 0
        for b in range(NUM_BELTS):
            cum = T_STARTUP
            for oi in belt_queues[b]:
                n = ototal(self.demands[oi])
                t = T_DIFF_BELT + (n - 1) * T_SAME_BELT
                cum += t
                total_ct += cum
            max_ct = max(max_ct, cum)
        return {'total_completion_time': total_ct, 'makespan': max_ct}


# ============================================================
# SIMULATED ANNEALING (Zhou 2017 Algorithm 1)
# ============================================================
class SA:
    def __init__(self, sim, dem, obj='total_completion_time',
                 iters=20000, T0=100, alpha=0.9997):
        self.sim = sim
        self.dem = dem
        self.n = len(dem)
        self.obj = obj
        self.iters = iters
        self.T0 = T0
        self.alpha = alpha

    def init_sol(self):
        """LPT assignment + SPT sequencing per belt."""
        sz = [(ototal(self.dem[i]), i) for i in range(self.n)]
        sz.sort(reverse=True)
        qs = [[] for _ in range(4)]
        ld = [0] * 4
        for s, o in sz:
            b = min(range(4), key=lambda x: ld[x])
            qs[b].append(o)
            ld[b] += s
        for b in range(4):
            qs[b].sort(key=lambda o: ototal(self.dem[o]))
        return qs

    def neighbor(self, sol):
        nw = [list(q) for q in sol]
        ne = [b for b in range(4) if nw[b]]
        if not ne:
            return nw
        r = random.random()
        if r < 0.5:
            sb = random.choice(ne)
            si = random.randint(0, len(nw[sb]) - 1)
            o = nw[sb].pop(si)
            db = random.randint(0, 3)
            di = random.randint(0, len(nw[db]))
            nw[db].insert(di, o)
        elif r < 0.8:
            b1 = random.choice(ne)
            i1 = random.randint(0, len(nw[b1]) - 1)
            b2 = random.choice(ne)
            i2 = random.randint(0, len(nw[b2]) - 1)
            nw[b1][i1], nw[b2][i2] = nw[b2][i2], nw[b1][i1]
        else:
            b = random.choice(ne)
            if len(nw[b]) >= 2:
                i = random.randint(0, len(nw[b]) - 2)
                j = random.randint(i + 1, len(nw[b]) - 1)
                nw[b][i:j + 1] = reversed(nw[b][i:j + 1])
        return nw

    def run(self, verbose=True):
        cur = self.init_sol()
        co = self.sim.fast_eval(cur, self.obj)[self.obj]
        best = [list(q) for q in cur]
        bo = co
        T = self.T0

        if verbose:
            ld = [sum(ototal(self.dem[o]) for o in q) for q in cur]
            print(f"  LPT init: obj={co:.0f}, belt_loads={ld}")

        for k in range(self.iters):
            cand = self.neighbor(cur)
            cao = self.sim.fast_eval(cand, self.obj)[self.obj]
            d = cao - co
            if d < 0 or (T > 1e-10 and random.random() < math.exp(-d / T)):
                cur = cand
                co = cao
            if co < bo:
                best = [list(q) for q in cur]
                bo = co
            T *= self.alpha
            if verbose and (k + 1) % 5000 == 0:
                print(f"  Iter {k + 1}: best={bo:.0f}, T={T:.1f}")

        if verbose:
            init_obj = self.sim.fast_eval(self.init_sol(), self.obj)[self.obj]
            imp = (init_obj - bo) / init_obj * 100 if init_obj > 0 else 0
            print(f"  Done: {bo:.0f} ({imp:.1f}% improvement over LPT)")
        return best, bo


# ============================================================
# TOTE LOADING SEQUENCE OPTIMIZER
# ============================================================
def optimize_loading_order(sim, belt_queues, demands):
    """
    Optimize the physical loading sequence.

    Tests multiple strategies and picks the best one:
    1. Naive: all belt 0 items, then belt 1, etc.
    2. Round-robin: 1 item per belt, cycling
    3. Smart interleave: group by belt but interleave
    4. SPT-aware: load items for smallest active orders first
    """
    strategies = {}

    # Strategy 1: Naive sequential
    naive = sim.naive_loading_order(belt_queues)
    res_naive = sim.simulate(belt_queues, naive)
    strategies['naive_sequential'] = (naive, res_naive)

    # Strategy 2: Smart interleave (round-robin)
    smart = sim.smart_loading_order(belt_queues)
    res_smart = sim.simulate(belt_queues, smart)
    strategies['round_robin_interleave'] = (smart, res_smart)

    # Strategy 3: Belt-grouped (all of belt 0, all of belt 1, etc) 
    # but belt order optimized by smallest-first
    belt_sizes = []
    for b in range(NUM_BELTS):
        if belt_queues[b]:
            oi = belt_queues[b][0]
            belt_sizes.append((ototal(demands[oi]), b))
    belt_sizes.sort()  # smallest belt load first

    grouped = []
    for _, b in belt_sizes:
        if belt_queues[b]:
            oi = belt_queues[b][0]
            for itype, qty in sorted(demands[oi].items()):
                for _ in range(qty):
                    grouped.append((itype, b))
    res_grouped = sim.simulate(belt_queues, grouped)
    strategies['belt_grouped_spt'] = (grouped, res_grouped)

    # Strategy 4: Full interleave with SPT priority
    belt_items = {}
    for b in range(NUM_BELTS):
        if belt_queues[b]:
            oi = belt_queues[b][0]
            items = []
            for itype, qty in sorted(demands[oi].items()):
                for _ in range(qty):
                    items.append((itype, b))
            belt_items[b] = items

    # Sort belts by number of items (smallest first)
    sorted_belts = sorted(belt_items.keys(), key=lambda b: len(belt_items[b]))
    spt_interleave = []
    max_len = max(len(v) for v in belt_items.values()) if belt_items else 0
    for idx in range(max_len):
        for b in sorted_belts:
            if idx < len(belt_items[b]):
                spt_interleave.append(belt_items[b][idx])
    res_spt = sim.simulate(belt_queues, spt_interleave)
    strategies['spt_interleave'] = (spt_interleave, res_spt)

    # Pick best by makespan
    best_name = min(strategies, key=lambda k: strategies[k][1]['makespan'])
    return best_name, strategies


# ============================================================
# SMALL TEST INSTANCE GENERATOR
# ============================================================
def generate_small_instance(data, demands, belt_queues, max_orders=4, max_items=12):
    """
    Extract a small subset of the full instance for physical conveyor testing.
    Picks the first order from each belt (up to 4 orders).
    """
    selected = []
    for b in range(NUM_BELTS):
        if belt_queues[b] and len(selected) < max_orders:
            oi = belt_queues[b][0]
            if ototal(demands[oi]) + sum(ototal(demands[o]) for o in selected) <= max_items:
                selected.append(oi)

    # If not enough, add more
    for b in range(NUM_BELTS):
        for oi in belt_queues[b]:
            if oi not in selected and len(selected) < max_orders:
                if ototal(demands[oi]) + sum(ototal(demands[o]) for o in selected) <= max_items:
                    selected.append(oi)

    small_queues = [[] for _ in range(NUM_BELTS)]
    for i, oi in enumerate(selected):
        small_queues[i % NUM_BELTS].append(oi)

    small_demands = {oi: demands[oi] for oi in selected}
    total = sum(ototal(demands[oi]) for oi in selected)

    return selected, small_queues, small_demands, total


# ============================================================
# CSV OUTPUT
# ============================================================
def gen_csv(belt_queues, demands, filename):
    """Generate input CSV for conveyor system."""
    rows = []
    mx = max(len(q) for q in belt_queues) if any(belt_queues) else 0
    for r in range(mx):
        for b in range(NUM_BELTS):
            if r < len(belt_queues[b]):
                oi = belt_queues[b][r]
                c = [0] * 8
                for t, q in demands[oi].items():
                    if 0 <= t < 8:
                        c[t] = q
                rows.append([b] + c)
    with open(filename, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['conv_num', 'cirle', 'pentagon', 'trapezoid',
                     'triangle', 'star', 'moon', 'heart', 'cross'])
        for row in rows:
            w.writerow([f"{row[0]}"] + [f" {v}" for v in row[1:]])
    return rows


# ============================================================
# TOTE INSTRUCTION SHEET
# ============================================================
def generate_tote_instructions(data, demands, belt_queues, loading_order, selected_orders=None):
    """
    Generate a human-readable instruction sheet for physically
    operating the conveyor.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("TOTE LOADING INSTRUCTIONS")
    lines.append("=" * 60)
    lines.append("")

    # Map: what's in each tote
    tote_contents = get_tote_contents(data)
    orders_to_show = selected_orders or list(range(data['n_orders']))

    lines.append("STEP 1: BELT ASSIGNMENTS (enter into conveyor CSV)")
    lines.append("-" * 50)
    for b in range(NUM_BELTS):
        if belt_queues[b]:
            oi = belt_queues[b][0]
            if oi in orders_to_show:
                items = ", ".join(f"{q}x {sname(t)}" for t, q in sorted(demands[oi].items()))
                lines.append(f"  Belt {b} → Order {oi}: {items}")
    lines.append("")

    lines.append("STEP 2: PICK ITEMS INTO TOTES")
    lines.append("-" * 50)
    totes_needed = set()
    for oi in orders_to_show:
        if oi < len(data['orders_totes']):
            for t in data['orders_totes'][oi]:
                totes_needed.add(t)

    for tote_id in sorted(totes_needed):
        contents = tote_contents.get(tote_id, {})
        if contents:
            items = ", ".join(f"{q}x {sname(t)}" for t, q in sorted(contents.items()))
            lines.append(f"  Tote {tote_id}: {items}")
    lines.append("")

    lines.append("STEP 3: LOAD ITEMS ONTO RAMP (in this order!)")
    lines.append("-" * 50)
    for i, (itype, target_belt) in enumerate(loading_order):
        lines.append(f"  {i + 1:3d}. Load 1x {sname(itype):12s} → (for Belt {target_belt})")
    lines.append("")

    lines.append("STEP 4: OBSERVE AND RECORD")
    lines.append("-" * 50)
    lines.append("  - Do NOT touch items once on conveyor")
    lines.append("  - Record which items get sorted to which belt")
    lines.append("  - Note any items that recirculate (go around twice)")
    lines.append("  - Collect output CSV from IDEAS Clinic staff")

    return "\n".join(lines)


# ============================================================
# EXPORT FOR VISUALIZATION
# ============================================================
def export_viz_data(data, demands, belt_queues, sim_result, loading_order):
    """Export data as JSON for the React visualization."""
    orders = []
    for i in range(data['n_orders']):
        items_list = []
        for t, q in sorted(demands[i].items()):
            items_list.append({'type': sname(t), 'type_id': t, 'quantity': q})
        orders.append({
            'id': i,
            'items': items_list,
            'total': ototal(demands[i]),
            'totes': data['orders_totes'][i],
            'completion_time': sim_result['order_completion_times'].get(i, None)
        })

    belts = []
    for b in range(NUM_BELTS):
        belts.append({
            'id': b,
            'order_queue': belt_queues[b],
        })

    loading = []
    for itype, target in loading_order:
        loading.append({'type': sname(itype), 'type_id': itype, 'target_belt': target})

    return {
        'orders': orders,
        'belts': belts,
        'loading_sequence': loading,
        'metrics': {
            'makespan': sim_result['makespan'],
            'total_completion_time': sim_result['total_completion_time'],
            'avg_completion_time': sim_result['avg_completion_time'],
            'items_sorted': sim_result['items_sorted'],
            'total_items': sim_result['total_items'],
        },
        'completion_sequence': sorted(
            [(oi, t) for oi, t in sim_result['order_completion_times'].items()],
            key=lambda x: x[1]
        )
    }


# ============================================================
# MAIN
# ============================================================
def main(seed=100, outdir='.'):
    print("=" * 65)
    print("MSE433 M3 - Conveyor Belt Optimizer v3 (Complete Solution)")
    print("=" * 65)
    print(f"Seed: {seed}\n")

    # 1. Generate data
    data = generate_data(seed)
    demands = get_demands(data)
    total_items = sum(ototal(d) for d in demands)

    print(f"Generated: {data['n_orders']} orders, {total_items} items, "
          f"{data['n_itemtypes']} types, {data['n_totes']} totes\n")

    for i in range(data['n_orders']):
        items = ", ".join(f"{q}x{sname(t)}" for t, q in sorted(demands[i].items()))
        print(f"  O{i:2d}: {ototal(demands[i]):2d} items ({items}) "
              f"| Totes: {data['orders_totes'][i]}")

    sim = ConveyorSim(demands)

    # 2. Optimize belt assignment
    print(f"\n{'=' * 65}")
    print("PHASE 1: Belt Assignment Optimization (SA)")
    print("=" * 65)

    random.seed(42)
    sa = SA(sim, demands, 'total_completion_time', iters=20000)
    best_sol, _ = sa.run()

    random.seed(43)
    sa2 = SA(sim, demands, 'makespan', iters=20000)
    sol2, _ = sa2.run()

    # Full sim both
    res1 = sim.simulate(best_sol)
    res2 = sim.simulate(sol2)

    if res1['makespan'] <= res2['makespan']:
        opt_sol, opt_res = best_sol, res1
        print(f"\n  Selected: total_completion_time optimizer")
    else:
        opt_sol, opt_res = sol2, res2
        print(f"\n  Selected: makespan optimizer")

    # Also compare vs naive
    naive_sol = [[] for _ in range(4)]
    for i in range(data['n_orders']):
        naive_sol[i % 4].append(i)
    naive_res = sim.simulate(naive_sol)

    print(f"\n  Naive:     makespan={naive_res['makespan']:.0f}s ({naive_res['makespan']/60:.1f}min)")
    print(f"  Optimized: makespan={opt_res['makespan']:.0f}s ({opt_res['makespan']/60:.1f}min)")
    imp = (naive_res['makespan'] - opt_res['makespan']) / naive_res['makespan'] * 100
    print(f"  Improvement: {imp:.1f}%")

    for b in range(NUM_BELTS):
        os_ = " → ".join(f"O{o}" for o in opt_sol[b])
        ld = sum(ototal(demands[o]) for o in opt_sol[b])
        print(f"  Belt {b} [{ld:2d} items]: {os_}")

    # 3. Optimize loading order
    print(f"\n{'=' * 65}")
    print("PHASE 2: Tote Loading Sequence Optimization")
    print("=" * 65)

    best_strategy, all_strategies = optimize_loading_order(sim, opt_sol, demands)

    for name, (_, res) in sorted(all_strategies.items(),
                                  key=lambda x: x[1][1]['makespan']):
        tag = " ← BEST" if name == best_strategy else ""
        print(f"  {name:30s}: makespan={res['makespan']:6.0f}s "
              f"({res['makespan']/60:.1f}min), "
              f"recirc={res['recirculation_events']}{tag}")

    best_loading, best_load_res = all_strategies[best_strategy]
    print(f"\n  Best strategy: {best_strategy}")
    print(f"  Final makespan: {best_load_res['makespan']:.0f}s ({best_load_res['makespan']/60:.1f} min)")

    cts = sorted(best_load_res['order_completion_times'].items(), key=lambda x: x[1])
    print(f"\n  Order completion sequence:")
    for o, ct in cts:
        print(f"    {ct:6.0f}s : O{o} ({ototal(demands[o])} items: "
              f"{', '.join(f'{q}x{sname(t)}' for t, q in sorted(demands[o].items()))})")

    # 4. Generate full CSV
    full_csv_path = f"{outdir}/conveyor_input_seed{seed}.csv"
    gen_csv(opt_sol, demands, full_csv_path)
    print(f"\n  Full CSV: {full_csv_path}")

    # 5. Generate small test instance
    print(f"\n{'=' * 65}")
    print("PHASE 3: Small Test Instance for Physical Conveyor")
    print("=" * 65)

    selected, small_queues, small_dem, small_total = generate_small_instance(
        data, demands, opt_sol, max_orders=4, max_items=12)

    print(f"  Selected orders: {selected} ({small_total} items)")
    for oi in selected:
        items = ", ".join(f"{q}x{sname(t)}" for t, q in sorted(demands[oi].items()))
        print(f"    O{oi}: {items}")

    small_sim = ConveyorSim(demands)
    small_loading = small_sim.smart_loading_order(small_queues)
    small_res = small_sim.simulate(small_queues, small_loading)

    small_csv_path = f"{outdir}/conveyor_input_small_seed{seed}.csv"
    gen_csv(small_queues, demands, small_csv_path)
    print(f"\n  Small CSV: {small_csv_path}")
    print(f"  Estimated time: ~{small_res['makespan']:.0f}s ({small_res['makespan']/60:.1f} min)")

    for b in range(NUM_BELTS):
        if small_queues[b]:
            oi = small_queues[b][0]
            items = ", ".join(f"{q}x{sname(t)}" for t, q in sorted(demands[oi].items()))
            print(f"  Belt {b} → O{oi}: {items}")

    # 6. Generate tote instructions
    instructions = generate_tote_instructions(
        data, demands, small_queues, small_loading, selected)
    inst_path = f"{outdir}/loading_instructions_seed{seed}.txt"
    with open(inst_path, 'w') as f:
        f.write(instructions)
    print(f"\n  Instructions: {inst_path}")

    # 7. Export viz data
    viz_data = export_viz_data(data, demands, opt_sol, best_load_res, best_loading)
    viz_path = f"{outdir}/viz_data_seed{seed}.json"
    with open(viz_path, 'w') as f:
        json.dump(viz_data, f, indent=2)
    print(f"  Viz JSON:   {viz_path}")

    return {
        'opt_sol': opt_sol, 'opt_res': best_load_res,
        'small_queues': small_queues, 'small_res': small_res,
        'data': data, 'demands': demands,
        'loading_order': best_loading, 'small_loading': small_loading,
    }


if __name__ == '__main__':
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    outdir = sys.argv[2] if len(sys.argv) > 2 else '.'
    main(seed=seed, outdir=outdir)
