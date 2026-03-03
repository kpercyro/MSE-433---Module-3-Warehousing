#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import heapq
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple

SHAPE_NAMES = [
    "circle",
    "pentagon",
    "trapezoid",
    "triangle",
    "star",
    "moon",
    "heart",
    "cross",
]


def shape_name_from_idx(shape_idx: int) -> str:
    if 0 <= shape_idx < len(SHAPE_NAMES):
        return SHAPE_NAMES[shape_idx]
    return f"shape_{shape_idx}"


@dataclass(frozen=True)
class SolverParams:
    iterations: int = 5000
    seed: int = 42
    t0: float = 1.0
    alpha: float = 0.99

    # Conveyor simulation parameters
    item_release_interval: float = 1.0
    tote_change_time: float = 2.0
    entry_to_first_belt: float = 3.0
    belt_spacing: float = 2.5
    return_to_first_belt: float = 2.5
    max_item_passes: int = 200

    infeasible_penalty: float = 1_000_000.0


@dataclass
class ProblemInstance:
    num_belts: int
    num_orders: int
    num_itemtypes: int
    order_demands: List[Dict[int, int]]
    order_total_items: List[int]
    tote_inventory: Dict[int, Dict[int, int]]
    tote_ids: List[int]
    tote_to_orders: Dict[int, set]
    total_items: int


@dataclass
class Solution:
    belt_queues: List[List[int]]
    tote_sequence: List[int]


@dataclass
class SimulationResult:
    feasible: bool
    makespan: float
    avg_completion: float
    completion_times: List[Optional[float]]
    claims: List[Tuple[int, int, float]]  # (conv_num, item_type, time)
    release_steps: List[Tuple[int, int, int, int, float]]
    # (global_release_order, tote_id, release_order_in_tote, item_type, release_time)
    dropped_items: int


def read_sparse_rows_csv(path: str) -> List[List[int]]:
    rows: List[List[int]] = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            parsed: List[int] = []
            for cell in row:
                s = cell.strip()
                if s == "":
                    continue
                try:
                    parsed.append(int(float(s)))
                except ValueError:
                    continue
            rows.append(parsed)
    return rows


def load_problem(
    order_itemtypes_path: str,
    order_quantities_path: str,
    orders_totes_path: str,
    num_belts: int = 4,
) -> ProblemInstance:
    item_rows = read_sparse_rows_csv(order_itemtypes_path)
    qty_rows = read_sparse_rows_csv(order_quantities_path)
    tote_rows = read_sparse_rows_csv(orders_totes_path)

    num_orders = max(len(item_rows), len(qty_rows), len(tote_rows))
    order_demands: List[Dict[int, int]] = []
    order_total_items: List[int] = []

    tote_inventory: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    tote_to_orders: Dict[int, set] = defaultdict(set)

    max_itemtype = -1

    for o in range(num_orders):
        types = item_rows[o] if o < len(item_rows) else []
        qtys = qty_rows[o] if o < len(qty_rows) else []
        totes = tote_rows[o] if o < len(tote_rows) else []

        n = min(len(types), len(qtys), len(totes))
        if n == 0:
            order_demands.append({})
            order_total_items.append(0)
            continue

        demand: Dict[int, int] = defaultdict(int)

        for j in range(n):
            item = int(types[j])
            qty = int(qtys[j])
            tote = int(totes[j])

            if qty <= 0:
                continue

            demand[item] += qty
            tote_inventory[tote][item] += qty
            tote_to_orders[tote].add(o)
            max_itemtype = max(max_itemtype, item)

        demand = dict(demand)
        order_demands.append(demand)
        order_total_items.append(sum(demand.values()))

    tote_inventory_final = {t: dict(inv) for t, inv in tote_inventory.items()}
    tote_ids = sorted(tote_inventory_final.keys())
    num_itemtypes = max_itemtype + 1 if max_itemtype >= 0 else 0
    total_items = sum(order_total_items)

    return ProblemInstance(
        num_belts=num_belts,
        num_orders=num_orders,
        num_itemtypes=num_itemtypes,
        order_demands=order_demands,
        order_total_items=order_total_items,
        tote_inventory=tote_inventory_final,
        tote_ids=tote_ids,
        tote_to_orders=dict(tote_to_orders),
        total_items=total_items,
    )


def build_initial_solution(problem: ProblemInstance) -> Solution:
    # Greedy order assignment by current belt load.
    belt_queues: List[List[int]] = [[] for _ in range(problem.num_belts)]
    belt_load = [0] * problem.num_belts

    # Place larger orders first.
    order_ids = list(range(problem.num_orders))
    order_ids.sort(key=lambda o: problem.order_total_items[o], reverse=True)

    # Enforce at least one order per belt when possible.
    if len(order_ids) >= problem.num_belts:
        for b in range(problem.num_belts):
            o = order_ids[b]
            belt_queues[b].append(o)
            belt_load[b] += problem.order_total_items[o]
        remaining_orders = order_ids[problem.num_belts :]
    else:
        remaining_orders = order_ids

    for o in remaining_orders:
        b = min(range(problem.num_belts), key=lambda x: belt_load[x])
        belt_queues[b].append(o)
        belt_load[b] += problem.order_total_items[o]

    # Build order rank based on queue depth then belt index.
    rank = {}
    for b, q in enumerate(belt_queues):
        for pos, o in enumerate(q):
            rank[o] = pos * problem.num_belts + b

    # Greedy tote sequence: totes that feed earlier orders first.
    tote_scored = []
    fallback_rank = 10**9
    for tote in problem.tote_ids:
        orders = problem.tote_to_orders.get(tote, set())
        score = min((rank.get(o, fallback_rank) for o in orders), default=fallback_rank)
        tote_scored.append((score, tote))

    tote_scored.sort(key=lambda x: (x[0], x[1]))
    tote_sequence = [t for _, t in tote_scored]

    return Solution(belt_queues=belt_queues, tote_sequence=tote_sequence)


def solution_signature(
    sol: Solution,
) -> Tuple[Tuple[Tuple[int, ...], ...], Tuple[int, ...]]:
    return (
        tuple(tuple(q) for q in sol.belt_queues),
        tuple(sol.tote_sequence),
    )


def clone_solution(sol: Solution) -> Solution:
    return Solution(
        belt_queues=[list(q) for q in sol.belt_queues],
        tote_sequence=list(sol.tote_sequence),
    )


def is_better(a: SimulationResult, b: SimulationResult) -> bool:
    if a.feasible and not b.feasible:
        return True
    if not a.feasible and b.feasible:
        return False

    if a.makespan != b.makespan:
        return a.makespan < b.makespan
    return a.avg_completion < b.avg_completion


def scalar_objective(res: SimulationResult, total_items: int) -> float:
    if not res.feasible:
        return res.makespan
    denom = max(total_items, 1)
    return (res.makespan / denom) + 0.1 * (res.avg_completion / denom)


def choose_item_type_for_release(
    tote_counts: Dict[int, int],
    active_order: List[Optional[int]],
    active_pos: List[int],
    belt_queues: List[List[int]],
    remaining: List[Dict[int, int]],
) -> Optional[int]:
    candidates = [item for item, qty in tote_counts.items() if qty > 0]
    if not candidates:
        return None

    best_item = None
    best_rank = None

    num_belts = len(belt_queues)

    for item in candidates:
        # Tier 0: needed by currently active orders; prefer earliest belt index.
        active_belt = None
        active_need = 0
        for b in range(num_belts):
            o = active_order[b]
            if o is None:
                continue
            need = remaining[o].get(item, 0)
            if need > 0:
                active_belt = b
                active_need = need
                break

        if active_belt is not None:
            rank = (0, active_belt, -active_need, item)
        else:
            # Tier 1: needed by future orders; prefer shorter queue distance.
            future_best = None
            for b in range(num_belts):
                if active_pos[b] < 0:
                    start_pos = 0
                else:
                    start_pos = active_pos[b] + 1

                q = belt_queues[b]
                for pos in range(start_pos, len(q)):
                    o = q[pos]
                    if remaining[o].get(item, 0) > 0:
                        dist = pos - active_pos[b] if active_pos[b] >= 0 else pos + 1
                        cand = (dist, b)
                        if future_best is None or cand < future_best:
                            future_best = cand
                        break

            if future_best is not None:
                rank = (1, future_best[0], future_best[1], item)
            else:
                # Tier 2: not needed anywhere (or already fulfilled)
                rank = (2, 0, 0, item)

        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_item = item

    return best_item


def simulate_solution(
    problem: ProblemInstance, sol: Solution, params: SolverParams
) -> SimulationResult:
    remaining = [dict(d) for d in problem.order_demands]
    remaining_total = [sum(d.values()) for d in remaining]
    completion_times: List[Optional[float]] = [None] * problem.num_orders

    belt_queues = sol.belt_queues
    num_belts = problem.num_belts

    active_order: List[Optional[int]] = []
    active_pos: List[int] = []
    for b in range(num_belts):
        if belt_queues[b]:
            active_order.append(belt_queues[b][0])
            active_pos.append(0)
        else:
            active_order.append(None)
            active_pos.append(-1)

    def activate_next_order(belt: int, t_now: float) -> None:
        while True:
            if active_pos[belt] < 0:
                return
            nxt = active_pos[belt] + 1
            if nxt >= len(belt_queues[belt]):
                active_pos[belt] = -1
                active_order[belt] = None
                return
            active_pos[belt] = nxt
            o = belt_queues[belt][nxt]
            active_order[belt] = o
            if remaining_total[o] == 0:
                if completion_times[o] is None:
                    completion_times[o] = t_now
                continue
            return

    # Handle zero-demand orders at time 0.
    for b in range(num_belts):
        while active_order[b] is not None:
            o = active_order[b]
            if o is None or remaining_total[o] > 0:
                break
            if completion_times[o] is None:
                completion_times[o] = 0.0
            activate_next_order(b, 0.0)

    # Tote mutable inventory for this simulation.
    tote_remaining: Dict[int, Dict[int, int]] = {
        t: dict(inv) for t, inv in problem.tote_inventory.items()
    }

    # Event queue: (time, priority, serial, event_type, payload)
    events: List[Tuple[float, int, int, str, dict]] = []
    serial = 0

    def push_event(time: float, priority: int, event_type: str, payload: dict) -> None:
        nonlocal serial
        heapq.heappush(events, (time, priority, serial, event_type, payload))
        serial += 1

    # Build tote unload schedule as release slots.
    t_cursor = 0.0
    for i, tote in enumerate(sol.tote_sequence):
        if tote not in tote_remaining:
            continue
        n_items = sum(tote_remaining[tote].values())
        if n_items <= 0:
            continue

        if i > 0:
            t_cursor += params.tote_change_time

        for k in range(n_items):
            push_event(
                t_cursor + k * params.item_release_interval,
                1,  # release after belt-pass events at same time
                "release",
                {"tote": tote},
            )

        t_cursor += n_items * params.item_release_interval

    claims: List[Tuple[int, int, float]] = []
    release_steps: List[Tuple[int, int, int, int, float]] = []
    release_order_in_tote: Dict[int, int] = defaultdict(int)
    global_release_order = 0
    dropped_items = 0

    while events:
        t_now, _, _, event_type, payload = heapq.heappop(events)

        if event_type == "release":
            tote = payload["tote"]
            counts = tote_remaining.get(tote, {})
            item = choose_item_type_for_release(
                counts,
                active_order,
                active_pos,
                belt_queues,
                remaining,
            )
            if item is None:
                continue

            counts[item] -= 1
            if counts[item] <= 0:
                del counts[item]

            release_order_in_tote[tote] += 1
            global_release_order += 1
            release_steps.append(
                (
                    global_release_order,
                    tote,
                    release_order_in_tote[tote],
                    item,
                    t_now,
                )
            )

            token = {"item": item, "passes": 0}
            push_event(
                t_now + params.entry_to_first_belt,
                0,
                "belt_pass",
                {"belt": 0, "token": token},
            )

        elif event_type == "belt_pass":
            belt = payload["belt"]
            token = payload["token"]
            item = token["item"]

            o = active_order[belt]
            if o is not None and remaining[o].get(item, 0) > 0:
                remaining[o][item] -= 1
                remaining_total[o] -= 1
                if remaining[o][item] == 0:
                    del remaining[o][item]

                claims.append((belt, item, t_now))

                if remaining_total[o] == 0 and completion_times[o] is None:
                    completion_times[o] = t_now
                    activate_next_order(belt, t_now)
                continue

            token["passes"] += 1
            if token["passes"] > params.max_item_passes:
                dropped_items += 1
                continue

            if belt < num_belts - 1:
                nb = belt + 1
                dt = params.belt_spacing
            else:
                nb = 0
                dt = params.return_to_first_belt

            push_event(
                t_now + dt,
                0,
                "belt_pass",
                {"belt": nb, "token": token},
            )

    incomplete = [i for i, ct in enumerate(completion_times) if ct is None]
    feasible = len(incomplete) == 0

    if feasible:
        makespan = max(completion_times) if completion_times else 0.0
        avg_completion = mean(completion_times) if completion_times else 0.0
    else:
        penalty = (
            params.infeasible_penalty + 1000.0 * len(incomplete) + 10.0 * dropped_items
        )
        makespan = penalty
        avg_completion = penalty

    claims.sort(key=lambda x: x[2])

    return SimulationResult(
        feasible=feasible,
        makespan=float(makespan),
        avg_completion=float(avg_completion),
        completion_times=completion_times,
        claims=claims,
        release_steps=release_steps,
        dropped_items=dropped_items,
    )


def random_neighbor(sol: Solution, rng: random.Random, num_belts: int) -> Solution:
    ns = clone_solution(sol)
    move_type = rng.choice(
        ["move_order", "swap_order", "swap_tote", "insert_tote", "reorder_within_belt"]
    )

    # Build order locations helper.
    def order_locations(queues: List[List[int]]) -> List[Tuple[int, int]]:
        locs = []
        for b, q in enumerate(queues):
            for i in range(len(q)):
                locs.append((b, i))
        return locs

    if move_type == "move_order":
        total_assigned_orders = sum(len(q) for q in ns.belt_queues)
        if total_assigned_orders >= num_belts:
            src_candidates = [b for b in range(num_belts) if len(ns.belt_queues[b]) > 1]
        else:
            src_candidates = [b for b in range(num_belts) if ns.belt_queues[b]]

        if not src_candidates:
            return ns

        src_b = rng.choice(src_candidates)
        src_i = rng.randrange(len(ns.belt_queues[src_b]))
        order = ns.belt_queues[src_b].pop(src_i)

        dst_b = rng.randrange(num_belts)
        dst_i = rng.randrange(len(ns.belt_queues[dst_b]) + 1)
        ns.belt_queues[dst_b].insert(dst_i, order)

    elif move_type == "swap_order":
        locs = order_locations(ns.belt_queues)
        if len(locs) < 2:
            return ns
        (b1, i1), (b2, i2) = rng.sample(locs, 2)
        ns.belt_queues[b1][i1], ns.belt_queues[b2][i2] = (
            ns.belt_queues[b2][i2],
            ns.belt_queues[b1][i1],
        )

    elif move_type == "reorder_within_belt":
        b = rng.randrange(num_belts)
        if len(ns.belt_queues[b]) < 2:
            return ns
        i, j = rng.sample(range(len(ns.belt_queues[b])), 2)
        order = ns.belt_queues[b].pop(i)
        ns.belt_queues[b].insert(j, order)

    elif move_type == "swap_tote":
        if len(ns.tote_sequence) < 2:
            return ns
        i, j = rng.sample(range(len(ns.tote_sequence)), 2)
        ns.tote_sequence[i], ns.tote_sequence[j] = (
            ns.tote_sequence[j],
            ns.tote_sequence[i],
        )

    elif move_type == "insert_tote":
        if len(ns.tote_sequence) < 2:
            return ns
        i, j = rng.sample(range(len(ns.tote_sequence)), 2)
        tote = ns.tote_sequence.pop(i)
        ns.tote_sequence.insert(j, tote)

    return ns


def run_simulated_annealing(
    problem: ProblemInstance, params: SolverParams, verbose: bool = False
):
    rng = random.Random(params.seed)

    current = build_initial_solution(problem)
    cache: Dict[
        Tuple[Tuple[Tuple[int, ...], ...], Tuple[int, ...]],
        Tuple[float, SimulationResult],
    ] = {}

    def eval_solution(sol: Solution) -> Tuple[float, SimulationResult]:
        sig = solution_signature(sol)
        if sig in cache:
            return cache[sig]
        sim = simulate_solution(problem, sol, params)
        obj = scalar_objective(sim, problem.total_items)

        # Strongly prefer using all belts when enough orders exist.
        if problem.num_orders >= problem.num_belts:
            empty_belts = sum(1 for q in sol.belt_queues if not q)
            if empty_belts > 0:
                obj += 10_000.0 * empty_belts

        cache[sig] = (obj, sim)
        return obj, sim

    current_obj, current_res = eval_solution(current)
    best = clone_solution(current)
    best_res = current_res
    best_obj = current_obj

    t = params.t0

    for it in range(1, params.iterations + 1):
        cand = random_neighbor(current, rng, problem.num_belts)
        cand_obj, cand_res = eval_solution(cand)

        delta = cand_obj - current_obj
        if delta <= 0:
            accept = True
        else:
            p = math.exp(-delta / max(t, 1e-12))
            accept = rng.random() < p

        if accept:
            current = cand
            current_obj = cand_obj
            current_res = cand_res

            if is_better(current_res, best_res):
                best = clone_solution(current)
                best_res = current_res
                best_obj = current_obj

        t *= params.alpha

        if verbose and (it % 500 == 0 or it == params.iterations):
            print(
                f"[iter={it:5d}] T={t:.6f} "
                f"current(mk={current_res.makespan:.3f}, avg={current_res.avg_completion:.3f}, feas={current_res.feasible}) "
                f"best(mk={best_res.makespan:.3f}, avg={best_res.avg_completion:.3f}, feas={best_res.feasible})"
            )

    return best, best_res, best_obj


def write_plan_csv(path: str, problem: ProblemInstance, solution: Solution) -> None:
    # Header must match the example input exactly (including "cirle" typo).
    header = [
        "conv_num",
        "cirle",
        "pentagon",
        "trapezoid",
        "triangle",
        "star",
        "moon",
        "heart",
        "cross",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        # One row per order assignment in queue order for each conveyor.
        for conv_num, queue in enumerate(solution.belt_queues):
            for order_idx in queue:
                demand = problem.order_demands[order_idx]
                row = [conv_num + 1] + [
                    demand.get(shape_idx, 0) for shape_idx in range(8)
                ]
                writer.writerow(row)


def write_tote_sequence_csv(
    path: str, problem: ProblemInstance, solution: Solution
) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tote_load_order", "tote_id", "items_in_tote", "shape_mix"])
        for load_order, tote_id in enumerate(solution.tote_sequence, start=1):
            tote_inv = problem.tote_inventory.get(tote_id, {})
            items_in_tote = sum(tote_inv.values())
            shape_mix = "; ".join(
                f"{shape_name_from_idx(shape)}:{qty}"
                for shape, qty in sorted(tote_inv.items())
                if qty > 0
            )
            writer.writerow([load_order, tote_id, items_in_tote, shape_mix])


def write_tote_item_release_csv(
    path: str, release_steps: Sequence[Tuple[int, int, int, int, float]]
) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "global_release_order",
                "tote_id",
                "release_order_in_tote",
                "shape_name",
                "release_time",
            ]
        )
        for global_order, tote_id, in_tote_order, shape, t in release_steps:
            writer.writerow(
                [
                    global_order,
                    tote_id,
                    in_tote_order,
                    shape_name_from_idx(shape),
                    f"{t:.6f}",
                ]
            )


def resolve_path(base_dir: str, maybe_relative: str) -> str:
    if os.path.isabs(maybe_relative):
        return maybe_relative
    return os.path.join(base_dir, maybe_relative)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulated annealing solver for tote unloading and conveyor order fulfillment."
    )
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--order-itemtypes", default="order_itemtypes.csv")
    parser.add_argument("--order-quantities", default="order_quantities.csv")
    parser.add_argument("--orders-totes", default="orders_totes.csv")
    parser.add_argument("--num-belts", type=int, default=4)

    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t0", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.99)

    parser.add_argument("--item-release-interval", type=float, default=1.0)
    parser.add_argument("--tote-change-time", type=float, default=2.0)
    parser.add_argument("--entry-to-first-belt", type=float, default=3.0)
    parser.add_argument("--belt-spacing", type=float, default=2.5)
    parser.add_argument("--return-to-first-belt", type=float, default=2.5)
    parser.add_argument("--max-item-passes", type=int, default=200)

    parser.add_argument("--output-plan", default="plan.csv")
    parser.add_argument("--output-tote-sequence", default="tote_sequence.csv")
    parser.add_argument("--output-tote-items", default="tote_item_release.csv")
    parser.add_argument("--verbose", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    order_itemtypes_path = resolve_path(args.base_dir, args.order_itemtypes)
    order_quantities_path = resolve_path(args.base_dir, args.order_quantities)
    orders_totes_path = resolve_path(args.base_dir, args.orders_totes)

    output_plan_path = resolve_path(args.base_dir, args.output_plan)
    output_tote_sequence_path = resolve_path(args.base_dir, args.output_tote_sequence)
    output_tote_items_path = resolve_path(args.base_dir, args.output_tote_items)

    problem = load_problem(
        order_itemtypes_path=order_itemtypes_path,
        order_quantities_path=order_quantities_path,
        orders_totes_path=orders_totes_path,
        num_belts=args.num_belts,
    )

    params = SolverParams(
        iterations=args.iterations,
        seed=args.seed,
        t0=args.t0,
        alpha=args.alpha,
        item_release_interval=args.item_release_interval,
        tote_change_time=args.tote_change_time,
        entry_to_first_belt=args.entry_to_first_belt,
        belt_spacing=args.belt_spacing,
        return_to_first_belt=args.return_to_first_belt,
        max_item_passes=args.max_item_passes,
    )

    best_solution, best_result, _ = run_simulated_annealing(
        problem, params, verbose=args.verbose
    )

    write_plan_csv(output_plan_path, problem, best_solution)
    write_tote_sequence_csv(output_tote_sequence_path, problem, best_solution)
    write_tote_item_release_csv(output_tote_items_path, best_result.release_steps)

    print("=== SA Warehouse Solver Summary ===")
    print(f"Orders: {problem.num_orders}")
    print(f"Totes with inventory: {len(problem.tote_ids)}")
    print(f"Total required items: {problem.total_items}")
    print(f"Feasible: {best_result.feasible}")
    print(f"Makespan: {best_result.makespan:.6f}")
    print(f"Average completion time: {best_result.avg_completion:.6f}")
    print(f"Dropped items: {best_result.dropped_items}")
    print(f"Plan CSV written to: {output_plan_path}")
    print(f"Tote sequence CSV written to: {output_tote_sequence_path}")
    print(f"Tote item release CSV written to: {output_tote_items_path}")


if __name__ == "__main__":
    main()
