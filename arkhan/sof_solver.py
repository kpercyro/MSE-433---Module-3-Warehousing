#!/usr/bin/env python3
"""
Shortest-Order-First (SOF) heuristic for the warehouse conveyor problem.

Assigns orders to belt positions by fewest items first, sequences totes
greedily, simulates the single entry conveyor, and writes a markdown
instructions file.

Usage:
    python sof_solver.py --base-dir .
"""
from __future__ import annotations

import argparse
import csv
import heapq
import os
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from statistics import mean
from typing import Dict, List, Optional, Tuple

# ── Shape names (indices match the CSV column order) ──────────────────────
SHAPES = ["Circle", "Pentagon", "Trapezoid", "Triangle", "Star", "Moon", "Heart", "Cross"]


def shape_name(idx: int) -> str:
    return SHAPES[idx] if 0 <= idx < len(SHAPES) else f"Shape{idx}"


# ── Data structures ───────────────────────────────────────────────────────
@dataclass
class Problem:
    num_belts: int
    num_orders: int
    order_demands: List[Dict[int, int]]   # order → {item_type: qty}
    order_total: List[int]
    tote_inv: Dict[int, Dict[int, int]]   # tote → {item_type: qty}
    tote_ids: List[int]
    tote_to_orders: Dict[int, set]
    total_items: int


@dataclass
class ConveyorParams:
    item_interval: float = 1.0
    tote_change: float = 2.0
    entry_to_first: float = 3.0
    belt_spacing: float = 2.5
    return_to_first: float = 2.5
    max_passes: int = 200


# ── CSV loader ────────────────────────────────────────────────────────────
def _read_rows(path: str) -> List[List[int]]:
    rows: List[List[int]] = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            vals = []
            for c in row:
                c = c.strip()
                if c == "":
                    continue
                try:
                    vals.append(int(float(c)))
                except ValueError:
                    pass
            rows.append(vals)
    return rows


def load_problem(types_csv: str, qty_csv: str, totes_csv: str, num_belts: int = 4) -> Problem:
    trows = _read_rows(types_csv)
    qrows = _read_rows(qty_csv)
    orows = _read_rows(totes_csv)
    n = max(len(trows), len(qrows), len(orows))

    demands: List[Dict[int, int]] = []
    totals: List[int] = []
    tote_inv: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    tote_orders: Dict[int, set] = defaultdict(set)

    for o in range(n):
        ts = trows[o] if o < len(trows) else []
        qs = qrows[o] if o < len(qrows) else []
        tt = orows[o] if o < len(orows) else []
        k = min(len(ts), len(qs), len(tt))
        d: Dict[int, int] = defaultdict(int)
        for j in range(k):
            if qs[j] <= 0:
                continue
            d[ts[j]] += qs[j]
            tote_inv[tt[j]][ts[j]] += qs[j]
            tote_orders[tt[j]].add(o)
        demands.append(dict(d))
        totals.append(sum(d.values()))

    inv_final = {t: dict(v) for t, v in tote_inv.items()}
    return Problem(
        num_belts=num_belts,
        num_orders=n,
        order_demands=demands,
        order_total=totals,
        tote_inv=inv_final,
        tote_ids=sorted(inv_final),
        tote_to_orders=dict(tote_orders),
        total_items=sum(totals),
    )


# ── Shortest-Order-First heuristic ───────────────────────────────────────
def build_sof_solution(prob: Problem) -> Tuple[List[List[int]], List[int]]:
    """Assign orders shortest-first, round-robin into lightest belt."""
    ids = sorted(range(prob.num_orders), key=lambda o: prob.order_total[o])
    queues: List[List[int]] = [[] for _ in range(prob.num_belts)]
    load = [0] * prob.num_belts

    for o in ids:
        b = min(range(prob.num_belts), key=lambda x: load[x])
        queues[b].append(o)
        load[b] += prob.order_total[o]

    # Rank orders by (queue-depth, belt) for tote sequencing.
    rank: Dict[int, int] = {}
    for b, q in enumerate(queues):
        for pos, o in enumerate(q):
            rank[o] = pos * prob.num_belts + b

    scored = []
    for t in prob.tote_ids:
        best = min((rank.get(o, 10**9) for o in prob.tote_to_orders.get(t, set())), default=10**9)
        scored.append((best, t))
    scored.sort()
    tote_seq = [t for _, t in scored]

    return queues, tote_seq


# ── Choose which item to release next from a tote ────────────────────────
def _pick_item(
    counts: Dict[int, int],
    active: List[Optional[int]],
    apos: List[int],
    queues: List[List[int]],
    rem: List[Dict[int, int]],
) -> Optional[int]:
    cands = [i for i, q in counts.items() if q > 0]
    if not cands:
        return None
    best, best_r = None, None
    nb = len(queues)
    for item in cands:
        r = None
        for b in range(nb):
            o = active[b]
            if o is not None and rem[o].get(item, 0) > 0:
                r = (0, b, -rem[o][item], item)
                break
        if r is None:
            fb = None
            for b in range(nb):
                sp = 0 if apos[b] < 0 else apos[b] + 1
                for p in range(sp, len(queues[b])):
                    if rem[queues[b][p]].get(item, 0) > 0:
                        d = p - apos[b] if apos[b] >= 0 else p + 1
                        c = (d, b)
                        if fb is None or c < fb:
                            fb = c
                        break
            r = (1, fb[0], fb[1], item) if fb else (2, 0, 0, item)
        if best_r is None or r < best_r:
            best_r, best = r, item
    return best


# ── Simulate ──────────────────────────────────────────────────────────────
def simulate(
    prob: Problem,
    queues: List[List[int]],
    tote_seq: List[int],
    p: ConveyorParams,
) -> Tuple[bool, float, float, int, List[Tuple[int, int, int, int, float]]]:
    """Returns (feasible, makespan, avg_completion, dropped, release_steps)."""
    rem = [dict(d) for d in prob.order_demands]
    rem_tot = [sum(d.values()) for d in rem]
    ct: List[Optional[float]] = [None] * prob.num_orders

    active: List[Optional[int]] = []
    apos: List[int] = []
    for b in range(prob.num_belts):
        if queues[b]:
            active.append(queues[b][0]); apos.append(0)
        else:
            active.append(None); apos.append(-1)

    def advance(belt: int, t: float):
        while True:
            if apos[belt] < 0:
                return
            nxt = apos[belt] + 1
            if nxt >= len(queues[belt]):
                apos[belt] = -1; active[belt] = None; return
            apos[belt] = nxt
            o = queues[belt][nxt]; active[belt] = o
            if rem_tot[o] == 0:
                if ct[o] is None: ct[o] = t
                continue
            return

    for b in range(prob.num_belts):
        while active[b] is not None and rem_tot[active[b]] == 0:
            if ct[active[b]] is None: ct[active[b]] = 0.0
            advance(b, 0.0)

    tote_rem = {t: dict(inv) for t, inv in prob.tote_inv.items()}
    events: list = []
    ser = 0

    def push(time, pri, etype, data):
        nonlocal ser
        heapq.heappush(events, (time, pri, ser, etype, data)); ser += 1

    tc = 0.0
    for i, t in enumerate(tote_seq):
        ni = sum(tote_rem.get(t, {}).values())
        if ni <= 0:
            continue
        if i > 0:
            tc += p.tote_change
        for k in range(ni):
            push(tc + k * p.item_interval, 1, "R", {"tote": t})
        tc += ni * p.item_interval

    steps: List[Tuple[int, int, int, int, float]] = []
    tote_cnt: Dict[int, int] = defaultdict(int)
    glo = 0
    dropped = 0

    while events:
        t_now, _, _, etype, data = heapq.heappop(events)
        if etype == "R":
            tote = data["tote"]
            c = tote_rem.get(tote, {})
            item = _pick_item(c, active, apos, queues, rem)
            if item is None:
                continue
            c[item] -= 1
            if c[item] <= 0:
                del c[item]
            tote_cnt[tote] += 1; glo += 1
            steps.append((glo, tote, tote_cnt[tote], item, t_now))
            push(t_now + p.entry_to_first, 0, "B", {"belt": 0, "item": item, "passes": 0})

        elif etype == "B":
            belt = data["belt"]; item = data["item"]
            o = active[belt]
            if o is not None and rem[o].get(item, 0) > 0:
                rem[o][item] -= 1; rem_tot[o] -= 1
                if rem[o][item] == 0:
                    del rem[o][item]
                if rem_tot[o] == 0 and ct[o] is None:
                    ct[o] = t_now
                    advance(belt, t_now)
                continue
            data["passes"] += 1
            if data["passes"] > p.max_passes:
                dropped += 1; continue
            nb = belt + 1 if belt < prob.num_belts - 1 else 0
            dt = p.belt_spacing if belt < prob.num_belts - 1 else p.return_to_first
            push(t_now + dt, 0, "B", {"belt": nb, "item": item, "passes": data["passes"]})

    feasible = all(c is not None for c in ct)
    mk = max((c for c in ct if c is not None), default=0.0)
    avg = mean([c for c in ct if c is not None]) if feasible else 0.0
    return feasible, mk, avg, dropped, steps


# ── Markdown writer ───────────────────────────────────────────────────────
def write_markdown(
    path: str,
    prob: Problem,
    queues: List[List[int]],
    tote_seq: List[int],
    feasible: bool,
    makespan: float,
    avg_ct: float,
    dropped: int,
    steps: List[Tuple[int, int, int, int, float]],
) -> None:
    L: List[str] = []
    a = L.append

    a("# Shortest-Order-First — Loading Instructions\n")

    # ── Results ──
    a("## Results\n")
    a("| Metric | Value |")
    a("|---|---|")
    a(f"| Orders | {prob.num_orders} |")
    a(f"| Totes | {len(prob.tote_ids)} |")
    a(f"| Total Items | {prob.total_items} |")
    a(f"| Feasible | {'Yes' if feasible else '**No**'} |")
    a(f"| Makespan | {makespan:.1f} s |")
    a(f"| Avg Completion | {avg_ct:.1f} s |")
    a(f"| Dropped Items | {dropped} |")
    a("")

    # ── Tote contents ──
    a("---\n")
    a("## What Is in Each Tote\n")
    a("| Tote | Contents | Qty |")
    a("|---|---|---|")
    for t in sorted(prob.tote_inv):
        inv = prob.tote_inv[t]
        parts = [f"{shape_name(s)} ×{q}" for s, q in sorted(inv.items()) if q > 0]
        a(f"| {t} | {', '.join(parts)} | {sum(inv.values())} |")
    a("")

    # ── Belt assignments ──
    a("---\n")
    a("## Belt Sort-Position Assignments\n")
    a("> All items enter via **one conveyor**; each belt position sorts items for its queued orders.\n")
    for b, q in enumerate(queues):
        a(f"### Belt {b+1} ({len(q)} order{'s' if len(q) != 1 else ''})\n")
        if not q:
            a("_Empty._\n"); continue
        a("| # | Order Items | Qty |")
        a("|---|---|---|")
        for pos, o in enumerate(q, 1):
            d = prob.order_demands[o]
            parts = [f"{shape_name(s)} ×{q}" for s, q in sorted(d.items()) if q > 0]
            a(f"| {pos} | {', '.join(parts) or '—'} | {sum(d.values())} |")
        a("")

    # ── Tote loading order ──
    a("---\n")
    a("## Tote Loading Order\n")
    a("| # | Tote | Items |")
    a("|---|---|---|")
    for i, t in enumerate(tote_seq, 1):
        a(f"| {i} | {t} | {sum(prob.tote_inv.get(t, {}).values())} |")
    a("")

    # ── Per-tote item release order ──
    a("---\n")
    a("## Item Release Order Per Tote\n")
    a("> For each tote, the exact order to place its items onto the entry conveyor.\n")

    grouped: Dict[int, List[Tuple[int, int, int, float]]] = OrderedDict()
    for t in tote_seq:
        grouped[t] = []
    for glo, t, intote, shape, tm in steps:
        grouped.setdefault(t, []).append((intote, shape, glo, tm))

    for i, t in enumerate(tote_seq, 1):
        items = grouped.get(t, [])
        tot = sum(prob.tote_inv.get(t, {}).values())
        a(f"### Tote {t}  (Load #{i}, {tot} item{'s' if tot != 1 else ''})\n")
        if not items:
            a("_No items._\n"); continue
        a("| # | Item | Time |")
        a("|---|---|---|")
        for intote, shape, _, tm in items:
            a(f"| {intote} | {shape_name(shape)} | {tm:.1f} s |")
        a("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


# ── CLI ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Shortest-Order-First conveyor solver")
    ap.add_argument("--base-dir", default=".")
    ap.add_argument("--order-itemtypes", default="order_itemtypes.csv")
    ap.add_argument("--order-quantities", default="order_quantities.csv")
    ap.add_argument("--orders-totes", default="orders_totes.csv")
    ap.add_argument("--num-belts", type=int, default=4)
    ap.add_argument("--output", default="sof_instructions.md")
    args = ap.parse_args()

    def p(rel):
        return os.path.join(args.base_dir, rel)

    prob = load_problem(p(args.order_itemtypes), p(args.order_quantities),
                        p(args.orders_totes), args.num_belts)
    queues, tote_seq = build_sof_solution(prob)
    params = ConveyorParams()
    feasible, mk, avg, dropped, steps = simulate(prob, queues, tote_seq, params)

    out = p(args.output)
    write_markdown(out, prob, queues, tote_seq, feasible, mk, avg, dropped, steps)

    print(f"SOF Solver  |  Orders: {prob.num_orders}  Totes: {len(prob.tote_ids)}  "
          f"Items: {prob.total_items}")
    print(f"Feasible: {feasible}  Makespan: {mk:.1f}s  Avg: {avg:.1f}s  "
          f"Dropped: {dropped}")
    print(f"Instructions written to: {out}")


if __name__ == "__main__":
    main()
