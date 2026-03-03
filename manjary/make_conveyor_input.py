# make_conveyor_input.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from optimizer import load_inputs, build_demands, write_conveyor_input_csv, csv_equal
from policies import queues_round_robin, queues_random, queues_lpt_balance


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate IDEAS conveyor input CSV from MSE433 M3 generator files.")
    p.add_argument("--data-dir", type=str, default=".", help="Folder containing order_itemtypes.csv, order_quantities.csv, orders_totes.csv")
    p.add_argument("--out", type=str, default="conveyor_input.csv", help="Output CSV path")
    p.add_argument("--policy", type=str, default="lpt",
                   choices=["lpt", "round_robin", "random"],
                   help="Lane assignment policy")
    p.add_argument("--conv-base", type=int, default=1, choices=[0, 1],
                   help="0 => conv_num is 0..3; 1 => conv_num is 1..4")
    p.add_argument("--seed", type=int, default=999, help="Seed for random policy (ignored otherwise)")
    p.add_argument("--different-from", type=str, default=None,
                   help="Path to an existing CSV. If generated CSV matches it, script will adjust (for random policy) until different.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    ref_path: Optional[Path] = Path(args.different_from) if args.different_from else None

    order_itemtypes, order_quantities, _orders_totes = load_inputs(data_dir)
    demands = build_demands(order_itemtypes, order_quantities, n_types=8)
    n_orders = len(demands)

    if args.policy == "round_robin":
        belt_queues = queues_round_robin(n_orders, num_belts=4)
    elif args.policy == "lpt":
        belt_queues, loads = queues_lpt_balance(demands, num_belts=4)
        print(f"[info] LPT belt loads (items): {loads}")
        print(f"[info] LPT belt queues: {belt_queues}")
    elif args.policy == "random":
        belt_queues = queues_random(n_orders, seed=args.seed, num_belts=4)
    else:
        raise ValueError("Unknown policy")

    write_conveyor_input_csv(belt_queues, demands, out_path, conv_num_base=args.conv_base)
    print(f"[ok] wrote {out_path}")

    # Ensure "different" if requested
    if ref_path and ref_path.exists():
        if csv_equal(out_path, ref_path):
            print("[warn] Generated CSV matches reference.")
            if args.policy != "random":
                print("[warn] For guaranteed difference, rerun with --policy random (and a different --seed) or change policy.")
            else:
                # keep bumping seed until different
                seed = args.seed
                while True:
                    seed += 1
                    belt_queues = queues_random(n_orders, seed=seed, num_belts=4)
                    write_conveyor_input_csv(belt_queues, demands, out_path, conv_num_base=args.conv_base)
                    if not csv_equal(out_path, ref_path):
                        print(f"[ok] regenerated with seed={seed} to ensure difference.")
                        break


if __name__ == "__main__":
    main()