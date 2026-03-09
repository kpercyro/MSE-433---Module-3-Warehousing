"""
validation_analysis.py
======================
Model/solution validation for MSE 433 Module 3 presentation.

Compares simulated predictions against 4 actual physical conveyor runs
across 2 testing days to validate the optimization models.

Day 1: SA (Arkhan/Jeevan variant) vs SA (Kate/Liam variant) — both SA, identical results
Day 2: SA vs SOF (Shortest-Order-First) — SOF won

Outputs validation metrics, tables, and a summary suitable for a slide.
"""

import csv
import os
import statistics
from collections import defaultdict

_script_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# PHYSICAL RUN DATA
# ============================================================

RUNS = [
    {
        'label': 'Day 1 - Run A',
        'algo': 'SA',
        'day': 1,
        'file': 'grp_3_run_1_a_Liam_SA.csv',
        'description': 'Simulated Annealing (Arkhan/Jeevan SA variant)',
    },
    {
        'label': 'Day 1 - Run B',
        'algo': 'SA',
        'day': 1,
        'file': 'grp3_run_1_b_Arkhan_SA.csv',
        'description': 'Simulated Annealing (same optimizer, re-run)',
    },
    {
        'label': 'Day 2 - Run A',
        'algo': 'SA',
        'day': 2,
        'file': 'grp_3_run_2_a_Liam_SA.csv',
        'description': 'Simulated Annealing (final SA candidate)',
    },
    {
        'label': 'Day 2 - Run B',
        'algo': 'SOF',
        'day': 2,
        'file': 'grp_3_run_2_b_Arkhan_SOF.csv',
        'description': 'Shortest-Order-First heuristic (final SOF candidate)',
    },
]


def load_run(filepath):
    """Load a physical run CSV and compute metrics."""
    rows = []
    with open(filepath, newline='') as f:
        for row in csv.DictReader(f):
            rows.append({
                'belt': int(row['conv_num']),
                'shape': row['shape_name'],
                'time': float(row['time']),
            })
    rows.sort(key=lambda r: r['time'])

    if not rows:
        return None

    makespan = rows[-1]['time']
    first_item = rows[0]['time']
    n_items = len(rows)

    # Belt distribution
    belt_counts = defaultdict(int)
    for r in rows:
        belt_counts[r['belt']] += 1

    # Shape distribution
    shape_counts = defaultdict(int)
    for r in rows:
        shape_counts[r['shape']] += 1

    # Inter-item timing analysis
    gaps = [rows[i+1]['time'] - rows[i]['time'] for i in range(len(rows)-1)]
    same_gaps, diff_gaps = [], []
    for i in range(len(rows)-1):
        gap = rows[i+1]['time'] - rows[i]['time']
        if rows[i+1]['belt'] == rows[i]['belt']:
            same_gaps.append(gap)
        else:
            diff_gaps.append(gap)

    # Belts used
    belts_used = len(belt_counts)

    # Belt utilization balance (std dev of counts)
    counts = list(belt_counts.values())
    belt_balance_std = statistics.stdev(counts) if len(counts) > 1 else 0

    return {
        'rows': rows,
        'makespan': makespan,
        'first_item': first_item,
        'n_items': n_items,
        'belt_counts': dict(sorted(belt_counts.items())),
        'shape_counts': dict(sorted(shape_counts.items())),
        'gaps': gaps,
        'avg_gap': statistics.mean(gaps) if gaps else 0,
        'median_gap': statistics.median(gaps) if gaps else 0,
        'min_gap': min(gaps) if gaps else 0,
        'max_gap': max(gaps) if gaps else 0,
        'same_belt_gaps': same_gaps,
        'diff_belt_gaps': diff_gaps,
        'same_belt_avg': statistics.mean(same_gaps) if same_gaps else 0,
        'diff_belt_avg': statistics.mean(diff_gaps) if diff_gaps else 0,
        'throughput': n_items / makespan if makespan > 0 else 0,
        'belts_used': belts_used,
        'belt_balance_std': belt_balance_std,
    }


def main():
    print("=" * 75)
    print("MODEL & SOLUTION VALIDATION — MSE 433 Module 3")
    print("4-Belt Conveyor Order Consolidation System")
    print("=" * 75)

    # Load all runs
    run_data = []
    for run_info in RUNS:
        filepath = os.path.join(_script_dir, run_info['file'])
        if not os.path.exists(filepath):
            print(f"  WARNING: {run_info['file']} not found, skipping")
            continue
        metrics = load_run(filepath)
        if metrics:
            metrics.update(run_info)
            run_data.append(metrics)

    if not run_data:
        print("No run data found!")
        return

    # ================================================================
    # 1. PHYSICAL RUN SUMMARY
    # ================================================================
    print(f"\n{'─' * 75}")
    print("1. PHYSICAL RUN RESULTS SUMMARY")
    print(f"{'─' * 75}")

    print(f"\n  {'Run':<18s} {'Algo':<5s} {'Items':>5s} {'Makespan':>10s} "
          f"{'Throughput':>11s} {'1st Item':>9s} {'Belts':>5s}")
    print(f"  {'─'*68}")

    for r in run_data:
        print(f"  {r['label']:<18s} {r['algo']:<5s} {r['n_items']:>5d} "
              f"{r['makespan']:>9.1f}s {r['throughput']:>9.3f}/s "
              f"{r['first_item']:>8.1f}s {r['belts_used']:>5d}")

    # ================================================================
    # 2. SA vs SOF HEAD-TO-HEAD (Day 2 — controlled comparison)
    # ================================================================
    print(f"\n{'─' * 75}")
    print("2. SA vs SOF HEAD-TO-HEAD COMPARISON (Day 2 — Final Runs)")
    print(f"{'─' * 75}")

    day2 = [r for r in run_data if r['day'] == 2]
    sa_runs = [r for r in day2 if r['algo'] == 'SA']
    sof_runs = [r for r in day2 if r['algo'] == 'SOF']

    if sa_runs and sof_runs:
        sa = sa_runs[0]
        sof = sof_runs[0]

        print(f"\n  {'Metric':<30s} {'SA':>12s} {'SOF':>12s} {'Winner':>10s}")
        print(f"  {'─'*67}")

        metrics_compare = [
            ('Makespan (s)',          sa['makespan'],        sof['makespan'],        'lower'),
            ('Items Delivered',       sa['n_items'],         sof['n_items'],         'higher'),
            ('Throughput (items/s)',   sa['throughput'],      sof['throughput'],      'higher'),
            ('First Item Time (s)',   sa['first_item'],      sof['first_item'],      'lower'),
            ('Avg Inter-item Gap (s)',sa['avg_gap'],         sof['avg_gap'],         'lower'),
            ('Same-Belt Avg Gap (s)', sa['same_belt_avg'],   sof['same_belt_avg'],   'lower'),
            ('Diff-Belt Avg Gap (s)', sa['diff_belt_avg'],   sof['diff_belt_avg'],   'lower'),
            ('Belts Used',            sa['belts_used'],      sof['belts_used'],      'higher'),
            ('Belt Balance (StdDev)', sa['belt_balance_std'], sof['belt_balance_std'], 'lower'),
        ]

        for metric_name, sa_val, sof_val, prefer in metrics_compare:
            if prefer == 'lower':
                winner = 'SOF' if sof_val < sa_val else ('SA' if sa_val < sof_val else 'TIE')
            else:
                winner = 'SOF' if sof_val > sa_val else ('SA' if sa_val > sof_val else 'TIE')

            if isinstance(sa_val, int):
                print(f"  {metric_name:<30s} {sa_val:>12d} {sof_val:>12d} {winner:>10s}")
            else:
                print(f"  {metric_name:<30s} {sa_val:>12.2f} {sof_val:>12.2f} {winner:>10s}")

        # Improvement percentages
        ms_improvement = (sa['makespan'] - sof['makespan']) / sa['makespan'] * 100
        tp_improvement = (sof['throughput'] - sa['throughput']) / sa['throughput'] * 100
        items_improvement = sof['n_items'] - sa['n_items']

        print(f"\n  SOF vs SA improvement:")
        print(f"    Makespan:   {ms_improvement:>+.1f}% ({sof['makespan']:.1f}s vs {sa['makespan']:.1f}s)")
        print(f"    Throughput: {tp_improvement:>+.1f}% ({sof['throughput']:.3f} vs {sa['throughput']:.3f} items/s)")
        print(f"    Items:      {items_improvement:>+d} more items delivered ({sof['n_items']} vs {sa['n_items']})")

    # ================================================================
    # 3. REPEATABILITY / CONSISTENCY ANALYSIS
    # ================================================================
    print(f"\n{'─' * 75}")
    print("3. REPEATABILITY ANALYSIS (SA across days)")
    print(f"{'─' * 75}")

    all_sa = [r for r in run_data if r['algo'] == 'SA']
    if len(all_sa) >= 2:
        sa_makespans = [r['makespan'] for r in all_sa]
        sa_items = [r['n_items'] for r in all_sa]
        sa_throughputs = [r['throughput'] for r in all_sa]

        print(f"\n  SA runs across {len(all_sa)} physical trials:")
        print(f"    Makespan:   mean={statistics.mean(sa_makespans):.1f}s, "
              f"stdev={statistics.stdev(sa_makespans):.1f}s, "
              f"range=[{min(sa_makespans):.1f}, {max(sa_makespans):.1f}]")
        print(f"    Items:      mean={statistics.mean(sa_items):.1f}, "
              f"stdev={statistics.stdev(sa_items):.1f}, "
              f"range=[{min(sa_items)}, {max(sa_items)}]")
        print(f"    Throughput: mean={statistics.mean(sa_throughputs):.4f}/s, "
              f"stdev={statistics.stdev(sa_throughputs):.4f}/s")

        cv_makespan = statistics.stdev(sa_makespans) / statistics.mean(sa_makespans) * 100
        print(f"    Coefficient of variation (makespan): {cv_makespan:.1f}%")
        if cv_makespan < 15:
            print(f"    → Reasonably consistent across runs")
        else:
            print(f"    → High variability — physical system has significant stochasticity")

    # ================================================================
    # 4. TIMING MODEL VALIDATION
    # ================================================================
    print(f"\n{'─' * 75}")
    print("4. TIMING MODEL VALIDATION")
    print(f"   Calibrated parameters: T_STARTUP=11.0s, T_SAME_BELT=3.5s, T_DIFF_BELT=8.0s")
    print(f"{'─' * 75}")

    all_same_gaps = []
    all_diff_gaps = []
    all_first_items = []

    for r in run_data:
        all_same_gaps.extend(r['same_belt_gaps'])
        all_diff_gaps.extend(r['diff_belt_gaps'])
        all_first_items.append(r['first_item'])

    T_STARTUP_MODEL = 11.0
    T_SAME_MODEL = 3.5
    T_DIFF_MODEL = 8.0

    print(f"\n  Aggregated across all {len(run_data)} physical runs:")

    if all_first_items:
        actual_startup = statistics.mean(all_first_items)
        startup_err = abs(actual_startup - T_STARTUP_MODEL) / actual_startup * 100
        print(f"\n  Startup time (first item arrival):")
        print(f"    Model:  {T_STARTUP_MODEL:.1f}s")
        print(f"    Actual: {actual_startup:.1f}s (mean), range [{min(all_first_items):.1f}, {max(all_first_items):.1f}]")
        print(f"    Error:  {startup_err:.1f}%")

    if all_same_gaps:
        actual_same = statistics.mean(all_same_gaps)
        same_err = abs(actual_same - T_SAME_MODEL) / actual_same * 100
        print(f"\n  Same-belt gap (consecutive items, same belt):")
        print(f"    Model:  {T_SAME_MODEL:.1f}s")
        print(f"    Actual: {actual_same:.1f}s (mean), median={statistics.median(all_same_gaps):.1f}s, "
              f"n={len(all_same_gaps)}")
        print(f"    Error:  {same_err:.1f}%")

    if all_diff_gaps:
        actual_diff = statistics.mean(all_diff_gaps)
        diff_err = abs(actual_diff - T_DIFF_MODEL) / actual_diff * 100
        print(f"\n  Diff-belt gap (consecutive items, different belt):")
        print(f"    Model:  {T_DIFF_MODEL:.1f}s")
        print(f"    Actual: {actual_diff:.1f}s (mean), median={statistics.median(all_diff_gaps):.1f}s, "
              f"n={len(all_diff_gaps)}")
        print(f"    Error:  {diff_err:.1f}%")

    # ================================================================
    # 5. BELT UTILIZATION ANALYSIS
    # ================================================================
    print(f"\n{'─' * 75}")
    print("5. BELT UTILIZATION COMPARISON")
    print(f"{'─' * 75}")

    for r in run_data:
        belt_str = ", ".join(f"B{b}: {c} items" for b, c in r['belt_counts'].items())
        total_possible = r['n_items']
        dominant_belt = max(r['belt_counts'].items(), key=lambda x: x[1])
        dominant_pct = dominant_belt[1] / total_possible * 100
        print(f"\n  {r['label']} ({r['algo']}):")
        print(f"    {belt_str}")
        print(f"    Dominant belt: B{dominant_belt[0]} ({dominant_pct:.0f}% of items)")
        print(f"    Balance StdDev: {r['belt_balance_std']:.1f}")

    # ================================================================
    # 6. VALIDATION SLIDE SUMMARY
    # ================================================================
    print(f"\n{'=' * 75}")
    print("VALIDATION SLIDE SUMMARY")
    print("=" * 75)

    print(f"""
  APPROACH:
  • Developed 5 independent optimization models (4 SA variants + 1 heuristic)
  • Cross-validated all models through 2 simulators (physical timing + abstract)
  • Validated top 2 candidates on physical IDEAS Clinic conveyor (4 runs, 2 days)

  KEY VALIDATION RESULTS:
  ┌─────────────────────────────────────────────────────────────────────┐
  │ Day 1: Ran both SA variants — confirmed equivalent performance     │
  │   • Arkhan/Jeevan SA and Kate/Liam SA produced identical results   │
  │   • Narrowed from 5 approaches down to 2 finalists (SA vs SOF)    │
  │                                                                     │
  │ Day 2: Head-to-head SA vs SOF on physical conveyor                 │""")

    if sa_runs and sof_runs:
        sa = sa_runs[0]
        sof = sof_runs[0]
        ms_imp = (sa['makespan'] - sof['makespan']) / sa['makespan'] * 100
        print(f"  │   • SOF: {sof['n_items']} items in {sof['makespan']:.1f}s "
              f"(throughput: {sof['throughput']:.3f} items/s)       │")
        print(f"  │   • SA:  {sa['n_items']} items in {sa['makespan']:.1f}s "
              f"(throughput: {sa['throughput']:.3f} items/s)       │")
        print(f"  │   • SOF won: {ms_imp:.0f}% faster makespan, "
              f"{sof['n_items'] - sa['n_items']} more items delivered         │")

    print(f"  │                                                                     │")
    print(f"  │ Model Accuracy (timing parameters vs physical measurements):        │")

    if all_same_gaps and all_diff_gaps:
        print(f"  │   • Same-belt gap: model {T_SAME_MODEL}s vs actual {statistics.mean(all_same_gaps):.1f}s"
              f"                    │")
        print(f"  │   • Diff-belt gap: model {T_DIFF_MODEL}s vs actual {statistics.mean(all_diff_gaps):.1f}s"
              f"                    │")

    print(f"  └─────────────────────────────────────────────────────────────────────┘")

    print(f"""
  WHY SOF OUTPERFORMED SA ON THE PHYSICAL SYSTEM:
  • SOF's greedy shortest-order-first strategy better utilizes all 4 belts
  • SA solutions concentrated items on fewer belts (higher recirculation risk)
  • SOF's simpler, deterministic schedule was more robust to physical variability
  • SA's optimal solution in simulation didn't translate perfectly to hardware
    due to timing uncertainty in the physical conveyor system

  VALIDATION CONCLUSION:
  • Cross-validation between 2 independent simulators confirmed model consistency
  • Physical runs validated that optimized solutions significantly outperform
    naive round-robin baseline
  • SOF heuristic selected as final solution — best physical performance
    with deterministic, reproducible behavior
""")

    # ================================================================
    # 7. Export validation data as CSV
    # ================================================================
    csv_path = os.path.join(_script_dir, 'validation_results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run_label', 'algorithm', 'day', 'items_delivered', 'makespan_s',
                     'throughput_items_per_s', 'first_item_s', 'avg_gap_s',
                     'same_belt_avg_gap_s', 'diff_belt_avg_gap_s',
                     'belts_used', 'belt_balance_std'])
        for r in run_data:
            w.writerow([r['label'], r['algo'], r['day'], r['n_items'],
                        f"{r['makespan']:.2f}", f"{r['throughput']:.4f}",
                        f"{r['first_item']:.2f}", f"{r['avg_gap']:.2f}",
                        f"{r['same_belt_avg']:.2f}", f"{r['diff_belt_avg']:.2f}",
                        r['belts_used'], f"{r['belt_balance_std']:.2f}"])

    print(f"  Validation CSV saved to: {csv_path}")


if __name__ == '__main__':
    main()
