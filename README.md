# MSE 433 - Module 3: Warehouse Conveyor Belt Optimization

Case study for MSE 433 exploring order consolidation on a 4-belt conveyor loop system at the IDEAS Clinic.

## Problem

A warehouse conveyor system has 4 belts forming a loop. Items are loaded onto a ramp, circulate on the conveyor, and pneumatic arms push items off when a scanner detects a match with the active order on that belt. The goal is to minimize the total time (makespan) to fulfill all orders by optimizing:

1. **Belt assignment** - which orders go on which belt and in what sequence
2. **Tote loading order** - the sequence items are physically placed on the ramp

## Team Members and Approaches

| Member | Approach | Key Idea |
|--------|----------|----------|
| **Arkhan** | Event-driven SA + SOF heuristic | Multi-restart SA with event simulation; also developed Shortest-Order-First greedy heuristic |
| **Jeevan** | Multi-restart SA | 3 restarts x 2 objectives (makespan + total CT), 40K iterations, physical timing model |
| **Kate** | Joint tote+order SA | Single-phase SA optimizing tote sequence and order priority simultaneously |
| **Liam** | Event-driven SA | Event-driven simulation with heapq, SA belt optimization |
| **Manjary** | LPT heuristic | Longest Processing Time greedy assignment (deterministic baseline) |

## Repository Structure

```
.
├── compare_solutions.py            # Cross-evaluation of all 5 solvers
├── stochastic_comparison.py        # 100 optimizer-seed trials (same problem)
├── stochastic_data_seeds.py        # 100 data-seed trials (different problems)
├── validation_analysis.py          # Physical run validation analysis
│
├── data/
│   ├── seed100/                    # Shared problem instance (seed=100)
│   │   ├── order_itemtypes.csv
│   │   ├── order_quantities.csv
│   │   └── orders_totes.csv
│   ├── physical_runs/              # Output CSVs from 4 physical conveyor runs
│   │   ├── grp_3_run_1_a_Liam_SA.csv
│   │   ├── grp3_run_1_b_Arkhan_SA.csv
│   │   ├── grp_3_run_2_a_Liam_SA.csv
│   │   └── grp_3_run_2_b_Arkhan_SOF.csv
│   ├── MSE433_M3_data_generator.ipynb
│   ├── MSE433_M3_Example input.csv
│   └── MSE433_M3_Example output.csv
│
├── results/
│   ├── comparison/                 # Single-run comparison outputs
│   │   ├── comparison_results.csv
│   │   └── validation_results.csv
│   ├── stochastic/                 # 100 optimizer-seed trials
│   │   ├── stochastic_results.csv
│   │   ├── stochastic_summary.csv
│   │   └── *.png                   # 7 plots (boxplot, violin, histograms, etc.)
│   └── dataseed/                   # 100 data-seed trials
│       ├── dataseed_results.csv
│       ├── dataseed_summary.csv
│       └── *.png                   # 8 plots (boxplot, pairwise, scatter, etc.)
│
├── arkhan/                         # Arkhan's SA solver + SOF heuristic
├── jeevan/                         # Jeevan's multi-restart SA + physical simulator
├── kate/                           # Kate's joint tote+order SA
├── liam/                           # Liam's event-driven SA solver
└── manjary/                        # Manjary's LPT heuristic
```

## Results

### Single-Instance Comparison (seed=100: 11 orders, 36 items)

| Rank | Solution | Composite | Makespan | Improvement |
|------|----------|-----------|----------|-------------|
| 1 | Jeevan (SA) | 69.7 | 186.5s | +22.5% |
| 2 | Kate (SA) | 70.8 | 186.5s | +22.5% |
| 3 | Arkhan (SA) | 73.3 | 182.0s | +24.3% |
| 4 | Liam (SA) | 73.3 | 182.0s | +24.3% |
| 5 | Arkhan (SOF) | 76.2 | 200.0s | +16.8% |
| 6 | Manjary (LPT) | 108.4 | 258.5s | -7.5% |
| - | Naive baseline | 100.0 | 240.5s | 0% |

Composite = 35% makespan + 35% avg completion + 15% total CT + 15% spread, normalized to naive = 100.

### Stochastic Validation (100 trials)

**Optimizer-seed variation** (same problem, different SA seeds):

| Solver | Makespan (mean +/- std) | Composite (mean +/- std) |
|--------|------------------------|--------------------------|
| Kate (SA) | 185.2 +/- 4.1s | 72.5 +/- 2.2 |
| Arkhan (SA) | 186.6 +/- 8.7s | 75.4 +/- 4.7 |
| Jeevan (SA) | 204.2 +/- 1.2s | 78.5 +/- 0.4 |
| Arkhan (SOF) | 200.0 +/- 0.0s | 76.2 +/- 0.0 |

**Data-seed variation** (100 different problem instances):

| Solver | Makespan (mean +/- std) | Composite (mean +/- std) | Makespan Improvement |
|--------|------------------------|--------------------------|---------------------|
| Kate (SA) | 235.9 +/- 39.3s | 79.6 +/- 7.0 | +17.5% |
| Arkhan (SOF) | 250.5 +/- 41.3s | 80.4 +/- 6.6 | +12.4% |
| Arkhan (SA) | 245.7 +/- 47.4s | 85.0 +/- 10.7 | +14.3% |
| Jeevan (SA) | 254.8 +/- 42.3s | 83.0 +/- 7.2 | +10.9% |
| Manjary (LPT) | 325.7 +/- 51.8s | 120.0 +/- 11.6 | -14.0% |

All SA solvers and SOF beat the naive baseline in 100/100 instances.

### Physical Validation (4 runs, 2 days)

| Run | Items | Makespan | Throughput |
|-----|-------|----------|------------|
| Day 1 Run A (SA) | 30 | 162.5s | 0.185/s |
| Day 1 Run B (SA) | 29 | 206.1s | 0.141/s |
| Day 2 Run A (SA) | 28 | 167.2s | 0.168/s |
| Day 2 Run B (SOF) | 31 | 116.8s | 0.265/s |

SOF outperformed SA by 30% on physical hardware (116.8s vs 167.2s), delivering more items (31 vs 28) with higher throughput.

## Running the Scripts

```bash
# Full cross-evaluation (includes stochastic validation summary)
python compare_solutions.py

# 100 optimizer-seed stochastic trials
python stochastic_comparison.py

# 100 data-seed robustness trials
python stochastic_data_seeds.py

# Physical run validation analysis
python validation_analysis.py
```

## Dependencies

- Python 3.8+
- matplotlib
- numpy
