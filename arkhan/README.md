# Jeevan's Conveyor Belt Optimizer

Simulated annealing approach to the 4-belt conveyor order consolidation problem, based on Zhou (2017).

## Overview

The optimizer runs in two phases:

1. **Belt Assignment (SA)** - Assigns orders to belts and determines processing sequence using simulated annealing with LPT+SPT initialization. Neighborhoods: insertion (50%), swap (30%), reversal (20%). 20K iterations with exponential cooling.
2. **Loading Sequence** - Tests 4 strategies (naive sequential, round-robin interleave, belt-grouped SPT, SPT interleave) and selects the best by makespan.

Also generates a small test instance (4 orders, ~8 items) for running on the physical conveyor.

## Files

| File | Description |
|------|-------------|
| `conveyor_optimizer.py` | Main optimizer - data generation, SA, simulation, CSV/instruction output |
| `conveyor_dashboard.jsx` | React visualization dashboard (Recharts) |
| `conveyor_input_seed100.csv` | Generated conveyor input (full, 11 orders) |
| `conveyor_input_small_seed100.csv` | Small test instance input (4 orders) |
| `loading_instructions_seed100.txt` | Step-by-step tote loading instructions for physical demo |

## Setup

### Python Optimizer

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### React Dashboard

```bash
npm create vite@latest dashboard -- --template react
cd dashboard
npm install
npm install recharts
```

Then copy `conveyor_dashboard.jsx` into `dashboard/src/App.jsx`:

```bash
cp ../conveyor_dashboard.jsx src/App.jsx
```

## Usage

### Run the Optimizer

Default seed=100:

```bash
python conveyor_optimizer.py
```

With a custom seed and output directory:

```bash
python conveyor_optimizer.py 42 ./output
```

### Run the Dashboard

```bash
cd dashboard
npm run dev
```

Opens at `http://localhost:5173`.

### Output

- `conveyor_input_seed<N>.csv` - Full conveyor input CSV
- `conveyor_input_small_seed<N>.csv` - Small test instance CSV
- `loading_instructions_seed<N>.txt` - Physical loading instructions
- `viz_data_seed<N>.json` - JSON data for the React dashboard

## Results (seed=100)

- **11 orders, 36 total items, 8 item types, 16 totes**
- Optimized makespan: **6.2 min** (vs 7.4 min naive = **16.1% improvement**)
- Belt loads balanced at 10/9/8/9 items
