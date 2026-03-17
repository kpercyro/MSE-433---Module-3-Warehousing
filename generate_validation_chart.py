"""Generate sim vs actual makespan bar chart for the report."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

_script_dir = os.path.dirname(os.path.abspath(__file__))

# Data from physical runs vs simulation
runs = [
    "Day 1\nRun A (SA)",
    "Day 1\nRun B (SA)",
    "Day 2\nRun A (SA)",
    "Day 2\nRun B (SOF)",
]
actual   = [162.5, 206.1, 167.2, 116.8]
simulated = [182.0, 182.0, 182.0, 200.0]
errors   = [12.0, 11.7, 8.9, 71.2]

x = np.arange(len(runs))
width = 0.32

fig, ax = plt.subplots(figsize=(10, 5.5))

bars_actual = ax.bar(x - width/2, actual, width, label='Actual (Physical)',
                     color='#2196F3', edgecolor='black', linewidth=0.6, zorder=3)
bars_sim = ax.bar(x + width/2, simulated, width, label='Simulated (Model)',
                  color='#FF9800', edgecolor='black', linewidth=0.6, zorder=3)

# Annotate bars with values
for bar in bars_actual:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 2, f'{h:.1f}s',
            ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#1565C0')

for bar in bars_sim:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 2, f'{h:.1f}s',
            ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#E65100')

# Error annotations between bar pairs
for i in range(len(runs)):
    mid_x = x[i]
    max_y = max(actual[i], simulated[i])
    ax.annotate(f'{errors[i]:.1f}% error',
                xy=(mid_x, max_y + 14),
                ha='center', va='bottom', fontsize=8.5,
                color='#D32F2F', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFEBEE',
                          edgecolor='#D32F2F', linewidth=0.8))

ax.set_ylabel('Makespan (seconds)', fontsize=12)
ax.set_title('Simulated vs Actual Physical Makespan', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(runs, fontsize=10)
ax.legend(fontsize=10, loc='upper left')
ax.set_ylim(0, 260)
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()
out = os.path.join(_script_dir, 'results', 'comparison', 'sim_vs_actual_makespan.png')
plt.savefig(out, dpi=200)
plt.close()
print(f"Saved: {out}")
