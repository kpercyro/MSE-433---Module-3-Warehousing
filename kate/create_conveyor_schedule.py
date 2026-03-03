import pandas as pd
from pathlib import Path

# determine directory containing this script so we can load CSVs reliably
base_dir = Path(__file__).parent

# ============================================================
# LOAD FILES
# ============================================================

tote_seq = pd.read_csv(base_dir / "tote_sequence.csv", header=None)[0].tolist()
order_seq = pd.read_csv(base_dir / "order_sequence.csv", header=None)[0].tolist()
item_release = pd.read_csv(base_dir / "item_release_sequence.csv")

order_itemtypes = pd.read_csv(base_dir / "order_itemtypes.csv", header=None)
order_quantities = pd.read_csv(base_dir / "order_quantities.csv", header=None)
orders_totes = pd.read_csv(base_dir / "orders_totes.csv", header=None)

item_columns = [
    "circle","pentagon","trapezoid",
    "triangle","star","moon","heart","cross"
]

# ============================================================
# BUILD REMAINING DEMAND
# ============================================================

remaining = {}

for _, row in item_release.iterrows():
    order_id = int(row["order"])
    remaining[order_id] = remaining.get(order_id, 0) + int(row["quantity"])

# ============================================================
# INITIALIZE LANES (4 ACTIVE ORDERS)
# ============================================================

lanes = {}
next_order_index = 4

for i in range(4):
    lanes[i+1] = order_seq[i]

# ============================================================
# BUILD FULL TIME-BASED SCHEDULE
# ============================================================

schedule_rows = []

for tote in tote_seq:

    tote_items = item_release[item_release["tote"] == tote]

    for _, item in tote_items.iterrows():

        order = int(item["order"])
        qty = int(item["quantity"])

        # ✅ Normalize item type immediately
        # read item_type if present; may be numeric code
        item_type = item.get("item_type", None)

        for _ in range(qty):

            # Find which lane this order is on
            lane = None
            for l, o in lanes.items():
                if o == order:
                    lane = l
                    break

            if lane is None:
                continue

            # Fallback lookup if missing or NaN
            if item_type is None or (isinstance(item_type, float) and pd.isna(item_type)) or str(item_type).strip() == "":
                for col in range(order_itemtypes.shape[1]):
                    if pd.isna(order_itemtypes.iloc[order, col]):
                        continue
                    if (
                        int(order_quantities.iloc[order, col]) > 0 and
                        int(orders_totes.iloc[order, col]) == tote
                    ):
                        item_type = order_itemtypes.iloc[order, col]
                        order_quantities.iloc[order, col] -= 1
                        break

            # normalize numeric codes and labels to actual column name
            def normalize_type(val):
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    return None
                # try numeric index
                try:
                    idx = int(float(val))
                except Exception:
                    s = str(val).strip().lower()
                    return s if s in item_columns else None
                if 0 <= idx < len(item_columns):
                    return item_columns[idx]
                return None

            item_type = normalize_type(item_type)

            if item_type is None:
                print("WARNING: Unknown item type:", item.get("item_type", None))

            # Build row with exactly one 1
            row = {"conv_num": lane}
            for col in item_columns:
                row[col] = 1 if col == item_type else 0

            schedule_rows.append(row)

            # Reduce remaining quantity
            remaining[order] -= 1

            # If order complete, load next order
            if remaining[order] == 0:
                lanes[lane] = None
                if next_order_index < len(order_seq):
                    lanes[lane] = order_seq[next_order_index]
                    next_order_index += 1

# ============================================================
# SAVE FINAL FILE
# ============================================================

# convert per-item rows into time-step aggregates by combining consecutive
# entries for the same conveyor and summing the type counts
if schedule_rows:
    df_schedule = pd.DataFrame(schedule_rows)
    df_schedule['group'] = (df_schedule['conv_num'] != df_schedule['conv_num'].shift()).cumsum()
    aggregated = df_schedule.groupby('group').agg({
        'conv_num': 'first',
        **{col: 'sum' for col in item_columns}
    }).reset_index(drop=True)
else:
    aggregated = pd.DataFrame(columns=['conv_num'] + item_columns)

# write output (respect script directory)
output_path = base_dir / "conveyor_full_schedule.csv"
aggregated.to_csv(output_path, index=False)
# also mirror to workspace root so user looking there will see it
try:
    aggregated.to_csv(Path.cwd() / "conveyor_full_schedule.csv", index=False)
except Exception:
    pass

print("Full conveyor schedule generated correctly.")