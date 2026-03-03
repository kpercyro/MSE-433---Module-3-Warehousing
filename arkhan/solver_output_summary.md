# Warehouse Conveyor Solver — Output Summary

## Solver Results

| Metric | Value |
|---|---|
| Orders | 11 |
| Totes | 12 |
| Total Items | 36 |
| Feasible | Yes |
| Makespan | 60.0 s |
| Avg Completion Time | 30.6 s |
| Dropped Items | 0 |

---

## Tote Inventory (Items on Each Tote)

| Tote ID | Items | Total Qty |
|---|---|---|
| 0 | Pentagon ×2, Triangle ×3 | 5 |
| 1 | Pentagon ×2, Heart ×3 | 5 |
| 3 | Moon ×1 | 1 |
| 4 | Triangle ×3 | 3 |
| 7 | Moon ×2 | 2 |
| 8 | Circle ×3, Trapezoid ×1, Star ×2 | 6 |
| 9 | Circle ×3 | 3 |
| 10 | Pentagon ×1, Moon ×1 | 2 |
| 11 | Pentagon ×1 | 1 |
| 12 | Trapezoid ×1 | 1 |
| 13 | Pentagon ×2, Triangle ×2, Moon ×1 | 5 |
| 14 | Trapezoid ×1, Star ×1 | 2 |

---

## Belt Sort-Position Assignments

> Items travel along **one entry conveyor** and are sorted at 
> **4 belt positions** where orders are fulfilled.

### Belt Position 1 (8 orders)

| Queue Pos | Order Items | Total |
|---|---|---|
| 1 | Pentagon ×1 | 1 |
| 2 | Moon ×2 | 2 |
| 3 | Pentagon ×2 | 2 |
| 4 | Trapezoid ×1, Triangle ×3, Star ×1 | 5 |
| 5 | Pentagon ×2, Triangle ×3 | 5 |
| 6 | Trapezoid ×1, Star ×2 | 3 |
| 7 | Heart ×3 | 3 |
| 8 | Circle ×3, Pentagon ×2, Moon ×1 | 6 |

### Belt Position 2 (1 order)

| Queue Pos | Order Items | Total |
|---|---|---|
| 1 | Pentagon ×1, Trapezoid ×1 | 2 |

### Belt Position 3 (1 order)

| Queue Pos | Order Items | Total |
|---|---|---|
| 1 | Triangle ×2, Moon ×1 | 3 |

### Belt Position 4 (1 order)

| Queue Pos | Order Items | Total |
|---|---|---|
| 1 | Circle ×3, Moon ×1 | 4 |

---

## Tote Loading Sequence (Order Totes Are Placed on the Conveyor)

| Load Order | Tote ID | Items in Tote |
|---|---|---|
| 1 | 10 | 2 |
| 2 | 13 | 5 |
| 3 | 14 | 2 |
| 4 | 4 | 3 |
| 5 | 0 | 5 |
| 6 | 7 | 2 |
| 7 | 8 | 6 |
| 8 | 11 | 1 |
| 9 | 12 | 1 |
| 10 | 1 | 5 |
| 11 | 9 | 3 |
| 12 | 3 | 1 |

---

## Item Release Order Per Tote

> For each tote (in loading sequence), the exact order to place items onto the entry conveyor.

### Tote 10 (Load #1, 2 items)

| Release # (in tote) | Item | Time (s) |
|---|---|---|
| 1 | Pentagon | 0.0 |
| 2 | Moon | 1.0 |

### Tote 13 (Load #2, 5 items)

| Release # (in tote) | Item | Time (s) |
|---|---|---|
| 1 | Moon | 4.0 |
| 2 | Pentagon | 5.0 |
| 3 | Pentagon | 6.0 |
| 4 | Triangle | 7.0 |
| 5 | Triangle | 8.0 |

### Tote 14 (Load #3, 2 items)

| Release # (in tote) | Item | Time (s) |
|---|---|---|
| 1 | Trapezoid | 11.0 |
| 2 | Star | 12.0 |

### Tote 4 (Load #4, 3 items)

| Release # (in tote) | Item | Time (s) |
|---|---|---|
| 1 | Triangle | 15.0 |
| 2 | Triangle | 16.0 |
| 3 | Triangle | 17.0 |

### Tote 0 (Load #5, 5 items)

| Release # (in tote) | Item | Time (s) |
|---|---|---|
| 1 | Pentagon | 20.0 |
| 2 | Pentagon | 21.0 |
| 3 | Triangle | 22.0 |
| 4 | Triangle | 23.0 |
| 5 | Triangle | 24.0 |

### Tote 7 (Load #6, 2 items)

| Release # (in tote) | Item | Time (s) |
|---|---|---|
| 1 | Moon | 27.0 |
| 2 | Moon | 28.0 |

### Tote 8 (Load #7, 6 items)

| Release # (in tote) | Item | Time (s) |
|---|---|---|
| 1 | Star | 31.0 |
| 2 | Star | 32.0 |
| 3 | Trapezoid | 33.0 |
| 4 | Circle | 34.0 |
| 5 | Circle | 35.0 |
| 6 | Circle | 36.0 |

### Tote 11 (Load #8, 1 item)

| Release # (in tote) | Item | Time (s) |
|---|---|---|
| 1 | Pentagon | 39.0 |

### Tote 12 (Load #9, 1 item)

| Release # (in tote) | Item | Time (s) |
|---|---|---|
| 1 | Trapezoid | 42.0 |

### Tote 1 (Load #10, 5 items)

| Release # (in tote) | Item | Time (s) |
|---|---|---|
| 1 | Heart | 45.0 |
| 2 | Heart | 46.0 |
| 3 | Heart | 47.0 |
| 4 | Pentagon | 48.0 |
| 5 | Pentagon | 49.0 |

### Tote 9 (Load #11, 3 items)

| Release # (in tote) | Item | Time (s) |
|---|---|---|
| 1 | Circle | 52.0 |
| 2 | Circle | 53.0 |
| 3 | Circle | 54.0 |

### Tote 3 (Load #12, 1 item)

| Release # (in tote) | Item | Time (s) |
|---|---|---|
| 1 | Moon | 57.0 |

---

## Full Item Release Sequence (All Totes, Chronological)

| # | Tote | Item Released | Time (s) |
|---|---|---|---|
| 1 | 10 | Pentagon | 0.0 |
| 2 | 10 | Moon | 1.0 |
| 3 | 13 | Moon | 4.0 |
| 4 | 13 | Pentagon | 5.0 |
| 5 | 13 | Pentagon | 6.0 |
| 6 | 13 | Triangle | 7.0 |
| 7 | 13 | Triangle | 8.0 |
| 8 | 14 | Trapezoid | 11.0 |
| 9 | 14 | Star | 12.0 |
| 10 | 4 | Triangle | 15.0 |
| 11 | 4 | Triangle | 16.0 |
| 12 | 4 | Triangle | 17.0 |
| 13 | 0 | Pentagon | 20.0 |
| 14 | 0 | Pentagon | 21.0 |
| 15 | 0 | Triangle | 22.0 |
| 16 | 0 | Triangle | 23.0 |
| 17 | 0 | Triangle | 24.0 |
| 18 | 7 | Moon | 27.0 |
| 19 | 7 | Moon | 28.0 |
| 20 | 8 | Star | 31.0 |
| 21 | 8 | Star | 32.0 |
| 22 | 8 | Trapezoid | 33.0 |
| 23 | 8 | Circle | 34.0 |
| 24 | 8 | Circle | 35.0 |
| 25 | 8 | Circle | 36.0 |
| 26 | 11 | Pentagon | 39.0 |
| 27 | 12 | Trapezoid | 42.0 |
| 28 | 1 | Heart | 45.0 |
| 29 | 1 | Heart | 46.0 |
| 30 | 1 | Heart | 47.0 |
| 31 | 1 | Pentagon | 48.0 |
| 32 | 1 | Pentagon | 49.0 |
| 33 | 9 | Circle | 52.0 |
| 34 | 9 | Circle | 53.0 |
| 35 | 9 | Circle | 54.0 |
| 36 | 3 | Moon | 57.0 |
