# Shortest-Order-First — Loading Instructions

## Results

| Metric | Value |
|---|---|
| Orders | 11 |
| Totes | 12 |
| Total Items | 36 |
| Feasible | Yes |
| Makespan | 84.0 s |
| Avg Completion | 44.0 s |
| Dropped Items | 0 |

---

## What Is in Each Tote

| Tote | Contents | Qty |
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

> All items enter via **one conveyor**; each belt position sorts items for its queued orders.

### Belt 1 (3 orders)

| # | Order Items | Qty |
|---|---|---|
| 1 | Pentagon ×1 | 1 |
| 2 | Triangle ×2, Moon ×1 | 3 |
| 3 | Pentagon ×2, Triangle ×3 | 5 |

### Belt 2 (3 orders)

| # | Order Items | Qty |
|---|---|---|
| 1 | Moon ×2 | 2 |
| 2 | Trapezoid ×1, Star ×2 | 3 |
| 3 | Trapezoid ×1, Triangle ×3, Star ×1 | 5 |

### Belt 3 (3 orders)

| # | Order Items | Qty |
|---|---|---|
| 1 | Pentagon ×1, Trapezoid ×1 | 2 |
| 2 | Heart ×3 | 3 |
| 3 | Circle ×3, Pentagon ×2, Moon ×1 | 6 |

### Belt 4 (2 orders)

| # | Order Items | Qty |
|---|---|---|
| 1 | Pentagon ×2 | 2 |
| 2 | Circle ×3, Moon ×1 | 4 |

---

## Tote Loading Order

| # | Tote | Items |
|---|---|---|
| 1 | 10 | 2 |
| 2 | 7 | 2 |
| 3 | 11 | 1 |
| 4 | 12 | 1 |
| 5 | 1 | 5 |
| 6 | 3 | 1 |
| 7 | 13 | 5 |
| 8 | 8 | 6 |
| 9 | 9 | 3 |
| 10 | 0 | 5 |
| 11 | 4 | 3 |
| 12 | 14 | 2 |

---

## Item Release Order Per Tote

> For each tote, the exact order to place its items onto the entry conveyor.

### Tote 10  (Load #1, 2 items)

| # | Item | Time |
|---|---|---|
| 1 | Pentagon | 0.0 s |
| 2 | Moon | 1.0 s |

### Tote 7  (Load #2, 2 items)

| # | Item | Time |
|---|---|---|
| 1 | Moon | 4.0 s |
| 2 | Moon | 5.0 s |

### Tote 11  (Load #3, 1 item)

| # | Item | Time |
|---|---|---|
| 1 | Pentagon | 8.0 s |

### Tote 12  (Load #4, 1 item)

| # | Item | Time |
|---|---|---|
| 1 | Trapezoid | 11.0 s |

### Tote 1  (Load #5, 5 items)

| # | Item | Time |
|---|---|---|
| 1 | Pentagon | 14.0 s |
| 2 | Pentagon | 15.0 s |
| 3 | Heart | 16.0 s |
| 4 | Heart | 17.0 s |
| 5 | Heart | 18.0 s |

### Tote 3  (Load #6, 1 item)

| # | Item | Time |
|---|---|---|
| 1 | Moon | 21.0 s |

### Tote 13  (Load #7, 5 items)

| # | Item | Time |
|---|---|---|
| 1 | Triangle | 24.0 s |
| 2 | Triangle | 25.0 s |
| 3 | Moon | 26.0 s |
| 4 | Pentagon | 27.0 s |
| 5 | Pentagon | 28.0 s |

### Tote 8  (Load #8, 6 items)

| # | Item | Time |
|---|---|---|
| 1 | Star | 31.0 s |
| 2 | Star | 32.0 s |
| 3 | Trapezoid | 33.0 s |
| 4 | Circle | 34.0 s |
| 5 | Circle | 35.0 s |
| 6 | Circle | 36.0 s |

### Tote 9  (Load #9, 3 items)

| # | Item | Time |
|---|---|---|
| 1 | Circle | 39.0 s |
| 2 | Circle | 40.0 s |
| 3 | Circle | 41.0 s |

### Tote 0  (Load #10, 5 items)

| # | Item | Time |
|---|---|---|
| 1 | Triangle | 44.0 s |
| 2 | Triangle | 45.0 s |
| 3 | Triangle | 46.0 s |
| 4 | Pentagon | 47.0 s |
| 5 | Pentagon | 48.0 s |

### Tote 4  (Load #11, 3 items)

| # | Item | Time |
|---|---|---|
| 1 | Triangle | 51.0 s |
| 2 | Triangle | 52.0 s |
| 3 | Triangle | 53.0 s |

### Tote 14  (Load #12, 2 items)

| # | Item | Time |
|---|---|---|
| 1 | Star | 56.0 s |
| 2 | Trapezoid | 57.0 s |
