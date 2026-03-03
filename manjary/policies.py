
import random
from typing import Dict, List, Tuple


def ototal(d: Dict[int, int]) -> int:
    return sum(int(v) for v in d.values())


def queues_round_robin(n_orders: int, num_belts: int = 4) -> List[List[int]]:
    qs = [[] for _ in range(num_belts)]
    for i in range(n_orders):
        qs[i % num_belts].append(i)
    return qs


def queues_random(n_orders: int, seed: int = 123, num_belts: int = 4) -> List[List[int]]:
    rng = random.Random(seed)
    qs = [[] for _ in range(num_belts)]
    for i in range(n_orders):
        qs[rng.randrange(num_belts)].append(i)
    return qs


def queues_lpt_balance(demands: List[Dict[int, int]], num_belts: int = 4) -> Tuple[List[List[int]], List[int]]:
    """
    LPT (largest processing time first) assignment to least-loaded belt.
    'Processing time' approximated by total number of items in the order.
    """
    sizes = sorted([(ototal(d), i) for i, d in enumerate(demands)], reverse=True)
    qs = [[] for _ in range(num_belts)]
    load = [0] * num_belts

    for sz, oi in sizes:
        b = min(range(num_belts), key=lambda x: load[x])
        qs[b].append(oi)
        load[b] += sz

    return qs, load