import numpy as np
from queue import Queue
from queue import PriorityQueue, LifoQueue
from copy import deepcopy
from functools import cached_property
from tqdm import tqdm


def greedy(capacity, item_count, items):
    n = len(items)
    taken = [0] * n
    filled = 0
    value = 0
    for item in tqdm(sorted(items, key=lambda x: x.density, reverse=True)):
        if filled + item.weight <= capacity:
            taken[item.index] = 1
            value += item.value
            filled += item.weight

    return value, 0, taken
