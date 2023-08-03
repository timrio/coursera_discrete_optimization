import numpy as np
from queue import Queue
from queue import PriorityQueue, LifoQueue
from copy import deepcopy
from functools import cached_property
from tqdm import tqdm


class SolverDP:
    def __init__(self, capacity, item_count, items):
        self.capacity = capacity
        self.item_count = item_count
        self.items = sorted(items, key=lambda x: x.weight, reverse=True)
        self.value_table = np.zeros((capacity + 1, item_count + 1))

    def run(self):
        for number_items in tqdm(range(1, self.item_count + 1)):
            last_item_weight = self.items[number_items - 1].weight
            last_item_value = self.items[number_items - 1].value
            self.value_table[: last_item_weight - 1, number_items] = self.value_table[
                : last_item_weight - 1, number_items - 1
            ]
            for size in range(last_item_weight, self.capacity + 1):
                previous_bag_value = self.value_table[size, number_items - 1]
                remaining_size = int(size - (last_item_weight))
                updated_value = (
                    self.value_table[remaining_size, number_items - 1] + last_item_value
                )
                if updated_value > previous_bag_value:
                    self.value_table[size, number_items] = updated_value
                    continue

                self.value_table[size, number_items] = previous_bag_value

        # backward computation
        taken = [0] * self.item_count
        current_size = self.capacity
        for number_items in reversed(range(1, self.item_count + 1)):
            if (
                self.value_table[current_size, number_items]
                != self.value_table[current_size, number_items - 1]
            ):
                taken[self.items[number_items - 1].index] += 1
                current_size -= self.items[number_items - 1].weight

        return int(self.value_table[self.capacity, self.item_count]), 1, taken
