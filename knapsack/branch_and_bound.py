import numpy as np
from queue import Queue
from queue import PriorityQueue, LifoQueue
from copy import deepcopy
from functools import cached_property


class Node:
    def __init__(self, level, value, optimal_bound, weight, taken):
        self.level = level
        self.value = value
        self.optimal_bound = optimal_bound
        self.weight = weight
        self.taken = taken


class SolverBranchAndBound:
    def __init__(self, capacity, item_count, items):
        self.capacity = capacity
        self.item_count = item_count
        self.items = sorted(items, key=lambda x: x.density, reverse=True)

    def optimal_bound(self, node: Node):
        bound = node.value
        running_weight = node.weight
        for item in self.items[node.level + 1 :]:
            bound += item.value
            if running_weight + item.weight <= self.capacity:
                bound += item.value
                running_weight += item.weight
            else:
                ratio = (
                    self.capacity - running_weight
                ) / item.weight  # compute ratio of item
                bound += item.value * ratio
                break
        return bound

    def run(self):
        node_queue = LifoQueue()
        running_node = Node(-1, 0, 0, 0, [0] * self.item_count)
        node_queue.put(running_node)
        max_value = 0
        optimal_node = running_node

        while not node_queue.empty():
            running_node = node_queue.get()

            # If the node is a leaf node, skip it
            if (
                running_node.level == self.item_count - 1
                or running_node.optimal_bound < max_value
            ):
                continue

            for add_child_bool in [0, 1]:
                added_item = self.items[running_node.level + 1]

                if (
                    running_node.weight + add_child_bool * added_item.weight
                    > self.capacity
                ):
                    continue

                child_node = Node(
                    running_node.level + 1,
                    running_node.value + add_child_bool * added_item.value,
                    running_node.optimal_bound,
                    running_node.weight + add_child_bool * added_item.weight,
                    [],
                )

                if not add_child_bool:
                    child_node.optimal_bound = self.optimal_bound(child_node)

                if child_node.optimal_bound < max_value:
                    continue

                new_taken = deepcopy(running_node.taken)
                new_taken[added_item.index] = add_child_bool
                child_node.taken = new_taken
                node_queue.put(child_node)

                if child_node.value > max_value:
                    max_value = child_node.value
                    optimal_node = child_node

        return max_value, 1, optimal_node.taken
