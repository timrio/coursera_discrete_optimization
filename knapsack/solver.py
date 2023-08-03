#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
from branch_and_bound import SolverBranchAndBound
from dp_solver import SolverDP
from dataclasses import dataclass
from greedy_solver import greedy


@dataclass
class Item:
    index: int
    value: int
    weight: int

    @property
    def density(self):
        if self.weight > 0:
            return self.value / self.weight
        return 0


def parser(input_data):
    # parse the input
    lines = input_data.split("\n")

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count + 1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i - 1, int(parts[0]), int(parts[1])))

    return capacity, item_count, items


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    capacity, item_count, items = parser(input_data)

    # solution
    print(len(items))
    print(capacity)
    if capacity > 500000:
        print("using greedy")
        # optimal_value, approx, taken = SolverBranchAndBound(capacity, item_count, items).run()
        optimal_value, approx, taken = greedy(capacity, item_count, items)
    else:
        print("using dynamic programing")
        optimal_value, approx, taken = SolverDP(capacity, item_count, items).run()

    # prepare the solution in the specified output format
    output_data = str(optimal_value) + " " + str(approx) + "\n"
    output_data += " ".join(map(str, taken))
    return output_data


if __name__ == "__main__":
    # import sys

    # if len(sys.argv) > 1:
    #     file_location = sys.argv[1].strip()
    #     with open(
    #         file_location,
    #         "r",
    #     ) as input_data_file:
    #         input_data = input_data_file.read()
    #     print(solve_it(input_data))
    # else:
    #     print(
    #         "This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)"
    #     )

    import sys

    # file_location = sys.argv[1].strip()
    with open(
        "data/ks_4_0",
        "r",
    ) as input_data_file:
        input_data = input_data_file.read()
    print(solve_it(input_data))
