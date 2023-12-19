#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
from ortools_solver import solve
from mysolver_refacto import solve_k_opt, iterative_k_opt
from simmulated_annealing import (
simulated_annealing_with_lin_kernighan
)
from kopt import solve_LKH

Point = namedtuple("Point", ["x", "y"])


def length(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split("\n")

    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount + 1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    # build a trivial solution
    # visit the nodes in the order they appear in the file
    if nodeCount < 550:
        if nodeCount==100:
            print("ok")
            value, route = solve_LKH(points,restarts=50)
        else:
            value, route = solve_LKH(points)
    else:
        value, route = solve(nodeCount, points)

    # prepare the solution in the specified output format
    output_data = "%.2f" % value + " " + str(0) + "\n"
    output_data += " ".join(map(str, route))

    return output_data


import sys

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, "r") as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print(
            "This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)"
        )
