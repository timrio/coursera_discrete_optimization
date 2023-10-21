import sys
import math
from simmulated_annealing import (
simulated_annealing_with_lin_kernighan
)
from kopt import solve_LKH

file_location = "tsp/data/tsp_144_1"


with open(file_location, "r") as input_data_file:
    input_data = input_data_file.read()


# read input

lines = input_data.split("\n")

nodeCount = int(lines[0])

points = []
for i in range(1, nodeCount + 1):
    line = lines[i]
    parts = line.split()
    points.append((float(parts[0]), float(parts[1])))


obj, route = solve_LKH(points)
