#!/usr/bin/python
# -*- coding: utf-8 -*-

from mysolutionsolver import Solver
from homemade_constraint_programing import run_home_made_solver


def parser(input_data):
    # parse the input
    lines = input_data.split("\n")

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))

    return node_count, edge_count, edges


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    node_count, edge_count, edges = parser(input_data)
    print(f"{node_count} - {edge_count}")

    # build a trivial solution
    # every node has its own color
    if node_count < 1:
        opt_sol, solution = Solver(node_count, edge_count, edges).solve()
    else:
        opt_sol, solution = run_home_made_solver(node_count, edge_count, edges)

    # prepare the solution in the specified output format
    output_data = str(int(opt_sol)) + " " + str(0) + "\n"
    output_data += " ".join(map(str, solution))

    return output_data


import sys

if __name__ == "__main__":
    import sys

    # with open("coloring/data/gc_50_1", "r") as input_data_file:
    #     input_data = input_data_file.read()
    # print(solve_it(input_data))

    import sys

    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, "r") as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print(
            "This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)"
        )
