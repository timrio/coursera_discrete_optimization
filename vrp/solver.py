#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
from or_tools_solver import main_solver


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    outputData = main_solver(input_data)

    return outputData


import sys

if __name__ == '__main__':
    # file_location = "/Users/timotheerio/Code_Bases/Coursera_Optimization/vrp/data/vrp_21_4_1"
    # with open(file_location, 'r') as input_data_file:
    #     input_data = input_data_file.read()
    # print(solve_it(input_data))

    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:

        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/vrp_5_4_1)')

