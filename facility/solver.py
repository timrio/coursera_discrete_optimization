#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import math
from or_tools_solver import main_solver

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def solve_it(input_data):
    output_data = main_solver(input_data)
    return output_data


import sys

if __name__ == '__main__':
    # file_location = "/Users/timotheerio/Code_Bases/Coursera_Optimization/facility/data/fl_16_1"
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
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/fl_16_2)')

