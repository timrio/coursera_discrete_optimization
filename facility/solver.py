#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import math
import numpy as np
import os
from MIP_Solver.solver.model import Model
from MIP_Solver.utils.pre_processing import compute_distance_to_facilities

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])


def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    facility_count = int(parts[0])
    customer_count = int(parts[1])
    
    facilities = []
    for i in range(1, facility_count+1):
        # facilities list format: index, setup_cost, capacity, coordinates
        parts = lines[i].split()
        facilities.append([i-1, float(parts[0]), int(parts[1]),[float(parts[2]), float(parts[3])]])

    customers = []
    for i in range(facility_count+1, facility_count+1+customer_count):
        # customer list format: index, demand, coordinates
        parts = lines[i].split()
        customers.append([i-1-facility_count, int(parts[0]), [float(parts[1]), float(parts[2])]])

    situation = [facilities,customers]

    ### To do, optimize here
    matrix = compute_distance_to_facilities(situation)
    model = Model('test')
    model.define_variables(situation)
    model.define_constraints(situation)
    model.add_slack_variables()
    A,X,b = model.create_matrix_notation()
    

    # calculate the cost of the solution
    obj = sum([f.setup_cost*used[f.index] for f in facilities])
    for customer in customers:
        obj += length(customer.location, facilities[solution[customer.index]].location)

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


import sys

if __name__ == '__main__':
    with open("/Users/timotheerio/Library/CloudStorage/OneDrive-TheBostonConsultingGroup,Inc/Documents/formations/Coursera_Optimization/facility/data/fl_16_1", 'r') as input_data_file:
        input_data = input_data_file.read()
    print(solve_it(input_data))

    # import sys
    # if len(sys.argv) > 1:
    #     file_location = sys.argv[1].strip()
    #     with open(file_location, 'r') as input_data_file:
    #         input_data = input_data_file.read()
    #     print(solve_it(input_data))
    # else:
    #     print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/fl_16_2)')

