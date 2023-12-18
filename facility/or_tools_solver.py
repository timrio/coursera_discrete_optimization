import numpy as np
import pandas as pd
from ortools.sat.python import cp_model
from collections import namedtuple
import math

Point = namedtuple("Point", ["x", "y"])
Facility = namedtuple("Facility", ["index", "setup_cost", "capacity", "location"])
Customer = namedtuple("Customer", ["index", "demand", "location"])


def euclidian_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def create_distance_matrix(facilities, customers):
    matrix = np.zeros((len(facilities), len(customers)))
    for i, facility in enumerate(facilities):
        for j, customer in enumerate(customers):
            matrix[i, j] = euclidian_distance(
                facility.location.x,
                facility.location.y,
                customer.location.x,
                customer.location.y,
            )
    return matrix


def create_setup_cost_array(facilities):
    return np.array([facility.setup_cost for facility in facilities])


def create_customer_demand_array(customers):
    return np.array([customer.demand for customer in customers])


def create_capacity_array(facilities):
    return np.array([facility.capacity for facility in facilities])


def create_problem_matricies(input_data):
    # parse the input
    lines = input_data.split("\n")

    parts = lines[0].split()
    facility_count = int(parts[0])
    customer_count = int(parts[1])

    facilities = []
    for i in range(1, facility_count + 1):
        parts = lines[i].split()
        facilities.append(
            Facility(
                i - 1,
                float(parts[0]),
                int(parts[1]),
                Point(float(parts[2]), float(parts[3])),
            )
        )

    customers = []
    for i in range(facility_count + 1, facility_count + 1 + customer_count):
        parts = lines[i].split()
        customers.append(
            Customer(
                i - 1 - facility_count,
                int(parts[0]),
                Point(float(parts[1]), float(parts[2])),
            )
        )

    distance_matrix = create_distance_matrix(facilities, customers)
    setup_cost_array = create_setup_cost_array(facilities)
    customer_demand_array = create_customer_demand_array(customers)
    capacity_array = create_capacity_array(facilities)
    return distance_matrix, setup_cost_array, customer_demand_array, capacity_array, facilities, customers


def create_model():
    model = cp_model.CpModel()
    return model


def create_variables(model, facilities, customers):
    # create variables
    x = {}
    for i in range(len(facilities)):
        for j in range(len(customers)):
            # x defines whether a customer is assigned to a facility
            x[i, j] = model.NewBoolVar("x[%i,%i]" % (i, j))
    y = {}
    for i in range(len(facilities)):
        # y defines wether a facility is open
        y[i] = model.NewBoolVar("y[%i]" % i)
    return x, y


def create_constraints(model, facilities, customers, x,y):
    # create constraints
    for j in range(len(customers)):
        model.Add(sum(x[i, j] for i in range(len(facilities))) == 1)

    for i in range(len(facilities)):
        for j in range(len(customers)):
            model.Add(x[i, j] <= y[i])

    for i in range(len(facilities)):
        model.Add(
            sum(x[i, j] * customers[j].demand for j in range(len(customers)))
            <= facilities[i].capacity
        )
    return model


def create_objective(model, customers, facilities, distance_matrix, setup_cost_array, customer_demand_array,x,y):
    # create objective
    model.Minimize(
        sum(
            sum(x[i, j] * distance_matrix[i, j] for i in range(len(facilities)))
            for j in range(len(customers))
        )
        + sum(y[i] * setup_cost_array[i] for i in range(len(facilities)))
    )
    return model


def solve_model(model):
    # solve model
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 300.0
    status = solver.Solve(model)
    return solver, status


def main_solver(input_data):
    (
        distance_matrix,
        setup_cost_array,
        customer_demand_array,
        capacity_array,
        facilities, 
        customers
    ) = create_problem_matricies(input_data)
    model = create_model()
    x, y = create_variables(model, facilities, customers)
    model = create_constraints(model, facilities, customers,x,y)
    model = create_objective(
        model, customers, facilities, distance_matrix, setup_cost_array, customer_demand_array,x,y
    )
    solver, status = solve_model(model)

    if status == cp_model.OPTIMAL:

        optimal_value = solver.ObjectiveValue()
        assignements = []
        for j in range(len(customers)):
            value = solver.Value(
                        np.argmax(
                            [solver.Value(x[i, j]) for i in range(len(facilities))]
                        )
                    )
            assignements.append(value)

        output_data = '%.2f' % optimal_value + ' ' + str(0) + '\n'
        output_data += ' '.join(map(str, assignements))
        return output_data
    else:
        print("No optimal solution found!")
        return None
