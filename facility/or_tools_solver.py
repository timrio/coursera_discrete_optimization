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
    for f in facilities:
        for c in customers:
            matrix[f.index, c.index] = euclidian_distance(
                f.location.x,
                f.location.y,
                c.location.x,
                c.location.y,
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
    return (
        distance_matrix,
        setup_cost_array,
        customer_demand_array,
        capacity_array,
        facilities,
        customers,
    )


def create_model():
    model = cp_model.CpModel()
    return model


def create_variables(model, facilities, customers):
    # create variables
    x = {}
    for f in facilities:
        for c in customers:
            # x defines whether a customer is assigned to a facility
            x[f.index, c.index] = model.NewBoolVar("x[%i,%i]" % (f.index, c.index))
    y = {}
    for f in facilities:
        # y defines wether a facility is open
        y[f.index] = model.NewBoolVar("y[%i]" % f.index)
    return x, y


def create_constraints(model, facilities, customers, x, y):
    # create constraints
    for c in customers:
        model.Add(sum(x[f.index, c.index] for f in facilities) == 1)

    for f in facilities:
        for c in customers:
            model.Add(x[f.index, c.index] <= y[f.index])

    for f in facilities:
        model.Add(
            sum(x[f.index, c.index] * c.demand for c in customers)
            <= f.capacity
        )
    return model


def create_objective(
    model,
    facilities,
    customers,
    distance_matrix,
    setup_cost_array,
    customer_demand_array,
    x,
    y,
):
    # create objective
    model.Minimize(
        sum(
            sum(x[f.index, c.index] * distance_matrix[f.index, c.index] for f in facilities)
            for c in customers
        )
        + sum(y[f.index] * f.setup_cost for f in facilities)
    )
    return model


def solve_model(model):
    # solve model
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 300.0
    status = solver.Solve(model)
    return solver, status


def main_solver(input_data, number_of_tiles_per_border = 4):
    (
        distance_matrix,
        setup_cost_array,
        customer_demand_array,
        capacity_array,
        facilities,
        customers,
    ) = create_problem_matricies(input_data)

    already_assigned_facilities = [False for i in range(len(facilities))]
    already_assigned_customers = [False for i in range(len(customers))]

    # find x min of facility and customer
    assets = facilities + customers
    x_min, x_max = min([asset.location.x for asset in assets]), max(
        [asset.location.x for asset in assets]
    )
    y_min, y_max = min([asset.location.y for asset in assets]), max(
        [asset.location.y for asset in assets]
    )

    x_tile_lenght = (x_max - x_min) / number_of_tiles_per_border
    y_tile_lenght = (y_max - y_min) / number_of_tiles_per_border

    optimal_value = 0
    assignements = [-1 for i in range(len(customers))]

    for i in range(number_of_tiles_per_border):
        for j in range(number_of_tiles_per_border):
            tile_id = i * number_of_tiles_per_border + j
            print(f"tile {tile_id} of {number_of_tiles_per_border**2}")
            tile_x_min = x_min + i * x_tile_lenght
            tile_x_max = x_min + (i + 1) * x_tile_lenght
            tile_y_min = y_min + j * y_tile_lenght
            tile_y_max = y_min + (j + 1) * y_tile_lenght

            # create subproblem         
            sub_facilities = [
                facility
                for facility in facilities
                if tile_x_min <= facility.location.x <= tile_x_max
                and tile_y_min <= facility.location.y <= tile_y_max
                and not already_assigned_facilities[facility.index]
            ]
            sub_customers = [
                customer
                for customer in customers
                if tile_x_min <= customer.location.x <= tile_x_max
                and tile_y_min <= customer.location.y <= tile_y_max
                and not already_assigned_customers[customer.index]
            ]
            for customer in sub_customers:
                already_assigned_customers[customer.index] = True
            for facility in sub_facilities:
                already_assigned_facilities[facility.index] = True

            model = create_model()
            x, y = create_variables(model, sub_facilities, sub_customers)
   
            model = create_constraints(model, sub_facilities, sub_customers, x, y)
            model = create_objective(
                model,
                sub_facilities,
                sub_customers,
                distance_matrix,
                setup_cost_array,
                customer_demand_array,
                x,
                y,
            )
            solver, status = solve_model(model)

            if status == cp_model.OPTIMAL:
                optimal_value += solver.ObjectiveValue()
                for c in sub_customers:
                    for f in sub_facilities:
                        if solver.Value(x[f.index, c.index]) == 1:
                            assignements[c.index] = f.index
                            break
            else:
                print("No optimal solution found!")
                return None
    output_data = "%.2f" % optimal_value + " " + str(0) + "\n"
    output_data += " ".join(map(str, assignements))
    return output_data
