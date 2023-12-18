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
    return (
        distance_matrix,
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


def create_constraints(model, facilities, customers, x, y, has_been_opened):
    # create constraints
    for c in customers:
        model.Add(sum(x[f.index, c.index] for f in facilities) == 1)

    for f in facilities:
        for c in customers:
            model.Add(x[f.index, c.index] <= y[f.index])

    for f in facilities:
        model.Add(sum(x[f.index, c.index] * c.demand for c in customers) <= f.capacity)

    for f_index in range(len(has_been_opened)):
        if has_been_opened[f_index]:
            y[f_index] = 1

    return model


def create_objective(
    model,
    facilities,
    customers,
    distance_matrix,
    x,
    y,
):
    # create objective
    model.Minimize(
        sum(
            sum(
                x[f.index, c.index] * distance_matrix[f.index, c.index]
                for f in facilities
            )
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


def main_solver(input_data):
    (
        distance_matrix,
        facilities,
        customers,
    ) = create_problem_matricies(input_data)

    # determine number of tiles
    print(len(customers))
    print(len(facilities))
    if len(customers) < 1000 and len(facilities) <= 100:
        number_of_tiles_per_border = 1
    elif len(customers) == 1000 and len(facilities)==100:
        number_of_tiles_per_border = 2
    elif len(customers) == 2000 and len(facilities)==2000:
        number_of_tiles_per_border = 10
    elif len(customers) == 3000 and len(facilities)==500:
        number_of_tiles_per_border = 7
    else:
        number_of_tiles_per_border = 4

    already_assigned_customers = [False for i in range(len(customers))]

    # find x min of facility and customer
    x_min, x_max = min([customer.location.x for customer in customers]), max(
        [customer.location.x for customer in customers]
    )
    y_min, y_max = min([customer.location.y for customer in customers]), max(
        [customer.location.y for customer in customers]
    )

    x_tile_lenght = (x_max - x_min) / number_of_tiles_per_border
    y_tile_lenght = (y_max - y_min) / number_of_tiles_per_border

    optimal_value = 0
    assignements = [-1 for i in range(len(customers))]
    has_been_opened = [False for i in range(len(facilities))]
    used_capacity = [0 for i in range(len(facilities))]
    for i in range(number_of_tiles_per_border):
        for j in range(number_of_tiles_per_border):
            tile_id = i * number_of_tiles_per_border + j
            print(f"tile {tile_id} of {number_of_tiles_per_border**2}")
            tile_x_min = x_min + i * x_tile_lenght
            tile_x_max = x_min + (i + 1) * x_tile_lenght
            tile_y_min = y_min + j * y_tile_lenght
            tile_y_max = y_min + (j + 1) * y_tile_lenght

            # create subproblem
            sub_customers = [
                customer
                for customer in customers
                if tile_x_min <= customer.location.x <= tile_x_max
                and tile_y_min <= customer.location.y <= tile_y_max
                and not already_assigned_customers[customer.index]
            ]
            for customer in sub_customers:
                already_assigned_customers[customer.index] = True

            for f_index in range(len(used_capacity)):
                old_cap = facilities[f_index].capacity
                new_cap = old_cap - used_capacity[f_index]
                facilities[f_index] = facilities[f_index]._replace(capacity=new_cap)
                if has_been_opened[f_index]:
                    facilities[f_index] = facilities[f_index]._replace(setup_cost=0)

            used_capacity = [0 for i in range(len(facilities))]

            model = create_model()
            x, y = create_variables(model, facilities, sub_customers)

            model = create_constraints(
                model, facilities, sub_customers, x, y, has_been_opened
            )
            model = create_objective(
                model,
                facilities,
                sub_customers,
                distance_matrix,
                x,
                y,
            )
            solver, status = solve_model(model)

            if status == cp_model.OPTIMAL:
                optimal_value += solver.ObjectiveValue()
                for c in sub_customers:
                    for f in facilities:
                        if solver.Value(x[f.index, c.index]) == 1:
                            assignements[c.index] = f.index
                            used_capacity[f.index] += c.demand
                            has_been_opened[f.index] = True
                            break
            else:
                print("No optimal solution found!")
                return None
    output_data = "%.2f" % optimal_value + " " + str(0) + "\n"
    output_data += " ".join(map(str, assignements))
    return output_data
