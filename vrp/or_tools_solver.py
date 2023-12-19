import numpy as np
import pandas as pd
from collections import namedtuple
import math
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])

def euclidian_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def create_distance_matrix(customers):
    matrix = np.zeros((len(customers), len(customers)))
    for i in range(len(customers)):
        for j in range(len(customers)):
            matrix[i, j] = euclidian_distance(
                customers[i].x,
                customers[i].y,
                customers[j].x,
                customers[j].y,
            )
    return matrix


def create_data_model(customers, vehicle_count, vehicle_capacity):
    data = {}
    data['distance_matrix'] = create_distance_matrix(customers)
    data['num_vehicles'] = vehicle_count
    data['depot'] = customers[0].index
    data["demands"] = [
        customer.demand for customer in customers
    ]
    data["vehicle_capacities"] = [vehicle_capacity] * vehicle_count
    return data


def read_data(input_data):
    # parse the input
    lines = input_data.split("\n")

    parts = lines[0].split()
    customer_count = int(parts[0])
    vehicle_count = int(parts[1])
    vehicle_capacity = int(parts[2])

    customers = []
    for i in range(1, customer_count + 1):
        line = lines[i]
        parts = line.split()
        customers.append(
            Customer(
                i - 1,
                int(parts[0]),
                float(parts[1]),
                float(parts[2]),
            )
        )

    return customers, vehicle_count, vehicle_capacity

def format_solution(data, manager, routing, solution):
    """Prints solution on console."""
    print(f"Objective: {solution.ObjectiveValue()}")
    total_distance = 0
    routes = []
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        route = []
        route_distance = 0
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            total_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
        route.append(manager.IndexToNode(index))
        routes.append(route)

    outputData = '%.2f' % total_distance + ' ' + str(0) + '\n'
    for route in routes:
        outputData += ' '.join([str(node_id) for node_id in route]) + '\n'
    return outputData


def main_solver(input_data):
    """Entry point of the program."""
    # Instantiate the data problem.
    customers, vehicle_count, vehicle_capacity = read_data(input_data)
    data = create_data_model(customers, vehicle_count, vehicle_capacity)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data["demands"][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data["vehicle_capacities"],  # vehicle maximum capacities
        True,  # start cumul to zero
        "Capacity",
    )
    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        sol = format_solution(data, manager, routing, solution)
        return sol
    else:
        print("No solution found !")
        return None
