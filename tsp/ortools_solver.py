"""Simple Travelling Salesperson Problem (TSP) between cities."""

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
from tqdm import tqdm


# define utils functions
def euclidian_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def get_distance_matrix(points):
    n = len(points)
    distance_matrix = np.zeros((n, n))
    for i in tqdm(range(n)):
        for j in range(i + 1, n):
            distance = euclidian_distance(
                points[i][0], points[i][1], points[j][0], points[j][1]
            )
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance
    return distance_matrix


def get_route_length(route, distance_matrix):
    length = 0
    for i in range(len(route) - 1):
        length += distance_matrix[route[i]][route[i + 1]]
    length += distance_matrix[route[-1]][route[0]]
    return length


def create_data_model(nodeCount, points):
    """Stores the data for the problem."""
    data = {}
    data["distance_matrix"] = get_distance_matrix(points)
    data["num_vehicles"] = 1
    data["depot"] = 0
    return data


def process_solution(manager, routing, solution):
    """Prints solution on console."""

    distance = solution.ObjectiveValue()

    index = routing.Start(0)
    route = []

    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
    return distance, route


def get_first_route_greedy(points, distance_matrix):
    route = [0]
    while len(route) < len(points):
        min_dist = np.inf
        min_point = None
        for running_point in range(len(points)):
            if running_point not in route:
                dist = distance_matrix[route[-1]][running_point]
                if dist < min_dist:
                    min_dist = dist
                    min_point = running_point
        if min_point:
            route.append(min_point)
    return route


def solve(nodeCount, points):
    """Entry point of the program."""
    print(nodeCount)
    distance_matrix = get_distance_matrix(points)
    # if nodeCount > 30000:
    #     print("greedy")
    #     route = get_first_route_greedy(points, distance_matrix)
    #     return get_route_length(route, distance_matrix), route
    # Instantiate the data problem.
    data = create_data_model(nodeCount, points)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = 60

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        dist, route = process_solution(manager, routing, solution)

    return get_route_length(route, distance_matrix), route
