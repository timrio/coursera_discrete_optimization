import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm


# define utils functions
def euclidian_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def get_distance_matrix(points):
    n = len(points)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i][j] = euclidian_distance(
                points[i][0], points[i][1], points[j][0], points[j][1]
            )
    return distance_matrix


def get_route_length(route, distance_matrix):
    length = 0
    for i in range(len(route) - 1):
        length += distance_matrix[route[i]][route[i + 1]]
    length += distance_matrix[route[-1]][route[0]]
    return length


def get_random_route(points):
    random_route = np.arange(len(points))
    np.random.shuffle(random_route)
    return random_route


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


def visualize_route(route, points):
    plt.figure(figsize=(10, 10))
    plt.scatter([x[0] for x in points], [x[1] for x in points], c="red")
    for i in range(len(route) - 1):
        plt.plot(
            [points[route[i]][0], points[route[i + 1]][0]],
            [points[route[i]][1], points[route[i + 1]][1]],
            c="blue",
        )
    plt.plot(
        [points[route[-1]][0], points[route[0]][0]],
        [points[route[-1]][1], points[route[0]][1]],
        c="blue",
    )
    plt.show()


# 2 opt tabu search
def initialize_tabu_list(node_count, tabu_duration=100):
    tabu_list = {}
    for i in range(node_count - 1):
        for j in range(i + 1, node_count):
            tabu_list[(i, j)] = 0
    return tabu_list


def get_neighbourhood(route):
    neighbourhood = {}
    for i in range(len(route) - 1):
        for j in range(i + 1, len(route)):
            neighbour = best_route[:i] + best_route[i:j][::-1] + best_route[j:]
            neighbourhood[(i, j)] = neighbour
    return neighbourhood


def get_neighbourhood_optimized(route, distance_matrix, max_candidates=50):
    n = len(route)
    neighbourhood = []

    # Calculate the current route length
    current_distance = get_route_length(route, distance_matrix)

    for i in range(n - 1):
        for j in range(i + 1, n):
            new_route = route.copy()
            new_route[i], new_route[j] = new_route[j], new_route[i]

            # Calculate the improvement in distance
            new_distance = current_distance - (
                distance_matrix[route[i], route[j]]
                - distance_matrix[new_route[i], new_route[j]]
                + distance_matrix[new_route[i], route[i]]
                + distance_matrix[new_route[j], route[j]]
            )

            neighbourhood.append((new_distance, (i, j), new_route))

    # Sort and select the most promising candidates
    neighbourhood.sort()

    result = {}
    num_candidates = min(max_candidates, len(neighbourhood))
    for _, (i, j), new_route in neighbourhood[:num_candidates]:
        result[(i, j)] = new_route

    return result


def k_opt(route, route_node_index, distance_matrix, tabu_list, tabu_duration):
    keep_going = True
    best_new_distance = get_route_length(route, distance_matrix)
    best_new_route = route.copy()
    running_route = route.copy()
    max_iter = 5
    iteration = 0
    while keep_going:
        keep_going = False
        successor_index = route_node_index + 1
        if route_node_index == len(route) - 1:
            successor_index = 0

        route_node = running_route[route_node_index]
        successor_node = running_route[successor_index]

        distance_route_successor = distance_matrix[route_node][successor_node]

        new_successor = None
        new_successor_index = None
        best_new_distance_sucessor = np.inf
        successors_list = []
        for i, k in enumerate(route):
            if (
                i != successor_index
                and i != route_node_index
                and i != successor_index + 1
            ):
                if counter < tabu_list[(min(route_node, k), max(route_node, k))]:
                    continue

                new_distance_sucessor = distance_matrix[successor_node][k]
                if new_distance_sucessor < distance_route_successor:
                    successors_list.append((k, i))
                    keep_going = True
                    # if new_distance_sucessor < best_new_distance_sucessor:
                    #     best_new_distance_sucessor = new_distance_sucessor
                    #     new_successor = k
                    #     new_successor_index = i
                    #     keep_going = True

        if keep_going:
            iteration += 1
            random_index = np.random.choice(len(successors_list))
            new_successor, new_successor_index = successors_list[random_index]

            new_route = running_route.copy()
            if new_successor_index >= successor_index:
                new_route[successor_index:new_successor_index] = running_route[
                    successor_index:new_successor_index
                ][::-1]
            else:
                new_route[successor_index:] = running_route[successor_index:][::-1]

            new_distance = get_route_length(new_route, distance_matrix)
            running_route = new_route.copy()
            tabu_list[
                (min(route_node, new_successor), max(route_node, new_successor))
            ] = (counter + tabu_duration)

            if new_distance <= best_new_distance:
                # final_swap = swap_list.copy()
                best_new_distance = new_distance
                best_new_route = new_route.copy()

    return best_new_route, get_route_length(best_new_route, distance_matrix)


def solve_k_opt(node_count, points, tabu_duration=50, max_iterations=10000):
    global counter
    counter = 0

    # inializa values
    distance_matrix = get_distance_matrix(points)
    best_route = get_first_route_greedy(points, distance_matrix)
    # best_route = get_random_route(points)
    best_distance = get_route_length(best_route, distance_matrix)
    tabu_list = initialize_tabu_list(node_count, tabu_duration)

    # running route
    running_route = best_route.copy()
    running_distance = best_distance

    # main loop
    for counter in tqdm(range(max_iterations)):
        for route_node_index in range(node_count):
            running_route, running_distance = k_opt(
                running_route,
                route_node_index,
                distance_matrix,
                tabu_list,
                tabu_duration,
            )
            if running_distance < best_distance:
                print(f"best: {best_distance}")
                best_distance = running_distance
                best_route = running_route.copy()
    visualize_route(best_route, points)
    return best_distance, best_route


def solve_2_opt(node_count, points, tabu_duration=50, max_iterations=10000):
    global counter
    counter = 0

    # inializa values
    distance_matrix = get_distance_matrix(points)
    # best_route = get_first_route_greedy(points, distance_matrix)
    best_route = get_first_route_greedy(points, distance_matrix)
    best_distance = get_route_length(best_route, distance_matrix)
    tabu_list = initialize_tabu_list(node_count, tabu_duration)

    # running route
    running_route = best_route.copy()
    running_distance = best_distance

    # main loop
    for counter in tqdm(range(max_iterations)):
        neighbourhood = get_neighbourhood(running_route, distance_matrix)
        best_neighbour_distance = np.inf
        best_move = None
        for move, neighbour in neighbourhood.items():
            if counter >= tabu_list[move]:
                neighbour_distance = get_route_length(neighbour, distance_matrix)
                if neighbour_distance < best_neighbour_distance:
                    best_neighbour_distance = neighbour_distance
                    best_neighbour = neighbour
                    best_move = move
        if best_move:
            running_route = best_neighbour
            running_distance = best_neighbour_distance
            tabu_list[best_move] = counter + tabu_duration

        if running_distance < best_distance:
            best_route = running_route
            best_distance = running_distance
            distance_to_last_update = 0
        else:
            distance_to_last_update += 1

        # if counter % 200 == 0:
        # visualize_route(best_route, points)

    return best_distance, best_route
