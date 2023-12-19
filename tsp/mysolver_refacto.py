import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import random


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
            for k in range(j + 1, node_count):
                tabu_list[(i, j, k)] = 0
    return tabu_list


def k_opt(route, route_node_index, distance_matrix, max_iter=5):
    keep_going = True
    route_node = route[route_node_index]
    best_new_route = route.copy()
    running_route = route.copy()
    min_delta = np.inf
    iteration = 0

    successor_index = route_node_index + 1
    if route_node_index == len(route) - 1:
        successor_index = 0

    while keep_going and iteration <= max_iter:
        keep_going = False
        iteration += 1

        # define successor
        successor_node = running_route[successor_index]

        # compute distance to successor
        distance_route_successor = distance_matrix[route_node][successor_node]

        new_successor = None
        new_successor_index = None
        best_new_distance_sucessor = np.inf

        # create a list of index and node and suffle it
        shuffled_list = [
            (i, k)
            for i, k in enumerate(running_route)
            if (
                i != successor_index
                and i != route_node_index
                and i != successor_index + 1
            )
        ]
        random.shuffle(shuffled_list)
        for i, k in shuffled_list:
            new_distance_sucessor = distance_matrix[successor_node][k]

            if new_distance_sucessor < distance_route_successor:
                keep_going = True
                new_successor = k
                new_successor_index = i
                break

        if keep_going:
            if new_successor_index > successor_index:
                running_route[successor_index:new_successor_index] = running_route[
                    successor_index:new_successor_index
                ][::-1]
            else:
                running_route[successor_index:] = running_route[successor_index:][::-1]

            new_successor_pred = running_route[
                new_successor_index - 1 if new_successor_index != 0 else -1
            ]

            delta = -(
                distance_matrix[successor_node][new_successor]
                + distance_matrix[route_node][new_successor_pred]
                - distance_matrix[route_node][successor_node]
                - distance_matrix[new_successor_pred][successor_node]
            )

            if delta < min_delta and delta < 0:
                min_delta = delta
                best_new_route = running_route.copy()

    return best_new_route, get_route_length(best_new_route, distance_matrix)


def solve_k_opt(node_count, points, distance_matrix, tabu_duration=50, max_iterations=10000):
    global counter
    counter = 0

    # best_route = get_first_route_greedy(points, distance_matrix)
    best_route = get_random_route(points)
    best_distance = get_route_length(best_route, distance_matrix)

    # running route
    running_route = best_route.copy()
    running_distance = best_distance

    distance_to_last_update = 0

    # main loop
    for counter in tqdm(range(max_iterations)):
        suffle_nodes = [i for i in range(node_count)]
        random.shuffle(suffle_nodes)
        for route_node_index in suffle_nodes:
            running_route, running_distance = k_opt(
                best_route,
                route_node_index,
                distance_matrix,
            )
            if running_distance < best_distance:
                distance_to_last_update = 0
                print(f"best: {best_distance}")
                best_distance = running_distance
                best_route = running_route.copy()

        distance_to_last_update += 1

        if distance_to_last_update == 50:
            best_route = best_route[::-1]

        if distance_to_last_update == 100:
            return best_distance, best_route

    return best_distance, best_route


def iterative_k_opt(node_count, points, number_of_k_opt=8):
    sol = None
    dist = np.inf
    # inializa values
    distance_matrix = get_distance_matrix(points)
    for i in range(number_of_k_opt):
        best_distance, best_route = solve_k_opt(node_count, points, distance_matrix)
        if best_distance < dist:
            dist = best_distance
            sol = best_route.copy()

    visualize_route(sol, points)
    return dist, sol
