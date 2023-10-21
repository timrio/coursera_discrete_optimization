import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

# Define your utility functions (get_distance_matrix, get_route_length, etc.)


def get_initial_solution(points):
    n = len(points)
    route = np.arange(n)
    np.random.shuffle(route)
    return route


def random_neighbour(route):
    i, j = np.random.choice(len(route), size=2, replace=False)
    new_route = route.copy()
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route


# define utils functions
def euclidian_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def get_route_length(route, distance_matrix):
    length = 0
    for i in range(len(route) - 1):
        length += distance_matrix[route[i]][route[i + 1]]
    length += distance_matrix[route[-1]][route[0]]
    return length


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


def apply_2_opt(route, i, j):
    new_route = route.copy()
    new_route[i : j + 1] = new_route[i : j + 1][::-1]
    return new_route


import numpy as np

# Define your utility functions (get_distance_matrix, get_route_length, etc.)


def get_initial_solution(points, distance_matrix):
    n = len(points)
    route = np.arange(n)
    np.random.shuffle(route)
    return route


def metropolis_criteria(delta, temperature):
    if delta < 0:
        return True
    return np.random.random() <= np.exp(-delta / temperature)


def lin_kernighan_heuristic(
    route, distance_matrix, temperature, cool_down_factor=0.999, max_iterations=1000
):
    n = len(route)
    best_route = route.copy()
    best_distance = get_route_length(best_route, distance_matrix)

    last_change = 0
    last_improvement = 0

    for counter in tqdm(range(max_iterations), position=1):
        for i in range(n - 2):
            current_route = best_route
            current_distance = best_distance
            for j in range(i + 2, n):
                delta = -(
                    distance_matrix[current_route[i], current_route[i + 1]]
                    + distance_matrix[current_route[j], current_route[(j + 1) % n]]
                ) + (
                    distance_matrix[current_route[i], current_route[j]]
                    + distance_matrix[current_route[i + 1], current_route[(j + 1) % n]]
                )
                if metropolis_criteria(delta, temperature):
                    current_route = apply_2_opt(current_route, i + 1, j)
                    current_distance = get_route_length(current_route, distance_matrix)
                    time_since_last_change = 0

                    if current_distance < best_distance:
                        best_route = current_route
                        best_distance = current_distance
                        last_improvement = counter
                else:
                    last_change = counter

        temperature = (
            temperature * cool_down_factor if temperature >= 1 else temperature
        )

        if (counter - last_change) == 100:
            break
        if (counter - last_improvement) == 200:
            current_route = best_route
            current_distance = best_distance

    return best_route, best_distance, temperature


def simulated_annealing_with_lin_kernighan(
    points, num_restarts=10, num_reheats=5, initial_temperature=None
):
    distance_matrix = get_distance_matrix(points)

    if initial_temperature is None:
        mean_edge_length = np.mean(distance_matrix)
        initial_temperature = mean_edge_length

    best_solution = None
    best_distance = float("inf")

    for _ in range(num_restarts):
        initial_temperature = 100
        current_solution = get_initial_solution(points, distance_matrix)
        for _ in range(num_reheats):
            current_solution, current_distance, temperature = lin_kernighan_heuristic(
                current_solution, distance_matrix, initial_temperature
            )
            if current_distance < best_distance:
                best_solution = current_solution
                best_distance = current_distance

            initial_temperature = temperature * (1.5 ** (num_reheats - 1))

    visualize_route(best_solution, points)
    return best_distance, best_solution
