def lin_kernighan_heuristic(
    route, distance_matrix, initial_temperature, cool_down=0.999, max_iterations=1000
):
    n = len(route)
    best_route = route.copy()
    best_distance = get_route_length(best_route, distance_matrix)
    temperature = initial_temperature

    for _ in tqdm(range(max_iterations)):
        improved = False
        suffle_nodes = [i for i in range(n)]
        random.shuffle(suffle_nodes)
        for route_node_index in suffle_nodes:
            running_route, running_distance = k_opt(
                best_route,
                route_node_index,
                distance_matrix,
            )
            if running_distance < best_distance or np.random.random() < np.exp(
                (best_distance - running_distance) / temperature
            ):
                best_distance = running_distance
                best_route = running_route.copy()
                improved = True

        temperature *= cool_down

        if not improved:
            break

    return best_distance, best_route


def k_opt(route, route_node_index, distance_matrix):
    keep_going = True
    route_node = route[route_node_index]
    best_new_route = route.copy()
    running_route = route.copy()
    min_delta = np.inf
    iteration = 0

    successor_index = route_node_index + 1
    if route_node_index == len(route) - 1:
        successor_index = 0

    forbiden_swap = set()

    while keep_going and len(forbiden_swap) < 3:
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
                pred_index = i - 1 if i != 0 else -1
                if (
                    min(running_route[pred_index], running_route[successor_index]),
                    max(running_route[pred_index], running_route[successor_index]),
                ) in forbiden_swap:
                    continue
                keep_going = True
                new_successor = k
                new_successor_index = i
                break

        if keep_going:
            pred_index = new_successor_index - 1 if new_successor_index != 0 else -1

            # compute delta
            delta = -(
                distance_matrix[
                    running_route[route_node_index], running_route[successor_index]
                ]
                + distance_matrix[
                    running_route[new_successor_index], running_route[pred_index]
                ]
            ) + (
                distance_matrix[running_route[route_node], running_route[pred_index]]
                + distance_matrix[
                    running_route[successor_index], running_route[new_successor_index]
                ]
            )

            # update route
            forbiden_swap.add(
                (
                    min(running_route[pred_index], running_route[successor_index]),
                    max(running_route[pred_index], running_route[successor_index]),
                )
            )
            if new_successor_index > successor_index:
                running_route[successor_index:new_successor_index] = running_route[
                    successor_index:new_successor_index
                ][::-1]
            else:
                running_route[successor_index:] = running_route[successor_index:][::-1]

            if delta < min_delta:
                min_delta = delta
                best_new_route = running_route.copy()

    return best_new_route, get_route_length(best_new_route, distance_matrix)