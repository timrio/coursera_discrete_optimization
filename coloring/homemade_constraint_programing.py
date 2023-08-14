import numpy as np
from copy import deepcopy

# input format node_count, edge_count, edges


def build_model_input(node_count, edge_count, edges, initial_number_of_color):
    """Build model main objects."""

    # initialiize allowed color for each node (index by node id)
    node_domain = [
        {i for i in range(initial_number_of_color)} for i in range(node_count)
    ]

    assigned_nodes = [False for i in range(node_count)]

    # neighbor dict
    neighbor_list = [[] for i in range(node_count)]
    for a, b in edges:
        neighbor_list[b].append(a)
        neighbor_list[a].append(b)

    return node_domain, neighbor_list, assigned_nodes


def propagate_constraint(node_domain, neighbor_list, assigned_nodes):
    has_made_an_update = True
    while has_made_an_update:
        has_made_an_update = False
        for node_idx in range(len(node_domain)):
            if not assigned_nodes[node_idx]:
                domain = node_domain[node_idx]
                if len(domain) == 1:
                    assigned_nodes[node_idx] = True
                    has_made_an_update = True
                    for neighbor in neighbor_list[node_idx]:
                        node_domain[neighbor] = node_domain[neighbor] - domain

    return (
        node_domain,
        neighbor_list,
        assigned_nodes,
    )


def is_feasible(node_domain, neighbor_list, assigned_nodes):
    for node_idx, domain in enumerate(node_domain):
        if len(domain) == 0:
            return False
    return True


def choose_next_node(node_domain, neighbor_list, assigned_nodes):
    number_of_neighbor_list = []
    for node_idx in range(len(node_domain)):
        number_of_neighbor_list.append(
            (node_idx, len(neighbor_list[node_idx]), -len(node_domain[node_idx]))
        )
    number_of_neighbor_list.sort(key=lambda a: (a[1], a[2]), reverse=True)
    for node_idx, _, _ in number_of_neighbor_list:
        if not assigned_nodes[node_idx]:
            yield node_idx


def choose_next_color(node_domain):
    # start with simple iterator
    selected_node_domain = list(node_domain)
    selected_node_domain.sort()
    for color in selected_node_domain:
        yield color


def run_recursion(node_domain, neighbor_list, assigned_nodes):
    # propagate constraint
    node_domain, neighbor_list, assigned_nodes = propagate_constraint(
        node_domain, neighbor_list, assigned_nodes
    )

    # check if feasible
    if not is_feasible(node_domain, neighbor_list, assigned_nodes):
        return False

    if sum(assigned_nodes) == len(assigned_nodes):
        return node_domain, neighbor_list, assigned_nodes

    for node_idx in choose_next_node(node_domain, neighbor_list, assigned_nodes):
        color_domain = set(node_domain[node_idx])
        for color in color_domain:
            # copy for recursion
            sub_node_domain = deepcopy(node_domain)
            sub_assigned_nodes = deepcopy(assigned_nodes)

            sub_node_domain[node_idx] = set([color])

            next_sol = run_recursion(
                sub_node_domain,
                neighbor_list,
                sub_assigned_nodes,
            )
            if next_sol:
                return next_sol

            # if not feasible we backtrack
            assigned_nodes[node_idx] = False
            node_domain[node_idx] = color_domain


def run_home_made_solver(node_count, edge_count, edges):
    for initial_number_of_color in range(4, node_count + 1):
        node_domain, neighbor_list, assigned_nodes = build_model_input(
            node_count, edge_count, edges, initial_number_of_color
        )

        solution = run_recursion(node_domain, neighbor_list, assigned_nodes)

        if solution:
            # a solution has been found
            node_domain = solution[0]
            break

    # get node assignement
    node_value = [0] * node_count
    for node_idx, domain in enumerate(node_domain):
        node_value[node_idx] = domain.pop()
    return int(initial_number_of_color), node_value
