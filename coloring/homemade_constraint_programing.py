import numpy as np
from copy import deepcopy
import random
import sys

sys.setrecursionlimit(10000)

# input format node_count, edge_count, edges


def build_model_input(node_count, edge_count, edges, initial_number_of_color, strat):
    """Build model main objects."""

    # initialiize allowed color for each node (index by node id)
    node_domain = [
        {i for i in range(initial_number_of_color)} for i in range(node_count)
    ]

    assigned_nodes = [False for i in range(node_count)]

    # neighbor dict
    global neighbor_list
    neighbor_list = [[] for i in range(node_count)]
    for a, b in edges:
        neighbor_list[b].append(a)
        neighbor_list[a].append(b)

    # assign first node
    first_node_idx = next(strat(node_domain, assigned_nodes))

    assigned_nodes[first_node_idx] = True
    node_domain[first_node_idx] = {0}
    for neighbor in neighbor_list[first_node_idx]:
        node_domain[neighbor] = node_domain[neighbor] - {0}

    return node_domain, neighbor_list, assigned_nodes


def propagate_constraint(node_domain, assigned_nodes):
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
        assigned_nodes,
    )


def is_feasible(node_domain, assigned_nodes):
    for node_idx, domain in enumerate(node_domain):
        if len(domain) == 0:
            return False
    return True


def choose_next_node_color_min(node_domain, assigned_nodes):
    number_of_neighbor_list = []
    for node_idx in range(len(node_domain)):
        number_of_neighbor_list.append(
            (
                node_idx,
                -len(neighbor_list[node_idx]),
                len(node_domain[node_idx]),
            )
        )
    number_of_neighbor_list.sort(key=lambda a: (a[2], a[1]))
    for node_idx, _, _ in number_of_neighbor_list:
        if not assigned_nodes[node_idx]:
            yield node_idx


def choose_next_node_color_max(node_domain, assigned_nodes):
    number_of_neighbor_list = []
    for node_idx in range(len(node_domain)):
        number_of_neighbor_list.append(
            (
                node_idx,
                len(neighbor_list[node_idx]),
                len(node_domain[node_idx]),
            )
        )
    number_of_neighbor_list.sort(key=lambda a: (a[2], a[1]), reverse=True)
    for node_idx, _, _ in number_of_neighbor_list:
        if not assigned_nodes[node_idx]:
            yield node_idx


def choose_next_node_domain_global_min(node_domain, assigned_nodes):
    number_of_neighbor_list = []
    for node_idx in range(len(node_domain)):
        number_of_neighbor_list.append(
            (
                node_idx,
                len(neighbor_list[node_idx]),
                len(node_domain[node_idx]),
            )
        )
    number_of_neighbor_list.sort(key=lambda a: (a[1], a[2]))
    for node_idx, _, _ in number_of_neighbor_list:
        if not assigned_nodes[node_idx]:
            yield node_idx


def choose_next_node_domain_global_max(node_domain, assigned_nodes):
    number_of_neighbor_list = []
    for node_idx in range(len(node_domain)):
        number_of_neighbor_list.append(
            (
                node_idx,
                len(neighbor_list[node_idx]),
                -len(node_domain[node_idx]),
            )
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


def compute_number_of_color(node_domain, assigned_nodes):
    color_set = set([])
    for node_idx, dom in enumerate(node_domain):
        if assigned_nodes[node_idx]:
            color_set.update(dom)
    return len(color_set)


def run_recursion(node_domain, assigned_nodes, strat):
    # propagate constraint

    if sum(assigned_nodes) == len(assigned_nodes):
        return node_domain, assigned_nodes

    # check if feasible
    if not is_feasible(node_domain, assigned_nodes):
        return False

    for node_idx in strat(node_domain, assigned_nodes):
        color_domain = list(node_domain[node_idx])
        for color in color_domain:
            # copy for recursion
            sub_node_domain = deepcopy(node_domain)
            sub_assigned_nodes = deepcopy(assigned_nodes)

            sub_node_domain[node_idx] = set([color])

            sub_node_domain, sub_assigned_nodes = propagate_constraint(
                sub_node_domain, sub_assigned_nodes
            )

            next_sol = run_recursion(sub_node_domain, sub_assigned_nodes, strat)

            if next_sol:
                return next_sol
        return False


def chose_next_node_color(node_domain, assigned_nodes, color):
    for i, dom in enumerate(node_domain):
        if color in dom and not assigned_nodes[i]:
            yield i


def run_home_made_solver(node_count, edge_count, edges):
    min_number_of_color = node_count
    best_sol = None

    for i, strat in enumerate(
        [
            choose_next_node_domain_global_max,
            choose_next_node_domain_global_min,
            choose_next_node_color_max,
            choose_next_node_color_min,
        ]
    ):
        print(f"strategy:{i} tried")
        node_domain, neighbor_list, assigned_nodes = build_model_input(
            node_count, edge_count, edges, min_number_of_color, strat
        )
        solution = run_recursion(node_domain, assigned_nodes, strat)

        node_domain = solution[0]
        assigned_nodes = solution[1]

        number_of_color = int(compute_number_of_color(node_domain, assigned_nodes))

        if number_of_color < min_number_of_color:
            print(f"new_min:{number_of_color}")
            min_number_of_color = number_of_color
            best_sol = deepcopy(solution)
    # get node assignement
    node_domain = best_sol[0]
    assigned_nodes = best_sol[1]
    node_value = [0] * node_count
    for node_idx, domain in enumerate(node_domain):
        node_value[node_idx] = domain.pop()

    return min_number_of_color, node_value
