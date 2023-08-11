import numpy as np
from copy import deepcopy

# input format node_count, edge_count, edges


def build_model_input(node_count, edge_count, edges, initial_number_of_color):
    """Build model main objects."""
    # allowed colors: same for each node
    color_set = {i for i in range(initial_number_of_color)}

    # initialiize allowed color for each node (index by node id)
    node_domain = [color_set for i in range(node_count)]

    assigned_nodes = [False for i in range(node_count)]

    # neighbor dict
    neighbor_list = [[] for i in range(node_count)]
    for a, b in edges:
        neighbor_list[b].append(a)
        neighbor_list[a].append(b)

    # set of color used
    color_used = set([])

    return node_domain, neighbor_list, assigned_nodes, color_used


def assigned_color_if_domain_is_singleton(
    node_domain, neighbor_list, assigned_nodes, color_used
):
    """If a domain is reduced to a single color, we assign it to the node, and return wether such move was possuble."""
    has_made_an_update = False
    for node_idx, domain in enumerate(node_domain):
        if len(domain) == 1 and not assigned_nodes[node_idx]:
            assigned_nodes[node_idx] = True
            color_used.update(domain)
            has_made_an_update = True
    return node_domain, neighbor_list, assigned_nodes, color_used, has_made_an_update


def update_domain(node_domain, neighbor_list, assigned_nodes, color_used):
    for node_idx, domain in enumerate(node_domain):
        for neighbor in neighbor_list[node_idx]:
            if assigned_nodes[neighbor]:
                domain = domain - node_domain[neighbor]
                node_domain[node_idx] = domain

    return node_domain, neighbor_list, assigned_nodes, color_used


def check_status(node_domain, neighbor_list, assigned_nodes, color_used):
    feasible = True
    for node_idx, domain in enumerate(node_domain):
        if len(domain) == 0:
            return False, False
    return sum(assigned_nodes)==len(assigned_nodes), feasible


def propagate_constraint(node_domain, neighbor_list, assigned_nodes, color_used):
    has_made_an_update = True
    while has_made_an_update:
        # update domains
        (
            node_domain,
            neighbor_list,
            assigned_nodes,
            color_used,
        ) = update_domain(node_domain, neighbor_list, assigned_nodes, color_used)

        # try to assign a color
        (
            node_domain,
            neighbor_list,
            assigned_nodes,
            color_used,
            has_made_an_update,
        ) = assigned_color_if_domain_is_singleton(
            node_domain, neighbor_list, assigned_nodes, color_used
        )

    # check problem status
    all_updated, feasible = check_status(
        node_domain, neighbor_list, assigned_nodes, color_used
    )

    return all_updated, feasible, node_domain, neighbor_list, assigned_nodes, color_used


def choose_next_node(node_domain, neighbor_list, assigned_nodes):
    # start with simple iterator
    number_of_neighbor_list = []
    for node_idx in range(len(node_domain)):
        number_of_neighbor_list.append(
            (node_idx, len(neighbor_list[node_idx]), len(node_domain[node_idx]))
        )
    number_of_neighbor_list.sort(key=lambda a: (a[1], a[2]), reverse=True)
    for node_idx, _, _ in number_of_neighbor_list:
        if not assigned_nodes[node_idx]:
            yield node_idx, node_domain[node_idx]


def choose_next_color(node_domain, color_used):
    # start with simple iterator
    for color in node_domain & color_used:
        yield color
    for color in node_domain - color_used:
        yield color


def run_recursion(node_domain, neighbor_list, assigned_nodes, color_used):
    all_updated = False
    for node_idx, domain in choose_next_node(
        node_domain, neighbor_list, assigned_nodes
    ):
        for color in choose_next_color(domain, color_used):
            # copy input for recursion
            sub_node_domain = deepcopy(node_domain)
            sub_assigned_nodes = deepcopy(assigned_nodes)
            sub_color_used = deepcopy(color_used)

            # assign color
            sub_assigned_nodes[node_idx] = True
            sub_node_domain[node_idx] = {color}
            sub_color_used.add(color)

            # propagate constraint
            (
                all_updated,
                feasible,
                sub_node_domain,
                neighbor_list,
                sub_assigned_nodes,
                sub_color_used,
            ) = propagate_constraint(
                sub_node_domain, neighbor_list, sub_assigned_nodes, sub_color_used
            )

            if all_updated and feasible:
                return (
                    sub_node_domain,
                    neighbor_list,
                    sub_assigned_nodes,
                    sub_color_used,
                )

            if feasible:
                return run_recursion(
                    sub_node_domain, neighbor_list, sub_assigned_nodes, sub_color_used
                )


def run_home_made_solver(node_count, edge_count, edges):
    for initial_number_of_color in range(4, node_count + 1):
        node_domain, neighbor_list, assigned_nodes, color_used = build_model_input(
            node_count, edge_count, edges, initial_number_of_color
        )

        solution = run_recursion(node_domain, neighbor_list, assigned_nodes, color_used)

        if solution is not None:
            # a solution has been found
            node_domain = solution[0]
            color_used = solution[-1]
            break

    # get number of color used
    number_of_color_used = len(color_used)

    # get node assignement
    node_value = [0] * node_count
    for node_idx, domain in enumerate(node_domain):
        node_value[node_idx] = domain.pop()
    return int(number_of_color_used), node_value
