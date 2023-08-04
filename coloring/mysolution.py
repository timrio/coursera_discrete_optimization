import numpy as np
from pyomo.environ import (
    ConcreteModel,
    Set,
    Binary,
    Var,
    Constraint,
    Model,
    Objective,
    minimize,
    maximize,
)
from pyomo.opt import SolverFactory


class Solver:
    def __init__(self, node_count, edge_count, edges):
        self.node_count = node_count
        self.edge_count = edge_count
        self.edges = edges
        self.model = ConcreteModel()

    def solve(self):
        number_of_color = 10
        self.model = ConcreteModel()

        # define and declaire domain
        domain = Domain(self.model, self.node_count, self.edge_count, self.edges)
        domain.declare_domain()

        # define and declaire variable
        variable = Variables(self.model)
        variable.declare_variables()

        # define big M param
        self.model.param_big_M = self.node_count

        # define and declaire constraints
        constraints = Constraints(self.model)
        constraints.declare_constraints()

        # define and declaire variable
        objective = Objectives(self.model)
        objective.declare_objective()

        # solve model
        instance = self.model.create_instance()
        opt = SolverFactory("cbc")
        results = opt.solve(instance)

        opt_value, solution = self.extract_solution(instance)

        return opt_value, solution

    def extract_solution(self, instance):
        solution = [0] * self.node_count
        sol = instance.var_color_node.extract_values()
        for (node, color), is_chosen in sol.items():
            if is_chosen:
                solution[node] = color

        return instance.objective.expr(), solution


class Domain:
    def __init__(self, model, node_count, edge_count, edges):
        self.model = model
        self.node_count = node_count
        self.edge_count = edge_count
        self.edges = edges

    def declare_domain(self):
        self.model.set_color_domain = Set(initialize=self.color_domain)
        self.model.set_node_domain = Set(initialize=self.node_domain)
        self.model.set_node_color_domain = Set(initialize=self.node_color_domain)
        self.model.set_neighbor_domain = Set(initialize=self.neighbor_domain)
        self.model.set_neighbor_color_domain = Set(
            initialize=self.neighbor_color_domain
        )
        self.model.set_all_successive_colors = Set(
            initialize=self.succesive_colors_domain
        )

    @property
    def color_domain(self):
        return set(np.arange(0, self.node_count, 1))

    @property
    def node_domain(self):
        return set(np.arange(0, self.node_count, 1))

    @property
    def node_color_domain(self):
        return set(
            (node, color) for node in self.node_domain for color in self.color_domain
        )

    @property
    def neighbor_domain(self):
        return set(self.edges)

    @property
    def neighbor_color_domain(self):
        return set(
            [
                (node1, node2, color)
                for (node1, node2) in self.edges
                for color in self.color_domain
            ]
        )

    @property
    def succesive_colors_domain(self):
        return set([(color, color + 1) for color in range(self.node_count - 1)])


class Variables:
    def __init__(self, model):
        self.model = model

    def declare_variables(self):
        self.model.var_color_used = Var(
            self.model.set_color_domain,
            domain=Binary,
        )
        self.model.var_color_node = Var(
            self.model.set_node_color_domain,
            domain=Binary,
        )


class Constraints:
    def __init__(self, model):
        self.model = model

    def declare_constraints(self):
        self.model.constraint_adjacency_color = Constraint(
            self.model.set_neighbor_color_domain,
            rule=self.declare_adjacency_color_constraint(),
        )
        self.model.constraint_one_color_per_node = Constraint(
            self.model.set_node_domain,
            rule=self.declare_one_color_per_node_constraint(),
        )
        self.model.constraint_define_color_used_upper = Constraint(
            self.model.set_color_domain,
            rule=self.declare_define_color_used_constraint_upper(),
        )

        self.model.constraint_define_color_used_lower = Constraint(
            self.model.set_color_domain,
            rule=self.declare_define_color_used_constraint_lower(),
        )

        self.model.constraint_symetry_breaking = Constraint(
            self.model.set_all_successive_colors,
            rule=self.declare_symetry_breaking_constraint(),
        )

    def declare_adjacency_color_constraint(self):
        def adjacency_color_constraint(model, node1, node2, color):
            return (
                model.var_color_node[node1, color] + model.var_color_node[node2, color]
                <= 1
            )

        return adjacency_color_constraint

    def declare_one_color_per_node_constraint(self):
        def define_one_color_per_node_constraint(model, node):
            return (
                sum(
                    [
                        model.var_color_node[node, color]
                        for color in model.set_color_domain
                    ]
                )
                == 1
            )

        return define_one_color_per_node_constraint

    def declare_define_color_used_constraint_upper(self):
        def define_color_used_constraint_upper(model, color):
            return (
                sum(
                    [
                        model.var_color_node[node, color]
                        for node in model.set_node_domain
                    ]
                )
                <= model.param_big_M * model.var_color_used[color]
            )

        return define_color_used_constraint_upper

    def declare_define_color_used_constraint_lower(self):
        def define_color_used_constraint_lower(model, color):
            return (
                sum(
                    [
                        model.var_color_node[node, color]
                        for node in model.set_node_domain
                    ]
                )
                >= model.var_color_used[color]
            )

        return define_color_used_constraint_lower

    def declare_symetry_breaking_constraint(self):
        def symetry_breaking_constraint(model, color1, color2):
            return model.var_color_used[color1] >= model.var_color_used[color2]

        return symetry_breaking_constraint


class Objectives:
    def __init__(self, model):
        self.model = model

    def declare_objective(self):
        self.model.objective = Objective(rule=self.set_objective(), sense=minimize)

    def set_objective(self):
        def minimize_color_used(model):
            return sum(
                [model.var_color_used[color] for color in model.set_color_domain]
            )

        return minimize_color_used
