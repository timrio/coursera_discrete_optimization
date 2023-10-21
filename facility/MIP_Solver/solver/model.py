from MIP_Solver.solver.constraint import Constraint
from MIP_Solver.solver.variable import Variable

class Model:
    def __init__(self, name):
        self.name = name
        self.variables = {}
        self.M = 10**10


    def define_variables(self, situation):
        self.variables = {}
        facilities, customers = situation[0], situation[1]
        for f in range(len(facilities)):
            self.variables[f"open_{f}"] = Variable(f"open_{f}", True, -1) # initial value
            for c in range(len(customers)):
                self.variables[f"served_by_{f}x{c}"] = Variable(f"served_by_{f}x{c}", True, -1) # initial value
        return

    def define_constraints(self, situation):
        self.constraints = {}
        facilities, customers = situation[0], situation[1]

        # max cap constraints
        for facility_index, facilty in enumerate(facilities):
            coeffs = [c[1] for c in customers]
            variables = [self.variables[f"served_by_{facility_index}x{j}"] for j in range(len(customers))]
            self.constraints[f"max_cap_{facility_index}"] = Constraint(f"max_cap_{facility_index}","<=", coeffs, variables, facilty[2])


        # max one facility per customer
        for customer_index, customer in enumerate(customers):
            coeffs = [1 for f in facilities]
            variables = [self.variables[f"served_by_{j}x{customer_index}"] for j in range(len(facilities))]
            self.constraints[f"one_facility_per_customer_{customer_index}"] = Constraint(f"one_facility_per_customer_{customer_index}","==", coeffs, variables, 1)

        # is open definition
        for facility_index, facilty in enumerate(facilities):
            coeffs = [self.M]
            coeffs += [-1 for c in customers]
            variables = [self.variables[f"open_{facility_index}"]]
            variables += [self.variables[f"served_by_{facility_index}x{j}"] for j in range(len(customers))]
            self.constraints[f"is_open_definition_{facility_index}"] = Constraint(f"is_open_definition_{facility_index}","<=", coeffs, variables, 0)

        return 

    def add_slack_variables(self):
        slack_counter = 0
        for name, constraint in self.constraints.items():
            if constraint.constraint_type != "==":
                # create slack variable
                self.variables[f"slack_{slack_counter}"] = Variable(f"slack_{slack_counter}", False, -1)
                # update constraint to take into account slack variable
                coeffs = constraint.left_coeffs
                left_variables = constraint.left_variables
                coeffs.append(1)
                left_variables.append(self.variables[f"slack_{slack_counter}"])
                right_member = constraint.right_member
                self.constraints[name] = Constraint(name,"==",coeffs, left_variables, right_member)
                slack_counter+=1
        return


    def create_matrix_notation(self):
        X = [variable for variable in self.variables]
        X.sort()

        constraints_ordered = [constraint for constraint in self.constraints]
        constraints_ordered.sort()
        
        A = [[self.constraints[constraint].coeff_var_dict[x] for x in X] for constraint in constraints_ordered]
        b = [self.constraints[constraint].right_member for constraint in constraints_ordered]

        return A,X,b

