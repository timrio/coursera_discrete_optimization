import numpy as np
from numpy.linalg import inv

class Model:

    def __init__(self, A, b , c):
        self.A = A
        self.b = b
        self.c = c
        self.basis_indexes = []
        self.non_basis_indexes = []
        self.initial_variable_indexes = [i for i in range(A.shape[1])]
        self.initial_constraints_indexes = [i for i in range(A.shape[0])]
        self.H = None
        self.current_optimal_value = None
        self.current_solution = None


    @property
    def number_of_variables(self): 
        return self.A.shape[1]

    @property
    def number_of_constraints(self):
        return self.A.shape[0]

    @property
    def current_solution_ordered(self):
        x = np.r_[np.array([0 for i in range(self.number_of_variables - self.number_of_constraints)]),self.current_solution]
        indexes = self.non_basis_indexes + self.basis_indexes
        x_ordered = [0 for i in range(self.number_of_variables)]
        for i, correct_index in enumerate(indexes):
            x_ordered[correct_index] = x[i]
        return x_ordered

    @property
    def solution_is_integer(self):
        for val in self.current_solution_ordered[:len(self.initial_variable_indexes)]:
            if np.round(val,0)!=val:
                return False
        return True


    def initiate_model_and_slack_variables(self):
        initial_number_of_constraints = len(self.initial_constraints_indexes)
        self.non_basis_indexes = [i for i in range(self.A.shape[1])]
        for i in range(initial_number_of_constraints):
            new_column = np.zeros(self.A.shape[0])
            new_column[i] = 1
            self.A = np.c_[self.A, new_column]
            self.c = np.r_[self.c, [0]]
        self.basis_indexes = list(set([i for i in range(self.A.shape[1])])-set(self.non_basis_indexes))
        return

        

    def simplex_iteration(self):   
        if any(self.b<0):           
            print("cannot find initial basis")
            return False

        keep_going = True

        while keep_going:
            A_b, A_n = self.A[:,self.basis_indexes], self.A[:,self.non_basis_indexes]
            c_b, c_n = self.c[self.basis_indexes], self.c[self.non_basis_indexes]

            if np.linalg.det(A_b) == 0:
                return 'matrice non inversible'

            A_b_inv = inv(A_b)
            pi = np.dot(c_b,A_b_inv) 
            self.current_solution = np.dot(A_b_inv,b)
            
            # express x basis according to xn
            # x_b = x_b_opt - H @ x_n
            self.H = np.dot(A_b_inv, A_n)

            # compute reduced costs
            reduced_cost = c_n - np.dot(pi, A_n)

            if (reduced_cost >= 0).all():
                keep_going = False

            else:
                # we need find entering and exiting variable
                entering_index = np.argmin(reduced_cost)
                ratio = (self.current_solution / self.H[:, entering_index])
                
                # Avoid division by zero in case of non-positive entries in H[:, entering_index]
                ratio[(ratio < 0)] = np.inf
                exiting_index = np.argmin([value if self.H[i, entering_index]>0 else np.inf for i,value in enumerate(ratio)])

                # permute entering and exiting values
                self.non_basis_indexes[entering_index], self.basis_indexes[exiting_index] = self.basis_indexes[exiting_index],self.non_basis_indexes[entering_index]

        # compute current optimal value 
        self.current_optimal_value = np.dot(pi,self.b)
        return

    def find_gomory_cuts(self):
        gomory_cuts = []
        for i, basic_value in enumerate(self.current_solution):
            # check if basic value is an int
            if np.round(basic_value,0)!=basic_value:
                new_cut = {}
                new_cut["b"] = -(basic_value - np.floor(basic_value))
                for j, val in enumerate(self.H[i,:]):
                    new_cut[self.non_basis_indexes[j]] = -(val - np.floor(val))
                gomory_cuts.append(new_cut)
        return gomory_cuts


    def add_gomory_cuts(self):
        gomory_cuts = self.find_gomory_cuts()
        if len(gomory_cuts)==0:
            return
        for cut in gomory_cuts:
            # add the constraint
            new_line = np.zeros(self.A.shape[1])
            new_b = None
            for k, v in cut.items():
                if k == "b":
                    new_b = v
                    continue
                new_line[k] = v


            self.A = np.r_[self.A, [new_line]]
            new_col = np.zeros(self.A.shape[0])
            new_col[-1] = 1
            self.A = np.c_[self.A, new_col]
            #update b
            self.b = np.append(self.b, new_b)
            # update c
            self.c = np.append(self.c, 0)
            self.basis_indexes.append(self.number_of_variables-1)

        # reconstruct A,b and c
        return 

    def dual_simplex_python(self):
        # initialize dual simplex 
        keep_going = True
        while keep_going:
            
            A_b, A_n = self.A[:,self.basis_indexes], self.A[:,self.non_basis_indexes]
            c_b, c_n = self.c[self.basis_indexes], self.c[self.non_basis_indexes]

            A_b_inv = inv(A_b)
            pi = np.dot(c_b,A_b_inv) 
            self.current_solution = np.dot(A_b_inv,self.b)
            
            # express x basis according to xn
            # x_b = x_b_opt - H @ x_n
            self.H = np.dot(A_b_inv, A_n)

            # compute reduced costs
            reduced_cost = c_n - np.dot(pi, A_n)

            if all(self.current_solution>=0):
                keep_going = False
            else:
                # find the variable that will leave, the basis, this is the one the first negative beta
                min_val = np.inf
                exiting_index = None
                for i in range(self.number_of_constraints):
                    if self.current_solution[i] < min_val and self.current_solution[i]<0:
                        exiting_index = i
                        min_val = self.current_solution[i]
                        break
                if all(val>=0 for val in self.H[exiting_index,:]):
                    print("model infeasible")
                    break
                else:
                    # we choose the variable that enters the basis
                    min_val = np.inf
                    entering_index = None
                    for i in range(self.number_of_variables-self.number_of_constraints):
                        if self.H[exiting_index,i] < 0 and reduced_cost[i]/np.abs(self.H[exiting_index,i]) < min_val:
                            min_val = reduced_cost[i]/np.abs(self.H[exiting_index,i]) 
                            entering_index = i 

                    # permute entering and exiting values
                    self.non_basis_indexes[entering_index], self.basis_indexes[exiting_index] = self.basis_indexes[exiting_index],self.non_basis_indexes[entering_index]
        # compute current optimal value 
        self.current_optimal_value = np.dot(pi,self.b)
        return

    def express_situation_according_to_initial_variables(self):
        temp_non_basis_indexes = self.initial_variable_indexes
        temp_basis_indexes = list(set([i for i in range(self.A.shape[1])])-set(temp_non_basis_indexes))

        A_b, A_n = self.A[:,temp_basis_indexes], self.A[:,temp_non_basis_indexes]
        c_b, c_n = self.c[temp_basis_indexes], self.c[temp_non_basis_indexes]

        A_b_inv = inv(A_b)

        ordinate = np.dot(A_b_inv,self.b)
        
        # express x basis according to xn
        # x_b = x_b_opt - H @ x_n
        H = np.dot(A_b_inv, A_n)
        
        return H, ordinate


    def solve(self):
        print("add slack variables")
        self.initiate_model_and_slack_variables()

        print("run simplex")
        self.simplex_iteration()
        print(f"current solution: {self.current_solution_ordered}")
        print(f"current optimal value: {self.current_optimal_value}")

        
        while not self.solution_is_integer:
            print("possible gomory cuts found, running dual simplex")
            self.add_gomory_cuts()
            self.dual_simplex_python()
            print(f"current solution: {self.current_solution_ordered}")
            print(f"current optimal value: {self.current_optimal_value}")

        print('optimal integer solution found')

        return self.current_solution_ordered, self.current_optimal_value


    def draw_situation(self):
        pass

###### define model and data

# define problem initial state
A = np.array([[3,2], [-3,2]])
b = np.array([6,0])
c = np.array([0,-1])

model = Model(A,b,c)

x_ordered, optimal_value = model.solve()
model.express_situation_according_to_initial_variables()
print(x_ordered)
print(optimal_value)