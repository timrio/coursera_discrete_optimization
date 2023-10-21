import numpy as np
from numpy.linalg import inv



# define problem initial state
A = np.array([[3,2,1,0], [-3,2,0,1]])
b = np.array([6,0])
c = np.array([0,-1,0,0])

def simplex_iteration(A,b,c):
    m = A.shape[0]
    n = A.shape[1]

    permutation = [i for i in range(n)]    
    if b.any()<0:
        print("cannot find initial basis")
        return False

    keep_going = True

    while keep_going:

        A_b, A_n = A[:,permutation[-m:]], A[:,permutation[:-m]]
        c_b, c_n = c[permutation[-m:]], c[permutation[:-m]]
        if np.linalg.det(A_b) == 0:
            return 'matrice non inversible'
        A_b_inv = inv(A_b)
        pi = np.dot(c_b,A_b_inv) 
        x_b_opt = np.dot(A_b_inv,b)
        
        # express x basis according to xn
        # x_b = x_b_opt - H @ x_n
        H = np.dot(A_b_inv, A_n)

        # compute reduced costs
        reduced_cost = c_n - np.dot(pi, A_n)

        if (reduced_cost >= 0).all():
            keep_going = False

        else:
            # we need find entering and exiting variable
            entering_index = np.argmin(reduced_cost)
            ratio = (x_b_opt / H[:, entering_index])
            
            # Avoid division by zero in case of non-positive entries in H[:, entering_index]
            ratio[(ratio < 0)] = np.inf
            exiting_index = np.argmin([value if H[i, entering_index]>0 else np.inf for i,value in enumerate(ratio)])

            # permute entering and exiting values
            permutation[entering_index], permutation[(n-m)+exiting_index] = permutation[(n-m)+exiting_index],permutation[entering_index]

    return H, pi, permutation, x_b_opt, m, n
    

def find_gomory_cuts(H, permutation, x_b_opt, m):
    gomory_cuts = []
    non_basic_variables = permutation[:m]
    for i, basic_value in enumerate(x_b_opt):
        # check if basic value is an int
        if np.round(basic_value,0)!=basic_value:
            new_cut = {}
            new_cut["b"] = -(basic_value - np.floor(basic_value))
            for j, val in enumerate(H[i,:]):
                new_cut[j] = -(val - np.floor(val))
            gomory_cuts.append(new_cut)
    return gomory_cuts


def add_gomory_cuts(H, x_b_opt, permuation, m, n, c, gomory_cuts):
    updated_H = H
    updated_x_b_opt = x_b_opt
    updated_c = c
    if len(gomory_cuts)==0:
        return updated_H, updated_x_b_opt, updated_c, m,n

    for cut in gomory_cuts:
        # add the constraint
        new_line = np.zeros(updated_H.shape[1])
        new_b = None
        for k, v in cut.items():
            if k == "b":
                new_b = v
                continue
            new_line[k] = v

        updated_H = np.r_[updated_H, [new_line]]

        #update b
        updated_x_b_opt = np.append(updated_x_b_opt, new_b)

        # update c
        updated_c = np.append(updated_c, 0)

    permutation.append(n)
    return updated_H, updated_x_b_opt, updated_c, m+1, n+1, permutation


def dual_simplex_python(H, x_b_opt, updated_c, permutation, m,n):


    # reconstruct A, b and c
    b = x_b_opt
    c = updated_c
    A = np.zeros((m,n))
    basis_col = np.eye(m)
    
    # start with non basis columns
    for i, col_index in enumerate(permutation[:-m]):
        A[:,col_index] = H[:,i]

    # continue with non basis columns
    for i, col_index in enumerate(permutation[-m:]):
        A[:,col_index] = basis_col[:,i]
        

    keep_going = True
    while keep_going:
        
        A_b, A_n = A[:,permutation[-m:]], A[:,permutation[:-m]]
        c_b, c_n = c[permutation[-m:]], c[permutation[:-m]]

        A_b_inv = inv(A_b)
        pi = np.dot(c_b,A_b_inv) 
        x_b_opt = np.dot(A_b_inv,b)
        
        # express x basis according to xn
        # x_b = x_b_opt - H @ x_n
        H = np.dot(A_b_inv, A_n)

        # compute reduced costs
        reduced_cost = c_n - np.dot(pi, A_n)


        if all(x_b_opt>=0):
            keep_going = False
        else:
            # find the variable that will leave, the basis, this is the one the first negative beta
            min_val = np.inf
            exiting_index = None
            for i in range(m):
                if x_b_opt[i] < min_val and x_b_opt[i]<0:
                    exiting_index = i
                    min_val = x_b_opt[i]
                    break
            if all(val>=0 for val in H[exiting_index,:]):
                print("model infeasible")
                break
            else:
                # we choose the variable that enters the basis
                min_val = np.inf
                entering_index = None
                for i in range(n-m):
                    if H[exiting_index,i] < 0 and reduced_cost[i]/np.abs(H[exiting_index,i]) < min_val:
                        min_val = reduced_cost[i]/np.abs(H[exiting_index,i]) 
                        entering_index = i 

                # permute entering and exiting values
                permutation[entering_index], permutation[(n-m)+exiting_index] = permutation[(n-m)+exiting_index],permutation[entering_index]
    return H, pi, permutation, x_b_opt, m, n
    


def get_sorted_value_from_simplex_iteration(pi,b,x_b_opt, permutation, number_of_basic_variables, number_of_variables):
    optimal_value = np.dot(pi,b)

    x = [0 for i in range(number_of_variables-number_of_basic_variables)] + list(x_b_opt)
    x_ordered = [0 for i in range(number_of_variables)]
    for i, correct_index in enumerate(permutation):
        x_ordered[correct_index] = x[i]

    return optimal_value, x_ordered

    

H, pi, permutation, x_b_opt, m, n = simplex_iteration(A,b,c)

gomory_cuts = find_gomory_cuts(H, permutation, x_b_opt, m)
H, x_b_opt, c, m, n, permutation = add_gomory_cuts(H, x_b_opt, permutation, m, n, c, gomory_cuts)
H, pi, permutation, x_b_opt, m, n = dual_simplex_python(H, x_b_opt, c, permutation, m,n)


gomory_cuts = find_gomory_cuts(H, permutation, x_b_opt, m)
H, x_b_opt, c, m, n, permutation = add_gomory_cuts(H, x_b_opt, permutation, m, n, c, gomory_cuts)
H, pi, permutation, x_b_opt, m, n = dual_simplex_python(H, x_b_opt, c, permutation, m,n)