import math
import numpy as np
from collections import defaultdict 


class Constraint:
    def __init__(self, name, constraint_type, left_coeffs, left_variables, right_member):

        self.name = name
        self.constraint_type = constraint_type
        self.left_coeffs = left_coeffs
        self.left_variables = left_variables
        self.right_member = right_member

    @property
    def coeff_var_dict(self):
        d = defaultdict(lambda: 0)
        for i,k in enumerate(self.left_variables):
            d[k.name] = self.left_coeffs[i]
        return d