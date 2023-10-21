import numpy as np
import math


def euclidian_distance(point1, point2):
    return math.sqrt((point1[0]- point2[0])**2 + (point1[1] - point2[1])**2)


def compute_distance_to_facilities(situation):
    facilities, customers = situation[0], situation[1]
    f = len(facilities)
    c = len(customers)
    distance_matrix = np.zeros((f,c))
    for i in range(f):
        for j in range(c):
            distance_matrix[i,j] = euclidian_distance(facilities[i][3], customers[j][2])
    return distance_matrix