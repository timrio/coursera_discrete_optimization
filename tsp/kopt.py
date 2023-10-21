from copy import deepcopy

from base import TSP
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import random


def makePair(i, j):
    if i > j:
        return (j, i)
    else:
        return (i, j)


class Tour:
    """
    Class to represent a tour in LKH.
    """

    def __init__(self, tour):
        self.tour = tour
        self.size = len(tour)
        self.length = TSP.pathCost(tour)
        self._makeEdges()

    def _makeEdges(self):
        """
        Create the list of edges from the current tour.

        >>> TSP.setEdges([[1] * 6] * 6)  # Dummy matrix
        >>> t = Tour([1, 2, 3, 4, 5])
        >>> t.edges == set([(1, 2), (2, 3), (3, 4), (4, 5), (1, 5)])
        True
        """
        self.edges = set()

        for i in range(self.size):
            self.edges.add(makePair(self.tour[i - 1], self.tour[i]))

    def at(self, i):
        return self.tour[i]

    def contains(self, edge):
        """
        Check if an edge belongs to the tour
        """
        return edge in self.edges

    def index(self, i):
        """
        Return the index of a node in a tour.
        """
        return self.tour.index(i)

    def around(self, node):
        """
        Return the predecessor and successor of the current node, given by
        index.

        Parameters:

            - node: node to look around

        Return: (pred, succ)
        """
        index = self.tour.index(node)

        pred = index - 1
        succ = index + 1

        if succ == self.size:
            succ = 0

        return (self.tour[pred], self.tour[succ])

    def pred(self, index, prev):
        """
        Return the predecessor or successor depending on the `prev` parameter.
        """
        return self.tour[index - 1] if prev else self.tour[index + 1]

    def generate(self, broken, joined):
        """
        Generate a temporary tour with the current exclusions and inclusions.

        Test optimal 2-opt
        >>> from tsp_local.twoopt import cross, start
        >>> TSP.setEdges(cross)
        >>> t = Tour(start)
        >>> _, tour = t.generate(set([(0, 2), (1, 3)]), set([(0, 1), (2, 3)]))
        >>> TSP.pathCost(tour)
        8

        Test disjoint tour
        >>> from tsp_local.threeopt import hexagon, start
        >>> TSP.setEdges(hexagon)
        >>> t = Tour(start)
        >>> t.generate(
        ...     set([(0, 3), (4, 5)]), set([(0, 5), (3, 4)])) #doctest:+ELLIPSIS
        (False, ...)

        Test optimal 3-opt
        >>> _, tour = t.generate(
        ...    set([(0, 3), (2, 4), (1, 5)]), set([(0, 5), (3, 4), (1, 2)]))
        >>> TSP.pathCost(tour)
        6
        """
        # New edges: old edges minus broken, plus joined
        edges = (self.edges - broken) | joined

        # If we do not have enough edges, we cannot form a tour -- should not
        # happen within LKH
        if len(edges) < self.size:
            return False, []

        successors = {}
        node = 0

        # Build the list of successors
        while len(edges) > 0:
            for i, j in edges:
                if i == node:
                    successors[node] = j
                    node = j
                    break
                elif j == node:
                    successors[node] = i
                    node = i
                    break

            edges.remove((i, j))

        # Similarly, if not every node has a successor, this can not work
        if len(successors) < self.size:
            return False, []

        succ = successors[0]
        new_tour = [0]
        visited = set(new_tour)

        # If we already encountered a node it means we have a loop
        while succ not in visited:
            visited.add(succ)
            new_tour.append(succ)
            succ = successors[succ]

        # If we visited all nodes without a loop we have a tour
        return len(new_tour) == self.size, new_tour


class KOpt(TSP):
    """
    K-opt move for the TSP, will become Lin-Kernighan.
    """

    def _optimise(self):
        """
        Global loop which restarts at each improving solution.
        """
        better = True
        self.solutions = set()

        # Rebuild the neighbours
        self.neighbours = {}

        for i in self.heuristic_path:
            self.neighbours[i] = []

            for j, dist in enumerate(TSP.edges[i]):
                if dist > 0 and j in self.heuristic_path:
                    self.neighbours[i].append(j)

        # Restart the loop each time we find an improving candidate
        while better:
            better = self.improve()
            # Paths always begin at 0 so this should manage to find duplicate
            # solutions
            print(f"best:{get_route_length(self.heuristic_path, matrix)}")
            self.solutions.add(str(self.heuristic_path))

        self.save(self.heuristic_path, self.heuristic_cost)

    def closest(self, t2i, tour, gain, broken, joined):
        """
        Find the closest neighbours of a node ordered by potential gain.  As a
        side-effect, also compute the partial improvement of joining a node.

        Parameters:

            - t2i: node to relink from

            - tour: current tour to optimise

            - gain: current gain

            - broken: set of edges to remove (X)

            - joined: set of edges to join (Y)

        Return: sorted list of neighbours based on potential improvement with
        next omission
        """
        # Select t_2i + 1
        neighbours = {}

        # Create the neighbours of t_2i
        for node in self.neighbours[t2i]:
            # Any new edge has to have a positive running sum, not be a broken
            # edge and not belong to the tour.
            yi = makePair(t2i, node)
            if yi in broken or tour.contains(yi):
                continue
            Gi = gain - TSP.dist(t2i, node)
            if Gi <= 0:
                continue

            for succ in tour.around(node):
                xi = makePair(node, succ)

                # TODO verify it is enough, but we do check if the tour is
                # valid first thing in `chooseX` so this should be sufficient
                #
                # Check that "x_i+1 exists"
                if xi not in broken and xi not in joined:
                    diff = TSP.dist(node, succ) - TSP.dist(t2i, node)

                    if node in neighbours and diff > neighbours[node][0]:
                        neighbours[node][0] = diff
                    else:
                        neighbours[node] = [diff, Gi]

        # Sort the neighbours by potential gain
        return sorted(neighbours.items(), key=lambda x: x[1][0], reverse=True)

    def improve(self):
        """
        Start the LKH algorithm with the current tour.
        """
        tour = Tour(self.heuristic_path)

        # Find all valid 2-opt moves and try them
        t1_suffle = self.heuristic_path.copy()
        random.shuffle(t1_suffle)
        for t1 in t1_suffle:
            around = tour.around(t1)

            for t2 in around:
                broken = set([makePair(t1, t2)])
                # Initial savings
                gain = TSP.dist(t1, t2)

                close = self.closest(t2, tour, gain, broken, set())

                # Number of neighbours to try
                tries = 30

                for t3, (_, Gi) in close:
                    # Make sure that the new node is none of t_1's neighbours
                    # so it does not belong to the tour.
                    if t3 in around:
                        continue

                    joined = set([makePair(t2, t3)])

                    if self.chooseX(tour, t1, t3, Gi, broken, joined):
                        # Return to Step 2, that is the initial loop
                        return True
                    # Else try the other options

                    tries -= 1
                    # Explored enough nodes, change t_2
                    if tries == 0:
                        break

        return False

    def chooseX(self, tour, t1, last, gain, broken, joined):
        """
        Choose an edge to omit from the tour.

        Parameters:

            - tour: current tour to optimise

            - t1: starting node for the current k-opt

            - last: tail of the last edge added (t_2i-1)

            - gain: current gain (Gi)

            - broken: potential edges to remove (X)

            - joined: potential edges to add (Y)

        Return: whether we found an improved tour
        """
        if len(broken) == 4:
            pred, succ = tour.around(last)

            # Give priority to the longest edge for x_4
            if TSP.dist(pred, last) > TSP.dist(succ, last):
                around = [pred]
            else:
                around = [succ]
        else:
            around = tour.around(last)

        for t2i in around:
            xi = makePair(last, t2i)
            # Gain at current iteration
            Gi = gain + TSP.dist(last, t2i)

            # Verify that X and Y are disjoint, though I also need to check
            # that we are not including an x_i again for some reason.
            if xi not in joined and xi not in broken:
                added = deepcopy(joined)
                removed = deepcopy(broken)

                removed.add(xi)
                added.add(makePair(t2i, t1))  # Try to relink the tour

                relink = Gi - TSP.dist(t2i, t1)
                is_tour, new_tour = tour.generate(removed, added)

                # The current solution does not form a valid tour
                if not is_tour and len(added) > 2:
                    continue

                # Stop the search if we come back to the same solution
                if str(new_tour) in self.solutions:
                    return False

                # Save the current solution if the tour is better, we need
                # `is_tour` again in the case where we have a non-sequential
                # exchange with i = 2
                if is_tour and relink > 0:
                    self.heuristic_path = new_tour
                    self.heuristic_cost -= relink

                    return True
                else:
                    # Pass on the newly "removed" edge but not the relink
                    choice = self.chooseY(tour, t1, t2i, Gi, removed, joined)

                    if len(broken) == 2 and choice:
                        return True
                    else:
                        # Single iteration for i > 2
                        return choice

        return False

    def chooseY(self, tour, t1, t2i, gain, broken, joined):
        """
        Choose an edge to add to the new tour.

        Parameters:

            - tour: current tour to optimise

            - t1: starting node for the current k-opt

            - t2i: tail of the last edge removed (t_2i)

            - gain: current gain (Gi)

            - broken: potential edges to remove (X)

            - joined: potential edges to add (Y)

        Return: whether we found an improved tour
        """
        ordered = self.closest(t2i, tour, gain, broken, joined)

        if len(broken) == 2:
            # Check the five nearest neighbours when i = 2
            top = 5
        else:
            # Otherwise the closest only
            top = 1

        for node, (_, Gi) in ordered:
            yi = makePair(t2i, node)
            added = deepcopy(joined)
            added.add(yi)

            # Stop at the first improving tour
            if self.chooseX(tour, t1, node, Gi, broken, added):
                return True

            top -= 1
            # Tried enough options
            if top == 0:
                return False

        return False


# define utils functions
def euclidian_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def get_route_length(route, distance_matrix):
    length = 0
    for i in range(len(route) - 1):
        length += distance_matrix[route[i]][route[i + 1]]
    length += distance_matrix[route[-1]][route[0]]
    return length


def get_distance_matrix(points):
    n = len(points)
    distance_matrix = np.zeros((n, n))
    for i in tqdm(range(n)):
        for j in range(i + 1, n):
            distance = euclidian_distance(
                points[i][0], points[i][1], points[j][0], points[j][1]
            )
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance
    return distance_matrix


def visualize_route(route, points):
    plt.figure(figsize=(10, 10))
    plt.scatter([x[0] for x in points], [x[1] for x in points], c="red")
    for i in range(len(route) - 1):
        plt.plot(
            [points[route[i]][0], points[route[i + 1]][0]],
            [points[route[i]][1], points[route[i + 1]][1]],
            c="blue",
        )
    plt.plot(
        [points[route[-1]][0], points[route[0]][0]],
        [points[route[-1]][1], points[route[0]][1]],
        c="blue",
    )
    plt.show()


def solve_LKH(points, restarts=5):
    global matrix
    matrix = get_distance_matrix(points)

    TSP.setEdges(matrix)

    best_dist = np.inf
    best_sol = None

    for restart in range(restarts):
        t = KOpt(range(len(matrix)))
        t._optimise()

        dist = get_route_length(t.heuristic_path, matrix)
        if dist < best_dist:
            best_dist = dist
            best_sol = t.heuristic_path

    return best_dist, best_sol
