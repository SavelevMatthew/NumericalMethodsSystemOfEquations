import numpy as np


class Jacobi:
    def __init__(self, equations):
        self.equations = equations.copy()
        self.values = np.array([1.0 for _ in range(len(equations))])
        self.orders = None
        self.L = np.zeros((len(equations), len(equations)))
        self.D = np.zeros((len(equations), len(equations)))
        self.D_inv = self.D
        self.R = np.zeros((len(equations), len(equations)))
        self.b = np.zeros((len(equations), 1))
        self.epsilon = 0.5 * (10 ** (-4))
        self.iterations = 0
        self.B = None
        self.g = None
        self.q = None

    def prepare(self):
        for i in range(len(self.equations)):
            row_index = self.orders.index(i)
            self.swap_rows(i, row_index)

    def check(self):
        maxes = np.argmax(np.abs(self.equations[:, :-1]), axis=1)
        self.orders = list(maxes)
        return len(set(maxes)) == len(maxes)

    def split(self):
        for i in range(len(self.equations)):
            for j in range(len(self.equations)):
                if j < i:
                    self.L[i, j] = self.equations[i, j]
                elif j == i:
                    self.D[i, j] = self.equations[i, j]
                else:
                    self.R[i, j] = self.equations[i, j]
        self.b = self.equations[:, -1]
        self.D_inv = np.linalg.inv(self.D)
        self.B = - np.dot(self.D_inv, (self.R + self.L))
        self.g = np.dot(self.D_inv, self.b)
        self.q = np.linalg.norm(self.B)

    def swap_rows(self, i, j):
        temp = self.equations[i].copy()
        self.equations[i] = self.equations[j]
        self.equations[j] = temp
        temp_row = self.orders[i]
        self.orders[i] = self.orders[j]
        self.orders[j] = temp_row

    def is_solution_good_enough(self, old_values, values):
        diffs = np.abs(old_values - values)
        return np.linalg.norm(diffs) <= (((1 - self.q) / self.q) * self.epsilon)

    def solve(self):
        self.prepare()
        self.split()
        old_values = self.values + 1
        self.iterations = 0
        while not self.is_solution_good_enough(old_values, self.values):
            old_values = self.values
            self.values = np.dot(self.B, self.values) + self.g
            self.iterations += 1
        return self.values

