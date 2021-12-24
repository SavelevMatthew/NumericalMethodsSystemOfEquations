import numpy as np


class ModifiedGaussian:
    def __init__(self, equations):
        self.equations = equations.copy()
        self.values = np.array([0.0 for _ in range(len(equations))])
        self.transitions = [i for i in range(len(equations))]

    def prepare(self):
        for i in range(len(self.equations)):
            slice = self.equations[i:, i:-1]
            x, y = np.unravel_index(np.abs(slice).argmax(), slice.shape)
            self.swap_rows(i, i + x)
            self.swap_cols(i, i + y)

            main_value = self.equations[i, i]
            for j in range(i + 1, len(self.equations)):
                value = self.equations[j, i]
                k = (1.0 / main_value) * value
                self.equations[j] = self.equations[j] - k * self.equations[i]

    def swap_rows(self, i, j):
        temp = self.equations[i].copy()
        self.equations[i] = self.equations[j]
        self.equations[j] = temp

    def swap_cols(self, i, j):
        temp = self.equations[:, i].copy()
        self.equations[:, i] = self.equations[:, j]
        self.equations[:, j] = temp
        temp_x = self.transitions[i]
        self.transitions[i] = self.transitions[j]
        self.transitions[j] = temp_x

    def solve(self):
        self.prepare()
        l = len(self.equations)
        for i in range(l - 1, -1, -1):
            self.values[i] = (self.equations[i][l] - np.dot(self.equations[i, :-1], self.values.T)) / self.equations[i][
                i]
        return [self.values[i] for i in self.transitions]
