import numpy as np


class Gaussian:
    def __init__(self, equations):
        self.equations = equations.copy()
        self.values = np.array([0.0 for _ in range(len(equations))])

    def solve(self):
        self.prepare()
        l = len(self.equations)
        for i in range(l - 1, -1, -1):
            self.values[i] = (self.equations[i][l] - np.dot(self.equations[i, :-1], self.values.T)) / self.equations[i][i]
        return self.values

    def prepare(self):
        for i in range(len(self.equations)):
            main_value = self.equations[i][i]
            for j in range(i + 1, len(self.equations)):
                value = self.equations[j][i]
                k = (1.0 / main_value) * value
                self.equations[j] = self.equations[j] - k * self.equations[i]
