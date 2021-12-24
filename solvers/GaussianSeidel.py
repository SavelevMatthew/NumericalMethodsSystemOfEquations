import numpy as np


class GaussianSeidel:
    def __init__(self, equations):
        self.equations = equations.copy()
        self.values = np.array([0.0 for _ in range(len(equations))])
        self.A = self.equations[:, :-1]
        self.shouldTransform = False
        self.b = self.equations[:, -1]
        self.epsilon = 0.5 * (10 ** (-4))
        self.R = np.zeros(self.A.shape)
        self.D = np.zeros(self.A.shape)
        self.L = np.zeros(self.A.shape)
        self.k = None
        self.iterations = None

    def check(self):
        w, _ = np.linalg.eig(self.A)
        sw, _ = np.linalg.eig(np.dot(self.A.T, self.A))
        if all(map(lambda x: x > 0, list(w))):
            print('Матрица A положительно определена. Можно применить метод')
            return True
        elif all(map(lambda x: x > 0, list(sw))):
            print('Матрица A*A положительно определена. Можно применить метод для (A*A)x = (A*)b')
            self.shouldTransform = True
            return True
        else:
            return False

    def prepare(self):
        if self.shouldTransform:
            self.b = np.dot(self.A.T, self.b)
            self.A = np.dot(self.A.T, self.A)
        self.split()

    def split(self):
        for i in range(len(self.A)):
            for j in range(len(self.A)):
                if j < i:
                    self.L[i, j] = self.A[i, j]
                elif j == i:
                    self.D[i, j] = self.A[i, j]
                else:
                    self.R[i, j] = self.A[i, j]
        self.k = np.linalg.inv(self.L + self.D)

    def is_good_enough_solution(self):
        return np.linalg.norm(np.dot(self.A, self.values) - self.b) <= self.epsilon

    def solve(self):
        self.prepare()
        self.iterations = 0
        while not self.is_good_enough_solution():
            self.values = np.dot(self.k, np.dot(-self.R, self.values) + self.b)
            self.iterations += 1
        return self.values