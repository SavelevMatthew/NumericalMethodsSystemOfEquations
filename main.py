import numpy as np
from solvers.Gaussian import Gaussian
from solvers.ModifiedGaussian import ModifiedGaussian
from solvers.Jacobi import Jacobi
from solvers.GaussianSeidel import GaussianSeidel
from utils import print_roots

equations_system = np.array([
    [-0.06, 1.64, 0.20, 2.12],
    [0.86, -0.3, 0.28, 2.26],
    [-0.34, -0.16, -0.5, -2.34],
])

print('Метод Гаусса')
gaussian = Gaussian(equations_system)
gaussian_roots = gaussian.solve()
print_roots(gaussian_roots)
print('=' * 64)

print('Метод Гаусса с выбором главного элемента по всей матрице')
modified = ModifiedGaussian(equations_system)
modified_roots = modified.solve()
print_roots(modified_roots)
print('=' * 64)

print('Метод Якоби')
jacobi = Jacobi(equations_system)
print('Проверка достаточного условия для сходимости метода (4.86)')
if jacobi.check():
    print('Можно применить метод Якоби, переставив строки так, '
          'чтобы в каждой строке на диагонали был максимальный по модулю элемент')
    jacobi_roots = jacobi.solve()
    print_roots(jacobi_roots)
    print(f'Итераций потребовалось: {jacobi.iterations}')
else:
    print('Исходная матрица не прошла проверку на сходимость метода (4.86) из лекций')
print('=' * 64)

print('Метод Гаусса-Зейделя')
gs = GaussianSeidel(equations_system)
print('Проверка достаточных условий для сходимости метода: положительная определенность A или A * A')
if gs.check():
    gs_roots = gs.solve()
    print_roots(gs_roots)
    print(f'Итераций потребовалось: {gs.iterations}')
else:
    print('Исходная матрица не прошла проверку на сходимость (положительную определенность)')
