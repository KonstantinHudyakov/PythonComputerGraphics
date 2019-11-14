import numpy as np
import matplotlib.pyplot as plt

from numericals.diff_schemas.lab10.common import build_grid, progonka, runge_rule, infinite_norm


def f(x):
    return -(x * x) + 2.5 * x + 1.25


def ans(x):
    return -4 * x * x - 14 * x - 3.52396 * np.exp(-x) + 68.524 * np.exp(0.25 * x) - 69


def build_coef_matrix(grid, p, q, f):
    size = len(grid)
    h = (grid[size - 1] - grid[0]) / (size - 1)

    matrix = np.zeros(shape=(size, size))
    matrix[0][0] = 1
    matrix[size - 1][size - 1] = 1
    for i in range(1, size - 1):
        matrix[i][i - 1] = -(1 / (h * h) + p / (2 * h))
        matrix[i][i] = q + 2 / (h * h)
        matrix[i][i + 1] = p / (2 * h) - 1 / (h * h)
    return matrix


def build_values_vector(grid, f, ua, ub):
    size = len(grid)
    values = np.zeros(shape=size)
    values[0] = ua
    values[size - 1] = ub
    for i in range(1, size - 1):
        values[i] = f(grid[i])
    return values


def base_finite_diff_method(a, b, ua, ub, p, q, f, n):
    grid = build_grid(a, b, n)
    matrix = build_coef_matrix(grid, p, q, f)
    values = build_values_vector(grid, f, ua, ub)
    ans = progonka(matrix, values)
    return grid, ans


a = 0
b = 2
ua = -4
ub = -0.5
p = -0.75
q = 0.25
eps = 0.001

x, y, n = runge_rule(a, b, ua, ub, p, q, f, eps, method=base_finite_diff_method, method_order=2)

diff = infinite_norm(ans(x), y)
print(n)
print(diff)

plt.figure(dpi=200)
plt.plot(x, ans(x), 'r--', label='Точное решение')
plt.plot(x, y, 'bo', label='МКР')
plt.ylabel('y')
plt.xlabel('x')
plt.legend(loc='best')

plt.show()
