import numpy as np
import matplotlib.pyplot as plt

from numericals.diff_schemas.lab10.common import build_grid, runge_diff
from numericals.diff_schemas.lab11.common import progonka


def build_coef_matrix(grid):
    size = len(grid)
    matrix = np.zeros(shape=(size, 3))
    matrix[0][0] = 1
    matrix[size - 1][2] = 1
    for i in range(1, size - 1):
        matrix[i][0] = -1
        matrix[i][1] = 2
        matrix[i][2] = -1
    return matrix


def build_values_vector(grid, f, ua, ub):
    size = len(grid)
    h = grid[1] - grid[0]
    values = np.zeros(shape=size)
    values[0] = ua
    values[size - 1] = ub
    for i in range(1, size - 1):
        values[i] = h * h * f(grid[i])
    return values


def base_finite_diff_method(a, b, ua, ub, f, n):
    # Строим сетку по заданному разбиению
    grid = build_grid(a, b, n)
    # Получаем три столбца - диагонали матрицы коэффециентов
    matrix = build_coef_matrix(grid)
    # Получаем вектор значений правой части
    values = build_values_vector(grid, f, ua, ub)
    # Решаем систему методом прогонки
    ans = progonka(matrix, values)
    return grid, ans


def runge_rule(a, b, ua, ub, f, eps, method, method_order):
    n = 2
    x1, y1 = method(a, b, ua, ub, f, n)
    x2, y2 = method(a, b, ua, ub, f, 2 * n)
    r = runge_diff(y1, y2, n, method_order)
    while r > eps:
        n *= 2
        y1 = y2
        x2, y2 = method(a, b, ua, ub, f, 2 * n)
        r = runge_diff(y1, y2, n, method_order)
    return x2, y2, 2 * n


a = 0
b = 2
ua = -3
ub = 3
n = 10
eps = 0.001
f = lambda x: 3 * x + x * x
g = lambda x: ua + (ub - ua) / b * x
ans = lambda x: 1 / 12 * (-(x ** 4) - 6 * (x ** 3) + 68 * x - 36)

h = b / n
h_grid = build_grid(a, b, n)
x, u, n = runge_rule(a, b, ua, ub, f, eps=eps, method=base_finite_diff_method, method_order=2)

plt.figure(dpi=200)
plt.plot(x, ans(x), 'r--', label='Точное решение t --> inf')
plt.plot(x, u, 'bo', label='МКР')
plt.ylabel('u')
plt.xlabel('x')
plt.legend(loc='best')

plt.show()
