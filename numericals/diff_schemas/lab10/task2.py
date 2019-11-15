import numpy as np
import matplotlib.pyplot as plt

from numericals.diff_schemas.lab10.common import build_grid, progonka, infinite_norm, runge_diff


def ans(x):
    return -4 * x * x - 14 * x - 3.52396 * np.exp(-x) + 68.524 * np.exp(0.25 * x) - 69


def build_coef_matrix(grid, p, q):
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
    # Строим сетку по заданному разбиению
    grid = build_grid(a, b, n)
    # Получаем трёхдиагональную матрицу коэффициентов системы уравнений
    matrix = build_coef_matrix(grid, p, q)
    # Получаем вектор значений правой части
    values = build_values_vector(grid, f, ua, ub)
    # Решаем систему методом прогонки
    ans = progonka(matrix, values)
    return grid, ans


def runge_rule(a, b, ua, ub, p, q, f, eps, method, method_order):
    n = 2
    x1, y1 = method(a, b, ua, ub, p, q, f, n)
    x2, y2 = method(a, b, ua, ub, p, q, f, 2 * n)
    r = runge_diff(y1, y2, n, method_order)
    while r > eps:
        n *= 2
        y1 = y2
        x2, y2 = method(a, b, ua, ub, p, q, f, 2 * n)
        r = runge_diff(y1, y2, n, method_order)
    return x2, y2, 2 * n


a = 0
b = 2
ua = -4
ub = -0.5
p = -0.75
q = 0.25
eps = 0.001
f = lambda x: -(x * x) + 2.5 * x + 1.25

x, y, n = runge_rule(a, b, ua, ub, p, q, f, eps, method=base_finite_diff_method, method_order=2)

diff = infinite_norm(ans(x) - y)
print('n = ' + str(n))
print('diff = ' + str(round(diff, 4)))

plt.figure(dpi=200)
plt.plot(x, ans(x), 'r--', label='Точное решение')
plt.plot(x, y, 'bo', label='МКР')
plt.ylabel('y')
plt.xlabel('x')
plt.legend(loc='best')

plt.show()
