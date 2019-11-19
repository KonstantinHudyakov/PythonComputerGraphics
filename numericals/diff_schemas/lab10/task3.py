import numpy as np
import matplotlib.pyplot as plt

from numericals.diff_schemas.lab10.common import build_grid, progonka, runge_diff, infinite_norm, lower_bound


def build_coef_matrix(grid, q, k1, k2, k3):
    def k(x):
        if x >= 0 and x <= 1:
            return k1(x)
        elif x > 1 and x <= 2:
            return k2(x)
        elif x > 2 and x <= 3:
            return k3(x)

    size = len(grid)
    h = (grid[size - 1] - grid[0]) / (size - 1)

    matrix = np.zeros(shape=(size, size))
    matrix[0][0] = 1
    matrix[size - 1][size - 1] = 1
    for i in range(1, size - 1):
        matrix[i][i - 1] = -k(grid[i] - h / 2) / (h * h)
        matrix[i][i] = q + (k(grid[i] - h / 2) + k(grid[i] + h / 2)) / (h * h)
        matrix[i][i + 1] = -k(grid[i] + h / 2) / (h * h)
    return matrix


def build_values_vector(grid, f, ua, ub):
    size = len(grid)
    values = np.zeros(shape=size)
    values[0] = ua
    values[size - 1] = ub
    for i in range(1, size - 1):
        values[i] = f(grid[i])
    return values


def special_finite_diff_method(n, a, b, ua, ub, q, f, k1, k2, k3):
    # Строим сетку по заданному разбиению
    grid = build_grid(a, b, n)
    # Получаем трёхдиагональную матрицу коэффициентов системы уравнений
    matrix = build_coef_matrix(grid, q, k1, k2, k3)
    # Получаем вектор значений правой части
    values = build_values_vector(grid, f, ua, ub)
    # Решаем систему методом прогонки
    ans = progonka(matrix, values)
    return grid, ans


def runge_rule(a, b, ua, ub, q, f, k1, k2, k3, eps, method, method_order):
    n = 2
    x1, y1 = method(n, a, b, ua, ub, q, f, k1, k2, k3)
    x2, y2 = method(2 * n, a, b, ua, ub, q, f, k1, k2, k3)
    r = runge_diff(y1, y2, n, method_order)
    while r > eps:
        n *= 2
        y1 = y2
        x2, y2 = method(2 * n, a, b, ua, ub, q, f, k1, k2, k3)
        r = runge_diff(y1, y2, n, method_order)
    return x2, y2, 2 * n


def plot_graphic(x, y, style, legend, xlabel='x', ylabel='y', markersize=1):
    plt.plot(x, y, style, label=legend, markersize=markersize)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(loc='best')


def print_x0_val(x, y, x0, comb_label):
    x0_ind = lower_bound(x, x0)
    x0_ind = x0_ind if abs(x[x0_ind] - x0) < abs(x[x0_ind - 1] - x0) else x0_ind - 1
    print(comb_label + ': y(x0) = ' + str(round(y[x0_ind], 3)))


a = 0
b = 3
x0 = 1.8
ua = 3
ub = 3
q = 0.25
eps = 0.001
f = lambda x: -(x * x) + 2.5 * x + 1.25
k1 = lambda x: 7 - x
k2 = lambda x: np.log(x * x + 2)
k3 = lambda x: 5 * (x + 2) * (x + 2)

x1, y1, n1 = runge_rule(a, b, ua, ub, q, f, k1, k2, k3, eps=eps, method=special_finite_diff_method, method_order=2)
x2, y2, n2 = runge_rule(a, b, ua, ub, q, f, k1, k3, k2, eps=eps, method=special_finite_diff_method, method_order=2)
x3, y3, n3 = runge_rule(a, b, ua, ub, q, f, k2, k1, k3, eps=eps, method=special_finite_diff_method, method_order=2)
x4, y4, n4 = runge_rule(a, b, ua, ub, q, f, k2, k3, k1, eps=eps, method=special_finite_diff_method, method_order=2)
x5, y5, n5 = runge_rule(a, b, ua, ub, q, f, k3, k1, k2, eps=eps, method=special_finite_diff_method, method_order=2)
x6, y6, n6 = runge_rule(a, b, ua, ub, q, f, k3, k2, k1, eps=eps, method=special_finite_diff_method, method_order=2)

print(n1)
print(n2)
print(n3)
print(n4)
print(n5)
print(n6)

plt.figure(dpi=200)
plot_graphic(x1, y1, 'ro', 'k1 k2 k3, n = ' + str(n1))
plot_graphic(x2, y2, 'bo', 'k1 k3 k2, n = ' + str(n2))
plot_graphic(x3, y3, 'go', 'k2 k1 k3, n = ' + str(n3))
plot_graphic(x4, y4, 'co', 'k2 k3 k1, n = ' + str(n4))
plot_graphic(x5, y5, 'mo', 'k3 k1 k2, n = ' + str(n5))
plot_graphic(x6, y6, 'yo', 'k3 k2 k1, n = ' + str(n6))
plt.axvline(x=x0)

plt.show()

print_x0_val(x1, y1, x0, comb_label='k1 k2 k3')
print_x0_val(x2, y2, x0, comb_label='k1 k3 k2')
print_x0_val(x3, y3, x0, comb_label='k2 k1 k3')
print_x0_val(x4, y4, x0, comb_label='k2 k3 k1')
print_x0_val(x5, y5, x0, comb_label='k3 k1 k2')
print_x0_val(x6, y6, x0, comb_label='k3 k2 k1')

# plot_graphic(build_grid(0, 1, 21), k1(build_grid(0, 1, 21)), 'r-', 'k1')
# plot_graphic(build_grid(1, 2, 21), k2(build_grid(1, 2, 21)), 'g-', 'k2')
# plot_graphic(build_grid(2, 3, 21), k3(build_grid(2, 3, 21)), 'b-', 'k3')
