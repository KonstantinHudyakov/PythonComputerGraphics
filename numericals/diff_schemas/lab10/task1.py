import numpy as np
import matplotlib.pyplot as plt

from numericals.diff_schemas.lab10.common import build_grid, infinite_norm, runge_diff


def f(x):
    return -(x * x) + 2.5 * x + 1.25


def ans(x):
    return -4 * x * x - 14 * x + 1.8 * np.exp(-x) + 63.2 * np.exp(0.25 * x) - 69


# Метод Эйлера решения задачи Коши для системы уравнений
def vector_euler(t0, T, y0, A, b, n):
    size = A.shape[0]  # Кол-во уравнений
    h = (T - t0) / n
    grid = build_grid(t0, T, n)
    y = [y0]
    for i in range(n):
        # Умножение матрицы на вектор
        # (np.identity(size) + h * A) @ y[i] - аналогично
        y.append(np.dot((np.identity(size) + h * A), y[i]) + h * b(grid[i]))
    # Возвращаем сетку и значения искомой функции в узлах сетки
    return grid, np.asarray(y).transpose()[0]


def runge_rule(t0, T, y0, A, b, eps, method, method_order):
    n = 1
    x1, y1 = method(t0, T, y0, A, b, n)
    x2, y2 = method(t0, T, y0, A, b, 2 * n)
    r = runge_diff(y1, y2, n, method_order)
    while r > eps:
        n *= 2
        y1 = y2
        x2, y2 = method(t0, T, y0, A, b, 2 * n)
        r = runge_diff(y1, y2, n, method_order)
    return x2, y2, 2 * n


t0 = 0
T = 2
u0 = -4
u1 = 0
p = -0.75
q = 0.25
eps = 0.001


y0 = np.array([u0, u1])
A = np.array([[0, 1], [q, p]])
b = lambda x: np.array([0, -f(x)])

t, y, n = runge_rule(t0, T, y0, A, b, eps, method=vector_euler, method_order=1)

diff = infinite_norm(ans(t), y)
print(n)
print(diff)

plt.figure(dpi=200)
plt.plot(t, ans(t), 'r--', label='Точное решение', linewidth = 7)
plt.plot(t, y, 'bo', label='Метод Эйлера', markersize=1)
plt.ylabel('y')
plt.xlabel('x')
plt.legend(loc='best')

plt.show()