import numpy as np
import matplotlib.pyplot as plt


# v1 и v2 - numpy arrays
def infinite_norm(v1, v2):
    diff = v1 - v2
    return abs(max(diff.max(), diff.min(), key=abs))


def build_grid(a, b, n):
    return np.linspace(a, b, num=n + 1)


def progonka(matrix, values):
    size = matrix.shape[0]
    a = np.zeros(shape=size)
    b = np.zeros(shape=size)
    ans = np.zeros(shape=size)

    temp = matrix[0][0]
    a[0] = -matrix[0][1] / temp
    b[0] = values[0] / temp
    for i in range(1, size - 1):
        temp = matrix[i][i] + matrix[i][i - 1] * a[i - 1]
        a[i] = -matrix[i][i + 1] / temp
        b[i] = (values[i] - matrix[i][i - 1] * b[i - 1]) / temp
    temp = matrix[size - 1][size - 1] + matrix[size - 1][size - 2] * a[size - 2]
    b[size - 1] = (values[size - 1] - matrix[size - 1][size - 2] * b[size - 2]) / temp

    ans[size - 1] = b[size - 1]
    for i in range(size - 2, -1, -1):
        ans[i] = a[i] * ans[i + 1] + b[i]

    return ans


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


def runge_diff(y1, y2, n, method_order):
    m = -np.inf
    for i in range(0, n + 1):
        eps = abs((y2[2 * i] - y1[i]) / (2 ** method_order - 1))
        if eps > m:
            m = eps
    return m

