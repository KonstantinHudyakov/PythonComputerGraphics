import numpy as np
import matplotlib.pyplot as plt
import math


# v1 и v2 - numpy arrays
def infinite_norm(v1, v2):
    diff = v1 - v2
    return abs(max(diff.max(), diff.min(), key=abs))


def euler(f, t0, T, y0, n):
    h = (T - t0) / n
    t = np.linspace(t0, T, num=n + 1)
    y = [y0]
    for i in range(0, n):
        y.append(y[i] + h * f(t[i], y[i]))
    return t, np.array(y)


def runge_kutta4(f, t0, T, y0, n):
    h = (T - t0) / n
    t = np.linspace(t0, T, num=n + 1)
    y = [y0]
    for i in range(0, n):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h / 2, y[i] + h * k1 / 2)
        k3 = f(t[i] + h / 2, y[i] + h * k2 / 2)
        k4 = f(t[i] + h, y[i] + h * k3)
        y.append(y[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
    return t, y


def runge_rule(f, method, method_order, t0, T, y0, eps):
    n = 1
    t1, y1 = method(f, t0, T, y0, n)
    t2, y2 = method(f, t0, T, y0, 2 * n)
    r = runge_diff(y1, y2, n, method_order)
    while r > eps:
        n *= 2
        y1 = y2
        t2, y2 = method(f, t0, T, y0, 2 * n)
        r = runge_diff(y1, y2, n, method_order)
    return t2, y2, 2 * n


def runge_diff(y1, y2, n, method_order):
    m = -np.inf
    for i in range(0, n + 1):
        eps = abs((y2[2 * i] - y1[i]) / (2 ** method_order - 1))
        if eps > m:
            m = eps
    return m
