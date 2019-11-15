import numpy as np
import matplotlib.pyplot as plt


# v1 и v2 - numpy arrays
def infinite_norm(v):
    return abs(max(v.max(), v.min(), key=abs))


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


def runge_diff(y1, y2, n, method_order):
    m = -1
    for i in range(0, n + 1):
        eps = abs((y2[2 * i] - y1[i]) / (2 ** method_order - 1))
        if eps > m:
            m = eps
    return m


def lower_bound(arr, target):
    left = 0
    right = len(arr) - 1
    while left < right + 1:
        mid = left + (right - left) // 2
        if target >= arr[mid] and target <= arr[mid + 1]:
            return mid + 1
        elif target > arr[mid]:
            left = mid + 1
        else:
            right = mid
    if target >= arr[left] and target <= arr[left + 1]:
        return right
    else:
        return left


