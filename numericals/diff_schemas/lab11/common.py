import numpy as np
import matplotlib.pyplot as plt


# matrix - матрица из 3-ёх столбцов, которые являются дигоналями исходной матрицы коэффицентов
def progonka(matrix, values):
    size = matrix.shape[0]
    a = np.zeros(shape=size)
    b = np.zeros(shape=size)
    ans = np.zeros(shape=size)

    temp = matrix[0][0]
    a[0] = -matrix[0][1] / temp
    b[0] = values[0] / temp
    for i in range(1, size - 1):
        temp = matrix[i][1] + matrix[i][0] * a[i - 1]
        a[i] = -matrix[i][2] / temp
        b[i] = (values[i] - matrix[i][0] * b[i - 1]) / temp
    temp = matrix[size - 1][2] + matrix[size - 1][1] * a[size - 2]
    b[size - 1] = (values[size - 1] - matrix[size - 1][1] * b[size - 2]) / temp

    ans[size - 1] = b[size - 1]
    for i in range(size - 2, -1, -1):
        ans[i] = a[i] * ans[i + 1] + b[i]

    return ans


def plot_graph(x, y, style, legend, xlabel='x', ylabel='y'):
    plt.figure(dpi=200)
    plt.plot(x, y, style, label=legend)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(loc='best')
    plt.show()