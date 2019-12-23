import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D


# v - ndarray
def infinite_norm(v):
    return abs(max(v.max(), v.min(), key=abs))


# matrix - ndarray
def euclidean_norm(matrix):
    sum = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            sum += matrix[i][j] ** 2
    return np.sqrt(sum)


# Возвращает вектор - сетку разбиения отрезка [a, b] на n частей
def build_grid(a, b, n):
    return np.linspace(a, b, num=n + 1)


def build_uniform_test_function(coef, n, m, a, b):
    p = np.pi * n / a
    q = np.pi * m / b
    answer = lambda x, y, t: coef * np.sin(p * x) * np.sin(q * y) * np.exp(-t * (p * p + q * q))
    base = lambda x, y: coef * np.sin(p * x) * np.sin(q * y)
    func = lambda x, y, t: 0
    return answer, func, base


# def build_test_function(coef, n, m, a, b):
#     p = np.pi * n / a
#     q = np.pi * m / b
#     answer = lambda x, y, t: coef * x * y * np.sin(p * x) * np.sin(q * y) * np.exp(-t * (p * p + q * q))
#     base = lambda x, y: coef * x * y * np.sin(p * x) * np.sin(q * y)
#     func = lambda x, y, t: -np.exp(-t * (p * p + q * q)) * (2 * p * y * np.cos(p * x) + 2 * q * x * np.cos(q * y))
#     return answer, func, base


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
    b[size - 1] = (values[size - 1]
                   - matrix[size - 1][1] * b[size - 2]) / temp

    ans[size - 1] = b[size - 1]
    for i in range(size - 2, -1, -1):
        ans[i] = a[i] * ans[i + 1] + b[i]

    return ans


# Генерирует анимацию графика поверхности в зависимоти от функции и разбиения
def animate_plotting_function(func, x_grid, y_grid, t_grid, z_min, z_max):
    X, Y = np.meshgrid(x_grid, y_grid)
    fig = plt.figure()
    ax = Axes3D(fig)

    def animate(frame_num):
        ax.clear()
        ax.set_zlim(bottom=z_min, top=z_max)
        surface = ax.plot_surface(X, Y, func(X, Y, t_grid[frame_num]), rstride=1, cstride=1, alpha=0.3, linewidth=0.5,
                                  edgecolors='r')
        return surface,

    anim = animation.FuncAnimation(fig, animate, frames=len(t_grid) - 1, interval=5, blit=True)
    anim.save('anim_test_func1.gif')


# Генерирует анимацию графика поверхности в зависимоти от матрицы слоёв решения
def animate_plotting_matrix(matrix, x_grid, y_grid, t_grid):
    X, Y = np.meshgrid(x_grid, y_grid)
    fig = plt.figure()
    ax = Axes3D(fig)

    def animate(frame_num):
        ax.clear()
        ax.set_zlim(bottom=matrix.min(), top=matrix.max())
        surface = ax.plot_surface(X, Y, matrix[frame_num], rstride=1, cstride=1, alpha=0.3, linewidth=0.5,
                                  edgecolors='r')
        return surface,

    anim = animation.FuncAnimation(fig, animate, frames=len(t_grid) - 1, interval=5, blit=True)
    anim.save('anim_method1.gif')


# Создаёт матрицу из 3-ёх столбцов - диагоналей матрицы коэффициентов для системы уравнений
def build_coef_matrix(grid, t_step):
    size = len(grid)
    h = grid[1] - grid[0]
    matrix = np.zeros(shape=(size, 3))
    matrix[0][0] = 1
    matrix[size - 1][2] = 1
    for i in range(1, size - 1):
        matrix[i][0] = -t_step / 2
        matrix[i][1] = t_step + h * h
        matrix[i][2] = -t_step / 2
    return matrix


# Создаёт вектор правой части системы уравнений для прогонки по X (промежуточный слой)
def build_values_vector_1(grid, f, prev, t_step, ind, cur_t):
    size = len(grid)
    h = grid[1] - grid[0]
    values = np.zeros(shape=size)
    values[0] = 0
    values[size - 1] = 0
    for i in range(1, size - 1):
        values[i] = t_step / 2 * prev[i][ind - 1] + (h * h - t_step) * prev[i][ind] + t_step / 2 * prev[i][
            ind + 1] + t_step / 2 * h * h * f(grid[i], grid[ind], cur_t + t_step / 2)
    return values


# Создаёт вектор правой части системы уравнений для прогонки по Y (финальный слой)
def build_values_vector_2(grid, f, prev, t_step, ind, cur_t):
    size = len(grid)
    h = grid[1] - grid[0]
    values = np.zeros(shape=size)
    values[0] = 0
    values[size - 1] = 0
    for i in range(1, size - 1):
        values[i] = t_step / 2 * prev[ind - 1][i] + (h * h - t_step) * prev[ind][i] + t_step / 2 * prev[ind + 1][
            i] + t_step / 2 * h * h * f(grid[ind], grid[i], cur_t + t_step / 2)
    return values


# Вычисляет один слой по времени
def calc_one_layer(prev, h_grid, h_step, t_step, f, step_num):
    n = len(h_grid)
    mid_layer = np.zeros(shape=(n, n))
    matrix = build_coef_matrix(h_grid, t_step)
    cur_t = t_step * (step_num - 1)

    for j in range(1, n - 1):
        values = build_values_vector_1(h_grid, f, prev, t_step, j, cur_t)
        mid_layer[j] = progonka(matrix, values)

    mid_layer = mid_layer.transpose()
    new_layer = np.zeros(shape=(n, n))
    for i in range(1, n - 1):
        values = build_values_vector_2(h_grid, f, mid_layer, t_step, i, cur_t)
        new_layer[i] = progonka(matrix, values)
    return new_layer


# Вычисляет все слои в зависимости от разбиения по времени и разбиения по X, Y
def calc_all_layers(t_grid, l, f, base, n):
    t_step = t_grid[1] - t_grid[0]
    m = len(t_grid)
    h_grid = build_grid(0, l, n)
    h_step = l / n
    u = np.zeros(shape=(m, n + 1, n + 1))
    X, Y = np.meshgrid(h_grid, h_grid)
    u[0] = base(X, Y)
    for i in range(1, m):
        u[i] = calc_one_layer(u[i - 1], h_grid, h_step, t_step, f, i)
    return u


def runge_diff(u1, u2, n, method_order):
    m = -1
    for k in range(u1.shape[0]):
        for i in range(n + 1):
            for j in range(n + 1):
                eps = abs(u2[k][2 * i][2 * j] - u1[k][i][j]) / (2 ** method_order - 1)
                if eps > m:
                    m = eps
    return m


# Вычисляет разницу по Рунге на основе Евклидовой нормы разности последних слоёв двух решений
def runge_diff_euc_norm(u1, u2, n, method_order):
    sum = 0
    last1 = u1[u1.shape[0] - 1]
    last2 = u2[u2.shape[0] - 1]
    for i in range(n + 1):
        for j in range(n + 1):
            sum += (last2[2 * i][2 * j] - last1[i][j]) ** 2
    return np.sqrt(sum) / (2 ** method_order - 1)


# Вычисляет решение задачи для данного разбиения по времени с заданной точностью eps.
def simulate(t_grid, l, f, base, eps):
    n = 2
    u1 = calc_all_layers(t_grid, l, f, base, n)
    u2 = calc_all_layers(t_grid, l, f, base, 2 * n)
    r = runge_diff_euc_norm(u1, u2, n, 2)
    while r > eps:
        n *= 2
        #print('n = ' + str(n) + ', r = ' + str(round(r, 3)))
        u1 = u2
        u2 = calc_all_layers(t_grid, l, f, base, 2 * n)
        r = runge_diff_euc_norm(u1, u2, n, 2)
    return u2, 2 * n


a = 1
b = 1
n = 1
m = 1
c = 1
eps = 0.01

# Максимальное время
T = 0.1
# Число разбиений по времени
M = 30
t_step = T / M
t_grid = build_grid(0, T, M)

#ans, f, ans0 = build_uniform_test_function(c, n, m, a, b)
p = np.pi * n / a
q = np.pi * m / b
ans = lambda x, y, t: c * np.sin(p * x) * np.sin(q * y)\
                      * np.exp(-t * (p * p + q * q))
base = lambda x, y: c * np.sin(p * x) * np.sin(q * y)
f = lambda x, y, t: 0
# s = lambda x, y, t: ans(x, y, t) + np.sin(3 * p * x) * np.sin(3 * q * y)
# f = lambda x, y, t: (9 * p * p + 9 * q * q) * np.sin(3 * p * x) * np.sin(3 * q * y)
# base = lambda x, y: ans0(x, y) + np.sin(3 * p * x) * np.sin(3 * q * y)
# ans, f, ans0 = build_test_function(c, n, m, a, b)

# s = lambda x, y, t: ans(x, y, t) + np.sin(0.5 * p * x) * np.sin(0.5 * q * y)
# f = lambda x, y, t: (0.25 * p * p + 0.25 * q * q) * np.sin(0.5 * p * x) * np.sin(0.5 * q * y)
# base = lambda x, y: ans0(x, y) + np.sin(0.5 * p * x) * np.sin(0.5 * q * y)
# s = ans
# base = ans0


# s = lambda x, y, t: ans(x, y, t) + np.sin(0.5 * p * x) * np.sin(0.5 * q * y)
# f = lambda x, y, t: 50
# base = lambda x, y: ans0(x, y) + np.sin(0.5 * p * x) * np.sin(0.5 * q * y)

u, size = simulate(t_grid, a, f, base, eps=eps)
# size = 128
# u = calc_all_layers(t_grid, a, f, base, size)

print(u.shape)

h_step = a / size
h_grid = build_grid(0, a, size)

animate_plotting_matrix(u, h_grid, h_grid, t_grid)
animate_plotting_function(ans, h_grid, h_grid, t_grid, u.min(), u.max())

# for i in range(M + 1):
#     u[i] = u[i].transpose()

X, Y = np.meshgrid(h_grid, h_grid)
answer = np.zeros(shape=(M + 1, size + 1, size + 1))
for i in range(M + 1):
    answer[i] = ans(X, Y, t_grid[i])

for k in range(M + 1):
    layer = u[k]
    ans_layer = answer[k]
    diff = layer - ans_layer
    print('t = ' + str(round(t_grid[k], 4)))
    euc_norm = euclidean_norm(diff)
    print('euclidean norm: ' + str(round(euc_norm, 4)))
# for i in range(size + 1):
#     for j in range(size + 1):
#         print('{:8}'.format(round(diff[i][j], 5)), end=' ')
#     print()
# print()



# fig = plt.figure()
# ax = Axes3D(fig)
# ax.set_zlim(bottom=u.min(), top=u.max())
# surface = ax.plot_surface(X, Y, u[ind], rstride=1, cstride=1, alpha=0.3, linewidth=0.5,
#                           edgecolors='r')
# plt.show()


# for i in range(M + 1):
#     diff = infinite_norm(answer[i] - u[i])
#     print('t = ' + str(round(t_grid[i], 2)) + ', diff = ' + str(round(diff, 3)))
