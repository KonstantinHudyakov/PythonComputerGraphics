import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from numericals.diff_schemas.lab10.common import build_grid
from numericals.diff_schemas.lab11.common import plot_graph


def one_step(h_grid, t_grid, prev, ua, ub, h_step, t_step, q, k, f, step_num):
    n = len(h_grid)
    g = t_step / (h_step * h_step)
    new = np.empty(shape=n)
    new[0] = ua
    new[n - 1] = ub
    for i in range(1, n - 1):
        new[i] = g * k(h_grid[i] - h_step / 2) * prev[i - 1] \
                 + (1 - g * (k(h_grid[i] + h_step / 2) + k(h_grid[i] - h_step / 2)) - t_step * q) * prev[i] \
                 + g * k(h_grid[i] + h_step / 2) * prev[i + 1] \
                 + t_step * f(h_grid[i]) * (1 - np.exp(-t_grid[step_num]))
    return new


def special_explicit_method(h_grid, t_grid, ua, ub, q, g, f, k1, k2, k3):
    def k(x):
        if x >= 0 and x <= 1:
            return k1(x)
        elif x > 1 and x <= 2:
            return k2(x)
        elif x > 2 and x <= 3:
            return k3(x)

    n = len(h_grid)
    layers = len(t_grid)
    h = abs(h_grid[1] - h_grid[0])
    t = abs(t_grid[1] - t_grid[0])
    u = np.empty(shape=(layers, n))
    u[0] = g(h_grid)
    for i in range(1, layers):
        u[i] = one_step(h_grid, t_grid, u[i - 1], ua, ub, h, t, q, k, f, i)
    return u


a = 0
b = 3
ua = 3
ub = 3
q = 0.25
f = lambda x: -(x * x) + 2.5 * x + 1.25
g = lambda x: ua + (ub - ua) / b * x
# best order - k2 k3 k1
k1 = lambda x: 7 - x
k2 = lambda x: np.log(x * x + 2)
k3 = lambda x: 5 * (x + 2) * (x + 2)
k4 = lambda x: 1

n = 60
m = 800

h = b / n
# t <= h * h / 2
t = h * h / 2 * 0.9
h_grid = build_grid(a, b, n)
t_grid = build_grid(0, m * t, m)
u = special_explicit_method(h_grid, t_grid, ua, ub, q, g, f, k4, k4, k4)

step = m // 5
for i in range(5):
    ind = (i + 1) * step - 1
    plot_graph(h_grid, u[ind], 'bo', 't = ' + str(round(t * ind, 2)), xlabel='x', ylabel='u')

fig = plt.figure()
ax = plt.axes(xlim=(a, b), ylim=(0, 2 * ub))


def animate(frame_num):
    ax.clear()
    graph = plt.plot(h_grid, u[frame_num], 'bo')
    return graph


anim = animation.FuncAnimation(fig, animate, frames=800, interval=20, blit=True)
anim.save('anim_task3.gif')