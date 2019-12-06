import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from numericals.diff_schemas.lab10.common import build_grid
from numericals.diff_schemas.lab11.common import plot_graph


def one_step(h_grid, t_grid, prev, ua, ub, h_step, t_step, f, step_num):
    n = len(h_grid)
    g = t_step / (h_step * h_step)
    new = np.empty(shape=n)
    new[0] = ua
    new[n - 1] = ub
    for i in range(1, n - 1):
        new[i] = g * prev[i - 1] + (1 - 2 * g) * prev[i] + g * prev[i + 1] \
                 + t_step * f(h_grid[i]) * (1 - np.exp(-t_grid[step_num]))
    return new


def explicit_method(h_grid, t_grid, ua, ub, g, f):
    n = len(h_grid)
    layers = len(t_grid)
    h = abs(h_grid[1] - h_grid[0])
    t = abs(t_grid[1] - t_grid[0])
    u = np.empty(shape=(layers, n))
    u[0] = g(h_grid)
    for i in range(1, layers):
        u[i] = one_step(h_grid, t_grid, u[i - 1], ua, ub, h, t, f, i)
    return u


a = 0
b = 2
ua = -3
ub = 3
n = 10
m = 100
f = lambda x: 3 * x + x * x
g = lambda x: ua + (ub - ua) / b * x
# решение задачи при t -> inf
ans = lambda x: 1 / 12 * (-(x ** 4) - 6 * (x ** 3) + 68 * x - 36)

h = b / n
# t <= h * h / 2
t = h * h / 2
h_grid = build_grid(a, b, n)
t_grid = build_grid(0, m * t, m)
u = explicit_method(h_grid, t_grid, ua, ub, g, f)

step = m // 5
for i in range(5):
    ind = (i + 1) * step - 1
    plot_graph(h_grid, u[ind], 'bo', 't = ' + str(round(t * ind, 2)), xlabel='x', ylabel='u')

fig = plt.figure()
ax = plt.axes(xlim=(a, b), ylim=(ua, ub))


def animate(frame_num):
    ax.clear()
    graph = plt.plot(h_grid, u[frame_num], 'bo')
    return graph


anim = animation.FuncAnimation(fig, animate, frames=100, interval=10, blit=True)
anim.save('anim_task1.gif')


tn = t * m
un = u[99]

plt.figure(dpi=200)
plt.plot(h_grid, ans(h_grid), 'r--', label='t --> inf')
plt.plot(h_grid, un, 'bo', label='t = ' + str(round(tn, 2)))
plt.ylabel('u')
plt.xlabel('x')
plt.legend(loc='best')
plt.show()
