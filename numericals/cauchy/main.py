import numpy as np
import matplotlib.pyplot as plt
from numericals.cauchy.diffEquation import *


def f(t, y):
    return -y / (t * np.log(t))


def sol(t):
    return 1 / np.log(t)


t0 = np.e
T = np.e * 2
y0 = 1
n = 5

t1, y1 = euler(f, t0, T, y0, n)
t2, y2 = runge_kutta4(f, t0, T, y0, n)
t3, y3, n3 = runge_rule(f, runge_kutta4, 1, t0, T, y0, 10 ** (-4))
print(n3)

plt.figure(dpi=300)
plt.plot(t1, sol(t1))
plt.plot(t1, y1, 'ro')
#plt.plot(t2, y2, 'bo')
plt.plot(t3, y3, 'go')
plt.xlabel('t')
plt.ylabel('y')

plt.show()

print(infinite_norm(y1, sol(t1)))
print(infinite_norm(y2, sol(t2)))
print(infinite_norm(y3, sol(t3)))