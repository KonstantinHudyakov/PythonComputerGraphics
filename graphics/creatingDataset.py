import sys
import numpy as np
import matplotlib.pyplot as plt
from graphics.genLines import gen_lines
from graphics.genDottedLines import gen_dotted_lines
from graphics.genLogLines import gen_log_lines
from graphics.genPartLogLine import gen_part_log_lines
from graphics.Trap import gen_trap
from graphics.Trap2 import gen_trap2

def gen_graphics_and_plot(gen_func, n, plot_ind):
    graphics = gen_func(n)
    j = 0
    for i in range(plot_ind, plot_ind + 10):
        img = graphics[j]
        j += 1
        plt.subplot(6, 10, i)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    return graphics


n = 10
plt.figure(dpi=300)
lines = gen_graphics_and_plot(gen_lines, n, 1)
dotted_lines = gen_graphics_and_plot(gen_dotted_lines, n, 11)
logs = gen_graphics_and_plot(gen_log_lines, n, 21)
part_logs = gen_graphics_and_plot(gen_part_log_lines, n, 31)
traps = gen_graphics_and_plot(gen_trap, n, 41)
traps2 = gen_graphics_and_plot(gen_trap2, n, 51)

plt.subplots_adjust(hspace=0.5)
plt.show()

