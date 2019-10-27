import sys
import numpy as np
import matplotlib.pyplot as plt
from graphics.genLines import gen_lines, gen_dotted_lines
from graphics.genLogLines import gen_log_lines, gen_part_log_lines
from graphics.genTrap import gen_trap, gen_trap2


def add_graphics_to_plot(gen_func, plot_ind):
    graphics = gen_func(10)
    j = 0
    for i in range(plot_ind, plot_ind + 10):
        img = graphics[j]
        j += 1
        plt.subplot(6, 10, i)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    return graphics


def plot_graphics():
    plt.figure(dpi=300)
    add_graphics_to_plot(gen_lines, 1)
    add_graphics_to_plot(gen_dotted_lines, 11)
    add_graphics_to_plot(gen_log_lines, 21)
    add_graphics_to_plot(gen_part_log_lines, 31)
    add_graphics_to_plot(gen_trap, 41)
    add_graphics_to_plot(gen_trap2, 51)

    plt.subplots_adjust(hspace=0.5)
    plt.show()


def gen_data(n):
    data = gen_lines(n)
    data = np.concatenate((data, gen_dotted_lines(n)))
    data = np.concatenate((data, gen_log_lines(n)))
    data = np.concatenate((data, gen_part_log_lines(n)))
    data = np.concatenate((data, gen_trap(n)))
    data = np.concatenate((data, gen_trap2(n)))
    return data


def gen_labels(n, num_of_classes):
    labels = np.empty(shape=1, dtype='uint8')
    for i in range(0, num_of_classes):
        labels = np.concatenate((labels, np.full(n, i)))
    return labels


def gen_dataset(n):
    n //= 6
    data = gen_data(n)
    labels = gen_labels(n, 6)

    shuffled_data = []
    shuffled_labels = []
    perm = np.random.permutation(len(data))
    for ind in perm:
        shuffled_data.append(data[ind])
        shuffled_labels.append(labels[ind])
    return shuffled_data, shuffled_labels
