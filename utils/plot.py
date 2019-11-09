import matplotlib.pyplot as plt
import numpy as np


def one_plot(n, y_label, g1_vals, g2_vals):
    plt.subplot(1, 2, n)
    if n == 2:
        lb, lb2 = 'loss', 'val_loss'
        yMin = min(min(g1_vals), min(g2_vals))
        yMax = 1.05 * max(max(g1_vals), max(g2_vals))
    else:
        lb, lb2 = 'accuracy', 'val_accuracy'
        yMin = min(min(g1_vals), min(g2_vals))
        yMax = 1.0
    plt.plot(g1_vals, color='r', label=lb, linestyle='--')
    plt.plot(g2_vals, color='b', label=lb2)
    plt.ylabel(y_label)
    plt.xlabel('Epoch')
    plt.ylim([0.95 * yMin, yMax])
    plt.legend()


def load_and_plot(path, loss_filename, val_loss_filename, acc_filename, val_acc_filename):
    loss = np.loadtxt(path + loss_filename)
    acc = np.loadtxt(path + acc_filename)
    val_loss = np.loadtxt(path + val_loss_filename)
    val_acc = np.loadtxt(path + val_acc_filename)

    one_plot(1, 'accuracy', acc, val_acc)
    one_plot(2, 'loss', loss, val_loss)
    plt.subplots_adjust(wspace=0.5)
    plt.show()