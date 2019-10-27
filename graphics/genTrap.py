import sys
import numpy as np
import matplotlib.pyplot as plt
import math


def draw_line(x1, y1, x2, y2, imgs, i):
    x = x1
    dx = 0.25
    while x <= x2:
        x += dx
        y = (x - x1) * (y2 - y1) / (x2 - x1) + y1
        ix = int(x)
        iy = int(y)
        if 0 <= iy <= 63:
            clr = np.random.choice(range(100, 255))
            imgs[i, ix, iy] = clr


def draw_straight_line(x, y1, y2, imgs, i):
    x = int(x)
    y1 = int(y1)
    y2 = int(y2)
    for y in range(y1, y2 + 1):
        clr = np.random.choice(range(100, 255))
        imgs[i, x, y] = clr


def gen_trap(N):
    img_rc = 64
    imgs = np.zeros((N, img_rc, img_rc), dtype='uint8')  # заполнение нулями массива размерности ()
    for i in range(N):
        x1 = np.random.uniform(5., 25.)
        x2 = np.random.uniform(35., 63.)
        y1 = np.random.uniform(3., 10.)
        y2 = np.random.uniform(16., 26.)
        y3 = np.random.uniform(32., 42.)
        y4 = np.random.uniform(48., 60.)
        draw_straight_line(x1, y1, y4, imgs, i)
        draw_straight_line(x2, y2, y3, imgs, i)
        draw_line(x1, y1, x2, y2, imgs, i)
        draw_line(x1, y4, x2, y3, imgs, i)
    return imgs


def gen_trap2(N):
    img_rc = 64
    imgs = np.zeros((N, img_rc, img_rc), dtype='uint8')

    for i in range(N):
        x1 = np.random.uniform(5., 25.)
        x2 = np.random.uniform(35., 63.)
        y1 = np.random.uniform(3., 10.)
        y2 = np.random.uniform(16., 26.)
        y3 = np.random.uniform(32., 42.)
        y4 = np.random.uniform(48., 60.)
        r = np.random.randint(1, 5)
        if r == 1:
            draw_straight_line(x2, y2, y3, imgs, i)
            draw_line(x1, y1, x2, y2, imgs, i)
            draw_line(x1, y4, x2, y3, imgs, i)
        elif r == 2:
            draw_straight_line(x1, y1, y4, imgs, i)
            draw_line(x1, y1, x2, y2, imgs, i)
            draw_line(x1, y4, x2, y3, imgs, i)
        elif r == 3:
            draw_straight_line(x1, y1, y4, imgs, i)
            draw_straight_line(x2, y2, y3, imgs, i)
            draw_line(x1, y4, x2, y3, imgs, i)
        elif r == 4:
            draw_straight_line(x1, y1, y4, imgs, i)
            draw_straight_line(x2, y2, y3, imgs, i)
            draw_line(x1, y1, x2, y2, imgs, i)
    return imgs

# imgs_labels = np.zeros(N, dtype = 'uint8')
# for i in range (N):
#    imgs_labels[i] = 1
# path = 'C://Users//kostj//Desktop//Python Computer Graphics//data//'
# fn = path + 'trap_train.bin'
# fn2 = path + 'trap_labels_train.bin'
# file_data = open(fn, 'wb')
# file_labels = open(fn2, 'wb')
# file_data.write(imgs)
# file_labels.write(imgs_labels)
# file_data.close()
# file_labels.close()
