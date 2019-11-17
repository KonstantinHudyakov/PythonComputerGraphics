import sys
import numpy as np
import matplotlib.pyplot as plt
import math


def gen_sinus(N):
    img_rc = 64
    imgs = np.zeros((N, img_rc, img_rc), dtype='uint8')
    signs = [1, -1]
    for i in range(N):
        a = np.random.uniform(7., 20.)
        b = np.random.uniform(0.16, 0.3)
        a *= signs[np.random.randint(0, 2)]
        b *= signs[np.random.randint(0, 2)]
        x = math.pi / abs(b) + 1
        dx = 0.25
        while x < img_rc - 2 and (b > 0 and b * x <= 3 * math.pi or b < 0 and b * x >= -3 * math.pi):
            x += dx
            y = 32. + a * math.sin(b * x)
            ix = int(x)
            iy = int(y)
            if iy >= 0 and iy <= 63:
                clr = np.random.choice(range(180, 255))
                imgs[i, ix, iy] = clr
                imgs[i, ix + 1, iy] = clr
                imgs[i, ix - 1, iy] = clr
            if iy + 1 >= 0 and iy + 1 <= 63:
                clr = np.random.choice(range(180, 255))
                imgs[i, ix, iy + 1] = clr
                imgs[i, ix + 1, iy + 1] = clr
                imgs[i, ix - 1, iy + 1] = clr
            if iy + 2 >= 0 and iy + 2 <= 63:
                clr = np.random.choice(range(180, 255))
                imgs[i, ix, iy + 2] = clr
                imgs[i, ix + 1, iy + 2] = clr
                imgs[i, ix - 1, iy + 2] = clr
    return imgs


def gen_part_sinus(N):
    img_rc = 64
    imgs = np.zeros((N, img_rc, img_rc), dtype='uint8')
    signs = [1, -1]
    for i in range(N):
        a = np.random.uniform(7., 20.)
        b = np.random.uniform(0.16, 0.3)
        a *= signs[np.random.randint(0, 2)]
        b *= signs[np.random.randint(0, 2)]
        x = math.pi / abs(b) + 1
        dx = 0.25
        part = np.random.randint(0, 2)
        while x < img_rc - 2 and (b > 0 and b * x <= 3 * math.pi or b < 0 and b * x >= -3 * math.pi):
            x += dx
            y = 32. + a * math.sin(b * x)
            ix = int(x)
            iy = int(y)
            if part == 0:
                if iy >= 0 and iy < 31:
                    clr = np.random.choice(range(180, 255))
                    imgs[i, ix, iy] = clr
                    imgs[i, ix + 1, iy] = clr
                    imgs[i, ix - 1, iy] = clr
                if iy + 1 >= 0 and iy + 1 < 31:
                    clr = np.random.choice(range(180, 255))
                    imgs[i, ix, iy + 1] = clr
                    imgs[i, ix + 1, iy + 1] = clr
                    imgs[i, ix - 1, iy + 1] = clr
                if iy + 2 >= 0 and iy + 2 < 31:
                    clr = np.random.choice(range(180, 255))
                    imgs[i, ix, iy + 2] = clr
                    imgs[i, ix + 1, iy + 2] = clr
                    imgs[i, ix - 1, iy + 2] = clr
            else:
                if iy > 32 and iy <= 63:
                    clr = np.random.choice(range(180, 255))
                    imgs[i, ix, iy] = clr
                    imgs[i, ix + 1, iy] = clr
                    imgs[i, ix - 1, iy] = clr
                if iy + 1 > 32 and iy + 1 <= 63:
                    clr = np.random.choice(range(180, 255))
                    imgs[i, ix, iy + 1] = clr
                    imgs[i, ix + 1, iy + 1] = clr
                    imgs[i, ix - 1, iy + 1] = clr
                if iy + 2 > 32 and iy + 2 <= 63:
                    clr = np.random.choice(range(180, 255))
                    imgs[i, ix, iy + 2] = clr
                    imgs[i, ix + 1, iy + 2] = clr
                    imgs[i, ix - 1, iy + 2] = clr
    return imgs

# a_all = []
# b_all = []
# imgs = gen_part_sinus(15, a_all, b_all)
# for i in range(15):
#     img = imgs[i]
#     plt.subplot(3, 5, i + 1)
#     title = 'a=' + a_all[i] + '\nb=' + b_all[i]
#     plt.title(title)
#     plt.imshow(img, cmap='gray')
#     plt.axis('off')
#
# plt.subplots_adjust(hspace=0.5)
# plt.show()