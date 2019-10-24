import sys
import numpy as np
import matplotlib.pyplot as plt
import math

def gen_part_log_lines(N):
    img_rc = 64
    imgs = np.zeros((N, img_rc, img_rc), dtype='uint8')
    for i in range(N):
        a = np.random.uniform(2., 5.)  # случайное число
        b = np.random.uniform(1., 3.)
        x = 0.
        dx = 0.25  # пискели
        while x < img_rc - 1:
            x += dx  # меняется по пикселям
            y = 32. + a * math.log(b * x)
            ix = int(x)
            iy = int(y)
            if 32 <= iy <= 63:
                clr = np.random.choice(range(100, 255))
                imgs[i, ix, iy] = clr
    return imgs



#imgs_labels = np.zeros(N, dtype = 'uint8')
#for i in range (N):
#    imgs_labels[i] = 1
#path = 'C://Users//kostj//Desktop//Python Computer Graphics//data//'
#fn = path + 'part_log_lines_train.bin'
#fn2 = path + 'part_log_lines_labels_train.bin'
#file_data = open(fn, 'wb')
#file_labels = open(fn2, 'wb')
#file_data.write(imgs)
#file_labels.write(imgs_labels)
#file_data.close()
#file_labels.close()
