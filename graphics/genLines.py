import sys
import numpy as np
import matplotlib.pyplot as plt
import math


def gen_lines(N):
    img_rc = 64
    imgs = np.zeros((N, img_rc, img_rc), dtype='uint8')
    for i in range(N):
        a = np.random.uniform(-2., 2.)  # случайное число
        b = np.random.uniform(20., 40.)
        x = 0.
        dx = 0.25  # пискели
        while x < img_rc - 1:
            x += dx  # меняется по пикселям
            y = a * x + b
            ix = int(x)
            iy = int(y)
            if iy >= 0 and iy <= 63:
                clr = np.random.choice(range(50, 255))
                imgs[i, ix, iy] = clr
    return imgs


N = 25
img_rc = 64
# y = a * x
imgs = np.zeros((N, img_rc, img_rc), dtype='uint8')  # заполнение нулями массива размерности ()
print(imgs.shape)
# print (imgs[0, 1, 25])
all_a = []
all_b = []
for i in range(N):
    a = np.random.uniform(-2., 2.)  # случайное число
    b = np.random.uniform(20., 40.)
    ##    k = np.random.choice([-1, 1]) #выбор из списка для знака
    ##    a = k*a
    #  print(a, b)
    all_a.append(str(round(a, 2)))
    all_b.append(str(round(b, 2)))
    x = 0.
    dx = 0.25  # пискели
    while x < img_rc - 1:
        x += dx  # меняется по пикселям
        y = a * x + b
        ix = int(x)
        iy = int(y)
        if iy >= 0 and iy <= 63:
            clr = np.random.choice(range(50, 255))
            imgs[i, ix, iy] = clr
for i in range(15):
    plt.subplot(5, 3, i + 1)  # задает число отдельных изображений в окне вывода
    img = imgs[i]
    plt.imshow(img, cmap='gray')
    ttl = 'a= ' + all_a[i] + '; b= ' + all_b[i]
    plt.title(ttl)
    plt.axis('off')
plt.subplots_adjust(hspace=0.5)
plt.show()

# imgs_labels = np.zeros(N, dtype = 'uint8')
# for i in range (N):
#    imgs_labels[i] = 0
# path = 'C://Users//kostj//Desktop//Python Computer Graphics//data//'
# fn = path + 'lines_train.bin'
# fn2 = path + 'lines_labels_train.bin'
# file_data = open(fn, 'wb')
# file_labels = open(fn2, 'wb')
# file_data.write(imgs)
# file_labels.write(imgs_labels)
# file_data.close()
# file_labels.close()
