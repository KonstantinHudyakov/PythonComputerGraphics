import keras
import matplotlib.pyplot as plt
import numpy as np
from tensorflow_core.python.keras.engine.input_layer import Input
from tensorflow_core.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow_core.python.keras.layers.convolutional import Conv1D, Conv2D
from tensorflow_core.python.keras.layers.normalization import BatchNormalization
from tensorflow_core.python.keras.layers.pooling import MaxPooling1D, MaxPooling2D
from tensorflow_core.python.keras.models import *
from tensorflow_core.python.keras.layers.core import *
from tensorflow_core.python.keras.optimizer_v2.adam import Adam

from utils.loadAndSave import load_graphics_data


# Создает модель генератора
def build_generator(img_shape, latent_dim):
    generator = Sequential()
    generator.add(Input(shape=(latent_dim,)))
    # generator.add(Dropout(0.4))
    generator.add(Dense(units=512))
    generator.add(LeakyReLU(alpha=0.2))
    generator.add(BatchNormalization(momentum=0.8))
    # generator.add(Dropout(0.3))
    generator.add(Dense(units=3072))
    generator.add(LeakyReLU(alpha=0.2))
    generator.add(BatchNormalization(momentum=0.8))
    # generator.add(Dropout(0.2))
    generator.add(Dense(units=6144))
    generator.add(LeakyReLU(alpha=0.2))
    generator.add(BatchNormalization(momentum=0.8))
    # generator.add(Flatten())
    # generator.add(Dropout(0.2))
    generator.add(Dense(units=4096, activation='tanh'))  # tanh
    generator.add(Reshape(target_shape=(img_shape[0], img_shape[1], 1)))

    print('Модель генератора')
    generator.summary()
    return generator


# Создает модель дискриминатора
def build_discriminator(img_shape, loss, optimizer):
    discriminator = Sequential()
    discriminator.add(Input(shape=img_shape))
    # discriminator.add(Dropout(0.4))
    # discriminator.add(Conv2D(3, kernel_size=(5, 5)))
    # discriminator.add(LeakyReLU(alpha=0.2))
    # discriminator.add(MaxPooling2D(pool_size=(3, 3), strides=1, padding='same'))
    discriminator.add(Flatten())
    # discriminator.add(Dropout(0.4))
    discriminator.add(Dense(2048))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dense(2048))
    discriminator.add(LeakyReLU(alpha=0.2))
    # discriminator.add(Dropout(0.3))
    # discriminator.add(Dense(128))
    # discriminator.add(LeakyReLU(alpha=0.2))
    # discriminator.add(Dropout(0.2))
    discriminator.add(Dense(1, activation='sigmoid'))  # sigmoid

    print('Модель дискриминатора')
    discriminator.summary()
    discriminator.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return discriminator


def save_sample_images(latent_dim, generator, epoch):
    r, c = 5, 5  # Выводим и сохраняем 25 изображений
    # latent_dim - размер шума, подаваемого на вход генератора
    noise = np.random.normal(0, 1, (r * c, latent_dim))
    gen_imgs = generator.predict(noise)
    # Возвращаемся к диапазону [0, 1]
    gen_imgs = 0.5 * gen_imgs + 0.5
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    # Сохраняем в папку images
    fig.savefig(pathToHistory + '%d.png' % epoch)
    plt.close()


def train(discriminator, generator, combined, epochs, batch_size,
          sample_interval, latent_dim, pathToHistory, x_train):
    # Приводим к диапазону [-1, 1]; activation = 'sigmoid' & activation = 'tanh'
    x_train = 2.0 * x_train - 1.0
    # Метки истинных и ложных изображений
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    d_loss = []
    d_acc = []
    g_loss = []
    cur = 0
    for epoch in range(epochs):
        # Обучаем дискриминатор
        # Выбираем batch_size случайных мзображений из обучающего множества
        # Формируем массив из batch_size целых чисел (индексов) из диапазона [0, x_train.shape[0]]
        # idx = np.random.randint(0, x_train.shape[0], batch_size)
        # # idx: [27011 19867 1049 10487 30340 12711 24354 3040 ...]
        # # Формируем массив imgs из batch_size изображений
        # imgs = x_train[idx]
        imgs = []
        for i in range(batch_size):
            imgs.append(x_train[cur])
            cur += 1
        if cur >= len(x_train):
            cur = 0

        # Шум (массив), подаваемый на вход генератора
        noise = np.random.normal(0, 1, (batch_size, latent_dim))  # numpy.ndarray: shape = (batch_size, latent_dim)
        # Генерируем batch_size изображений
        gen_imgs = generator.predict(noise)  # numpy.ndarray: shape = (batch_size, 64, 64, 1)
        # Обучаем дискриминатор, подавая ему сначала настоящие, а затем поддельные изображения

        shuffled_data = []
        shuffled_labels = []
        perm = np.random.permutation(batch_size * 2)
        for ind in perm:
            if ind < batch_size:
                shuffled_data.append(imgs[ind])
                shuffled_labels.append(valid[ind])
            else:
                shuffled_data.append(gen_imgs[ind - batch_size])
                shuffled_labels.append(fake[ind - batch_size])

        # d_hist_real = discriminator.train_on_batch(imgs, valid)  # Результат: list: [0.3801252, 0.96875]
        # d_hist_fake = discriminator.train_on_batch(gen_imgs, fake)  # Результат: list: [0.75470906, 0.1875]
        # # Усредняем результаты и получаем средние потери и точность
        # d_hist = 0.5 * np.add(d_hist_real, d_hist_fake)  # numpy.ndarray: [0.56741714 0.578125]

        d_hist = discriminator.train_on_batch(np.array(shuffled_data), np.array(shuffled_labels))

        # Обучение обобщенной модели. Реально обучается только генератор
        noise = np.random.normal(0, 1, (batch_size, latent_dim))  # numpy.ndarray: shape = (batch_size, latent_dim)
        # Обучение генератора. Метки изображений valid (единицы),
        # то есть изображения, порожденные генератором при его обучении, считаются истинными
        g_ls = combined.train_on_batch(noise, valid)  # numpy.float32: 0.6742059
        if epoch % 100 == 0:
            d_loss.append(d_hist[0])
            d_acc.append(d_hist[1])
            g_loss.append(g_ls)
        # Потери и точность дискриминатора и потери генератора
        if epoch % 100 == 0:
            print(str(epoch) + ' [D loss: ' + str(round(d_hist[0], 6)) + ', acc.: ' + str(
                round(100 * d_hist[1], 2)) + '%] [G loss: ' + str(round(g_ls, 6)) + ']')
        # Генерируем и сохраняем рисунок с 25-ю изображениями
        if epoch % sample_interval == 0:
            save_sample_images(latent_dim, generator, epoch)
            file_gen = pathToHistory + 'generator_model_%03d.h5' % epoch
            generator.save(file_gen)
    # Сохраняем обученный генератор в файл
    file_gen = pathToHistory + 'generator_model_%03d.h5' % epochs
    generator.save(file_gen)
    print('Модель генератора сохранена в файл', file_gen)

    # Вывод историй обучения в файлы
    fn_d_loss, fn_d_acc, fn_g_loss = 'd_loss.txt', 'd_acc.txt', 'g_loss.txt'
    print('Истории сохранены в файлы:\n' + fn_d_loss + '\n' + fn_d_acc + '\n' + fn_g_loss)
    print('Путь:', pathToHistory)
    with open(pathToHistory + fn_d_loss, 'w') as output:
        for val in d_loss: output.write(str(val) + '\n')
    with open(pathToHistory + fn_d_acc, 'w') as output:
        for val in d_acc: output.write(str(val) + '\n')
    with open(pathToHistory + fn_g_loss, 'w') as output:
        for val in g_loss: output.write(str(val) + '\n')
    # Вывод графиков историй обучения
    yMax = max(g_loss)
    cnt = len(g_loss)
    rng = np.arange(cnt)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(rng, d_loss, marker='o', c='blue', edgecolor='black')
    ax.scatter(rng, g_loss, marker='x', c='red')
    ax.set_title('Потери генератора (x) и дискриминатора (o)')
    ax.set_ylabel('Потери')
    ax.set_xlabel('Эпоха / 100')
    ax.set_xlim([-0.5, cnt])
    ax.set_ylim([0, 1.1 * yMax])
    fig.show()


pathToData = 'data//'
pathToHistory = 'history//'
img_rows = 64
img_cols = 64
channels = 1
num_classes = 6
img_shape = (img_rows, img_cols, channels)
optimizer = Adam(0.0002, 0.5)
loss = 'binary_crossentropy'
loss_g = 'binary_crossentropy'  # 'mse', 'poisson', 'binary_crossentropy'
# latent_dim - размер шума, подаваемого на вход генератора
# Шум - это вектор, формируемый на базе нормального распределения
# Число формируемых векторов равно batch_size
# Шум можно рассматривать как изображение размера 10*10
latent_dim = 100
epochs = 30001  # Число эпох обучения (30001)
batch_size = 30  # Размер пакета обучения (число генерируемых изображений)
sample_interval = 3000  # Интервал между сохранением сгенерированных изображений в файл

# Построение генератора
generator = build_generator(img_shape, latent_dim)
# Построение и компиляция дискриминатора
discriminator = build_discriminator(img_shape, loss, optimizer)

# Обобщенная модель
# Генератор принимает шум и возвращает (генерирует) изображения
# (их количество равно размеру пакета обучения batch_size)
combined = Sequential()
combined.add(Input(shape=(latent_dim,)))
combined.add(generator)
discriminator.trainable = False
combined.add(discriminator)

combined.compile(loss=loss_g, optimizer=optimizer)

# inp = Input(shape=(latent_dim,))
# img = generator(inp)
# В объединенной модели обучаем только генератор
# discriminator.trainable = False
# Дискриминатор принимает сгенерированное изображение
# и классифицирует его либо как истинное, либо как поддельное, возвращая validity
# output = 1, если дискриминатор посчитает, что изображение истинное, или 0 - в противном случае
# output = discriminator(img)  # <class 'tensorflow.python.framework.ops.Tensor'>: shape = (?, 1)
# Объединенная модель - стек генератора и дискриминатора
# combined = Model(inp, output)
# Поскольку метрика не задана, то после каждой эпохи вычисляются только потери
# combined.compile(loss=loss_g, optimizer=optimizer)

print('Обобщеная модель')
combined.summary()

x_train, y_train = load_graphics_data(pathToData, img_rows, img_cols)

x_train = x_train.reshape((len(x_train), img_cols, img_rows, channels)) / 255.0
y_train = keras.utils.to_categorical(y_train, num_classes=num_classes, dtype=np.uint8)

print('Обучение. Интервал между выводом изображений', sample_interval)
train(discriminator, generator, combined, epochs,
      batch_size, sample_interval, latent_dim, pathToHistory, x_train=x_train)
