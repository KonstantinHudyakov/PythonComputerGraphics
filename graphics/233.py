import sys
import numpy as np
import matplotlib.pyplot as plt
# import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow_core.python.keras.layers.convolutional import Conv2D
from tensorflow_core.python.keras.layers.pooling import MaxPooling2D
from tensorflow_core.python.keras.models import *
from tensorflow_core.python.keras.layers.core import *

from graphics.creatingDataset import *


def get_class_title(class_num):
    title = ''
    if class_num == 0:
        title = 'line'
    elif class_num == 1:
        title = 'dotted line'
    elif class_num == 2:
        title = 'sinus'
    elif class_num == 3:
        title = 'part sinus'
    elif class_num == 4:
        title = 'trapezium'
    elif class_num == 5:
        title = 'trapezium part'
    return title


def plot_history(history):
    val_acc = history.history['val_accuracy']
    acc = history.history['accuracy']
    val_loss = history.history['val_loss']
    loss = history.history['loss']
    plt.figure(dpi=200)

    plt.subplot(1, 2, 1)
    plt.plot(range(1, 11), val_acc, 'ro', label='val_acc')
    plt.plot(range(1, 11), acc, 'bo', label='acc')
    plt.legend(loc='best')
    plt.axis([0, 11, 0.5, 1])
    plt.xlabel('epoch')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, 11), val_loss, 'r--', label='val_loss')
    plt.plot(range(1, 11), loss, 'b--', label='loss')
    plt.legend(loc='best')
    plt.axis([0, 11, 0, 0.1])
    plt.xlabel('epoch')

    plt.subplots_adjust(hspace=0.5)
    plt.show()


def predict_and_plot(model, test_data):
    classes = model.predict_classes(test_data)
    plt.figure(dpi=200)
    for i in range(15):
        img = test_data[i]
        img = img.reshape((64, 64))
        plt.subplot(3, 5, i + 1)
        plt.title(get_class_title(classes[i]))
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.subplots_adjust(hspace=0.5)
    plt.show()


def calc_val_predict(model, test_data, test_labels):
    x_test = np.array(test_data, dtype='float').reshape((len(test_data), 64, 64, 1)) / 255.0
    size = len(x_test)
    right = 0
    predicted_labels = model.predict(x_test)
    false = []
    for i in range(len(x_test)):
        if np.argmax(predicted_labels[i]) == np.argmax(test_labels[i]):
            right += 1
        else:
            false.append(i)

    for i in range(min(6, len(false))):
        plt.subplot(2, 3, i + 1)
        plt.imshow(test_data[false[i]])
        plt.title(get_class_title(np.argmax(predicted_labels[false[i]])))
        plt.axis('off')
        for j in range(len(predicted_labels[false[i]])):
            predicted_labels[false[i]][j] = round(predicted_labels[false[i]][j], 2)
        print(str(i) + ')' + str(predicted_labels[false[i]]) + '\n  ' + str(test_labels[false[i]]))

    plt.subplots_adjust(hspace=0.5)
    plt.show()

    return right / size


# plot_graphics()

x_train, y_train = gen_dataset(3000)
x_test, y_test = gen_dataset(600)

# Сохраним изначальный массив изображений
test_data = x_test

x_train = np.array(x_train, dtype='float').reshape((len(x_train), 64, 64, 1)) / 255.0
x_test = np.array(x_test, dtype='float').reshape((len(x_test), 64, 64, 1)) / 255.0
y_train = keras.utils.to_categorical(y_train, num_classes=6, dtype='uint8')
y_test = keras.utils.to_categorical(y_test, num_classes=6, dtype='uint8')

model = Sequential()
# model.add(Flatten(input_shape=(64, 64, 1)))
# model.add(Dense(2048, input_shape=(4096, ) ,activation='relu'))
# model.add(Dense(512 ,activation='relu'))
# model.add(Dense(6 ,activation='softmax'))

model.add(Conv2D(4, kernel_size=(5, 5), strides=(1, 1), padding='same',
                 activation='relu', input_shape=(64, 64, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(6, activation='softmax'))

model.summary()
model.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])

datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1)

datagen.fit(x_train)

history = model.fit_generator(datagen.flow(x_train, y_train,
                                           batch_size=128), epochs=10, verbose=2, validation_data=(x_test, y_test))

history = history.history

acc_history = history['accuracy']
val_acc_history = history['val_accuracy']
loss_history = history['loss']
val_loss_history = history['val_loss']

max_acc_ind = np.argmax(acc_history)
max_val_acc_ind = np.argmax(val_acc_history)

print('max accuracy epoch = ' + str(max_acc_ind + 1))
print('max accuracy value = ' + str(round(acc_history[max_acc_ind], 4)))
print('max val_accuracy epoch = ' + str(max_val_acc_ind + 1))
print('max val_accuracy value = ' + str(round(val_acc_history[max_val_acc_ind], 4)))

images = datagen.flow(x_train, y_train, batch_size=16)
images = images[0][0]
k = -1
for i in range(len(images)):
    k += 1
    plt.subplot(2, 8, k + 1)
    img = images[i]
    img = img.reshape(64, 64)
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.axis('off')
    if k == 15: break
plt.subplots_adjust(hspace=0.1)  # wspace
plt.show()
