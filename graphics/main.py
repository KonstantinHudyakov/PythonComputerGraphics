import sys
import numpy as np
import matplotlib.pyplot as plt
import keras
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
    acc = history.history['val_accuracy']
    loss = history.history['val_loss']
    plt.figure(dpi=300)

    plt.subplot(1, 2, 1)
    plt.plot(range(1, 11), acc, 'ro')
    plt.axis([0, 11, 0.5, 1])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, 11), loss, 'r--')
    plt.axis([0, 11, 0, 0.1])
    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.subplots_adjust(hspace=0.5)
    plt.show()


def predict_and_plot(model, test_data):
    classes = model.predict_classes(test_data)
    plt.figure(dpi=300)
    for i in range(15):
        img = test_data[i]
        img = img.reshape((64, 64))
        plt.subplot(3, 5, i + 1)
        plt.title(get_class_title(classes[i]))
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.subplots_adjust(hspace=0.5)
    plt.show()


# plot_graphics()

x_train, y_train = gen_dataset(3000)
x_test, y_test = gen_dataset(600)

x_train = np.array(x_train, dtype='float').reshape((len(x_train), 64, 64, 1)) / 255.0
x_test = np.array(x_test, dtype='float').reshape((len(x_test), 64, 64, 1)) / 255.0
y_train = keras.utils.to_categorical(y_train, num_classes=6, dtype='uint8')
y_test = keras.utils.to_categorical(y_test, num_classes=6, dtype='uint8')

model = Sequential()
model.add(Conv2D(4, kernel_size=(5, 5), strides=(1, 1), padding='same',
                 activation='relu', input_shape=(64, 64, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(6, activation='softmax'))

model.summary()
model.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=2, validation_data=(x_test, y_test))

plot_history(history)
predict_and_plot(model, x_train)
