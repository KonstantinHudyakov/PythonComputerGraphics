import sys
import numpy as np
import matplotlib.pyplot as plt
import keras
from tensorflow_core.python.keras.layers.convolutional import Conv2D
from tensorflow_core.python.keras.layers.pooling import MaxPooling2D
from tensorflow_core.python.keras.models import *
from tensorflow_core.python.keras.layers.core import *
from graphics.creatingDataset import gen_dataset


x_train, y_train = gen_dataset(3000)
x_test, y_test = gen_dataset(600)

x_train = np.array(x_train, dtype='float') / 255.0#.reshape((len(x_train), 64 * 64)) / 255.0
x_test = np.array(x_test, dtype='float') / 255.0#.reshape((len(x_test), 64 * 64)) / 255.0
y_train = keras.utils.to_categorical(y_train, num_classes=6, dtype='uint8')
y_test = keras.utils.to_categorical(y_test, num_classes=6, dtype='uint8')


model = Sequential()
model.add(Conv2D(3, kernel_size = (5, 5), strides = (1, 1), padding = 'same',
                 activation = 'relu', input_shape = (64, 64)))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same'))
model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dense(6, activation = 'softmax'))

# model = Sequential()
# model.add(Dense(2048, input_shape=(4096,), activation='relu'))
# model.add(Dense(1024, activation='relu'))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(6, activation='softmax'))

model.summary()
model.compile(optimizer = 'Adam', loss = 'mse', metrics = ['accuracy'])

history = model.fit(x_train, y_train, batch_size = 128, epochs = 20, verbose = 2, validation_data = (x_test, y_test))