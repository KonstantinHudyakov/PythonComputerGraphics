import numpy as np
import matplotlib.pyplot as plt
import keras
from tensorflow_core.python.keras.engine.input_layer import Input
from tensorflow_core.python.keras.layers.convolutional import Conv1D, Conv2D
from tensorflow_core.python.keras.layers.normalization import BatchNormalization
from tensorflow_core.python.keras.layers.pooling import MaxPooling1D, MaxPooling2D
from tensorflow_core.python.keras.models import *
from tensorflow_core.python.keras.layers.core import *
from utils.loadAndSave import load_mnist_data, save_history
from utils.plot import one_plot

pathToData = 'data//'
pathToHistory = 'history//'
# pathToData = 'C://Users//student//Desktop//'
img_rows = img_cols = 28
num_classes = 10

x_train, y_train, x_test, y_test = load_mnist_data(pathToData, img_rows, img_cols, num_classes)

n_model = 1
epochs = 10
units = [0, 0, 32]  # Если 0, то слоя Dense нет
rate = [0, 0, 0.3]  # Если 0, то слоя Dropout нет
bn = [False, False, False]  # Если False, то слоя BatchNormalization нет

use_conv = 2 # 0 - Conv-слоев нет; 1 - Conv1D; 2 - Conv2D
filters = [0, 0, 32] # Если 0, то слоя Conv1D (Conv2D) нет
rate_conv = [0, 0, 0.3] # Если 0, то слоя Dropout перед слоем Conv нет
bn_conv = [False, False, False] # Если False, то слоя BatchNormalization после слоя Conv нет

input_shape = (img_rows * img_cols, 1)
# if use_conv == 1:
#     input_shape = (img_rows * img_cols, 1)
if use_conv == 2:
    input_shape = (img_rows, img_cols, 1)

model = Sequential()
model.add(Input(shape=input_shape))

if use_conv == 1:
    x_train = x_train.reshape(len(x_train), img_rows * img_cols, 1)
    x_test = x_test.reshape(len(x_test), img_rows * img_cols, 1)

for k in range(3):
    if filters[k] > 0:
        if rate_conv[k] > 0:
            model.add(Dropout(rate_conv[k]))
        if use_conv == 1:
            model.add(Conv1D(filters[k], kernel_size=4, activation='relu'))
            model.add(MaxPooling1D(pool_size=3, strides=1, padding='same'))
        else:
            model.add(Conv2D(filters[k], kernel_size=(4, 4), activation='relu'))
            model.add(MaxPooling2D(pool_size=(3, 3), strides=1, padding='same'))
        if bn_conv[k]:
            model.add(BatchNormalization())

model.add(Flatten())

for k in range(3):
    if units[k] > 0:
        if rate[k] > 0:
            model.add(Dropout(rate[k]))
        model.add(Dense(units[k], activation='relu'))
        if bn[k]:
            model.add(BatchNormalization())

#model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()
model.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=200, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
history = history.history

acc_history = history['accuracy']
val_acc_history = history['val_accuracy']
loss_history = history['loss']
val_loss_history = history['val_loss']

save_history(history, n_model, pathToHistory)

plt.figure(dpi=200)
one_plot(1, 'accuracy', acc_history, val_acc_history)
one_plot(2, 'loss', loss_history, val_loss_history)
plt.subplots_adjust(wspace=0.5)
plt.show()

max_acc_ind = np.argmax(acc_history)
max_val_acc_ind = np.argmax(val_acc_history)

print('max accuracy epoch = ' + str(max_acc_ind + 1))
print('max accuracy value = ' + str(round(acc_history[max_acc_ind], 4)))
print('max val_accuracy epoch = ' + str(max_val_acc_ind + 1))
print('max val_accuracy value = ' + str(round(val_acc_history[max_val_acc_ind], 4)))

model_filename = pathToHistory + 'mnist_conv2d_model.h5'
model.save(model_filename)
