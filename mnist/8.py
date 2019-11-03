import numpy as np
import matplotlib.pyplot as plt
import keras
from tensorflow_core.python.keras.layers.normalization import BatchNormalization
from tensorflow_core.python.keras.models import *
from tensorflow_core.python.keras.layers.core import *


def loadBinData(pathToData, img_rows, img_cols, num_classes):
    with open(pathToData + 'imagesTrain.bin', 'rb') as read_binary:
        x_train = np.fromfile(read_binary, dtype=np.uint8)
    with open(pathToData + 'labelsTrain.bin', 'rb') as read_binary:
        y_train = np.fromfile(read_binary, dtype=np.uint8)
    with open(pathToData + 'imagesTest.bin', 'rb') as read_binary:
        x_test = np.fromfile(read_binary, dtype=np.uint8)
    with open(pathToData + 'labelsTest.bin', 'rb') as read_binary:
        y_test = np.fromfile(read_binary, dtype=np.uint8)

    ##    print(x_train.shape) # (47040000,)
    ##    print(y_train.shape) # (60000,)
    x_train_shape = int(x_train.shape[0] / (img_rows * img_cols))  # 60000
    x_test_shape = int(x_test.shape[0] / (img_rows * img_cols))  # 10000
    x_train = x_train.reshape(x_train_shape, img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test_shape, img_rows, img_cols, 1)

    x_train = np.asarray(x_train, dtype='float32') / 255
    x_test = np.asarray(x_test, dtype='float32') / 255
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test


def save_history(history, n_model, path):
    suff = str(n_model) + '.txt'
    fn_loss = path + 'loss_' + suff
    fn_acc = path + 'acc_' + suff
    fn_val_loss = path + 'val_loss_' + suff
    fn_val_acc = path + 'val_acc_' + suff

    with open(fn_loss, 'w') as output:
        for val in history['loss']: output.write(str(val) + '\n')
    with open(fn_acc, 'w') as output:
        for val in history['accuracy']: output.write(str(val) + '\n')
    with open(fn_val_loss, 'w') as output:
        for val in history['val_loss']: output.write(str(val) + '\n')
    with open(fn_val_acc, 'w') as output:
        for val in history['val_accuracy']: output.write(str(val) + '\n')


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


pathToData = 'data//'
pathToHistory = 'history//'
# pathToData = 'C://Users//student//Desktop//'
img_rows = img_cols = 28
num_classes = 10
input_shape = (img_rows, img_cols, 1)

x_train, y_train, x_test, y_test = loadBinData(pathToData, img_rows, img_cols, num_classes)

n_model = 1
epochs = 100
units = [256, 128, num_classes]  # Если 0, то слоя Dense нет
rate = [0.4, 0.3, 0.2]  # Если 0, то слоя Dropout нет
bn = [False, True, False]  # Если False, то слоя BatchNormalization нет

model = Sequential()
model.add(Flatten(input_shape=input_shape))

for k in range(3):
    if units[k] > 0:
        if rate[k] > 0:
            model.add(Dropout(rate[k]))
        model.add(Dense(units[k], activation='relu'))
        if bn[k]:
            model.add(BatchNormalization())

model.summary()
model.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=128, epochs=epochs, verbose=2, validation_data=(x_test, y_test))
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
plt.savefig(pathToHistory + 'graphics_' + str(n_model))

max_acc_ind = np.argmax(acc_history)
max_val_acc_ind = np.argmax(val_acc_history)

print('max accuracy epoch = ' + str(max_acc_ind + 1))
print('max accuracy value = ' + str(round(acc_history[max_acc_ind], 4)))
print('max val_accuracy epoch = ' + str(max_val_acc_ind + 1))
print('max val_accuracy value = ' + str(round(val_acc_history[max_val_acc_ind], 4)))
