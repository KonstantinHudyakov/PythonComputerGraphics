import numpy as np
import matplotlib.pyplot as plt
import keras
from tensorflow_core.python.keras.engine.input_layer import Input
from tensorflow_core.python.keras.layers.convolutional import Conv1D, Conv2D
from tensorflow_core.python.keras.layers.normalization import BatchNormalization
from tensorflow_core.python.keras.layers.pooling import MaxPooling1D, MaxPooling2D
from tensorflow_core.python.keras.models import *
from tensorflow_core.python.keras.layers.core import *

from utils.loadAndSave import load_mnist_data, load_mnist_images, save_history
from utils.plot import one_plot


def plot_wrong_predicted(test_images, test_labels, predicted_labels):
    wrong = []
    for i in range(len(test_images)):
        if np.argmax(predicted_labels[i]) != np.argmax(test_labels[i]):
            wrong.append(i)

    for i in range(15):
        plt.subplot(3, 5, i + 1)
        plt.imshow(test_images[wrong[i]], cmap='gray')
        plt.title('predicted: ' + str(np.argmax(predicted_labels[wrong[i]])) + '\n real: ' + str(
            np.argmax(test_labels[wrong[i]])))
        plt.axis('off')
    plt.subplots_adjust(hspace=0.5)
    plt.show()


path_to_data = 'data//'
path_to_history = 'history//'
model_filename = 'mnist_conv2d_model.h5'
img_rows = img_cols = 28
input_shape = (img_rows, img_cols, 1)
num_classes = 10

# Необработанные изображения и метки
train_images, train_labels, test_images, test_labels = load_mnist_images(path_to_data, img_rows, img_cols, num_classes)
# Обработанные данные для модели
x_train, y_train, x_test, y_test = load_mnist_data(path_to_data, img_rows, img_cols, num_classes)

model = load_model(path_to_history + model_filename)
model.summary()

model.evaluate(x_train, y_train, batch_size=200, verbose=2)

predicted_labels = model.predict(x_test)
plot_wrong_predicted(test_images, y_test, predicted_labels)

# Добавляем в модель новый слой
layers = model.layers
inp = Input(shape=input_shape)
x = inp
for layer in layers[1:len(layers) - 1]:
    # L.trainable = False
    x = layer(x)
x = Dropout(0.2, name='dropout_2')(x)
output = layers[-1](x)

model = Model(inputs=inp, outputs=output)
model.summary()
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=200, epochs=10, verbose=1, validation_data=(x_test, y_test))
history = history.history

acc_history = history['accuracy']
val_acc_history = history['val_accuracy']
loss_history = history['loss']
val_loss_history = history['val_loss']

# save_history(history, n_model, pathToHistory)

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
