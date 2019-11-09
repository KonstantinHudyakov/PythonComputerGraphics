import numpy as np
import keras

def load_mnist_data(pathToData, img_rows, img_cols, num_classes):
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