import sys # Для sys.exit()
import numpy as np
import matplotlib.pyplot as plt
import keras
#
def loadBinData(pathToData, img_rows, img_cols, num_classes, show_img, useTestData):
    print('Загрузка данных из двоичных файлов...')
    with open(pathToData + 'imagesTrain.bin', 'rb') as read_binary:
        x_train = np.fromfile(read_binary, dtype = np.uint8)
    with open(pathToData + 'labelsTrain.bin', 'rb') as read_binary:
        y_train = np.fromfile(read_binary, dtype = np.uint8)
    with open(pathToData + 'imagesTest.bin', 'rb') as read_binary:
        x_test = np.fromfile(read_binary, dtype = np.uint8)
    with open(pathToData + 'labelsTest.bin', 'rb') as read_binary:
        y_test = np.fromfile(read_binary, dtype = np.uint8)
#
##    print(x_train.shape) # (47040000,)
##    print(y_train.shape) # (60000,)
    x_train_shape = int(x_train.shape[0] / (img_rows * img_cols)) # 60000
    x_test_shape = int(x_test.shape[0] / (img_rows * img_cols)) # 10000
    x_train = x_train.reshape(x_train_shape, img_rows , img_cols, 1)
    x_test = x_test.reshape(x_test_shape, img_rows , img_cols, 1)
##        print(x_train.shape) # (60'000, 28, 28, 1)
##        print(y_train.shape) # (60'000,)
##        print(x_test.shape) # (10'000, 28, 28, 1)
##        print(y_test.shape) # (10'000,)
    if show_img:
        if useTestData:
            print('Показываем примеры тестовых данных')
        else:
            print('Показываем примеры обучающих данных')
        # Выводим обучающие картинки по цифре
        names = []
        for i in range(10): names.append(chr(48 + i)) # ['0', '1', '2', ..., '9']

        needed_number = 9
        count = 0  #к-во изображений с нужной цифрой
        i = 0 

        #Добавляем в plot 50 картинок
        while count < 50:
            
            ind = y_test[i] if useTestData else y_train[i]
            img = x_test[i] if useTestData else x_train[i]
            img = img[:, :, 0]
            if ind == needed_number:
                plt.subplot(5, 10, count + 1)
                plt.imshow(img, cmap = plt.get_cmap('gray'))
                plt.title(names[ind])
                plt.axis('off')
                count+= 1
            i+= 1
                
        plt.subplots_adjust(hspace = 0.5) # wspace
        plt.show()

        #Собираем картинки с необх цифрой
        arr = []
        for i in range(0,y_test.shape[0]):
            if y_test[i] == needed_number:
                arr.append(x_test[i])

        #Суммируем все картинки и делим на их количество

        image = [0] * 28
        for i in range(28):
            image[i] = [0] * 28
        count = 0
        for i in arr:
            image = image + i
            count += 1
        image = image / count

        #Выводим
        image = image[:, :, 0]
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap = plt.get_cmap('gray'))
        plt.title("Тестовая выборка")
        plt.axis('off')
        
        #Аналогично
        arr = []
        for i in range(0,y_train.shape[0]):
            if y_train[i] == needed_number:
                arr.append(x_train[i])
                
        image = [0] * 28
        for i in range(28):
            image[i] = [0] * 28
        count = 1
        for i in arr:
            image = image + i
            count += 1
            
        image = image / count
        image = image[:, :, 0]
        plt.subplot(1, 2, 2)
        plt.imshow(image, cmap = plt.get_cmap('gray'))
        plt.title("Обучающая выборка")
        plt.axis('off')

        plt.subplots_adjust(hspace = 0.5) # wspace
        plt.show()
                
        
    # Преобразование целочисленных данных в float32 и приведение к диапазону [0.0, 1.0]
    x_train = np.asarray(x_train, dtype = 'float32') / 255
    x_test = np.asarray(x_test, dtype = 'float32') / 255
    # Преобразование в бинарное представление: метки - числа из диапазона [0, 9] в двоичный вектор размера num_classes
    # Так, в случае MNIST метка 5 (соответствует классу 6) будет преобразована в вектор [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
    #print(y_train[0]) # (MNIST) Напечатает: 5
    print('Преобразуем массивы меток в категориальное представление')
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    #print(y_train[0]) # (MNIST) Напечатает: [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
    return x_train, y_train, x_test, y_test
#
pathToData = 'C://Users//student//Desktop//'
img_rows = img_cols = 28
num_classes = 10
show_img = True
useTestData = False # True False
#
x_train, y_train, x_test, y_test = loadBinData(pathToData, img_rows, img_cols, num_classes, show_img, useTestData)
#



from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape
input_shape = (28, 28, 1)

inp = Input (shape = input_shape)
x = Flatten()(inp)
##### x = Reshape((img_rows * img_cols,))(inp)
####
x = Dense (units = 256, activation = 'relu')(x)
output = Dense (units = 10, activation = 'softmax')(x)
model = Model (inputs = inp, outputs = output)

model.summary()
model.compile(optimizer = 'Adam', loss = 'mse', metrics = ['accuracy'])
####
history = model.fit(x_train, y_train, batch_size = 256, epochs = 20, verbose = 2, validation_data = (x_test, y_test))
