import numpy as np
from utils.loadAndSave import gen_and_save_graphics, load_graphics_data

path = 'data//'
data_filename = 'graphics_train.bin'
labels_filename = 'graphics_train_labels.bin'

data1, labels1 = gen_and_save_graphics(12000, path, data_filename, labels_filename)
data2, labels2 = load_graphics_data(path, 64, 64)