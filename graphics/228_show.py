import keras
import matplotlib.pyplot as plt
import numpy as np
from tensorflow_core.python.keras.models import load_model


def show_images(images):
    fig, axs = plt.subplots(5, 5)
    cnt = 0
    for i in range(5):
        for j in range(5):
            axs[i, j].imshow(images[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    plt.show()


path_to_history = 'history//'
model_filename = 'generator_model_30001.h5'
img_rows = img_cols = 64
batch_size = 25
latent_dim = 100

model = load_model(path_to_history + model_filename)
model.summary()

for i in range(2):
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    images = model.predict(noise)
    images = 0.5 * images + 0.5
    show_images(images)

