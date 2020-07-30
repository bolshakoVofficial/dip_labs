from tensorflow.python.keras.models import Model, model_from_json
from tensorflow.python.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# для визуализации слоев нейросети

# выбираете, какую фотку скушает нейросеть
# image_file_name = 'Vlad.jpg'
image_file_name = 'Jim.jpg'
img = image.load_img(image_file_name, target_size=(250, 250))
plt.imshow(img)

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.

# Загружаем данные об архитектуре сети из файла json
json_file = open("model_json.json", "r")
loaded_model_json = json_file.read()
json_file.close()
# Создаем модель на основе загруженных данных
loaded_model = model_from_json(loaded_model_json)
# Загружаем веса в модель
loaded_model.load_weights("model_json.h5")
loaded_model.summary()

# Номера сверточных слоев - 0, 3, 6
for layer in range(0, 7, 3):
    activation_model = Model(inputs=loaded_model.input, outputs=loaded_model.layers[layer].output)
    activation_model.summary()

    activation = activation_model.predict(img_array)
    print(activation.shape)

    plt.matshow(activation[0, :, :, 18], cmap='viridis')

    #  вывод всех карт признаков
    images_per_row = 16
    n_filters = activation.shape[-1]
    size = activation.shape[1]
    n_cols = n_filters // images_per_row

    display_grid = np.zeros((n_cols * size, images_per_row * size))

    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = activation[0, :, :, col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image

    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
plt.show()
