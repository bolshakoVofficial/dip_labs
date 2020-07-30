from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np

# Каталог с данными для обучения, проверки, тестирования
train_dir = 'Training'
val_dir = 'Validation'
test_dir = 'Testing'

img_width, img_height = 250, 250  # Размеры изображения

# backend Keras Tensorflow, формат channels_last
input_shape = (img_width, img_height, 3)  # размер тензора, 3 канала для цвета

# параметры ниже меняйте под себя, чем больше фоток - тем, разумеется, лучше для нейроночки <3
epochs = 30  # Количество эпох
batch_size = 2  # Размер мини-выборки
nb_train_samples = 36  # Количество изображений для обучения
nb_validation_samples = 8  # Количество изображений для проверки
nb_test_samples = 8  # Количество изображений для тестирования


# сверточная нейронная сеть - это ...
# https://neurohive.io/ru/osnovy-data-science/glubokaya-svertochnaja-nejronnaja-set/
def create_nn():
    model = Sequential()
    # 3 каскада свертки и подвыборки, выделение важных признаков изображений
    #  Слой свертки, размер ядра 3х3, количество карт признаков - 32 шт., функция активации ReLU.
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #  Слой свертки, размер ядра 3х3, количество карт признаков - 32 шт., функция активации ReLU.
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #  Слой свертки, размер ядра 3х3, количество карт признаков - 64 шт., функция активации ReLU.
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # двухмерный вывод MaxPooling2D превращается в одномерный вектор

    model.add(Dense(64))  # полносвязный слой, 64 нейрона, output = activation(dot(input, kernel) + bias)
    model.add(Activation('relu'))  # функция активации полулинейная
    model.add(Dropout(0.5))  # слой предотвращение переобучения
    model.add(Dense(1))  # выходной слой, 1 нейрон
    model.add(Activation('sigmoid'))  # функция активации сигмоидальная

    # про функции активации, НУ ВДРУГ ИНТЕРЕСНО
    # https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


# про всё, что ниже, смотрите видосик и его канал и сайт
# https://www.youtube.com/watch?v=_bH2oh75Kdo
# https://www.asozykin.ru/courses/nnpython

def create_generator(gen, dir_name):
    generator = gen.flow_from_directory(
        dir_name,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')
    return generator


def fit_nn(loc_model, train_generator, val_generator):
    loc_model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=nb_validation_samples // batch_size)
    return loc_model


def get_data_and_label_from_gen(gen):
    x, y = zip(*(gen[i] for i in range(len(gen))))
    x_value, y_value = np.vstack(x), np.vstack(y)
    return x_value, y_value.reshape(-1)


def get_real_label(nlabel):
    return "Me" if nlabel == 1 else "Aramis"


def show_graph(image, orig_label, predict_label):
    for i, img in enumerate(image):
        plt.subplot(4, 3, i + 1)
        plt.imshow(img.reshape(img_width, img_height, 3), cmap='gray', interpolation='nearest')
        plt.title("Class: {} Prediction: {}".format(orig_label[i], predict_label[i]))
        plt.xlabel(get_real_label(predict_label[i]))
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    # Create data generator
    data_gen = ImageDataGenerator(rescale=1. / 255)
    train_gen = create_generator(data_gen, train_dir)
    val_gen = create_generator(data_gen, val_dir)
    test_gen = create_generator(data_gen, test_dir)

    # Load or create nn
    try:
        model = load_model("model.h5py")
    except (OSError, ImportError, ValueError):
        model = create_nn()
        model = fit_nn(model, train_gen, val_gen)
        model.save("model.h5py")

        model_json = model.to_json()
        json_file = open("model_json.json", "w")
        json_file.write(model_json)
        json_file.close()
        model.save_weights("model_json.h5")

    # Predict
    test_x, test_y = get_data_and_label_from_gen(test_gen)
    predict = np.round(model.predict(test_x, batch_size=batch_size)).reshape(-1)
    print("Исходная разметка: {} \nРешение машины: {}".format(test_y, predict))

    # Show results
    scores = model.evaluate_generator(test_gen)
    print("Точность: %.2f%%" % (scores[1] * 100))
    show_graph(test_x, test_y, predict)


# def get_data_and_label_from_gen(gen):
#     x, y = zip(*(gen[i] for i in range(len(gen))))
#     x_value, y_value = np.vstack(x), np.vstack(y)
#     return x_value, y_value.reshape(-1)
#
#
# model = load_model("garbage_cnn.h5")
# test_dir = 'test'
#
# data_gen = ImageDataGenerator(rescale=1. / 255)
# test_gen = create_generator(data_gen, test_dir)
#
# # Predict
# test_x, test_y = get_data_and_label_from_gen(test_gen)
# predict = np.round(model.predict(test_x, batch_size=batch_size)).reshape(-1)
# print("Исходная разметка: {} \nРешение машины: {}".format(test_y, predict))
#
# # Show results
# scores = model.evaluate_generator(test_gen)
# print("Точность: %.2f%%" % (scores[1] * 100))
# show_graph(test_x, test_y, predict)

