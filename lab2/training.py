from skimage import color, io, img_as_float
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

import testing

AMOUNT_OF_CLASS = 10


def create_dataset(sample_number, type_sample="Training/segmented"):
    dataset = []
    if sample_number == -1:
        os.chdir(type_sample)
    else:
        os.chdir(type_sample + "/" + str(sample_number))
    img_list = os.listdir()
    for img in img_list:
        dataset.append(io.imread(img))
    if sample_number == -1:
        os.chdir(os.getcwd() + "/../../")
    else:
        os.chdir(os.getcwd() + "/../../../")
    return dataset


def calc_avg_color(dataset):
    avg_color = 0
    cntr_white_pixels = 0
    for n, img in enumerate(dataset):
        for i in range(dataset[0].shape[0]):
            for j in range(dataset[0].shape[1]):
                try:
                    # print("img[i][j][0] ", img[i][j][0])
                    # print("img[i][j][1] ", img[i][j][1])
                    d = img[i][j][0] + img[i][j][1]  # white pixel on not
                    average = img.mean(axis=0).mean(axis=0)
                    print(average)
                except:
                    d = 0
                if d < 0.05:
                    cntr_white_pixels += 1
                else:
                    avg_color += img[i][j]
    denominator = dataset[0].shape[0] * dataset[0].shape[1] * len(dataset) - cntr_white_pixels
    # print(denominator, avg_color)
    avg_color /= denominator
    return avg_color


# находим средний цвет изображения
def calc_avg_color_mean(dataset):
    r, g, b = 0, 0, 0
    for n, img in enumerate(dataset):
        average = img.mean(axis=0).mean(axis=0)
        r += average[0]
        g += average[1]
        b += average[2]
    denom = len(dataset)
    avg_color = [r / denom, g / denom, b / denom]
    # avg_color = [round(r/denom), round(g/denom), round(b/denom)]
    return avg_color


# с помощью кластеризации (kmeans) выделяю 10 цветов и наиболее распространенный из них
# использую как доминирующий во всем изображении
def calc_avg_color_dominant(dataset):
    r, g, b = 0, 0, 0
    for n, img in enumerate(dataset):
        pixels = np.float32(img.reshape(-1, 3))
        n_colors = 10
        # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
        _, counts = np.unique(labels, return_counts=True)

        dominant = palette[np.argmax(counts)]
        r += dominant[0]
        g += dominant[1]
        b += dominant[2]
    denom = len(dataset)
    avg_color = [r / denom, g / denom, b / denom]
    return avg_color


def show_mean_dom_color(dataset):
    for n, img in enumerate(dataset):
        average = img.mean(axis=0).mean(axis=0)
        pixels = np.float32(img.reshape(-1, 3))
        n_colors = 10
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
        _, counts = np.unique(labels, return_counts=True)
        dominant = palette[np.argmax(counts)]

        avg_patch = np.ones(shape=img.shape, dtype=np.uint8) * np.uint8(average)

        indices = np.argsort(counts)[::-1]
        freqs = np.cumsum(np.hstack([[0], counts[indices] / counts.sum()]))
        rows = np.int_(img.shape[0] * freqs)

        dom_patch = np.zeros(shape=img.shape, dtype=np.uint8)
        for i in range(len(rows) - 1):
            dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])

        fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(12, 6))
        ax0.imshow(avg_patch)
        ax0.set_title('Average color')
        ax0.axis('off')
        ax1.imshow(dom_patch)
        ax1.set_title('Dominant colors')
        ax1.axis('off')
        ax2.imshow(img)
        ax2.axis('off')
        plt.show(fig)


# раскомментировать две строчки чтобы показывались средний и доминантный цвет
def get_train_avg_color_set(show_bit=0):
    train_dataset = []
    avg_color_set = []
    for i in range(AMOUNT_OF_CLASS):
        '''я использую разные методы для расчета среднего цвета
        средний цвет по изображению (calc_avg_color_mean)
        vs
        доминантный цвет в изображении (calc_avg_color_dominant)
        средний как ни странно работает лучше'''
        train_dataset.append(create_dataset(i))
        # avg_color_set.append(calc_avg_color_dominant(train_dataset[i]))
        avg_color_set.append(calc_avg_color_mean(train_dataset[i]))
        # show_mean_dom_color(train_dataset[i])

    if not show_bit:
        return avg_color_set

    for i in range(AMOUNT_OF_CLASS):
        print(testing.get_label_class(i))
        print(avg_color_set[i])
    return avg_color_set
