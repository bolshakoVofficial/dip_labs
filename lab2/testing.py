from skimage import color, io
import matplotlib.pyplot as plt

import training


def check_image(img, local_avg_color_set, show_bit=1):

    decision_class, decision_error = decision(img, local_avg_color_set)
    label = get_label_class(decision_class)

    print("Результат определения:")
    print(label)
    print("Ошибка определения: {:.3}".format(decision_error))

    if not show_bit:
        return

    fig, ax = plt.subplots()
    ax.set_title("RGB\nMSE: {:.3}, \nКласс: {}".format(decision_error, label))
    ax.imshow(img)
    ax.axis('off')
    io.show()


def decision(img, color_set):
    avg_img_color = training.calc_avg_color_mean([img])
    # avg_img_color = training.calc_avg_color_dominant([img])
    min_sum = 256
    decision_class = -1
    # print(avg_img_color)
    for n, sample in enumerate(color_set):
        r = abs(avg_img_color[0] - sample[0])
        g = abs(avg_img_color[1] - sample[1])
        b = abs(avg_img_color[2] - sample[2])
        mse = ((r**2 + g**2 + b**2) / 3)**0.5
        if mse < min_sum:
            min_sum = mse
            decision_class = n
    return decision_class, min_sum


def get_label_class(class_number):
    dict_class_label = {
        0: 'Салат с капустой',
        1: 'Красный компот',
        2: 'Гороховый суп',
        3: 'Макароны с котлетами',
        4: 'Борщ',
        5: 'Пюрешка',
        6: 'Огурцы и помидоры',
        7: 'Морковка',
        8: 'Чай',
        9: 'Светлый компот'
    }
    return dict_class_label[class_number]