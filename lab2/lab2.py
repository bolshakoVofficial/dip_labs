from skimage import io
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import training, testing



def segmentation(type_sample="Training"):
    os.chdir(type_sample)
    img_list = os.listdir()
    for i, img in enumerate(img_list):
        if os.path.isdir(img):
            continue
        image = cv2.imread(img)
        circles = detect_circles(img)
        try:
            os.mkdir("segmented")
        except OSError:
            pass
        os.chdir("segmented")
        for j, c in enumerate(circles[0, :]):
            x_left = int(c[0]) - int(c[2]) - 1
            x_right = int(c[0]) + int(c[2]) + 1
            y_top = int(c[1]) - int(c[2]) - 1
            y_bottom = int(c[1]) + int(c[2]) + 1
            cropped = image[y_top:y_bottom, x_left:x_right]
            cv2.imwrite(str(i) + str(j) + ".jpg", cropped)
        os.chdir(os.getcwd() + "/..")
    os.chdir(os.getcwd() + "/..")


def detect_circles(image, show_bit=0):
    src = cv2.imread(image)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1,
                               param1=100, param2=50, minDist=150,
                               minRadius=25, maxRadius=120)
    if not show_bit:
        return circles

    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(src, (i[0], i[1]), i[2], (255, 0, 0), 2)
        cv2.circle(src, (i[0], i[1]), 2, (255, 0, 0), 3)

    cv2.imshow('circles', src)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
    return circles


def cat_plot(names, values):
    plt.figure(figsize=(6, 4))
    plt.subplot()
    barlist = plt.bar(names, values)
    barlist[0].set_color('r')
    barlist[1].set_color('y')
    barlist[2].set_color('b')
    plt.suptitle('Accuracy')
    plt.show()


if __name__ == '__main__':
    avg_color_set = training.get_train_avg_color_set(show_bit=1)

    detect_circles('Testing/photo7.jpg', show_bit=1)
    segmentation(type_sample="Testing")

    print("\n")
    os.chdir("Testing/segmented")
    img_list = os.listdir()
    for img in img_list:
        image = io.imread(img)
        testing.check_image(image, avg_color_set)
        print()

    '''график строится тупо, было лень делать нормально о_о
    результаты честные. я просто запускал с одним и вторым способом нахождения "общего" цвета
    всего попыток 25, угадано было 12 и 15 картинок в зависимости от способа'''
    cat_plot(['Dominant color', 'Average color', 'Total'], [12, 15, 25])

    os.chdir(os.getcwd() + "/../../")
