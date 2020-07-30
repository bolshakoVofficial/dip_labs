from skimage import data, io, filters
from skimage.color import rgb2gray
from skimage.measure import compare_ssim
import glob, time


def get_edges(path_to_image):
    image = rgb2gray(data.load(path_to_image))
    # io.imshow(data.load(path_to_image))
    # io.show()
    # io.imshow(image)
    # io.show()

    # https://scikit-image.org/docs/dev/api/skimage.filters.html
    # или
    # https://scikit-image.org/docs/dev/api/skimage.feature.html
    # описание фильтров
    edges = filters.prewitt(image)
    # io.imshow(edges)
    # io.show()
    return edges


def similarity_slow(path_to_image, training_set):
    sim_sum = 0
    a = get_edges(path_to_image)
    for i in range(len(training_set)):
        b = get_edges(training_set[i])
        # https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.compare_ssim
        sim_sum += compare_ssim(a, b) + 1
    return sim_sum


start = time.time() % 60
training_apples = glob.glob("C:/Users/bolsh/Desktop/Study/DIP/lab1/Training/Apple Golden 1/*.jpg")
training_quince = glob.glob("C:/Users/bolsh/Desktop/Study/DIP/lab1/Training/Quince/*.jpg")
testing_apples = glob.glob("C:/Users/bolsh/Desktop/Study/DIP/lab1/Testing/Apple Golden 1/*.jpg")
testing_quince = glob.glob("C:/Users/bolsh/Desktop/Study/DIP/lab1/Testing/Quince/*.jpg")

fruit_identity = []
apple_ident_values = []
quince_ident_values = []
print("Check apples")

for i in range(len(testing_apples)):
    # for i in range(3):
    print('\r', i, "\t\t\t", end='')
    apple_sim = similarity_slow(testing_apples[i], training_apples)
    quince_sim = similarity_slow(testing_apples[i], training_quince)

    fruit_identity.append(1) if apple_sim > quince_sim else fruit_identity.append(0)
    apple_ident_values.append(apple_sim - quince_sim)
    print("\r", apple_sim > quince_sim)

print("Apples are checked")

print("Check Quinces")
for i in range(len(testing_quince)):
    # for i in range(3):
    print('\r', i, "\t\t\t", end='')
    apple_sim = similarity_slow(testing_quince[i], training_apples)
    quince_sim = similarity_slow(testing_quince[i], training_quince)

    fruit_identity.append(0) if apple_sim > quince_sim else fruit_identity.append(1)
    quince_ident_values.append(quince_sim - apple_sim)
    print("\r", apple_sim > quince_sim)

print("Quinces are checked")

end = time.time() % 60
print("Accuracy: ")
print(sum(fruit_identity) / len(fruit_identity))
print("Time elapsed:", end - start)
