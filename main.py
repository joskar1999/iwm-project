import numpy as np
import cv2

IMG_WIDTH = 1000
KERNEL_SHARPENING = np.array([[-1, -1, -1],
                              [-1, 9, -1],
                              [-1, -1, -1]])


def resize(image):
    h, w, d = image.shape
    image = cv2.resize(image, (IMG_WIDTH, int(IMG_WIDTH * h / w)))
    return image


def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def calculate_precision(image, labeled, mask):
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for i in range(len(image)):
        for j in range(len(image[i])):
            if mask[i][j] == 0:
                image[i][j] = 0

    for i in range(len(image)):
        for j in range(len(image[i])):
            if labeled[i][j] == 255 and image[i][j] == 255 and mask[i][j] == 255:
                true_positive += 1
            elif labeled[i][j] == 255 and image[i][j] == 0 and mask[i][j] == 255:
                false_negative += 1
            elif labeled[i][j] == 0 and image[i][j] == 255 and mask[i][j] == 255:
                false_positive += 1
            elif labeled[i][j] == 0 and image[i][j] == 0 and mask[i][j] == 255:
                true_negative += 1

    all_pixels = true_negative + true_positive + false_positive + false_negative

    print("all: ", all_pixels)
    print("true positive: ", true_positive)
    print("false positive: ", false_positive)
    print("true negative: ", true_negative)
    print("false negative: ", false_negative)
    print("accuracy: ", (true_positive + true_negative) / all_pixels)
    print("precision: ", true_positive / (true_positive + false_positive))
    print("specificity: ", true_negative / (false_positive + true_negative))
    print("sensitivity(recall): ", true_positive / (true_positive + false_negative))
    print()


def label_image(file_name, labeled_file_name, mask_file_name):
    image = cv2.imread(file_name, cv2.IMREAD_COLOR)
    labeled = cv2.imread(labeled_file_name, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_file_name, cv2.IMREAD_COLOR)

    image = resize(image)
    labeled = resize(labeled)
    mask = resize(mask)

    image = cv2.filter2D(image, -1, KERNEL_SHARPENING)
    image = cv2.fastNlMeansDenoisingColored(image, None, 8, 10, 7, 21)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    labeled = cv2.cvtColor(labeled, cv2.COLOR_BGR2GRAY)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 7)
    calculate_precision(image, labeled, mask)
    numpy_horizontal_concat = np.concatenate((image, labeled), axis=1)
    cv2.imshow('window', numpy_horizontal_concat)
    cv2.waitKey(0)


if __name__ == "__main__":
    label_image('./all/images/01_dr.JPG', './all/manual1/01_dr.tif', './all/mask/01_dr_mask.tif')
    label_image('./all/images/02_dr.JPG', './all/manual1/02_dr.tif', './all/mask/02_dr_mask.tif')
    label_image('./all/images/03_dr.JPG', './all/manual1/03_dr.tif', './all/mask/03_dr_mask.tif')
    label_image('./all/images/04_dr.JPG', './all/manual1/04_dr.tif', './all/mask/04_dr_mask.tif')
    label_image('./all/images/05_dr.JPG', './all/manual1/05_dr.tif', './all/mask/05_dr_mask.tif')
