import cv2
import numpy as np
import numba as nb
import pandas as pd


def local_minima(image):
    height, width = image.shape

    r, c = 1, 1

    board = np.zeros((height, width), np.uint8)
    while r + 1 < height:
        c = 1
        while c + 1 < width:
            value = image[r, c]
            minimum = np.amin(image[r-1:r+1, c-1:c+1])
            if minimum != value:
                board[r, c] = 255
            c = c + 1
        r = r + 1

    result = image | board
    return result


def preprocessing(image):

    height, width = image.shape
    resize_image = cv2.resize(
        image, (width // 3 * 2, height // 3 * 2), cv2.INTER_CUBIC)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(resize_image, kernel)
    return eroded


def find_target_pixel(preprocess_image, max_r=120):
    crop_image = preprocess_image[max_r:-max_r, max_r:-max_r]

    _, binary_image = cv2.threshold(crop_image, 127.5, 255, cv2.INTER_CUBIC)

    crop_N_binary = crop_image | binary_image

    minima_image = local_minima(crop_N_binary)
    return minima_image


@nb.jit(nopython=True)
def conv_2d(image, x, y, mask):
    height, width = mask.shape
    masking_image = image[y-height//2: y +
                          height//2, x-width//2: x+width//2] & mask
    count = np.count_nonzero(mask)

    return np.sum(masking_image)/count


def Daughman_Algorithm(image, preprocess, r_max=120, r_min=15):
    height, width = image.shape
    max_value, max_x, max_y, max_r = 0, 0, 0, 0
    daughman_values = np.zeros(
        (height-r_max*2, width-r_max*2, r_max - r_min + 1), np.float)

    # Calculate the gradient using Sobel operator
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.hypot(sobel_x, sobel_y)

    for r in range(r_min, r_max+1):
        # print(f"Processing... {round(r/120*100)}%", end="\r")
        mask = cv2.circle(np.zeros((r*2, r*2), np.uint8),
                          (r, r), r, (255), thickness=1)

        for y in range(r_max, height - r_max):
            for x in range(r_max, width - r_max):
                pre_loc_x, pre_loc_y = x-r_max, y-r_max
                if preprocess[pre_loc_y, pre_loc_x] != 255:

                    # Incorporate the gradient magnitude into the computation of daughman_values
                    daughman_values[pre_loc_y, pre_loc_x, r - r_min] = conv_2d(
                        image, x, y, mask) + gradient_magnitude[y, x]

    for r in range(r_min, r_max-1):
        for y in range(r_max, height - r_max):
            for x in range(r_max, width - r_max):
                pre_loc_x, pre_loc_y = x-r_max, y-r_max
                diff = daughman_values[pre_loc_y, pre_loc_x, r - r_min] - \
                    daughman_values[pre_loc_y, pre_loc_x, r - r_min + 1]
                if abs(diff) > max_value:
                    max_value = abs(diff)
                    max_x, max_y, max_r = x, y, r

    # circle = cv2.circle(image, (max_x, max_y), max_r, 255, 1)
    pupil = (max_x, max_y, max_r)
    max_value, max_r = 0, max_r + 10
    # for r in range(max_r, r_max):
    #     diff = daughman_values[max_y - r_max, max_x - r_max, r - r_min] - \
    #         daughman_values[max_y - r_max, max_x - r_max, r - r_min+1]
    #     if max_value < abs(diff):
    #         max_value = abs(diff)
    #         max_r = r

    # circle_image = cv2.circle(circle, (max_x, max_y), max_r, 255, 1)
    # iris = (max_x, max_y, max_r)
    # return pupil, iris, circle_image
    return pupil


def conv_2d_fast(image, x, y, mask):
    height, width = mask.shape
    masking_image = image[y-height//2: y +
                          height//2, x-width//2: x+width//2] & mask
    count = np.count_nonzero(mask)

    return np.sum(masking_image)/count


def Daughman_Algorithm_Fast(image, preprocess, r_max=120, r_min=15):
    height, width = image.shape
    max_value, max_x, max_y, max_r = 0, 0, 0, 0
    daughman_values = np.zeros(
        (height-r_max*2, width-r_max*2, r_max - r_min + 1), np.float)

    # Calculate the gradient using Sobel operator
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.hypot(sobel_x, sobel_y)

    for r in range(r_min, r_max+1):
        # print(f"Processing... {round(r/120*100)}%", end="\r")
        mask = cv2.circle(np.zeros((r*2, r*2), np.uint8),
                          (r, r), r, (255), thickness=1)

        for y in range(r_max, height - r_max):
            for x in range(r_max, width - r_max):
                pre_loc_x, pre_loc_y = x-r_max, y-r_max
                if preprocess[pre_loc_y, pre_loc_x] != 255:

                    # Incorporate the gradient magnitude into the computation of daughman_values
                    daughman_values[pre_loc_y, pre_loc_x, r - r_min] = conv_2d_fast(
                        image, x, y, mask) + gradient_magnitude[y, x]

    for r in range(r_min, r_max-1):
        for y in range(r_max, height - r_max):
            for x in range(r_max, width - r_max):
                pre_loc_x, pre_loc_y = x-r_max, y-r_max
                diff = daughman_values[pre_loc_y, pre_loc_x, r - r_min] - \
                    daughman_values[pre_loc_y, pre_loc_x, r - r_min + 1]
                if abs(diff) > max_value:
                    max_value = abs(diff)
                    max_x, max_y, max_r = x, y, r

    circle = cv2.circle(image, (max_x, max_y), max_r, 255, 1)
    pupil = (max_x, max_y, max_r)
    max_value, max_r = 0, max_r + 10
    for r in range(max_r, r_max):
        diff = daughman_values[max_y - r_max, max_x - r_max, r - r_min] - \
            daughman_values[max_y - r_max, max_x - r_max, r - r_min+1]
        if max_value < abs(diff):
            max_value = abs(diff)
            max_r = r

    circle_image = cv2.circle(circle, (max_x, max_y), max_r, 255, 1)
    iris = (max_x, max_y, max_r)
    return pupil, iris, circle_image
