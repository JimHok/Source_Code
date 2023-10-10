import cv2
import numpy as np

# step a: Adaptive thresholding


def adaptive_thresholding(img, c=60):

    # compute mean in neighborhood 23x23
    mean = cv2.blur(img, (23, 23))

    # create mask where intensity > mean + c
    mask = np.where(img > mean + c, 1, 0).astype(np.uint8)

    return mask

# step b: Region size filtering


def region_size_filtering(mask, min_size=10, max_size=1000):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_size or stats[i, cv2.CC_STAT_AREA] > max_size:
            mask[labels == i] = 0

    return mask

# step c: Morphological dilation


def morphological_dilation(mask, se_size=11):
    # create circular structuring element
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))

    # apply dilation
    mask = cv2.dilate(mask, se, iterations=1)

    return mask

# Reflection removal


def remove_reflections(img, mask):
    # OpenCV inpainting
    dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)

    return dst
