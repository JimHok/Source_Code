from scipy.interpolate import interp1d
import numpy as np


def interpolate_pixel(contour, N):
    # Create a function to interpolate the x and y coordinates separately
    interp_x = interp1d(
        np.arange(contour.shape[0]), contour[:, 0], kind='cubic')
    interp_y = interp1d(
        np.arange(contour.shape[0]), contour[:, 1], kind='cubic')

    # Create a new array with N evenly spaced points
    contour_inter = np.zeros((N, 2))
    contour_inter[:, 0] = interp_x(np.linspace(0, contour.shape[0]-1, N))
    contour_inter[:, 1] = interp_y(np.linspace(0, contour.shape[0]-1, N))

    # Rearrange the contour_inter array so that the pixel with the greatest x-axis value is the first element
    max_x_idx = np.argmax(contour_inter[:, 1])
    contour_inter = np.append(
        contour_inter[max_x_idx:], contour_inter[:max_x_idx], axis=0)

    return contour_inter


def normalization_seg(img, pupil_circle, iris_circle, M=64, N=400, offset=0):

    normalized = np.zeros((M, N))
    map_area = []

    for i in range(N):
        begin = pupil_circle
        end = iris_circle

        xspace = np.linspace(begin[i][0], end[i][0], M)
        yspace = np.linspace(begin[i][1], end[i][1], M)
        normalized[:, i] = [img[int(x), int(y)]
                            if 0 <= int(x) < img.shape[1] and 0 <= int(y) < img.shape[0]
                            else 0
                            for x, y in zip(xspace, yspace)]
        map_area.append([[int(y), int(x)]
                         if 0 <= int(x) < img.shape[1] and 0 <= int(y) < img.shape[0]
                         else 0
                         for x, y in zip(xspace, yspace)])

    return normalized, np.array(map_area)
