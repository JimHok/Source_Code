from io import StringIO
import streamlit as st
import pandas as pd
from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import math
import matplotlib


def read_image(path):
    img = cv2.imread(path)
    gray_eye_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_eye_image


def find_pupil_new(img):
    img = cv2.medianBlur(img, 15)
    img = cv2.Canny(img, 0, 50)
    param1 = 200  # 200
    param2 = 120  # 150
    circles = None
    while circles is None and param2 > 20:
        # HoughCircles
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 1,
                                   param1=param1, param2=param2,
                                   minRadius=20, maxRadius=60)

        param2 -= 1

    if circles is None:
        return None, None, None

    return circles.astype(int)[0][0]


def lash_removal(img, thresh=40):
    ref = img < thresh
    coords = np.where(ref == 1)
    rmov_img = img.astype(float)
    rmov_img[coords] = float('nan')
    return rmov_img


def lash_removal_daugman(img, thresh=40):
    ref = img < thresh
    coords = np.where(ref == 1)
    rmov_img = img.astype(float)
    rmov_img[coords] = float('nan')
    temp_img = rmov_img.copy()
    temp_img[coords] = 255/2
    avg = np.sum(temp_img) / (rmov_img.shape[0] * rmov_img.shape[1])
    rmov_img[coords] = avg

    noise_img = np.zeros(img.shape)
    noise_img[coords] = 1
    return rmov_img, noise_img.astype(bool)


def localization(img, N=400, alpha=1.6, beta=500, gamma=0.05):
    DoG = cv2.GaussianBlur(img, (3, 3), 0) - cv2.GaussianBlur(img, (25, 25), 0)
    median1 = cv2.medianBlur(DoG, 9)
    eroted = cv2.erode(median1, np.ones((3, 3), np.uint8), iterations=1)
    median2 = cv2.medianBlur(eroted, 5)
    dilated = cv2.dilate(median2, np.ones((3, 3), np.uint8), iterations=1)
    eroted = cv2.erode(dilated, np.ones((5, 5), np.uint8), iterations=1)
    result = cv2.bitwise_or(img, eroted)

    x, y, rad = find_pupil_new(img)

    if x is None:
        x, y = 350, 250

    s = np.linspace(0, 2*np.pi, 400)
    c = x + 150*np.cos(s)
    r = y + 150*np.sin(s)
    init = np.array([r, c]).T

    snake = active_contour(result, init, alpha=alpha, beta=beta, gamma=gamma)

    return init, snake, (x, y, rad)


def trans_axis(circle, theta):

    x0, y0, r = circle
    x = int(x0 + r * math.cos(theta))
    y = int(y0 + r * math.sin(theta))
    return x, y


def normalization(img, pupil_circle, iris_circle, M=64, N=400, offset=0):

    normalized = np.zeros((M, N))
    theta = np.linspace(0, 2 * np.pi, N)

    for i in range(N):
        curr_theta = theta[i] + offset
        if curr_theta > 2 * np.pi:
            curr_theta -= 2 * np.pi
        begin = trans_axis(pupil_circle, curr_theta)
        end = iris_circle

        xspace = np.linspace(begin[0], end[i][0], M)
        yspace = np.linspace(begin[1], end[i][1], M)
        normalized[:, i] = [img[int(y), int(x)]
                            if 0 <= int(x) < img.shape[1] and 0 <= int(y) < img.shape[0]
                            else 0
                            for x, y in zip(xspace, yspace)]
    return normalized


def masked(img, snake, circles):
    mask1 = np.zeros_like(img)
    mask1 = cv2.circle(mask1, (int(circles[0]), int(
        circles[1])), int(circles[2]), (255, 255, 255), -1)
    mask2 = np.zeros_like(img)
    mask2[snake[:, 0].astype(int), snake[:, 1].astype(int)] = 255

    contours, _ = cv2.findContours(mask2, 2, 2)
    for i in range(len(contours)):
        cv2.drawContours(mask2, contours, i, (255, 255, 255), 3, cv2.LINE_8)

    contours, _ = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        cv2.drawContours(mask2, [cnt], -1, 255, -1)

    mask = cv2.subtract(mask2, mask1)
    masked_gray = cv2.bitwise_and(img, img, mask=mask)
    return masked_gray


def gaborconvolve_f(img, minw_length, mult, sigma_f):
    """
    Convolve each row of an imgage with 1D log-Gabor filters.
    """
    rows, ndata = img.shape
    logGabor_f = np.zeros(ndata)
    filterb = np.zeros([rows, ndata], dtype=complex)

    radius = np.arange(ndata/2 + 1) / (ndata/2) / 2
    radius[0] = 1

    # filter wavelength
    wavelength = minw_length

    # radial filter component
    fo = 1 / wavelength
    logGabor_f[0: int(ndata/2) + 1] = np.exp((-(np.log(radius/fo))**2) /
                                             (2 * np.log(sigma_f)**2))
    logGabor_f[0] = 0

    # convolution for each row
    for r in range(rows):
        signal = img[r, 0:ndata]
        imagefft = np.fft.fft(signal)
        filterb[r, :] = np.fft.ifft(imagefft * logGabor_f)

    return filterb


def encode_iris(arr_polar, arr_noise, minw_length, mult, sigma_f):
    """
    Generate iris template and noise mask from the normalised iris region.
    """
    # convolve with gabor filters
    filterb = gaborconvolve_f(arr_polar, minw_length, mult, sigma_f)
    l = arr_polar.shape[1]
    template = np.zeros([arr_polar.shape[0], 2 * l])
    h = np.arange(arr_polar.shape[0])

    # making the iris template
    mask_noise = np.zeros(template.shape)
    filt = filterb[:, :]

    # quantization and check to se if the phase data is useful
    H1 = np.real(filt) > 0
    H2 = np.imag(filt) > 0

    H3 = np.abs(filt) < 0.0001
    for i in range(l):
        ja = 2 * i

        # biometric template
        template[:, ja] = H1[:, i]
        template[:, ja + 1] = H2[:, i]
        # noise mask_noise
        mask_noise[:, ja] = arr_noise[:, i] | H3[:, i]
        mask_noise[:, ja + 1] = arr_noise[:, i] | H3[:, i]

    return template, mask_noise


def shiftbits_ham(template, noshifts):
    templatenew = np.zeros(template.shape)
    width = template.shape[1]
    s = 2 * np.abs(noshifts)
    p = width - s

    if noshifts == 0:
        templatenew = template

    elif noshifts < 0:
        x = np.arange(p)
        templatenew[:, x] = template[:, s + x]
        x = np.arange(p, width)
        templatenew[:, x] = template[:, x - p]

    else:
        x = np.arange(s, width)
        templatenew[:, x] = template[:, x - s]
        x = np.arange(s)
        templatenew[:, x] = template[:, p + x]

    return templatenew


def HammingDistance(template1, mask1, template2, mask2):
    hd = np.nan

    # Shifting template left and right, use the lowest Hamming distance
    for shifts in range(-8, 9):
        template1s = shiftbits_ham(template1, shifts)
        mask1s = shiftbits_ham(mask1, shifts)

        mask = np.logical_and(mask1s, mask2)
        nummaskbits = np.sum(mask == 1)
        totalbits = template1s.size - nummaskbits

        C = np.logical_xor(template1s, template2)
        C = np.logical_and(C, np.logical_not(mask))
        bitsdiff = np.sum(C == 1)

        if totalbits == 0:
            hd = np.nan
        else:
            hd1 = bitsdiff / totalbits
            if hd1 > hd or np.isnan(hd):
                hd = hd1

    return hd


img_num = 1
plot_size = 20

st.title('Iris Recognition Demo')

param = pd.read_csv(
    'C:/Users/jimyj/Desktop/TAIST/Thesis/Source_Code/param.csv')
tb_alpha = param['alpha']
tb_beta = param['beta']
tb_gamma = param['gamma']

op_1_fol = st.sidebar.selectbox(
    "Select Reference Iris Image Folder",
    (f'Iris Folder {i}' for i in range(60)),
)

op_1_item = st.sidebar.selectbox(
    "Select Reference Iris Image file",
    (f'Iris Image {i}' for i in range(20)),
)

op_2_fol = st.sidebar.selectbox(
    "Select Match Iris Image Folder",
    (f'Iris Folder {i}' for i in range(60)),
)

op_2_item = st.sidebar.selectbox(
    "Select Match Iris Image file",
    (f'Iris Image {i}' for i in range(20)),
)

normalize_func = st.sidebar.container()
container = st.sidebar.container()

col1, col2, col3, col4, col5, col6 = st.sidebar.columns(
    [1.6, 1.6, 1.6, 1.6, 1.6, 1.6])

fields = ['Index', 'Alpha', 'Beta', 'Gamma', 'Load', 'Delete']
for col, field in zip([col1, col2, col3, col4, col5, col6], fields):
    col.write(field)

i = 0
while i < len(tb_alpha):
    col1, col2, col3, col4, col5, col6 = st.sidebar.columns(
        [1.6, 1.6, 1.6, 1.6, 1.6, 1.6])
    button_phold = col5.empty()
    if button_phold.button('Load', key=i):
        st.session_state.alpha = tb_alpha[i]
        st.session_state.beta = tb_beta[i]
        st.session_state.gamma = tb_gamma[i]
    button_phold1 = col6.empty()
    if button_phold1.button('Delete', key=i+100):
        tb_alpha.pop(i)
        tb_beta.pop(i)
        tb_gamma.pop(i)
        param = pd.DataFrame(
            {'alpha': tb_alpha, 'beta': tb_beta, 'gamma': tb_gamma})
        param.to_csv('param.csv', index=False)
        st.experimental_rerun()
    col1.write(i)
    col2.write(tb_alpha[i])
    col3.write(tb_beta[i])
    col4.write(tb_gamma[i])
    i += 1

alpha = container.slider('Alpha', 0.0, 10.0, 1.6, 0.05, key='alpha')
beta = container.slider('Beta', 0.0, 1000.0, 500.0, 5.0, key='beta')
gamma = container.slider('Gamma', 0.0, 0.5, 0.05, 0.01, key='gamma')

if container.button('Save Variables'):
    tb_alpha = tb_alpha.to_list()
    tb_alpha.append(alpha)
    tb_beta = tb_beta.to_list()
    tb_beta.append(beta)
    tb_gamma = tb_gamma.to_list()
    tb_gamma.append(gamma)
    param = pd.DataFrame(
        {'alpha': tb_alpha, 'beta': tb_beta, 'gamma': tb_gamma})
    param.to_csv('param.csv', index=False)
    st.experimental_rerun()

with st.spinner('Loading Image...'):
    img_1_fol = int(op_1_fol.split(' ')[2])
    img_1_item = int(op_1_item.split(' ')[2])

    img_2_fol = int(op_2_fol.split(' ')[2])
    img_2_item = int(op_2_item.split(' ')[2])

    img_1 = read_image(
        f'C:/Users/jimyj/Desktop/TAIST/Thesis/Source_Code/CASIA-IrisV2/device1/00{str(img_1_fol).zfill(2)}/00{str(img_1_fol).zfill(2)}_0{str(img_1_item).zfill(2)}.bmp')

    img_2 = read_image(
        f'C:/Users/jimyj/Desktop/TAIST/Thesis/Source_Code/CASIA-IrisV2/device1/00{str(img_2_fol).zfill(2)}/00{str(img_2_fol).zfill(2)}_0{str(img_2_item).zfill(2)}.bmp')

    matplotlib.rcParams.update({'font.size': 22})
    fig = plt.figure(figsize=(20, plot_size), constrained_layout=False)
    outer_grid = fig.add_gridspec(5, 2, wspace=0.1, hspace=-0.5)

    imgs = [img_1, img_2]
    templates = []
    masks = []

    for i in range(2):
        img = imgs[i]

        _, snake, circles = localization(
            img, N=400, alpha=alpha, beta=beta, gamma=gamma)

        pupil_circle = circles
        iris_circle = np.flip(np.array(snake).astype(int), 1)

        ax0 = fig.add_subplot(outer_grid[:3, i])
        ax1 = fig.add_subplot(outer_grid[3, i])
        ax2 = fig.add_subplot(outer_grid[4, i])

        ax0.imshow(img, cmap='gray')
        ax0.plot(snake[:, 1], snake[:, 0], '-b', lw=2)
        ax0.set_title(f'Reference Image', fontsize=40)

        if circles[0] is None:
            err_msg = f'<p style="color:Red; font-size: 20px;">No circles found in image</p>'
            st.markdown(err_msg, unsafe_allow_html=True)
            ax1.imshow(img, cmap='gray')
            ax1.axis([0, 400, 64, 0])
        else:
            circle = plt.Circle((circles[0], circles[1]),
                                circles[2], color='g', fill=False, linewidth=2)
            ax0.add_patch(circle)
            ax0.scatter(circles[0], circles[1], s=20, c='g', marker='o')

            # Image Preprocessing (Normalization)
            iris_norm = normalization(img, pupil_circle, iris_circle)

            # rmov_img = lash_removal(iris_norm, thresh=50)
            # rmov_img_test = lash_removal_daugman(iris_norm, thresh=30)

            # iris_norm_op = normalize_func.selectbox(
            #     "Select a Iris normalization method",
            #     (f'{i} Image' for i in ['Inverse', 'Non-Inverse'])).split(' ')[0]

            # if iris_norm_op == 'Inverse':
            #     iris_norm = 255-iris_norm
            #     rmov_img = 255-rmov_img
            #     rmov_img_test = 255-rmov_img_test

            ax1.imshow(iris_norm, cmap='gray')
            ax1.set_title(f'Normalized Image', fontsize=40)

            # Feature Extraction
            romv_img, noise_img = lash_removal_daugman(iris_norm, thresh=50)
            template, mask_noise = encode_iris(
                romv_img, noise_img, minw_length=18, mult=1, sigma_f=0.5)

            templates.append(template)
            masks.append(mask_noise)

            ax2.imshow(template, cmap='gray')
            ax2.set_title(f'Binary Encoded Image', fontsize=40)

            # Matching
            if len(templates) >= 2:
                hd_raw = HammingDistance(
                    templates[0], masks[0], templates[i], masks[i])
                ax0.set_title(
                    f'Hamming Dist: {round(hd_raw, 4)}', fontsize=40)

    # st.write(f"Hammimg Distance: {round(hd_raw, 4)}")

    st.pyplot(fig)
