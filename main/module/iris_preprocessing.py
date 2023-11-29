from mpire import WorkerPool
from dask.distributed import Client, LocalCluster
import dask
from dask.diagnostics import ProgressBar
import dask.bag as db
import multiprocessing as mp
import concurrent.futures
from tqdm.autonotebook import tqdm
import os
import threading
from itertools import product
import pandas as pd
import traceback
from skimage import measure

from module.Iris_recognition import *
from module.Periocular_recognition import *
from module.img_enhance.half_UGV import *
from module.img_enhance.reflection_removal import *
from module.densenet_seg.test import *
from module.densenet_seg.iris_seg import *


def create_iris_norm(img_folder):
    iris_norm_L = []
    iris_norm_R = []

    for dir1 in tqdm(os.listdir(img_folder)):
        for eye in os.listdir(os.path.join(img_folder, dir1)):
            for file in list(
                os.listdir(os.path.join(img_folder, dir1, eye))[i]
                for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            ):
                image_path = os.path.join(img_folder, dir1, eye, file)
                img = read_image(image_path)
                _, snake, circles = localization(img, N=400)
                pupil_circle = circles
                iris_circle = np.flip(np.array(snake).astype(int), 1)
                if circles[2] is None:
                    if eye == "L":
                        iris_norm_L.append(np.zeros((64, 400)))
                    else:
                        iris_norm_R.append(np.zeros((64, 400)))
                else:
                    # Image Preprocessing (Normalization)
                    iris_norm = normalization(img, pupil_circle, iris_circle)
                    if eye == "L":
                        iris_norm_L.append(iris_norm)
                    else:
                        iris_norm_R.append(iris_norm)

    return np.array(iris_norm_L), np.array(iris_norm_R)


def create_fold_norm(iris_norm_L, iris_norm_R, fold):
    iris_norm_L_fold = []
    iris_norm_R_fold = []
    for fol in range(1000):
        for item in range(4):
            iris_norm_L_fold.append(iris_norm_L[fol * 10 + item + fold])
            iris_norm_R_fold.append(iris_norm_R[fol * 10 + item + fold])
    return np.array(iris_norm_L_fold), np.array(iris_norm_R_fold)


def create_iris_norm_enhanced(img_folder, test_beg=None, test_til=None):
    iris_norm_L = []
    iris_norm_R = []

    total = len(os.listdir(img_folder)[test_beg:test_til]) * 2 * 10
    with tqdm(total=total, desc="Normalize Image") as pbar:
        for dir1 in os.listdir(img_folder)[test_beg:test_til]:
            for eye in os.listdir(os.path.join(img_folder, dir1)):
                for file in list(
                    os.listdir(os.path.join(img_folder, dir1, eye))[i]
                    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                ):
                    image_path = os.path.join(img_folder, dir1, eye, file)
                    image1 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                    # Get the reflection mask
                    mask = adaptive_thresholding(image1)
                    mask = region_size_filtering(mask)
                    mask = morphological_dilation(mask)

                    # Remove reflections
                    img_no_reflections = remove_reflections(image1, mask)
                    preprocess_image = preprocessing(img_no_reflections)
                    # preprocess_image = img_no_reflections
                    targeting_image = find_target_pixel(preprocess_image, 120)

                    # Process Daughman
                    pupil, iris, circle = Daughman_Algorithm(
                        preprocess_image, targeting_image, 120
                    )

                    img = read_image(image_path)
                    _, snake, circles = localization(
                        img,
                        N=400,
                        pupil_loc=(
                            pupil[0] // 2 * 3,
                            pupil[1] // 2 * 3,
                            pupil[2] // 2 * 3,
                        ),
                    )

                    pupil_circle = circles
                    iris_circle = np.flip(np.array(snake).astype(int), 1)

                    if circles[2] is None:
                        if eye == "L":
                            iris_norm_L.append(np.zeros((64, 400)))
                        else:
                            iris_norm_R.append(np.zeros((64, 400)))
                    else:
                        # Image Preprocessing (Normalization)
                        iris_norm = normalization(img, pupil_circle, iris_circle)
                        if eye == "L":
                            iris_norm_L.append(iris_norm)
                        else:
                            iris_norm_R.append(iris_norm)
                    pbar.update(1)

    return np.array(iris_norm_L), np.array(iris_norm_R)


def process_image(image_path, dir1, eye, file, df):
    image1 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Get the reflection mask
    mask = adaptive_thresholding(image1)
    mask = region_size_filtering(mask)
    mask = morphological_dilation(mask)

    # Remove reflections
    img_no_reflections = remove_reflections(image1, mask)
    preprocess_image = preprocessing(img_no_reflections)
    # preprocess_image = img_no_reflections
    targeting_image = find_target_pixel(preprocess_image, 120)

    # Process Daughman
    pupil, iris, circle = Daughman_Algorithm(preprocess_image, targeting_image, 120)

    img = read_image(image_path)
    _, snake, circles = localization(
        img, N=400, pupil_loc=(pupil[0] // 2 * 3, pupil[1] // 2 * 3, pupil[2] // 2 * 3)
    )

    pupil_circle = circles
    iris_circle = np.flip(np.array(snake).astype(int), 1)

    if circles[2] is None:
        if eye == "L":
            df["iris_norm_L"][int(dir1) * 10 + int(file[-6:-4])] = np.zeros((64, 400))
        else:
            df["iris_norm_R"][int(dir1) * 10 + int(file[-6:-4])] = np.zeros((64, 400))
    else:
        # Image Preprocessing (Normalization)
        iris_norm = normalization(img, pupil_circle, iris_circle)
        if eye == "L":
            df["iris_norm_L"][int(dir1) * 10 + int(file[-6:-4])] = iris_norm
        else:
            df["iris_norm_R"][int(dir1) * 10 + int(file[-6:-4])] = iris_norm


def create_iris_norm_enhanced_multi(img_folder, test_beg=None, test_til=None):
    total = len(os.listdir(img_folder)[test_beg:test_til]) * 2 * 10
    if test_beg is None:
        iris_norm_L = [[] for _ in range(total // 2)]
        iris_norm_R = [[] for _ in range(total // 2)]
        files_name = [[] for _ in range(total // 2)]
    else:
        iris_norm_L = [[] for _ in range(test_til * 10)]
        iris_norm_R = [[] for _ in range(test_til * 10)]
        files_name = [[] for _ in range(test_til * 10)]
    df = pd.DataFrame(
        {
            "files_name": files_name,
            "iris_norm_L": iris_norm_L,
            "iris_norm_R": iris_norm_R,
        }
    )
    dir1s = []
    files = []

    for dir1 in os.listdir(img_folder)[test_beg:test_til]:
        dir1s.append(dir1)
        for eye in os.listdir(os.path.join(img_folder, dir1)):
            for file in list(os.listdir(os.path.join(img_folder, dir1, eye))):
                if file.endswith(".jpg"):
                    files.append(file)

    with mp.Pool(processes=8) as pool:
        try:
            num_files_per_dir = 10
            dir_files = [
                files[i : i + num_files_per_dir]
                for i in range(0, len(files), num_files_per_dir)
            ]
            image_combinations = [
                (dir1s[i // 2], dir1[j])
                for i, dir1 in enumerate(dir_files)
                for j in range(len(dir1))
            ]

            results = list(
                tqdm(
                    pool.imap(process_image_multi, image_combinations),
                    total=total,
                    desc="Normalizing",
                )
            )
            for files, img_name, img_num, img in results:
                df["files_name"][img_num] = files
                df[img_name][img_num] = img

        except Exception as e:
            print(traceback.format_exc())
            print(e)
            if e != "KeyboardInterrupt":
                pd.to_pickle(df, "temp_data/iris_norm.pkl")

    return df


def process_image_multi(args):
    dir1, files = args
    image_path = os.path.join(
        "Iris-Dataset/CASIA-Iris-Thousand", dir1, files[5:6], files
    )
    image1 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Get the reflection mask
    mask = adaptive_thresholding(image1)
    mask = region_size_filtering(mask)
    mask = morphological_dilation(mask)

    # Remove reflections
    img_no_reflections = remove_reflections(image1, mask)
    preprocess_image = preprocessing(img_no_reflections)
    # preprocess_image = img_no_reflections
    targeting_image = find_target_pixel(preprocess_image, 120)

    # Process Daughman
    pupil = Daughman_Algorithm(preprocess_image, targeting_image)

    img = read_image(image_path)
    _, snake, circles = localization(
        img, N=400, pupil_loc=(pupil[0] // 2 * 3, pupil[1] // 2 * 3, pupil[2] // 2 * 3)
    )

    pupil_circle = circles
    iris_circle = np.flip(np.array(snake).astype(int), 1)

    if circles[2] is None:
        if files[5:6] == "L":
            return (
                files,
                "iris_norm_L",
                int(dir1) * 10 + int(files[-6:-4]),
                np.zeros((64, 400)),
            )
        else:
            return (
                files,
                "iris_norm_R",
                int(dir1) * 10 + int(files[-6:-4]),
                np.zeros((64, 400)),
            )

    else:
        # Image Preprocessing (Normalization)
        iris_norm = normalization(img, pupil_circle, iris_circle)
        if files[5:6] == "L":
            return files, "iris_norm_L", int(dir1) * 10 + int(files[-6:-4]), iris_norm

        else:
            return files, "iris_norm_R", int(dir1) * 10 + int(files[-6:-4]), iris_norm


def create_iris_norm_enhanced_dask(img_folder, test_beg=None, test_til=None):
    cluster = LocalCluster(n_workers=8)
    client = Client(cluster)
    total = len(os.listdir(img_folder)[test_beg:test_til]) * 2 * 10
    iris_norm_L = [[] for _ in range(total // 2)]
    iris_norm_R = [[] for _ in range(total // 2)]
    df = pd.DataFrame({"iris_norm_L": iris_norm_L, "iris_norm_R": iris_norm_R})
    dir1s = []
    files = []

    for dir1 in os.listdir(img_folder)[test_beg:test_til]:
        dir1s.append(dir1)
        for eye in os.listdir(os.path.join(img_folder, dir1)):
            for file in list(os.listdir(os.path.join(img_folder, dir1, eye))):
                files.append(file)

    with ProgressBar():
        image_combinations = [(dir1, file) for dir1 in dir1s for file in files]
        results = (
            db.from_sequence(image_combinations).map(process_image_multi).compute()
        )
        for img_name, img_num, img in results:
            df[img_name][img_num] = img

    return df


def create_iris_norm_seg(img_folder, test_beg=None, test_til=None):
    iris_norm_L = []
    iris_norm_R = []

    total = len(os.listdir(img_folder)[test_beg:test_til]) * 2 * 10
    with tqdm(total=total, desc="Normalize Image") as pbar:
        for dir1 in os.listdir(img_folder)[test_beg:test_til]:
            for eye in os.listdir(os.path.join(img_folder, dir1)):
                for file in list(
                    os.listdir(os.path.join(img_folder, dir1, eye))[i]
                    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                ):
                    image_path = os.path.join(img_folder, dir1, eye, file)
                    model_name = "densenet"
                    model_path = "D:/Users/jimyj/Desktop/TAIST/Thesis/Source_Code/main/module/densenet_seg/model/densenet_seg.pkl"

                    img, img_seg = run_prediction(
                        image_path, model_name, model_path, use_gpu=True
                    )

                    if (
                        img_seg is None
                        or measure.find_contours(img_seg, 0.6) == []
                        or measure.find_contours(img_seg, 0.9) == []
                    ):
                        if eye == "L":
                            iris_norm_L.append(np.zeros((64, 400)))
                        else:
                            iris_norm_R.append(np.zeros((64, 400)))
                        print(f"No segmentation: {image_path}")
                    else:
                        contours_iris = np.array(
                            [max(measure.find_contours(img_seg, 0.6), key=len)]
                        )
                        contours_pupil = np.array(
                            [max(measure.find_contours(img_seg, 0.9), key=len)]
                        )

                        contour_iris_inter = interpolate_pixel(contours_iris[0], 400)
                        contour_pupil_inter = interpolate_pixel(contours_pupil[0], 400)

                        iris_norm, map_area = normalization_seg(
                            img, contour_pupil_inter, contour_iris_inter
                        )

                        if eye == "L":
                            iris_norm_L.append(iris_norm * 255)
                        else:
                            iris_norm_R.append(iris_norm * 255)
                    pbar.update(1)

    return np.array(list(iris_norm_L)), np.array(list(iris_norm_R))


def save_iris_norm(iris_norm_new):
    iris_norm_L_new = np.array(
        list(
            iris_norm_new[~iris_norm_new["files_name"].apply(lambda x: len(x) == 0)][
                "iris_norm_L"
            ]
        )
    )
    iris_norm_R_new = np.array(
        list(
            iris_norm_new[~iris_norm_new["files_name"].apply(lambda x: len(x) == 0)][
                "iris_norm_R"
            ]
        )
    )

    # load the features from the file
    with np.load("temp_data/iris_norm_all.npz") as data:
        iris_norm_L = data["iris_norm_L"]
        iris_norm_R = data["iris_norm_R"]

    iris_norm_L = np.append(iris_norm_L, iris_norm_L_new, axis=0)
    iris_norm_R = np.append(iris_norm_R, iris_norm_R_new, axis=0)

    # save the features to a file
    np.savez(
        "temp_data/iris_norm_all.npz", iris_norm_L=iris_norm_L, iris_norm_R=iris_norm_R
    )
