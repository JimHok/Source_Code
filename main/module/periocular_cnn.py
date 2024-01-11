from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image
from tqdm.auto import trange, tqdm
from sklearn.metrics import classification_report, confusion_matrix
from keras import backend as K
from sklearnex import patch_sklearn

patch_sklearn()


def load_VASIA(img_folder, img_num):
    iris_data = []
    iris_label = [[], [], []]

    for dir1 in tqdm(os.listdir(img_folder)):
        left_eye_files = os.listdir(os.path.join(img_folder, dir1, "L"))
        right_eye_files = os.listdir(os.path.join(img_folder, dir1, "R"))
        if len(left_eye_files) < img_num or len(right_eye_files) < img_num:
            continue
        for eye in os.listdir(os.path.join(img_folder, dir1)):
            for file in list(
                os.listdir(os.path.join(img_folder, dir1, eye))[i]
                for i in range(img_num)
            ):
                image_path = os.path.join(img_folder, dir1, eye, file)
                if image_path.endswith(".jpg") == False:
                    continue
                img = image.load_img(image_path, target_size=(64, 64))
                img = image.img_to_array(img)
                iris_data.append(img)
                iris_label[0].append(dir1 + "0" if eye == "L" else dir1 + "1")
                iris_label[1].append(file[6:8])
                iris_label[2].append(image_path)

    return np.array(iris_data), np.array(iris_label)


def load_UBIPr(img_folder, img_num):
    files_list = []
    for files in os.listdir(img_folder):
        if files.endswith(".jpg") and "S1" in files:
            files = files.replace(".jpg", "")
            files = files.split("_")
            files[0] = int(files[0].replace("C", ""))
            files[1] = int(files[1].replace("S", ""))
            files[2] = int(files[2].replace("I", ""))
            files_list.append(files)
    files_list = sorted(files_list)
    files_db = pd.DataFrame(files_list, columns=["C", "S", "I"])
    counts = files_db["C"].value_counts()
    index = counts[counts != img_num].index
    new_index = []
    for i in index:
        new_index.append(i)
        if i % 2 == 0:
            new_index.append(i - 1)
        else:
            new_index.append(i + 1)
    files_db = files_db.loc[~files_db["C"].isin(list(set(new_index)))]

    iris_data = []
    iris_label = [[], [], []]
    for C, S, I in tqdm(files_db.values):
        image_path = f"{img_folder}/C{C}_S{S}_I{I}.jpg"
        img = image.load_img(image_path, target_size=(64, 64))
        img = image.img_to_array(img)
        iris_data.append(img)
        iris_label[0].append(
            str(C // 2).zfill(3) + "0"
            if C % 2 != 0
            else str((C - 1) // 2).zfill(3) + "1"
        )
        iris_label[1].append(f"{I}".zfill(2))
        iris_label[2].append(image_path)

    return np.array(iris_data), np.array(iris_label)


def load_UBIPr_peri(img_folder, img_num):
    files_list = []
    for files in os.listdir(img_folder):
        if files.endswith(".jpg") and "S1" in files:
            files = files.replace(".jpg", "")
            files = files.split("_")
            files[0] = int(files[0].replace("C", ""))
            files[1] = int(files[1].replace("S", ""))
            files[2] = int(files[2].replace("I", ""))
            files_list.append(files)
    files_list = sorted(files_list)
    files_db = pd.DataFrame(files_list, columns=["C", "S", "I"])
    counts = files_db["C"].value_counts()
    index = counts[counts != img_num].index
    new_index = []
    for i in index:
        new_index.append(i)
        if i % 2 == 0:
            new_index.append(i - 1)
        else:
            new_index.append(i + 1)
    files_db = files_db.loc[~files_db["C"].isin(list(set(new_index)))]

    iris_data = []
    iris_label = [[], [], []]
    for C, S, I in tqdm(files_db.values):
        image_path = f"{img_folder}/C{C}_S{S}_I{I}.jpg"
        img = image.load_img(image_path, target_size=(64, 128))
        img = image.img_to_array(img)
        iris_data.append(img)
        iris_label[0].append(str((C + 1) // 2).zfill(3))
        iris_label[1].append(f"{I}".zfill(2))
        iris_label[2].append(image_path)

    return np.array(iris_data), np.array(iris_label[0]), np.array(iris_label[1])


def combine_LR(X, y, classes, img_num):
    X_combined = []
    y_combined = []
    img_label = []
    for i in range(0, classes * img_num * 2, img_num * 2):
        for j in range(img_num):
            X_combined.append(np.concatenate((X[i + j], X[i + j + img_num]), axis=1))
            y_combined.append("".join([*y[0][i]][:3]))
            img_label.append(y[1][i + j].zfill(2))
    return np.array(X_combined), np.array(y_combined), np.array(img_label)


def make_pairs(images, labels, img_label, set=1):
    pairImages = []
    pairLabels = []
    imageLabels = []

    numClasses = max(np.unique(labels)) + 1
    idx = [np.where(labels == i)[0] for i in range(0, numClasses)]
    # loop over all images
    for idxA in range(len(images)):
        # grab the current image and label belonging to the current
        # iteration
        currentImage = images[idxA]
        label = labels[idxA]
        posIdx = idx[label]
        negIdx = np.where(labels != label)[0]
        for _ in range(set):
            while True:
                # randomly pick an image that belongs to the *same* class
                idxB = np.random.choice(posIdx)
                posIdx = np.delete(posIdx, np.where(posIdx == idxB))
                if idxB != idxA:
                    break
            posImage = images[idxB]
            # prepare a positive pair and update the images and labels
            # lists, respectively
            pairImages.append([currentImage, posImage])
            pairLabels.append([1])
            imageLabels.append(
                [
                    str(labels[idxA]).zfill(3) + img_label[idxA],
                    str(labels[idxB]).zfill(3) + img_label[idxB],
                ]
            )

            # grab the indices for each of the class labels *not* equal to
            # the current label and randomly pick an image corresponding
            # to a label *not* equal to the current label
            while True:
                idxC = np.random.choice(negIdx)
                negIdx = np.delete(negIdx, np.where(negIdx == idxC))
                if idxC != idxA:
                    break
            negImage = images[idxC]
            # prepare a negative pair of images and update our lists
            pairImages.append([currentImage, negImage])
            pairLabels.append([0])
            imageLabels.append(
                [
                    str(labels[idxA]).zfill(3) + img_label[idxA],
                    str(labels[idxC]).zfill(3) + img_label[idxC],
                ]
            )

    # return a 2-tuple of our image pairs and labels
    return (np.array(pairImages), np.array(pairLabels), np.array(imageLabels))


def set_gpu():
    # set the visible devices to the GPU
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[0], "GPU")
        except RuntimeError as e:
            print(e)


def euclidean_distance(vectors):
    # unpack the vectors into separate lists
    (featsA, featsB) = vectors
    # compute the sum of squared distances between the vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)
    # return the euclidean distance between the vectors
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))


def plot_training(H, plotPath):
    # construct a plot that plots and saves the training history
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H.history["loss"], label="train_loss")
    plt.plot(H.history["val_loss"], label="val_loss")
    plt.plot(H.history["accuracy"], label="train_acc")
    plt.plot(H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")


def feature_preprocessing(
    features_train_a,
    features_test_a,
    features_train_b,
    features_test_b,
    fusion_scores_train,
    fusion_scores_test,
    use_fusion=True,
):
    X_train_final = np.abs(features_train_a - features_train_b)
    X_train_final = X_train_final.reshape(X_train_final.shape[0], -1)

    X_test_final = np.abs(features_test_a - features_test_b)
    X_test_final = X_test_final.reshape(X_test_final.shape[0], -1)

    if use_fusion:
        fusion_train_final = fusion_scores_train.reshape(
            fusion_scores_train.shape[0], -1
        )
        X_train_final = np.concatenate((X_train_final, fusion_train_final), axis=1)

        fusion_test_final = fusion_scores_test.reshape(fusion_scores_test.shape[0], -1)
        X_test_final = np.concatenate((X_test_final, fusion_test_final), axis=1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_final)
    X_test_scaled = scaler.transform(X_test_final)

    return X_train_scaled, X_test_scaled


def get_best_param(X_train_scaled, y_train_final, model, param_grid, refit=True):
    # create an SVM classifier
    model = model

    # define the parameter grid to search over
    param_grid = param_grid

    # perform the grid search
    grid_search = GridSearchCV(model, param_grid, refit=refit, verbose=2)
    grid_search.fit(X_train_scaled, y_train_final)

    return grid_search


def plot_confusion_matrix(y_test_final, y_predict):
    # compute the confusion matrix
    cm = confusion_matrix(y_test_final, y_predict)

    # plot the confusion matrix
    fig, ax = plt.subplots()
    im = ax.imshow(
        cm,
        cmap=plt.cm.Blues,
        interpolation="nearest",
        extent=(
            -0.5,
            len(np.unique(y_test_final)) - 0.5,
            len(np.unique(y_test_final)) - 0.5,
            -0.5,
        ),
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    fig.colorbar(im)

    # display the number of samples in each cell
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 1000:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white")
            else:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    # set the x and y axis ticks to display only 1 and 0
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    plt.show()
