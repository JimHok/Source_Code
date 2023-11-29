from module.Iris_recognition import *
from module.Periocular_recognition import *
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import concurrent.futures
import os
import sys

sys.path.append("D:/Users/jimyj/Desktop/TAIST/Thesis/Source_Code/main")


def create_dataset(img_folder, fold):
    test_data = []
    test_label = []
    fold_list = [i for i in range(fold - 1, fold + 3)]

    for dir1 in tqdm(os.listdir(img_folder)):
        for eye in os.listdir(os.path.join(img_folder, dir1)):
            for file in list(
                os.listdir(os.path.join(img_folder, dir1, eye))[i] for i in fold_list
            ):
                image_path = os.path.join(img_folder, dir1, eye, file)
                if image_path.endswith(".jpg") == False:
                    continue
                img = image.load_img(image_path, target_size=(64, 64))
                img = image.img_to_array(img)
                test_data.append(img)
                test_label.append(dir1 + "0" if eye == "L" else dir1 + "1")

    np.save(f"temp_data/Fusion_x_fold{fold}.npy", test_data)
    np.save(f"temp_data/Fusion_y_fold{fold}.npy", test_label)
    return np.array(test_data), np.array(test_label)


def load_dataset(fold):
    Fusion_X, Fusion_y = combine_LR(
        np.load(f"temp_data/Fusion_x_fold{fold}.npy"),
        np.load(f"temp_data/Fusion_y_fold{fold}.npy"),
        1000,
        4,
    )

    model = VGG16(weights="imagenet", include_top=False, input_shape=(64, 128, 3))
    with tf.device("GPU:0"):
        features_test = model.predict(Fusion_X)

    return features_test, Fusion_X, Fusion_y


def create_fold_norm(fold):
    iris_norm_L_fold = []
    iris_norm_R_fold = []
    for fol in range(1000):
        for item in range(4):
            iris_norm_L_fold.append(iris_norm_L[fol * 10 + item + fold])
            iris_norm_R_fold.append(iris_norm_R[fol * 10 + item + fold])
    return np.array(iris_norm_L_fold), np.array(iris_norm_R_fold)


def accuracy_score_multi_thread(ref_num, test_num, same_num=2):
    # features_test, Fusion_X, Fusion_y = load_dataset()

    def process_images(args):
        img_1_fol, img_1_item, img_2_fol, img_2_item = args
        tar = 0
        trr = 0
        far = 0
        frr = 0
        predict = []
        ground_truth = []

        if img_1_fol == img_2_fol:
            ground_truth.append(1)
        else:
            ground_truth.append(0)

        img_1_L = iris_norm_L_fold[(img_1_fol) * 4 + img_1_item]
        img_1_R = iris_norm_R_fold[(img_1_fol) * 4 + img_1_item]
        img_2_L = iris_norm_L_fold[(img_2_fol) * 4 + img_2_item]
        img_2_R = iris_norm_R_fold[(img_2_fol) * 4 + img_2_item]

        iris_score = iris_match_preload(img_1_L, img_1_R, img_2_L, img_2_R)

        if iris_score == "Match":
            predict.append(1)
            if img_2_fol == img_1_fol:
                tar += 1
            else:
                far += 1
        elif iris_score == "Not Sure" or iris_score == "No Iris":
            peri_score = predict_image(
                features_test,
                str(img_1_fol).zfill(3),
                img_2_item + 4 * img_2_fol,
                fold_num,
            )
            if peri_score == "Match":
                predict.append(1)
                if img_2_fol == img_1_fol:
                    tar += 1
                else:
                    far += 1
            else:
                predict.append(0)
                if img_2_fol == img_1_fol:
                    frr += 1
                else:
                    trr += 1
        else:
            predict.append(0)
            if img_2_fol == img_1_fol:
                frr += 1
            else:
                trr += 1

        return tar, trr, far, frr, predict, ground_truth

    tar = 0
    trr = 0
    far = 0
    frr = 0
    predict = []
    ground_truth = []

    max_workers = 2 * os.cpu_count()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        image_combinations = product(
            range(ref_num), range(0, 1), range(test_num), range(1, same_num)
        )
        for t, true, f, false, p, g in tqdm(
            executor.map(process_images, image_combinations),
            total=ref_num * test_num * (same_num - 1),
        ):
            tar += t
            trr += true
            far += f
            frr += false
            predict.extend(p)
            ground_truth.extend(g)

    return [[trr, far], [frr, tar]], predict, ground_truth


def plot_confu(ground_truth, predict, url):
    # compute the confusion matrix
    cm = confusion_matrix(ground_truth, predict)

    # plot the confusion matrix
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["0", "1"])
    plt.yticks(tick_marks, ["0", "1"])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    plt.savefig(url, dpi=600)


def print_accuracy(ground_truth, confu):
    # calculate the true acceptance rate (TAR), true rejection rate (TRR), false acceptance rate (FAR), and false rejection rate (FRR)
    total_positive = np.sum(ground_truth)
    total_negative = len(ground_truth) - total_positive
    confu = np.array(confu)
    true_positive = confu[1, 1]
    true_negative = confu[0, 0]
    false_positive = confu[0, 1]
    false_negative = confu[1, 0]

    tar = true_positive / total_positive
    trr = true_negative / total_negative
    far = false_positive / total_negative
    frr = false_negative / total_positive
    # print the accuracy and error rates
    accuracy = (true_positive + true_negative) / len(ground_truth)
    error_rate = 1 - accuracy
    print(f"True Acceptance Rate (TAR): {tar*100}%")
    print(f"True Rejection Rate (TRR): {trr*100}%")
    print(f"False Acceptance Rate (FAR): {far*100}%")
    print(f"False Rejection Rate (FRR): {frr*100}%")
    print(f"Accuracy: {accuracy*100}%")
    print(f"Error Rate: {error_rate*100}%")
    # print(f'Recongition Rate: {(1-far-frr)*100}%')
    # print(f'Error: {(far+frr)*100}%')


fold_num = 2
ref_img_num = 200
test_img_num = 200

if f"Fusion_x_fold{fold_num}.npy" not in list(os.listdir("temp_data")):
    X_test, y_test = create_dataset("Iris-Dataset/CASIA-Iris-Thousand", fold_num)

features_test, Fusion_X, Fusion_y = load_dataset(fold_num)

iris_norm_L = np.load("temp_data/iris_norm_L_all.npy")
iris_norm_R = np.load("temp_data/iris_norm_R_all.npy")

iris_norm_L_fold, iris_norm_R_fold = create_fold_norm(fold_num - 1)

confu, predict, ground_truth = accuracy_score_multi_thread(ref_img_num, test_img_num, 4)
plot_confu(
    ground_truth,
    predict,
    f"Confu_matrix_{ref_img_num}_{test_img_num}_4_fold{fold_num}.png",
)
print_accuracy(ground_truth, confu)
