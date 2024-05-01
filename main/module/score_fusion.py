from module.Iris_recognition import *
from module.Periocular_recognition import *
from module.matching_algo import *
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import concurrent.futures
import os
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import classification_report


def sexy_majority_vote(hd, wed, jc, rg):
    vote = 0
    vote += (hd <= 4.71) + (wed <= 0.235) + (jc <= 0.651) + (rg <= 0.642)
    if vote >= 2:
        return 1
    elif vote < 2:
        return jc <= 0.651


def normal_majority_vote(hd, wed, jc, rg):
    vote = 0
    vote += (hd <= 4.71) + (wed <= 0.235) + (jc <= 0.651) + (rg <= 0.642)

    if vote > 2:
        return 1
    elif vote < 2:
        return 0
    else:
        return jc <= 0.651


def iris_score_fusion_preload(img_1_L, img_1_R, img_2_L, img_2_R):
    imgs = [[img_1_L, img_1_R], [img_2_L, img_2_R]]
    templates = [[], []]
    masks = [[], []]
    results = [[], [], [], []]

    for i in range(len(imgs)):
        for j in range(len(imgs[i])):
            img = imgs[i][j]

            if np.all(img == 0, axis=(0, 1)):
                return [[0.471, 0.471], [0.651, 0.651], [0.235, 0.235], [0.642, 0.642]]
            else:
                # Image Preprocessing (Normalization)
                iris_norm = img

                # Feature Extraction
                romv_img, noise_img = lash_removal_daugman(iris_norm, thresh=50)
                template, mask_noise = encode_iris(
                    romv_img, noise_img, minw_length=18, mult=1, sigma_f=0.5
                )

                templates[i].append(template)
                masks[i].append(mask_noise)

                # Matching
                if len(templates[1]) > 0:
                    hd_raw = HammingDistance(
                        templates[i - 1][j],
                        masks[i - 1][j],
                        templates[i][j],
                        masks[i][j],
                    )
                    results[0].append(hd_raw)

                    jd_raw = JaccardDistance(
                        templates[i - 1][j],
                        masks[i - 1][j],
                        templates[i][j],
                        masks[i][j],
                    )
                    results[1].append(jd_raw)

                    wed_raw = WeightedEuclideanDistance(
                        templates[i - 1][j],
                        masks[i - 1][j],
                        templates[i][j],
                        masks[i][j],
                    )
                    results[2].append(wed_raw)

                    tdi_raw = TanimotoDistance(
                        templates[i - 1][j],
                        masks[i - 1][j],
                        templates[i][j],
                        masks[i][j],
                    )
                    results[3].append(tdi_raw)

    # print(f'HD: {[round(r, 3) for r in results[0]]}\nJD: {[round(r, 3) for r in results[1]]}\nWED: {[round(r, 3) for r in results[2]]}\nTDI: {[round(r, 3) for r in results[3]]}')
    return results


def set_gpu():
    # set the visible devices to the GPU
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[0], "GPU")
        except RuntimeError as e:
            print(e)


def plot_cm(y_test_final, y_predict, fig=None, ax=None):
    # compute the confusion matrix
    cm = confusion_matrix(y_test_final, y_predict)
    plt.rcParams.update({"font.size": 12})

    if fig or ax is None:
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
            if cm[i, j] > len(y_predict) / 4:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white")
            else:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    # set the x and y axis ticks to display only 1 and 0
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    # print classification report
    report = classification_report(y_test_final, y_predict)
    ax.text(-1, 2.9, report, fontsize=12, ha="left", family="monospace")
    plt.show()


def plot_cm_mul(y_test_final, y_predict, classifiers, plot_num=1):
    fig, axs = plt.subplots(1, plot_num, figsize=(50, 10))
    plt.rcParams.update({"font.size": 27})

    for plot in range(plot_num):
        # compute the confusion matrix
        cm = confusion_matrix(y_test_final[plot], y_predict[plot])
        im = axs[plot].imshow(
            cm,
            cmap=plt.cm.Blues,
            interpolation="nearest",
            extent=(
                -0.5,
                len(np.unique(y_test_final[plot])) - 0.5,
                len(np.unique(y_test_final[plot])) - 0.5,
                -0.5,
            ),
        )
        axs[plot].set_title(
            f"Confusion Matrix for {classifiers[plot].__class__.__name__}"
        )
        axs[plot].set_xlabel("Predicted Label", fontsize=27)
        axs[plot].set_ylabel("True Label", fontsize=27)
        fig.colorbar(im)

        # display the number of samples in each cell
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j] > cm.max() / 2:
                    axs[plot].text(
                        j, i, str(cm[i, j]), ha="center", va="center", color="white"
                    )
                else:
                    axs[plot].text(
                        j, i, str(cm[i, j]), ha="center", va="center", color="black"
                    )

        # set the x and y axis ticks to display only 1 and 0
        axs[plot].tick_params(axis="both", labelsize=27)
        axs[plot].set_xticks([0, 1])
        axs[plot].set_yticks([0, 1])

        # print classification report
        report = classification_report(y_test_final[plot], y_predict[plot])
        axs[plot].text(2, 2.8, report, fontsize=27, ha="right", family="monospace")
    plt.show()


def print_accuracy(ground_truth, predictions):
    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(ground_truth, predictions)

    # Calculate the true positive, true negative, false positive, and false negative rates
    tp = conf_matrix[1, 1]
    tn = conf_matrix[0, 0]
    fp = conf_matrix[0, 1]
    fn = conf_matrix[1, 0]

    # Calculate the precision, recall, and F1 score
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Calculate the accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Print the confusion matrix and performance metrics
    # print(f'Confusion Matrix:\n{conf_matrix}')
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")


def accuracy_score(labels, model, iris_norm_L, iris_norm_R, X_test):
    predict = []
    ground_truth = []
    total_test_img = 10

    for pair in tqdm(range(len(labels))):
        img_1_fol = int(labels[pair][0][:-2])
        img_1_item = int(labels[pair][0][-2:])
        img_2_fol = int(labels[pair][1][:-2])
        img_2_item = int(labels[pair][1][-2:])

        img_1_L = iris_norm_L[(img_1_fol) * total_test_img + img_1_item]
        img_1_R = iris_norm_R[(img_1_fol) * total_test_img + img_1_item]
        img_2_L = iris_norm_L[(img_2_fol) * total_test_img + img_2_item]
        img_2_R = iris_norm_R[(img_2_fol) * total_test_img + img_2_item]

        if img_1_fol == img_2_fol:
            ground_truth.append(1)
        else:
            ground_truth.append(0)

        iris_score = iris_match_preload(img_1_L, img_1_R, img_2_L, img_2_R)
        if iris_score == "Match":
            predict.append(1)

        elif iris_score == "Not Sure" or iris_score == "No Iris":
            peri_score = peri_match_preload(model, X_test[pair].reshape(1, -1))
            if peri_score == "Match":
                predict.append(1)
            else:
                predict.append(0)

        else:
            predict.append(0)

    return predict, ground_truth


def score_fusion(
    iris_scores,
    peri_score,
    hd_thresh=0.471,
    jd_thresh=0.651,
    tdi_thresh=0.642,
    alpha_0=0.5,
    alpha_inc=10,
):
    try:
        # Calculate the deviation of the iris scores from the thresholds
        iris_score_dev = [
            [max(0, hd - hd_thresh), max(0, jd - jd_thresh), max(0, tdi - tdi_thresh)]
            for [hd, jd, tdi] in iris_scores
        ]

        # Calculate the average deviation of the iris scores from the thresholds
        num_pass = sum(
            [
                hd <= hd_thresh and jd <= jd_thresh and tdi <= tdi_thresh
                for [hd, jd, tdi] in iris_scores
            ]
        )
        if num_pass == 0:
            avg_dev = 0
        else:
            avg_dev = (
                sum(
                    [
                        sum(dev_score)
                        for dev_score, [hd, jd, tdi] in zip(iris_score_dev, iris_scores)
                        if hd >= hd_thresh and jd >= jd_thresh and tdi >= tdi_thresh
                    ]
                )
                / num_pass
            )

        # Calculate the weight alpha based on the average deviation score
        alpha = alpha_0 + alpha_inc * avg_dev

        # Combine iris and periocular scores using weighted sum
        iris_scores_weighted = [
            1 if hd <= hd_thresh and jd <= jd_thresh and tdi <= tdi_thresh else 0
            for [hd, jd, tdi] in iris_scores
        ]
        combined_score = alpha * max(iris_scores_weighted) + (1 - alpha) * peri_score

        # Threshold the combined score to get binary prediction
        prediction = 1 if combined_score >= 0.5 else 0

    except:
        print(iris_scores, peri_score)

    return prediction


def accuracy_score_formula(labels, model, iris_norm_L, iris_norm_R, X_test):
    predict = []
    ground_truth = []
    total_test_img = 10

    for pair in tqdm(range(len(labels))):
        img_1_fol = int(labels[pair][0][:-2])
        img_1_item = int(labels[pair][0][-2:])
        img_2_fol = int(labels[pair][1][:-2])
        img_2_item = int(labels[pair][1][-2:])

        img_1_L = iris_norm_L[(img_1_fol) * total_test_img + img_1_item]
        img_1_R = iris_norm_R[(img_1_fol) * total_test_img + img_1_item]
        img_2_L = iris_norm_L[(img_2_fol) * total_test_img + img_2_item]
        img_2_R = iris_norm_R[(img_2_fol) * total_test_img + img_2_item]

        if img_1_fol == img_2_fol:
            ground_truth.append(1)
        else:
            ground_truth.append(0)

        iris_score = iris_match_preload(
            img_1_L, img_1_R, img_2_L, img_2_R, formula=True
        )

        peri_score = peri_match_preload(
            model, X_test[pair].reshape(1, -1), formula=True
        )

        predict.append(score_fusion(iris_score, peri_score))

    return predict, ground_truth


def accuracy_score_multi_thread(
    labels, model, iris_norm_L, iris_norm_R, X_test, num_threads=2 * os.cpu_count()
):
    predict = []
    ground_truth = []
    total_test_img = 10

    def process_pair(pair):
        img_1_fol = int(labels[pair][0][:-2])
        img_1_item = int(labels[pair][0][-2:])
        img_2_fol = int(labels[pair][1][:-2])
        img_2_item = int(labels[pair][1][-2:])

        img_1_L = iris_norm_L[(img_1_fol) * total_test_img + img_1_item]
        img_1_R = iris_norm_R[(img_1_fol) * total_test_img + img_1_item]
        img_2_L = iris_norm_L[(img_2_fol) * total_test_img + img_2_item]
        img_2_R = iris_norm_R[(img_2_fol) * total_test_img + img_2_item]

        if img_1_fol == img_2_fol:
            ground_truth.append(1)
        else:
            ground_truth.append(0)

        iris_score = iris_match_preload(img_1_L, img_1_R, img_2_L, img_2_R)
        if iris_score == "Match":
            predict.append(1)

        elif iris_score == "Not Sure" or iris_score == "No Iris":
            peri_score = peri_match_preload(model, X_test[pair].reshape(1, -1))
            if peri_score == "Match":
                predict.append(1)
            else:
                predict.append(0)

        else:
            predict.append(0)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_pair, pair) for pair in range(len(labels))]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(labels)):
            pass

    return predict, ground_truth


def accuracy_score_preload(labels, model, iris_scores, X_test):
    predict = []
    ground_truth = []
    total_test_img = 10

    for pair in tqdm(range(len(labels))):
        img_1_fol = int(labels[pair][0][:-2])
        img_2_fol = int(labels[pair][1][:-2])

        if img_1_fol == img_2_fol:
            ground_truth.append(1)
        else:
            ground_truth.append(0)

        iris_score = iris_scores[pair]

        # Iris Match
        # if (
        #     (iris_score[0][0] <= 0.4 and iris_score[0][1] <= 0.4)
        #     and (iris_score[1][0] <= 0.6 and iris_score[1][1] <= 0.6)
        #     and (iris_score[3][0] <= 0.6 and iris_score[3][1] <= 0.6)
        # ):
        #     predict.append(1)
        # # Iris Not Match
        # elif (
        #     (iris_score[0][0] >= 0.5 and iris_score[0][1] >= 0.5)
        #     and (iris_score[1][0] >= 0.7 and iris_score[1][1] >= 0.7)
        #     and (iris_score[3][0] >= 0.7 and iris_score[3][1] >= 0.7)
        # ):
        #     predict.append(0)
        if (
            (iris_score[0][0] <= 0.45 and iris_score[0][1] <= 0.45)
            and (iris_score[1][0] <= 0.63 and iris_score[1][1] <= 0.63)
            and (iris_score[3][0] <= 0.63 and iris_score[3][1] <= 0.63)
        ):
            predict.append(1)
        # Iris Not Match
        elif (
            (iris_score[0][0] >= 0.49 and iris_score[0][1] >= 0.49)
            and (iris_score[1][0] >= 0.67 and iris_score[1][1] >= 0.67)
            and (iris_score[3][0] >= 0.67 and iris_score[3][1] >= 0.67)
        ):
            predict.append(0)
        # Iris Not Sure
        else:
            peri_score = peri_match_preload(model, X_test[pair].reshape(1, -1))
            if peri_score == "Match":
                predict.append(1)
            else:
                predict.append(0)

    return predict, ground_truth


def accuracy_score_iris_preload(labels, iris_scores):
    predict = []
    ground_truth = []

    for pair in tqdm(range(len(labels))):
        img_1_fol = int(labels[pair][0][:-2])
        img_2_fol = int(labels[pair][1][:-2])

        if img_1_fol == img_2_fol:
            ground_truth.append(1)
        else:
            ground_truth.append(0)

        iris_score = iris_scores[pair]

        # Iris Match
        # if (
        #     (iris_score[0][0] <= 0.46 or iris_score[0][1] <= 0.46)  # 0.45
        #     and (iris_score[1][0] <= 0.64 or iris_score[1][1] <= 0.64)  # 0.64
        #     and (iris_score[3][0] <= 0.64 or iris_score[3][1] <= 0.64)  # 0.64
        # ):
        #     predict.append(1)
        if (
            (iris_score[0][0] <= 0.47 or iris_score[0][1] <= 0.47)  # 0.45
            and (iris_score[1][0] <= 0.66 or iris_score[1][1] <= 0.66)  # 0.64
            and (iris_score[3][0] <= 0.66 or iris_score[3][1] <= 0.66)  # 0.64
        ):
            predict.append(1)
        # Iris Not Match
        else:
            predict.append(0)

    return predict, ground_truth


def accuracy_score_irisNoF_preload(labels, iris_scores):
    predict = []
    ground_truth = []

    for pair in tqdm(range(len(labels))):
        img_1_fol = int(labels[pair][0][:-2])
        img_2_fol = int(labels[pair][1][:-2])

        if img_1_fol == img_2_fol:
            ground_truth.append(1)
        else:
            ground_truth.append(0)

        iris_score = iris_scores[pair]

        # Iris Match
        if iris_score[0][0] <= 0.47 or iris_score[0][1] <= 0.47:  # 0.45
            predict.append(1)
        # Iris Not Match
        else:
            predict.append(0)

    return predict, ground_truth
