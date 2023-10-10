from module.Iris_recognition import *
from module.Periocular_recognition import *
# from vgg16 import *
import os
import concurrent.futures
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def load_dataset():
    Fusion_X, Fusion_y = combine_LR(np.load('temp_data/Fusion_x.npy'),
                                    np.load('temp_data/Fusion_y.npy'),
                                    1000, 4)

    model = VGG16(weights='imagenet', include_top=False,
                  input_shape=(64, 128, 3))
    # with tf.device('GPU:0'):
    features_test = model.predict(Fusion_X)

    return features_test, Fusion_X, Fusion_y


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

        img_1_L = iris_norm_L[(img_1_fol) * 4 + img_1_item]
        img_1_R = iris_norm_R[(img_1_fol) * 4 + img_1_item]
        img_2_L = iris_norm_L[(img_2_fol) * 4 + img_2_item]
        img_2_R = iris_norm_R[(img_2_fol) * 4 + img_2_item]

        iris_score = iris_match_preload(img_1_L, img_1_R, img_2_L, img_2_R)
        peri_score = predict_image(
            features_test, str(img_1_fol).zfill(3), img_2_item + 4 * img_2_fol
        )

        if iris_score == "Match":
            predict.append(1)
            if img_2_fol == img_1_fol:
                tar += 1
            else:
                far += 1
        elif iris_score == "Not Sure" or iris_score == "No Iris":
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
        for t, true, f, false, p, g in tqdm(executor.map(process_images, image_combinations),
                                            total=ref_num*test_num *
                                            (same_num-1),
                                            miniters=int(
                                                ref_num*test_num*(same_num-1)/6000),
                                            maxinterval=20000):
            tar += t
            trr += true
            far += f
            frr += false
            predict.extend(p)
            ground_truth.extend(g)

    return [[trr, far], [frr, tar]], predict, ground_truth


def accuracy_score_iris(ref_num, test_num, same_num=2, thresh=0.47):
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

        img_1_L = iris_norm_L[(img_1_fol) * 4 + img_1_item]
        img_1_R = iris_norm_R[(img_1_fol) * 4 + img_1_item]
        img_2_L = iris_norm_L[(img_2_fol) * 4 + img_2_item]
        img_2_R = iris_norm_R[(img_2_fol) * 4 + img_2_item]

        iris_score = iris_match_only(
            img_1_L, img_1_R, img_2_L, img_2_R, thresh=thresh)

        if iris_score == "Match":
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
        for t, true, f, false, p, g in tqdm(executor.map(process_images, image_combinations),
                                            total=ref_num*test_num *
                                            (same_num-1),
                                            miniters=int(
                                                ref_num*test_num*(same_num-1)/6000),
                                            maxinterval=20000):
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
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['0', '1'])
    plt.yticks(tick_marks, ['0', '1'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), horizontalalignment="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

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
    print(f'True Acceptance Rate (TAR): {tar*100}%')
    print(f'True Rejection Rate (TRR): {trr*100}%')
    print(f'False Acceptance Rate (FAR): {far*100}%')
    print(f'False Rejection Rate (FRR): {frr*100}%')
    print(f'Accuracy: {accuracy*100}%')
    print(f'Error Rate: {error_rate*100}%')
    # print(f'Recongition Rate: {(1-far-frr)*100}%')
    # print(f'Error: {(far+frr)*100}%')


features_test, Fusion_X, Fusion_y = load_dataset()

iris_norm_L = np.load('temp_data/iris_norm_L.npy')
iris_norm_R = np.load('temp_data/iris_norm_R.npy')

ref_img_num = 1000
test_img_num = 1000
# confu, predict, ground_truth = accuracy_score_multi_thread(ref_img_num, test_img_num, 4)
confu, predict, ground_truth = accuracy_score_iris(
    ref_img_num, test_img_num, same_num=4, thresh=0.5)
plot_confu(ground_truth, predict,
           f'Confu_matrix_{ref_img_num}_{test_img_num}_4_iris_5.png')
print_accuracy(ground_truth, confu)
