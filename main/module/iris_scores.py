import os
import pickle
import concurrent.futures
import multiprocessing as mp

from module.Iris_recognition import *
from module.Periocular_recognition import *
from module.score_fusion import *


def get_fusion_scores(iris_norm_L, iris_norm_R, labels):
    total_test_img = 10
    fusion_scores = []
    for pair in tqdm(range(len(labels))):
        img_1_fol = int(labels[pair][0][:-2])
        img_1_item = int(labels[pair][0][-2:])
        img_2_fol = int(labels[pair][1][:-2])
        img_2_item = int(labels[pair][1][-2:])

        img_1_L = iris_norm_L[(img_1_fol) * total_test_img + img_1_item]
        img_1_R = iris_norm_R[(img_1_fol) * total_test_img + img_1_item]
        img_2_L = iris_norm_L[(img_2_fol) * total_test_img + img_2_item]
        img_2_R = iris_norm_R[(img_2_fol) * total_test_img + img_2_item]
        fusion_scores.append(iris_score_fusion_preload(
            img_1_L, img_1_R, img_2_L, img_2_R))

    return np.array(fusion_scores)


def get_fusion_scores_mt(iris_norm_L, iris_norm_R, labels):
    total_test_img = 10
    fusion_scores = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for pair in range(len(labels)):
            img_1_fol = int(labels[pair][0][:-2])
            img_1_item = int(labels[pair][0][-2:])
            img_2_fol = int(labels[pair][1][:-2])
            img_2_item = int(labels[pair][1][-2:])

            img_1_L = iris_norm_L[(img_1_fol) * total_test_img + img_1_item]
            img_1_R = iris_norm_R[(img_1_fol) * total_test_img + img_1_item]
            img_2_L = iris_norm_L[(img_2_fol) * total_test_img + img_2_item]
            img_2_R = iris_norm_R[(img_2_fol) * total_test_img + img_2_item]
            futures.append(executor.submit(
                iris_score_fusion_preload, img_1_L, img_1_R, img_2_L, img_2_R))

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            fusion_scores.append(future.result())

    return np.array(fusion_scores)


# def get_fusion_scores_multi_process(iris_norm_L, iris_norm_R, labels):
#     total_test_img = 10
#     fusion_scores = []
#     with mp.Pool(processes=8) as pool:
#         results = [pool.apply_async(iris_score_fusion_preload, args=(iris_norm_L[(int(labels[pair][0][:-2])) * total_test_img + int(labels[pair][0][-2:])],
#                                                                      iris_norm_R[(
#                                                                          int(labels[pair][0][:-2])) * total_test_img + int(labels[pair][0][-2:])],
#                                                                      iris_norm_L[(
#                                                                          int(labels[pair][1][:-2])) * total_test_img + int(labels[pair][1][-2:])],
#                                                                      iris_norm_R[(int(labels[pair][1][:-2])) * total_test_img + int(labels[pair][1][-2:])])) for pair in range(len(labels))]
#         for result in tqdm(results):
#             fusion_scores.append(result.get())

#     return np.array(fusion_scores)


def get_fusion_scores_mp(iris_norm_L, iris_norm_R, labels, checkpoint_file):
    total_test_img = 10
    fusion_scores = []
    checkpoint_file = f"checkpoint/{checkpoint_file}"

    # Check if checkpoint file exists and load saved state of fusion_scores
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "rb") as f:
            fusion_scores = pickle.load(f)
        start_index = len(fusion_scores)
    else:
        start_index = 0

    with mp.Pool(processes=8) as pool:
        results = [pool.apply_async(iris_score_fusion_preload, args=(iris_norm_L[(int(labels[pair][0][:-2])) * total_test_img + int(labels[pair][0][-2:])],
                                                                     iris_norm_R[(
                                                                         int(labels[pair][0][:-2])) * total_test_img + int(labels[pair][0][-2:])],
                                                                     iris_norm_L[(
                                                                         int(labels[pair][1][:-2])) * total_test_img + int(labels[pair][1][-2:])],
                                                                     iris_norm_R[(int(labels[pair][1][:-2])) * total_test_img + int(labels[pair][1][-2:])])) for pair in range(start_index, len(labels))]

        for i, result in enumerate(tqdm(results, initial=start_index, total=len(labels))):
            # Checkpoint every 100 iterations
            if (i + start_index) % 100 == 0:
                with open(checkpoint_file, "wb") as f:
                    pickle.dump(fusion_scores, f)
            fusion_scores.append(result.get())

    # Save final state of fusion_scores to checkpoint file
    with open(checkpoint_file, "wb") as f:
        pickle.dump(fusion_scores, f)

    return np.array(fusion_scores)
