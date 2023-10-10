import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


def combine_LR(X, y, classes, img_num):
    X_combined = []
    y_combined = []
    for i in range(0, classes*img_num*2, img_num*2):
        for j in range(img_num):
            X_combined.append(np.concatenate((X[i+j], X[i+j+img_num]), axis=1))
            y_combined.append("".join([*y[i]][:3]))
    return np.array(X_combined), np.array(y_combined)


def predict_image(features_test, img_1, img_2, fold):
    clf = pickle.load(open(f'Model/5fold/svm_VGG16_fold{fold}.pickle', "rb"))

    ref_test = np.expand_dims(features_test[img_2], axis=0)

    y_predict_ref = clf.predict(ref_test.reshape(ref_test.shape[0], -1))

    if y_predict_ref[0] == img_1:
        return "Match"

    return "Not Match"


def feature_preprocessing(features_train_a, features_test_a, features_train_b, features_test_b, fusion_scores_train, fusion_scores_test, use_fusion=True):
    X_train_final = np.abs(features_train_a - features_train_b)
    X_train_final = X_train_final.reshape(X_train_final.shape[0], -1)

    X_test_final = np.abs(features_test_a - features_test_b)
    X_test_final = X_test_final.reshape(X_test_final.shape[0], -1)

    if use_fusion:
        fusion_train_final = fusion_scores_train.reshape(
            fusion_scores_train.shape[0], -1)
        X_train_final = np.concatenate(
            (X_train_final, fusion_train_final), axis=1)

        fusion_test_final = fusion_scores_test.reshape(
            fusion_scores_test.shape[0], -1)
        X_test_final = np.concatenate(
            (X_test_final, fusion_test_final), axis=1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_final)
    X_test_scaled = scaler.transform(X_test_final)

    return X_train_scaled, X_test_scaled


def peri_match_preload(model, X, formula=False):
    model = model
    y_predict = model.predict(X)
    if not formula:
        if y_predict == 1:
            return 'Match'
        else:
            return 'Non-Match'
    else:
        return y_predict
