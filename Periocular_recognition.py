import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.applications import ResNet101, ResNet50, VGG16, VGG19, InceptionV3, InceptionResNetV2
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess_input
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as inception_resnet_preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm.auto import trange, tqdm
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import sklearn.svm as svm
import sklearn.model_selection as model_selection
import sklearn.linear_model as linear_model
import pickle
import matplotlib.pyplot as plt
import matplotlib
import datetime

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
