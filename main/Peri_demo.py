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

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


@st.cache_data(ttl=datetime.timedelta(hours=1), show_spinner="Loading dataset...")
def load_dataset(train_split=0.6, val_split=0.2, test_split=0.2):
    X_train = np.load('temp_data/X_train.npy')
    X_val = np.load('temp_data/X_val.npy')
    X_test = np.load('temp_data/X_test.npy')

    y_train = np.load('temp_data/y_train.npy')
    y_val = np.load('temp_data/y_val.npy')
    y_test = np.load('temp_data/y_test.npy')

    X_train, y_train = combine_LR(X_train, y_train, 1000, int(train_split*10))
    X_val, y_val = combine_LR(X_val, y_val, 1000, int(val_split*10))
    X_test, y_test = combine_LR(X_test, y_test, 1000, int(test_split*10))

    X = np.concatenate((X_train, X_val, X_test), axis=0)
    y = np.concatenate((y_train, y_val, y_test), axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    with tf.device('GPU:0'):
        features_test = model.predict(X_test)

    return X_train, X_test, y_train, y_test, features_test


def combine_LR(X, y, classes, img_num):
    X_combined = []
    y_combined = []
    for i in range(0, classes*img_num*2, img_num*2):
        for j in range(img_num):
            X_combined.append(np.concatenate((X[i+j], X[i+j+img_num]), axis=1))
            y_combined.append("".join([*y[i]][:3]))
    return np.array(X_combined), np.array(y_combined)


model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 128, 3))

X_train, X_test, y_train, y_test, features_test = load_dataset()

img_num = 1
plot_size = 20

st.title('Periocular Recognition Demo')

ref_num = st.selectbox(
    "Select Reference Iris Image",
    (f'Reference Image {i}' for i in range(len(X_test))),
)

with st.spinner('Loading Image...'):
    clf = pickle.load(open('Model/svm_VGG16.pickle', "rb"))

    ref_num = int(ref_num.split(' ')[2])

    ref_img = X_test[ref_num]

    matplotlib.rcParams.update({'font.size': 22})
    fig = plt.figure(figsize=(20, plot_size*2), constrained_layout=False)
    outer_grid = fig.add_gridspec(3, 1, wspace=0.1, hspace=-0.8)

    ax1 = fig.add_subplot(outer_grid[0:3, 0])

    ref_test = np.expand_dims(features_test[ref_num], axis=0)

    y_predict_ref = clf.predict(ref_test.reshape(ref_test.shape[0], -1))

    ax1.imshow((ref_img*255).astype(np.uint8))
    ax1.set_title(
        f'Reference Image {y_test[ref_num]} ➜ {y_predict_ref[0]}', fontsize=40)

    if y_test[ref_num] == y_predict_ref:
        st.success('Match', icon='✅')
    else:
        st.error('Not Match', icon='🚨')

    st.pyplot(fig)
