# Iris Recognition and Periocular Recognition

## **Overview:**

Welcome to the Iris-Periocular Fusion Recognition Application! This application combines the power of iris and periocular recognition technologies to provide robust and secure identification solutions. By fusing these two biometric modalities, the system enhances accuracy, reliability, and versatility in various authentication scenarios.

## **Features:**

1. **Dual Biometric Fusion:** Leveraging both iris and periocular recognition, the application offers a comprehensive biometric identification system. This fusion enhances recognition accuracy and resilience to environmental variations, ensuring reliable authentication.
2. **Multi-Modal Enrollment:** Users can enroll their iris and periocular biometric data separately or simultaneously. The application supports flexible enrollment methods to accommodate diverse user preferences and operational requirements.

## New Version

Fusion of Iris and Periocular recognition

1. Preprocess the image

- Use Iris_norm.ipynb to preprocess the datasets

2. Train model for periocular recognition

- Use peri_cnn_new.ipynb to train the periocular recognition model

3. Test the model with test data

- Use score_level_fusion.ipynb to test the iris and periocular fusion system

## Old Version

### For streamlit demostration

Iris Recognition:

- streamlit run 'user directory/Daugman_demo.py'

Periocular Recognition:

- streamlit run 'user directory/Peri_demo.py'

### For jupyter notebook source code

Iris Recognition:

- Use Daugman_final.ipynb

Periocular Recognition:

- Use peri_cnn.ipynb
- For periocular recognition use the Load Image and Transfer learning and SVM code for maximum accuracy
