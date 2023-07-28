# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf
from keras.applications.inception_v3 import  preprocess_input
from keras.preprocessing.image import ImageDataGenerator

#Avoid Security
import ssl
import urllib.request

ssl._create_default_https_context = ssl._create_unverified_context
response = urllib.request.urlopen("https://example.com")

#Machine Learning

warnings.filterwarnings("ignore")

# Base Path for all files
data_dir = 'input/flowers-recognition/flowers'
print("directory path is:", data_dir)

# Data Augmentation Configuration
datagenerator = {
    "train": ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.1,
    ).flow_from_directory(
        directory=data_dir,
        target_size=(300, 300),
        subset='training',
    ),

    "valid": ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.1,
    ).flow_from_directory(
        directory=data_dir,
        target_size=(300, 300),
        subset='validation',
    ),
}
print("data augmentation configuration complete")