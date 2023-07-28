# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from keras import optimizers, regularizers, callbacks
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model

#Avoid Security
import ssl
import urllib.request

ssl._create_default_https_context = ssl._create_unverified_context
response = urllib.request.urlopen("https://example.com")

warnings.filterwarnings("ignore")

#Machine Learning

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

# Base Model: InceptionV3 (pretrained) with input image shape as (300, 300, 3)
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
print("base model imported")

# Setting the last few layers of InceptionV3 model to trainable
for layer in base_model.layers[:-5]:
    layer.trainable = False
print("base model trainable")

# Custom Model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.15),
    Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dense(5, activation='softmax'),  # 5 Output Neurons for 5 Classes
])
print("custom model created")

# Setting variables for the model
batch_size = 32
epochs = 10

# Seperating Training and Testing Data
train_generator = datagenerator["train"]
valid_generator = datagenerator["valid"]

# Calculating variables for the model
steps_per_epoch = train_generator.n // batch_size
validation_steps = valid_generator.n // batch_size

print("steps_per_epoch :", steps_per_epoch)
print("validation_steps :", validation_steps)

# Using the Adam Optimizer with learning rate scheduling
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0001,
    decay_steps=steps_per_epoch * 5,  # steps_per_epoch = 32
    decay_rate=0.9,
)
opt = optimizers.Adam(learning_rate=lr_schedule)
print("updated learning rate")

# Compiling the model
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])

# View Model Summary
model.summary()
plot_model(model, show_shapes=True, show_layer_names=True)

# Model Training
filepath = "./model_{epoch:02d}-{val_accuracy:.2f}.h5"
checkpoint1 = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint1]


# Early Stopping Callback
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# TensorBoard Callback
tensorboard_callback = callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)

history = model.fit_generator(
    generator=train_generator,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=valid_generator,
    validation_steps=validation_steps,
    callbacks=[checkpoint1, early_stopping, tensorboard_callback]
)

# Plot Training History
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, max(plt.ylim())])
plt.title('Training and Validation Loss')
plt.show()

# Calculate the Loss and Accuracy on the Validation Data
test_loss, test_acc = model.evaluate(valid_generator)
print('test accuracy : ', test_acc)
