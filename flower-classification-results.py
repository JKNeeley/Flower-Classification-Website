# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf
from tensorflow import keras

#Avoid Security
import ssl
import urllib.request

ssl._create_default_https_context = ssl._create_unverified_context
response = urllib.request.urlopen("https://example.com")

warnings.filterwarnings("ignore")

# Load the model with best validation accuracy
loaded_best_model = tf.keras.models.load_model("./model_10-0.90.h5")

# Custom function to load and predict label for the image
def predict(img_rel_path):
    # Import Image from the path with size of (300, 300)
    img = keras.preprocessing.image.load_img(img_rel_path, target_size=(300, 300))
    img = keras.preprocessing.image.img_to_array(img, dtype=np.uint8)

    # Scaling the Image Array values between 0 and 1
    img = np.array(img)/255.0

    # Plotting the Loaded Image
    plt.title("Loaded Image")
    plt.axis('off')
    plt.imshow(img.squeeze())
    plt.show()

    # Get the Predicted Label for the loaded Image
    p = loaded_best_model.predict(img[np.newaxis, ...])

    labels = {0: 'daisy', 1: 'dandelion', 2: 'rose', 3: 'sunflower', 4: 'tulip'}
    print("\n\nMaximum Probability: ", np.max(p[0], axis=-1))
    predicted_class = labels[np.argmax(p[0], axis=-1)]
    print("Classified:", predicted_class, "\n\n")

    classes = []
    prob = []
    print("\n-------------------Individual Probability--------------------------------\n")

    for i, j in enumerate(p[0], 0):
        print(labels[i].upper(), ':', round(j*100, 2), '%')
        classes.append(labels[i])
        prob.append(round(j*100, 2))

    # Plot the probabilities for the loaded image
    def plot_bar_x():
        index = np.arange(len(classes))
        plt.bar(index, prob)
        plt.xlabel('Labels', fontsize=8)
        plt.ylabel('Probability', fontsize=8)
        plt.xticks(index, classes, fontsize=8, rotation=20)
        plt.title('Probability for loaded image')
        plt.show()

    plot_bar_x()

# Model Testing with Unseen Dataset
predict("input/flowers-dataset/test/Image_1.jpg")
predict("input/flowers-dataset/test/Image_10.jpg")
predict("input/flowers-dataset/test/Image_121.jpg")
predict("input/flowers-dataset/test/Image_130.jpg")
