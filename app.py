import os
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
import numpy as np
# from keras.applications import ResNet50
import ssl

from keras.preprocessing import image
from tensorflow import keras

ssl._create_default_https_context = ssl._create_default_https_context = ssl._create_unverified_context

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

model = keras.models.load_model("./model_10-0.90.h5")

labels = {0: 'daisy', 1: 'dandelion', 2: 'rose', 3: 'sunflower', 4: 'tulip'}

def predict_image(image_path):
    img = image.load_img(image_path, target_size=(300, 300))
    img = image.img_to_array(img)
    img = np.array(img) / 255.0
    img = img[np.newaxis, ...]
    predictions = model.predict(img)
    predicted_class = labels[np.argmax(predictions[0], axis=-1)]

    # Create a dictionary with class labels and their probabilities
    probabilities = {label: prob for label, prob in zip(labels.values(), predictions[0])}

    return predicted_class, probabilities

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No image selected')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img = Image.open(file)
        img.thumbnail((300, 300))
        img.save(file_path)
        print('upload_image filename: ' + filename)
        predicted_class, probabilities = predict_image(file_path)
        return render_template("upload.html", prediction=predicted_class, probabilities=probabilities)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
	print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)