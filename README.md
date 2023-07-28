# Flower Classification Website
University of Missouri Independent Study of Machine Learning Final Project
By Jade Neeley

### Class Credit Notice
This project is being developed as the final project of _Independent Study: Machine Learning_ at the University of Missouri during the summer of 2024. This project proposal includes steps outside the scope of Machine Learning including Web Development and Cloud Deployment. In order for this project to be deemed worth Machine Learning course credit, the website must, at a minimum, be able to run locally and successfully integrate a machine learning model to accurately identify the flowers.

__________________

# Project Proposal

## Introduction
The purpose of this project is to develop a website that allows users to upload a photo of a flower and use a classification machine learning model to determine the type of flower in the image. The website will utilize the Flower Recognition dataset provided in the DPhi Data Sprint #25 which contains raw JPEG images of five types of flowers: daisy, dandelion, rose, sunflower, and tulip. Users will receive the predicted flower type as output after uploading their photo. By following the specified technology stack and project phases, I aim to create an intuitive and accurate flower classification website.

## Dataset Description
_Train_ contains subfolders for each flower type and is used for training the model.
_Test_ contains 924 flower images that require predictions. 

## Development Stack
Development: _GitHub_ will host the repository used for development.
Front-End: _HTML_, _CSS_, and _JavaScript_ will be used to create an interactive user interface.
Back-End: _Python_ will be used as the programming language for the back-end implementation of the website. _Flask_ will handle the server-side logic and serve the website's pages to the users.
Machine Learning: _TensorFlow_ will be used to build and train a model for flower classification. OpenCV will be utilized for image preprocessing tasks.
Deployment and Hosting: The website will be deployed and hosted on AWS using _Amazon S3_ for image storage and _Amazon EC2_ for hosting the application. 

## Project Phases and Tasks
### Phase 1: Dataset Preparation
- Explore and understand the dataset.
- Perform any necessary data preprocessing, such as resizing and normalization.
### Phase 2: Model Development
- Select and load a pre-trained image classification model from TensorFlow.
- Fine-tune the pre-trained model on the flower dataset using transfer learning techniques.
- Train the model on the prepared training set.
### Phase 3: Website Development 
- Design and implement the user interface using HTML, CSS, and JavaScript.
- Develop the back-end server using Flask to handle user requests.
- Integrate the trained machine learning model into the Flask server.
- Implement image preprocessing using OpenCV before passing the images to the model.
### Phase 4: Deployment and Hosting
- Deploy the website and configure it to handle user requests.
- Set up the necessary infrastructure, such as Amazon S3 for image storage.
- Perform testing and ensure the website is functioning correctly.

## Deliverables
Upon completion, the project will deliver the following:
- A fully functional website allowing users to upload flower images for classification.
- Proper image preprocessing using OpenCV.
- Integration of a machine learning model for flower classification.
- Deployment of the website on AWS.

__________________

# Sources Used

Efforts were made to generate files completely on my own, but due to time constraints and an unfamiliarity with Flask and Tensorflow, I used sources oustide of myself and previous coursework to create this successful application. Instead of generating the entire model from scratch, I focused on learning from others and learning the implementation of Tensorflow withing Flask and the creation of predictions and the custom layers.

## Datasets found from 
 - https://www.kaggle.com/datasets/imsparsh/flowers-dataset
 - https://www.kaggle.com/datasets/alxmamaev/flowers-recognition

## Flask demo found at 
- https://roytuts.com/upload-and-display-image-using-python-flask/

## Transfer Learning Model demos found at 
- https://www.kaggle.com/code/aryashah2k/flower-image-classification-transfer-learning/notebook
- https://www.kaggle.com/code/chetanambi/fiowers-classification-with-tf2-and-keras/notebook
