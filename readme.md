# Rock, Paper, Scissors Image Classification with TensorFlow

<p align="center">
  <img src="https://www.science.org/do/10.1126/science.aac4663/abs/sn-rockpaper.jpg" alt="Rock, Paper, Scissors" width="400">
</p>

Welcome to the Rock, Paper, Scissors image classification project! In this project, we will build and train a Convolutional Neural Network (CNN) using TensorFlow to classify images of rock, paper, and scissors.

## Table of Contents
- [Project Overview](##Project-Overview)
- [Prerequisites](##Prerequisites)
- [Dataset](##Dataset)
- [Data Preprocessing](##Data-Preprocessing)
- [Model Architecture](##Model-Architecture)
- [Training the Model](##Training-the-Model)
- [Model Evaluation](##Model-Evaluation)
- [Making Predictions](##Making-Predictions)
- [Conclusion](##Conclusion)
- [References](##References)

## Project Overview
In this project, we will create a CNN model to classify hand gestures of rock, paper, and scissors. The dataset used for this project is available on Kaggle. Here's an overview of the project steps:

## Prerequisites
Before running the code, make sure you have the necessary libraries installed and set up your Kaggle API key:
```py
  # Install the Kaggle library
  !pip install -q kaggle
  
  # Upload your Kaggle API key file (kaggle.json) to /root/.kaggle/
  !mkdir -p ~/.kaggle
  !cp kaggle.json ~/.kaggle/
  !ls ~/.kaggle
  
  # Set permissions for the Kaggle API key file
  !chmod 600 /root/.kaggle/kaggle.json
```

## Dataset
The dataset used in this project is the [Rock, Paper, Scissors](https://www.kaggle.com/drgfreeman/rockpaperscissors) dataset from Kaggle. It contains images of hand gestures representing rock, paper, and scissors.

## Data Preprocessing
We'll split the dataset into training and validation sets and apply data augmentation using ImageDataGenerator to improve model performance.

## Model Architecture
Our CNN model consists of convolutional layers, max-pooling layers, and dense layers. It's designed to learn and recognize patterns in the hand gesture images.

## Training the Model
We'll train the model on the training dataset, and early stopping will be applied to prevent overfitting.

## Model Evaluation
We'll evaluate the model's accuracy on both the training and validation sets to assess its performance.

## Making Predictions
You can upload your own hand gesture images and use the trained model to predict their classes.

## Conclusion
This project demonstrates how to build a CNN for image classification and provides an opportunity to explore computer vision techniques.

## References
- [Rock, Paper, Scissors Dataset on Kaggle](https://www.kaggle.com/drgfreeman/rockpaperscissors)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs/python/tf)

Feel free to explore the code and experiment with your own images. Have fun classifying rock, paper, and scissors gestures!

# Error Update
```bash
ValueError                                Traceback (most recent call last)
<ipython-input-9-a651793e253e> in <cell line: 6>()
      4 print("Train accuracy:", train_score[1] * 100, "%")
      5 
----> 6 validation_score = model.evaluate(validation_generator)
      7 print("Validation accuracy:", validation_score[1] * 100, "%"

ValueError: Asked to retrieve element 0, but the Sequence has length 0
```
