# Rock, Paper, Scissors Image Classification with TensorFlow

<p align="center">
  <img src="https://storage.googleapis.com/kaggle-datasets-images/107582/256559/14accfe0345e7cd534eaff2a9658a4cf/dataset-cover.png?t=2019-01-19-17-16-51" alt="Rock, Paper, Scissors" width="400">
</p>

Welcome to the Rock, Paper, Scissors image classification project! In this project, we will build and train a Convolutional Neural Network (CNN) using TensorFlow to classify images of rock, paper, and scissors.

## Prerequisites
Before we dive into the world of rock, paper, scissors, let's set up our environment:
1. Install the Kaggle library for easy dataset access:
   ```bash
   !pip install -q kaggle

2. Upload your Kaggle API key (kaggle.json) to enable dataset downloads. Place it in the /root/.kaggle/ directory in Colab:
   ```bash
   !mkdir -p ~/.kaggle
   !cp kaggle.json ~/.kaggle/
   !ls ~/.kaggle

3. Set permissions for the Kaggle API key file:
   ```bash
   !chmod 600 /root/.kaggle/kaggle.json

### Step 1: Download and Prepare the Dataset
We'll start by fetching the Rock, Paper, Scissors dataset from Kaggle, unzipping it, and organizing it into training and validation sets.

### Step 2: Load and Preprocess the Dataset
Our dataset is divided into training and validation sets. We'll use ImageDataGenerator to augment the data, helping our model generalize better.

### Step 3: Define and Train the Model
Our image classifier model is constructed using TensorFlow. It includes convolutional layers, max-pooling, and dense layers. After defining the architecture, we'll train it on the training data while using early stopping to prevent overfitting.

### Step 4: Evaluate the Model
To gauge our model's performance, we'll assess its accuracy on both the training and validation datasets.

### Step 5: Make Predictions on New Images
Now, the fun part! You can upload your own Rock, Paper, Scissors images to the Colab environment and let the trained model predict the results.

```bash
from google.colab import files
uploaded = files.upload()

for fn in uploaded.keys():
    # Load the image
    path = fn
    img = image.load_img(path, target_size=target_size)
    imgplot = plt.imshow(img)
    
    # Preprocess the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    # Make predictions
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)

    # Print the predicted class
    print(fn)
    print('Predicted class: ')
    predicted_class_index = np.argmax(classes)
    if predicted_class_index == 0:
        print("Paper")
    elif predicted_class_index == 1:
        print("Rock")
    elif predicted_class_index == 2:
        print("Scissors")
