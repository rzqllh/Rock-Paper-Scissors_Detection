# Rock, Paper, Scissors Image Classification with TensorFlow
This project demonstrates how to build and train a convolutional neural network (CNN) to classify images of rock, paper, and scissors using TensorFlow and the Kaggle dataset.

## Prerequisites
Before running the code, you'll need to set up your Kaggle API key file and install the required libraries. Follow these steps:
1. Install the Kaggle library:
   ```bash
   !pip install -q kaggle

2. Upload your Kaggle API key file (kaggle.json) manually to the /root/.kaggle/ directory in your Colab environment.
   ```bash
   !mkdir -p ~/.kaggle
   !cp kaggle.json ~/.kaggle/
   !ls ~/.kaggle
3. Set permissions for the Kaggle API key file:
   ```bash
   !chmod 600 /root/.kaggle/kaggle.json

### Step 1: Download and Prepare the Dataset
This step involves downloading the dataset from Kaggle, extracting it, and organizing it into training and validation sets.

### Step 2: Load and Preprocess the Dataset
The dataset is split into training and validation sets, and data augmentation is applied using ImageDataGenerator to improve model generalization.

### Step 3: Define and Train the Model
A CNN model is defined and compiled. It consists of convolutional layers, max-pooling layers, and dense layers. The model is trained on the training data, and early stopping is applied to prevent overfitting.

### Step 4: Evaluate the Model
The model's performance is evaluated on both the training and validation sets to measure its accuracy.

### Step 5: Make Predictions on New Images
You can upload new images to the Colab environment and use the trained model to predict their classes.
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
