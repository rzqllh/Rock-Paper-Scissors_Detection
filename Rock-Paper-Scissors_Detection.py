
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import zipfile
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from sklearn.model_selection import train_test_split

# Step 1: Download and prepare the dataset

!pip install -q kaggle

# Upload your Kaggle API key file (kaggle.json) manually to /root/.kaggle/ in your Colab environment
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!ls ~/.kaggle

# Set permissions for the Kaggle API key file
!chmod 600 /root/.kaggle/kaggle.json

# Download the dataset from Kaggle
!kaggle datasets download -d drgfreeman/rockpaperscissors

# Extract the dataset
with zipfile.ZipFile('rockpaperscissors.zip', 'r') as zip_ref:
    zip_ref.extractall('rockpaperscissors')

# Create train and val directories
train_dir = 'rockpaperscissors/train'
val_dir = 'rockpaperscissors/val'
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Split data into train and val
classes = ['rock', 'paper', 'scissors']
for c in classes:
    source_dir = f'rockpaperscissors/{c}'
    train_class_dir = f'{train_dir}/{c}'
    val_class_dir = f'{val_dir}/{c}'

    num_files = len(os.listdir(source_dir))
    num_train = int(0.8 * num_files)

    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)

    for file in os.listdir(source_dir)[:num_train]:
        source_file = os.path.join(source_dir, file)
        target_file = os.path.join(train_class_dir, file)
        os.rename(source_file, target_file)

    for file in os.listdir(source_dir)[num_train:]:
        source_file = os.path.join(source_dir, file)
        target_file = os.path.join(val_class_dir, file)
        os.rename(source_file, target_file)

# Step 2: Load and preprocess the dataset

base_dir = 'rockpaperscissors'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

# Split the dataset into train and validation sets
train_images, _ = train_test_split(os.listdir(train_dir), test_size=0.2, random_state=42)
val_images, _ = train_test_split(os.listdir(val_dir), test_size=0.2, random_state=42)

# Data augmentation for train and validation sets
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    vertical_flip=True,
    shear_range=0.2,
    validation_split=0.4
)

validation_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    vertical_flip=True,
    shear_range=0.2,
    validation_split=0.4
)

# Create data generators
target_size = (150, 150)
batch_size = 4

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = validation_datagen.flow_from_directory(
    val_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Step 3: Define and train the model

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.optimizers.RMSprop(learning_rate=0.0001),
    metrics=['accuracy']
)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=50,
    callbacks=[early_stopping_callback]
)

# Step 4: Evaluate the model

train_score = model.evaluate(train_generator)
print("Train accuracy:", train_score[1] * 100, "%")

validation_score = model.evaluate(validation_generator)
print("Validation accuracy:", validation_score[1] * 100, "%")

# Step 5: Make predictions on new images

from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
    path = fn
    img = image.load_img(path, target_size=target_size)
    imgplot = plt.imshow(img)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)

    print(fn)
    print('Predicted class: ')
    predicted_class_index = np.argmax(classes)
    if predicted_class_index == 0:
        print("Paper")
    elif predicted_class_index == 1:
        print("Rock")
    elif predicted_class_index == 2:
        print("Scissors")
