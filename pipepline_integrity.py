import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

# Define the image size for resizing
IMG_SIZE = 128
image_dir = r'C:\Users\wwwam\OneDrive\Desktop\Pipeline integrity\dataset'  # Dataset directory path

# Function to load images from dataset folders and assign labels
def load_images(image_dir, img_size):
    images = []
    labels = []
    for label, category in enumerate(['normal', 'compromised']):  # Label 0 for normal, 1 for compromised
        path = os.path.join(image_dir, category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                if img_array is not None:
                    resized_img = cv2.resize(img_array, (img_size, img_size))
                    images.append(resized_img)
                    labels.append(label)
            except Exception as e:
                print(f"Error loading image {img}: {e}")
    return np.array(images), np.array(labels)

# Load and preprocess the images
images, labels = load_images(image_dir, IMG_SIZE)
images = images / 255.0  # Normalize pixel values (0 to 1)
images = images.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Split the dataset (80% training, 20% testing)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Basic Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

datagen.fit(X_train)

# Function to define the CNN model
def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create the CNN model
input_shape = (IMG_SIZE, IMG_SIZE, 1)
model = create_cnn_model(input_shape)

# Early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the CNN model
model.fit(datagen.flow(X_train, y_train, batch_size=32),
          validation_data=(X_test, y_test),
          steps_per_epoch=int(len(X_train) / 32),
          epochs=15,  # Adjust as necessary
          callbacks=[early_stopping])

# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# Function to preprocess a new image for prediction
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    image = image.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    return image

# Prediction function
def predict_pipeline_status(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)[0]
    return "Compromised Pipeline" if prediction >= 0.85 else "Normal Pipeline"

# GUI setup
def open_file_dialog():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        image = Image.open(file_path)
        image = image.resize((250, 250))  # Resize for display in GUI
        img = ImageTk.PhotoImage(image)
        image_label.config(image=img)
        image_label.image = img

        # Get prediction
        status = predict_pipeline_status(file_path)
        result_label.config(text=f"Pipeline status: {status}")

# Set up the Tkinter GUI window
root = tk.Tk()
root.title("Pipeline Integrity Prediction")
root.geometry("400x500")

# Label to display the image
image_label = Label(root)
image_label.pack(pady=20)

# Button to open file dialog for drag-and-drop
open_button = tk.Button(root, text="Drag and Drop an Image Here", command=open_file_dialog)
open_button.pack()

# Label to display prediction result
result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack(pady=20)

# Run the Tkinter application
root.mainloop()
