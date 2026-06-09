import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    model = models.Sequential([
        #First Convulation Layer
        layers.Conv2D(32, (3, 3), activation ='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

        #Flattening and Dense Classification Layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3), #Helps prevent overfitting
        layers.Dense(num_classes, activation='softmax')
    ])

    return model