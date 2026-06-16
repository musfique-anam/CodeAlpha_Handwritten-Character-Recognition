import cv2
import numpy as np
import tensorflow as tf

def get_prediction(image_path, model_path='models/mnist_cnn_model.h5'):
    # 1. Load the trained model weights offline
    model = tf.keras.models.load_model(model_path)

    # 2. Read image as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None

    # 3. Resize to 28x28 pixels
    img_resized = cv2.resize(img, (28, 28))

    # 4. Invert colors if the image background is light paper
    if np.mean(img_resized) > 127:
        img_resized = cv2.bitwise_not(img_resized)

    # 5. Normalize pixel values to 0-1 range
    img_normalized = img_resized.astype('float32') / 255.0

    # 6. Reshape for CNN input batch shape (1, 28, 28, 1)
    img_input = np.expand_dims(np.expand_dims(img_normalized, axis=0), axis=-1)

    # 7. Predict the digit class
    prediction = model.predict(img_input, verbose=0)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    return predicted_digit, confidence