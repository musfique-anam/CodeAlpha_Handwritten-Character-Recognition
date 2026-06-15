import os
import sys
import cv2
import numpy as np
import tensorflow as tf

def predict_custom_image(image_path):
    model_path = 'models/mnist_cnn_model.h5'
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Trained engine missing at {model_path}. Please execute train.py first.")
        return

    # Load serialized model instance
    model = tf.keras.models.load_model(model_path)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[ERROR] Target invalid or unreadable file pathway: {image_path}")
        return
    
    # Resize spatial resolution 
    img_resized = cv2.resize(img, (28, 28))
    
    if np.mean(img_resized) > 127:
        img_resized = cv2.bitwise_not(img_resized)
        
    img_normalized = img_resized.astype('float32') / 255.0
    
    img_input = np.expand_dims(np.expand_dims(img_normalized, axis=0), axis=-1)
    
    prediction = model.predict(img_input, verbose=0)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    print("\n===============================")
    print(f" Target Path: {image_path}")
    print(f" Predicted Identity Class: {predicted_digit}")
    print(f" Network Confidence Value: {confidence:.2f}%")
    print("===============================")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("[USAGE RUNTIME] Command syntax needs parameter: python src/predict.py <path_to_your_image.png>")
    else:
        predict_custom_image(sys.argv[1])