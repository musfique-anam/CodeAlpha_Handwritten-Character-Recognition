import os
import sys
import numpy as np

# Adjust paths to make running from inside src/ safe
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import load_and_preprocess_data
from src.model import build_cnn_model
from utils.visualization import plot_training_history, plot_confusion_matrix

def main():
    # Ensure structural directory exists for artifacts
    os.makedirs('models', exist_ok=True)
    
    # 1. Pipeline Data Preparation
    print("[INFO] Fetching and processing MNIST datasets...")
    X_train, y_train, X_test, y_test = load_and_preprocess_data()
    
    # 2. Build and Compile Architecture
    print("[INFO] Initializing Convolutional Neural Network...")
    model = build_cnn_model()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # 3. Training Execution Loop
    print("[INFO] Initiating model optimization training...")
    history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
    
    # 4. Out-of-Sample Validation Evaluation
    print("[INFO] Running evaluation routines against test partitions...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[RESULT] Verification Test Final Accuracy: {test_acc * 100:.2f}%")
    
    # 5. Serialization of Saved Weights
    model.save('models/mnist_cnn_model.h5')
    print("[INFO] Architecture saved successfully to: models/mnist_cnn_model.h5")
    
    # 6. Report Analytics Charts Generation
    print("[INFO] Exporting training metrics history charts and matrices...")
    plot_training_history(history)
    
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    plot_confusion_matrix(y_test, y_pred)
    print("[SUCCESS] All tasks completed. Check your 'models/' folder for updates.")

if __name__ == '__main__':
    main()