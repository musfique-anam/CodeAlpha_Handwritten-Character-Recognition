import tensorflow as tf

def load_and_preprocess_data():
    #Load the build-in MNIST dataset
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    #Normalize pixel intensities from [0, 255] down to [0, 1]
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    #Reshape matrices to add the explicit channel dimension (28x28 -> 28x28x1)
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    return X_train, y_train, X_test, y_test