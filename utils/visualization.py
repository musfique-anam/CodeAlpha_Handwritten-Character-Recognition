import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplot(1, 2, figsize=(14, 5))

    #Accuracy tracking
    ax1.plot(history.history['accuracy'], label)