import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

import classes.Layer as Layer

def one_hot(y, classes = None):
    if classes is None:
        classes = np.max(y) + 1
    y_one_hot = np.zeros((y.size, classes))
    y_one_hot[np.arange(y.size), y] = 1
    return y_one_hot.T



def show_image(image, label):
    plt.title('Label is {label}'.format(label=label))
    plt.imshow(image.reshape(28,28).T, cmap='gray')
    plt.show()

def confusion_heatmap(confusion_matrix, labels, title, model_number, savepath, save=True):
    filepath = savepath+'/'+model_number
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, cmap='viridis', fmt='d', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.title(title)
    if save:
        plt.savefig(f"{filepath}/{title}.png")
    else:
        plt.show()

