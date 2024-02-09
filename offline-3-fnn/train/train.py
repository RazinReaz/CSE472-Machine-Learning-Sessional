import torchvision.datasets as ds
from torchvision import transforms
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tqdm import tqdm

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import classes.utils as utils
import classes.network as network
import classes.Layer as Layer
import classes.Initializer as Initializer
import classes.Loss as Loss


def load_train_val_dataset(dataset_name: str):
    if dataset_name == "letters":
        train_validation_dataset = ds.EMNIST(root='./offline-3-fnn/data', split='letters',
                              train=True,
                              transform=transforms.ToTensor(),
                              download = False)
        train_dataset, validation_dataset = train_test_split(train_validation_dataset, test_size=0.15, random_state=42)
        X_train = np.array([sample[0].numpy().flatten() for sample in train_dataset])
        y_train = np.array([sample[1] for sample in train_dataset]) - 1
        X_validation = np.array([sample[0].numpy().flatten() for sample in validation_dataset])
        y_validation = np.array([sample[1] for sample in validation_dataset]) - 1
        input_size = 784
        output_size = 26
        return X_train, y_train, X_validation, y_validation, input_size, output_size
    elif dataset_name == "digits":
        train_validation_dataset = load_digits()
        X_train, X_validation, y_train, y_validation = train_test_split(train_validation_dataset.data, train_validation_dataset.target, test_size=0.15, random_state=42)
        input_size = 64
        output_size = 10
        return X_train, y_train, X_validation, y_validation, input_size, output_size
    else:
        raise ValueError("Invalid dataset name")

if __name__ == "__main__":
    

    # input_size = 784        # 28 * 28
    # output_size = 26        # 26 letters
    X_train, y_train, X_validation, y_validation, input_size, output_size = load_train_val_dataset("letters")
    learning_rate = 5e-4
    batch_size = 1024
    epochs = 100
    
    model_number = 3
    filepath = 'offline-3-fnn/trained-models/letter-model-'+ str(model_number)+'.pkl'
    modelpath = 'offline-3-fnn/trained-models/letter-model-'+ str(model_number)+'.pkl'
    
    model = network.FNN(input_size, output_size, learning_rate, batch_size, epochs)
    model.sequential(Layer.DenseLayer(input_size, 464, initializer=Initializer.Xavier()),
                    Layer.Sigmoid(),
                    Layer.Dropout(0.7),
                    Layer.DenseLayer(464, 348, initializer=Initializer.He()),
                    Layer.ReLU(),
                    Layer.Dropout(0.52),
                    Layer.DenseLayer(348, output_size, initializer=Initializer.Xavier()),
                    Layer.Softmax())

    print("model built\n")
    model.describe()
    model.train(X_train, y_train, X_validation, y_validation)
    print("model trained")
    network.export_model(model, filepath)
    print("model weights and biases saved in ", filepath)
    with open(modelpath, 'rb') as f:
        model = network.create_model(f)
    model.graphs(model_number=str(model_number), savepath='offline-3-fnn/report/images')
    
    training_accuracy, training_loss, training_confusion = model.score(X_train, y_train)
    validation_accuracy, validation_loss, validation_confusion = model.score(X_validation, y_validation)
    validation_macro_f1 = model.macro_f1(X_validation, y_validation)

    print("training accuracy:\t", training_accuracy*100, "%")
    print("validation accuracy:\t", validation_accuracy*100, "%")
    print("training loss:\t\t", training_loss)
    print("validation loss:\t", validation_loss)
    print("validation macro f1:\t", validation_macro_f1)

    # characters = [chr(i+97) for i in range(26)]
    # utils.confusion_heatmap(training_confusion, labels=characters, title="Training Confusion Matrix", model_number=str(model_number))
    # utils.confusion_heatmap(validation_confusion, labels=characters, title="Validation Confusion Matrix", model_number=str(model_number))
