import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tqdm import tqdm

import classes.Loss as Loss
from classes.Layer import Dense_layer, Dropout, Activation
from classes.utils import one_hot



class FNN():
    def __init__(self, input_size, output_size, learning_rate=0.0005, batch_size=50, epochs=100, loss = Loss.CrossEntropyLoss()):
        self.input_size = input_size
        self.output_size = output_size
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        self.children = []

        self.built = False

        self.loss = loss
        self.training_losses = []
        self.validation_losses = []
        self.training_accuracies = []
        self.validation_accuracies = []
    
    def add_layer(self, layer):
        if len(self.children) == 0:
            assert isinstance(layer, Dense_layer)
            assert layer.input_size == self.input_size
            self.children.append(layer)
            return
        if isinstance(layer, Dense_layer):
            assert layer.input_size == self.children[-1].output_size
        elif isinstance(layer, Activation):
            assert isinstance(self.children[-1], Dense_layer)
            layer.set_size(self.children[-1].output_size)
        elif isinstance(layer, Dropout):
            assert isinstance(self.children[-1], Activation)
            layer.set_size(self.children[-1].output_size)
        self.children.append(layer)
    
    def sequential(self, *layers):
        for layer in layers:
            self.add_layer(layer)
        self.built = True
    
    def forward(self, input, training=True):
        """
        input: (input_size, batch_size)
        output: (output_size, batch_size)
        """
        assert self.built
        if len(input.shape) == 1:
            # if the input is a single sample as a 1d row, reshape it as a column
            input = input.reshape(input.shape[0], 1)
        elif input.shape[1] == self.input_size:
            # transforming the features as columns
            input = input.T

        assert input.shape[0] == self.input_size

        next = input
        for layer in self.children:
            if isinstance(layer, Dropout) and not training:
                continue
            next = layer(next)
        self.output = next
        return self.output
    
    def backward(self, target):
        """
        output: (classes, batch_size)
        target: (batch_size, 1) or (batch_size, classes)[one hot encoded] 
        """
        # print("\nIN BACKWARD")
        if len(target.shape) == 1:
            target = one_hot(target, classes = self.output_size)
        elif target.shape == (self.batch_size, self.output_size):
            # if the target is a one hot encoded matrix, transform it to a column
            target = target.T
        assert target.shape == self.output.shape

        # the final layer has cross entropy with one hot and softmax activation
        # we can do a shortcut and find the derivative of loss w.r.t z instead of a when this is the case
        delta = self.output - target   # target has to be one hot encoded
        for i, layer in enumerate(reversed(self.children)):
            if i == 0: continue
            delta = layer.backward(delta, self.learning_rate)

    def train(self, X, y, X_val, y_val):
        # train by mini batch
        for epoch in tqdm(range(self.epochs)):
            for i in range(0, X.shape[0], self.batch_size):
                self.forward(X[i:i+self.batch_size])
                self.backward(y[i:i+self.batch_size])

            self.training_losses.append(self.loss(self.forward(X, training=False), y))
            self.validation_losses.append(self.loss(self.forward(X_val, training=False), y_val))

            self.training_accuracies.append(self.accuracy(self.predict(X), y))
            self.validation_accuracies.append(self.accuracy(self.predict(X_val), y_val))



    def predict(self, X):
        return np.argmax(self.forward(X, training=False), axis=0)
            

    def accuracy(self, predictions, target):
        """
        output: (classes, batch_size)
        target: (batch_size, )
        """
        return np.mean(predictions == target)

    def score(self, X, y):
        output = self.predict(X)
        accuracy = self.accuracy(output, y)
        loss = self.loss(self.output, y)
        # confusion matrix
        confusion = np.zeros((self.output_size, self.output_size))
        for i in range(len(y)):
            confusion[output[i], y[i]] += 1
        confusion = confusion.astype(int)
        return accuracy, loss, confusion
    
    def macro_f1(self, X, y):
        output = self.predict(X)
        
        confusion = np.zeros((self.output_size, self.output_size))
        for i in range(len(y)):
            confusion[output[i], y[i]] += 1
        confusion = confusion.astype(int)
        precision = np.zeros(self.output_size)
        recall = np.zeros(self.output_size)
        for i in range(self.output_size):
            precision[i] = confusion[i, i] / np.sum(confusion[i, :])
            recall[i] = confusion[i, i] / np.sum(confusion[:, i])
        f1 = 2 * precision * recall / (precision + recall)
        return np.mean(f1)
    
    def save(self, filepath):
        # # dump only the weights and biases
        # weights = []
        # biases = []
        # input_sizes = []
        # output_sizes = []
        # for layer in self.children:
        #     if isinstance(layer, Dense_layer):
        #         weights.append(layer.weights)
        #         biases.append(layer.bias)
        #         input_sizes.append(layer.input_size)
        #         output_sizes.append(layer.output_size)
        # with open(filepath, 'wb') as f:
        #     pickle.dump((weights, biases, input_sizes, output_sizes), f)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
    def describe(self):
        print("learning rate:", self.learning_rate)
        print("batch size:", self.batch_size)
        print("epochs:", self.epochs)
        print()
        for layer in self.children:
            print(layer)
        print(self.loss)
        print()
    
    def graphs(self, model_number):
        plt.figure(figsize=(8, 6))
        plt.plot(self.training_losses, label="Training Loss")
        plt.plot(self.validation_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig('offline-3-fnn/report/images/'+model_number+'/loss.png')

        plt.figure(figsize=(8, 6))
        plt.plot(self.training_accuracies, label="Training Accuracy")
        plt.plot(self.validation_accuracies, label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig('offline-3-fnn/report/images/'+model_number+'/accuracy.png')
