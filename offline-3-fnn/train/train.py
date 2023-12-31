import torchvision.datasets as ds
from torchvision import transforms
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
import math



# train_validation_dataset = ds.EMNIST(root='./offline-3-fnn/data', split='letters',
#                               train=True,
#                               transform=transforms.ToTensor(),
#                               download = True)



# TODO
# 1. do 1 hidden layer first with 100 
# 2. implement dropout
# 3. implement batch normalization
# 4. pickle the model
# 5. implement batch stuff
# 6. loss interface
# 7. layer class has activation inside it?? to use layer-stack

def softmax(x):
    """
    x: (classes, batch_size)
    """
    x_max = np.max(x, axis=0, keepdims=True)
    x = np.exp(x - x_max)
    probabilities = x / np.sum(x, axis=0, keepdims=True)
    return probabilities

def relu(x):
    return np.maximum(x, 0)

def one_hot(y, classes = None):
    if classes is None:
        classes = np.max(y) + 1
    y_one_hot = np.zeros((y.size, classes))
    y_one_hot[np.arange(y.size), y] = 1
    return y_one_hot.T



class dense_layer():
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        np.random.seed(0)
        stdev = np.sqrt(2 / (input_size + output_size))
        self.weights = np.random.normal(loc=0, scale=stdev, size=(output_size, input_size))
        self.bias = np.random.normal(loc=0, scale=stdev, size=(output_size, 1))
        
    def __call__(self, input):
        """
        input: (input_size, batch_size)
        output: (output_size, batch_size)
        """
        assert input.shape[0] == self.input_size
        self.input = input
        self.output = np.dot(self.weights, input) + self.bias
        return self.output
    
    def backprop(self, delta_output, learning_rate):
        """
        delta_output: (output_size, batch_size)
        """
        # print("\nIN LAYER BACKPROP")
        assert delta_output.shape == self.output.shape
        batch_size = delta_output.shape[1]
        self.delta_weights = np.dot(delta_output, self.input.T) / batch_size
        self.delta_bias = np.sum(delta_output, axis=1, keepdims=True) / batch_size

        # print("delta_weights shape", self.delta_weights.shape)
        # print("weights shape", self.weights.shape)
        # print("delta_bias shape", self.delta_bias.shape)
        # print("bias shape", self.bias.shape)

        self.weights -= learning_rate * self.delta_weights
        self.bias -= learning_rate * self.delta_bias

        



class Activation():
    def __call__(self, input):
        raise NotImplementedError("Subclasses must implement __call__ method.")
    
    def derivative(self):
        raise NotImplementedError("Subclasses must implement derivative method.")


class Softmax(Activation):
    def __call__(self, input):
        assert len(input.shape) == 2
        self.input = input
        self.output = softmax(input)
        return self.output
    def derivative(self):
        return self.output * (1 - self.output) #! 1{i=j} - self.output?


class ReLU(Activation):
    def __call__(self, input):
        self.input = input
        self.output = relu(input)
        return self.output
    def derivative(self):
        return np.where(self.input > 0, 1, np.where(self.input < 0, 0, 0.5)) #! will this work?
    


class Loss():
    def __call__(self, output, target):
        sample_losses = self.calculate(output, target)
        return np.mean(sample_losses)
    def derivative(self, input):
        raise NotImplementedError("Subclasses must implement derivative method.")

class CrossEntropyLoss(Loss):
    def calculate(self, output, target):
        """
        output: (classes, batch_size)
        target: (batch_size, ) or (classes, batch_size)
        """
        if len(target.shape) == 1:
            loss = -np.log(output[[target], np.arange(target.size)])
        elif len(target.shape) == 2:
            loss = -np.sum(target * np.log(output), axis=0)
        return loss
    def derivative(self, input):
        return 1/input


input_size = 784        # 28 * 28
hidden_size = 100
output_size = 26        # 26 letters
learning_rate = 0.001
batch_size = 50

class FNN():
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001, batch_size=3, epochs=100):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        self.layer1 = dense_layer(input_size, hidden_size)
        self.activation1 = ReLU()
        self.layer2 = dense_layer(hidden_size, output_size)
        self.activation2 = Softmax()
        self.loss = CrossEntropyLoss()
    
    def forward(self, input):
        """
        input: (input_size, batch_size)
        output: (output_size, batch_size)
        """
        # print("\nIN FORWARD")
        if len(input.shape) == 1:
            # if the input is a single sample as a 1d row, reshape it as a column
            input = input.reshape(input.shape[0], 1)
        elif input.shape[1] == self.input_size:
            # transforming the features as columns
            input = input.T
        assert input.shape[0] == self.input_size

        z1 = self.layer1(input)
        a1 = self.activation1(z1)
        z2 = self.layer2(a1)
        a2 = self.activation2(z2)
        self.output = a2
        # print("z1 shape", z1.shape)
        # print("a1 shape", a1.shape)
        # print("z2 shape", z2.shape)
        # print("a2 shape", a2.shape)
        return self.output
    
    def backward(self, target):
        """
        output: (classes, batch_size)
        target: (batch_size, 1)
        """
        # print("\nIN BACKWARD")
        if len(target.shape) == 1:
            target = one_hot(target, classes = self.output_size)
        elif target.shape == (self.batch_size, self.output_size):
            # if the target is a one hot encoded matrix, transform it to a column
            target = target.T
        assert target.shape == self.output.shape
        # print("output shape", self.output.shape)
        # print("target shape", target.shape)

        # the final layer has cross entropy with one hot and softmax activation
        # we can do a shortcut and find the derivative of loss w.r.t z instead of a when this is the case
        self.delta_z2 = self.output - target   # target has to be one hot encoded
        self.delta_a1 = np.dot(self.layer2.weights.T, self.delta_z2)
        self.delta_z1 = self.delta_a1 * self.activation1.derivative()
        self.delta_input = np.dot(self.layer1.weights.T, self.delta_z1)

        # print("delta_z2", self.delta_z2)
        # print("delta_a1", self.delta_a1)
        # print("activation1 derivative", self.activation1.derivative())

        # print("delta_z2 shape", self.delta_z2.shape)
        # print("delta_z1 shape", self.delta_z1.shape)
        
        self.layer2.backprop(self.delta_z2, self.learning_rate)
        self.layer1.backprop(self.delta_z1, self.learning_rate)

    def train(self, X, y):
        # train by mini batch
        for epoch in range(self.epochs):
            for i in range(0, X.shape[0], self.batch_size):
                self.forward(X[i:i+self.batch_size])
                self.backward(y[i:i+self.batch_size])
                break
            if epoch + 1 % 10 == 0:
                print("Epoch", epoch + 1, " : Loss", self.loss(self.output, y[i:i+self.batch_size]))

    def predict(self, X):
        return self.forward(X)
            

    def accuracy(self, output, target):
        """
        output: (classes, batch_size)
        target: (batch_size, )
        """
        # print("\nIN ACCURACY")

        predictions = np.argmax(output, axis=0)
        return np.mean(predictions == target)



if __name__ == "__main__":
    # 
    X, y = load_digits(return_X_y=True)
    
    model = FNN(64, 24, 10, epochs=1000, batch_size=100, learning_rate=0.01)
    model.train(X, y)

    output = model.predict(X)
    accuracy = model.accuracy(output, y)
    print("accuracy", accuracy*100, "%")

    # for sample in range(10, 20):
    #     predictions = model.predict(X[sample])
    #     prediction = np.argmax(predictions, axis=0)[0]
    #     print("prediction", prediction, "  y[", sample ,"]", y[sample])
        