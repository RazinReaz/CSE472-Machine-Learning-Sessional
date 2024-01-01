import torchvision.datasets as ds
from torchvision import transforms
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pickle
from functools import partial



# train_validation_dataset = ds.EMNIST(root='./offline-3-fnn/data', split='letters',
#                               train=True,
#                               transform=transforms.ToTensor(),
#                               download = True)



# TODO
# 2. implement dropout
# 4. implement momentum
# 6. He initialization
# 3. implement batch normalization
# 5. implement batch stuff
# 7. layer class has activation inside it?? to use layer-stack
# 8. modify score function

def softmax(x):
    """
    x: (classes, batch_size)
    """
    x_max = np.max(x, axis=0, keepdims=True)
    x = np.exp(x - x_max)
    probabilities = x / np.sum(x, axis=0, keepdims=True)
    return probabilities

def softmax_prime(x):   #! this is concerning as it is not the derivative of softmax
    return softmax(x) * (1 - softmax(x))

def relu(x):
    return np.maximum(x, 0)
def relu_prime(x):
    return np.where(x > 0, 1, np.where(x < 0, 0, 0.5))

def tanh(x):
    return np.tanh(x)
def tanh_prime(x):
    return 1 - np.tanh(x)**2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def one_hot(y, classes = None):
    if classes is None:
        classes = np.max(y) + 1
    y_one_hot = np.zeros((y.size, classes))
    y_one_hot[np.arange(y.size), y] = 1
    return y_one_hot.T


class Initializer():
    def __init__(self, stddev_calculation_function, name):
        self.stdev_calculation_function = stddev_calculation_function
        self.name = name
    def __str__(self) -> str:
        return "Initializer:" + self.name
    def __call__(self, input_size, output_size):
        np.random.seed(0)
        stdev = self.stdev_calculation_function(input_size, output_size)
        return np.random.normal(loc=0, scale=stdev, size=(output_size, input_size))
    
class Xavier(Initializer):
    def xavier(self, input_size, output_size):
        return np.sqrt(2 / (input_size + output_size))
    def __init__(self):
        super().__init__(self.xavier, "Xavier")

class He(Initializer):
    def he(self, input_size, output_size):
        return np.sqrt(2 / input_size)
    def __init__(self):
        super().__init__(self.he, "He")

class LeCun(Initializer):
    def lecun(self, input_size, output_size):
        return 1 / np.sqrt(input_size)
    def __init__(self):
        super().__init__(self.lecun, "LeCun")
        

class Dense_layer():
    def __init__(self, input_size, output_size, initializer = Xavier()):
        self.input_size = input_size
        self.output_size = output_size
        self.initializer = initializer
        self.weights = self.initializer(input_size, output_size)
        self.bias = self.initializer(1, output_size)
        
        
    def __call__(self, input):
        """
        input: (input_size, batch_size)
        output: (output_size, batch_size)
        """
        assert input.shape[0] == self.input_size
        self.input = input
        self.output = np.dot(self.weights, input) + self.bias
        return self.output
    
    def __str__(self) -> str:
        return "Dense Layer: (" +str(self.input_size) + "," + str(self.output_size) + ")\n" + str(self.initializer) 
    
    def backward(self, delta_output, learning_rate):
        """
        delta_output: (output_size, batch_size)
        """
        assert delta_output.shape == self.output.shape
        batch_size = delta_output.shape[1]
        self.delta_weights = np.dot(delta_output, self.input.T) / batch_size
        self.delta_bias = np.sum(delta_output, axis=1, keepdims=True) / batch_size

        weights_copy = self.weights.copy()
        self.weights -= learning_rate * self.delta_weights
        self.bias -= learning_rate * self.delta_bias

        return np.dot(weights_copy.T, delta_output)

        



class Activation():
    def __init__(self, activation, derivative, name):
        self.activation = activation
        self.derivative = derivative
        self.name = name
    def __str__(self) -> str:
        return "Activation: "+ self.name
    def __call__(self, input):
        assert len(input.shape) == 2
        self.input = input
        self.output = self.activation(input)
        return self.output    
    def backward(self, delta_output, learning_rate):
        return delta_output * self.derivative(self.input)

class Softmax(Activation):
    def __init__(self):
        super().__init__(activation=softmax, derivative=softmax_prime, name="Softmax")

class ReLU(Activation):
    def __init__(self):
        super().__init__(activation=relu, derivative=relu_prime, name="ReLU")
        
class Tanh(Activation):
    def __init__(self):
        super().__init__(activation=tanh, derivative=tanh_prime, name="Tanh")

class Sigmoid(Activation):
    def __init__(self):
        super().__init__(activation=sigmoid, derivative=sigmoid_prime, name="Sigmoid")


class Loss():
    def __call__(self, output, target):
        sample_losses = self.calculate(output, target)
        return np.mean(sample_losses)
    def __str__(self) -> str:
        return "Loss: "

class CrossEntropyLoss(Loss):
    def __str__(self) -> str:
        return super().__str__() + "Cross Entropy Loss"
    def calculate(self, output, target):
        """
        output: (classes, batch_size)
        target: (batch_size, ) or (classes, batch_size)
        """
        if len(target.shape) == 1:
            # if the target in not one hot encoded
            loss = -np.log(output[[target], np.arange(target.size)])
        elif len(target.shape) == 2:
            # if the target is one hot encoded
            loss = -np.sum(target * np.log(output), axis=0)
        return loss




class FNN():
    def __init__(self, input_size, output_size, learning_rate=0.001, batch_size=50, epochs=100, loss = CrossEntropyLoss()):
        self.input_size = input_size
        self.output_size = output_size
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        self.children = []

        self.built = False

        self.loss = loss
    
    def add_layer(self, layer):
        if len(self.children) == 0:
            assert type(layer) != Activation
            assert layer.input_size == self.input_size
            self.children.append(layer)
            return
        if isinstance(layer, Dense_layer):
            assert layer.input_size == self.children[-2].output_size
        elif isinstance(layer, Activation):
            assert not isinstance(self.children[-1], Activation)
        self.children.append(layer)
    
    def sequential(self, *layers):
        for layer in layers:
            self.add_layer(layer)
        self.built = True
    
    def forward(self, input):
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
            # if isinstance(layer, Activation):
            #     delta = delta * layer.derivative()
            # elif isinstance(layer, Dense_layer):
            #     temp_delta = np.dot(layer.weights.T, delta)
            #     layer.backprop(delta, self.learning_rate)
            #     delta = temp_delta

        # self.delta_z2 = self.output - target
        # self.delta_a1 = np.dot(self.layer2.weights.T, self.delta_z2)
        # self.delta_z1 = self.delta_a1 * self.activation1.derivative()
        # self.delta_input = np.dot(self.layer1.weights.T, self.delta_z1)
        
        # self.layer2.backprop(self.delta_z2, self.learning_rate)
        # self.layer1.backprop(self.delta_z1, self.learning_rate)

    def train(self, X, y):
        # train by mini batch
        for epoch in range(self.epochs):
            for i in range(0, X.shape[0], self.batch_size):
                self.forward(X[i:i+self.batch_size])
                self.backward(y[i:i+self.batch_size])
            if (epoch + 1) % 10 == 0:
                print("Epoch", epoch + 1, " : Loss", self.loss(self.output, y[i:i+self.batch_size]))

    def predict(self, X):
        return np.argmax(self.forward(X), axis=0)
            

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
        return accuracy, loss
    
    def macro_f1(self, X, y):
        output = self.predict(X)
        y = one_hot(y, classes=self.output_size)
        tp = np.sum(output * y, axis=1)
        fp = np.sum(output * (1 - y), axis=1)
        fn = np.sum((1 - output) * y, axis=1)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        return np.mean(f1)
    
    def describe(self):
        print("learning rate:", self.learning_rate)
        print("batch size:", self.batch_size)
        print("epochs:", self.epochs)
        print()
        for layer in self.children:
            print(layer)
        print(self.loss)
        
def show_image(image, label):
    plt.title('Label is {label}'.format(label=label))
    plt.imshow(image.reshape(28,28).T, cmap='gray')
    plt.show()

if __name__ == "__main__":
    filepath = 'offline-3-fnn/trained-models/letter-model.pkl'
    train_validation_dataset = ds.EMNIST(root='./offline-3-fnn/data', split='letters',
                              train=True,
                              transform=transforms.ToTensor(),
                              download = True)

    train_dataset, validation_dataset = train_test_split(train_validation_dataset, test_size=0.15, random_state=42)

    # dataset[i] returns a tuple of (image, label)
    # image is a tensor of shape (1, 28, 28) we can convert it to a numpy array of shape (28, 28) by calling .numpy()

    X_train = np.array([sample[0].numpy().flatten() for sample in train_dataset])
    y_train = np.array([sample[1] for sample in train_dataset]) - 1
    X_validation = np.array([sample[0].numpy().flatten() for sample in validation_dataset])
    y_validation = np.array([sample[1] for sample in validation_dataset]) -1


    # sample = 19
    # show_image(X_train[sample], y_train[sample])

    input_size = 784        # 28 * 28
    output_size = 26        # 26 letters
    learning_rate = 0.001
    batch_size = 100
    epochs = 200
    
    model = FNN(input_size, output_size, learning_rate, batch_size, epochs)
    model.sequential(Dense_layer(input_size, 1024, initializer=He()),
                    ReLU(),
                    Dense_layer(1024, output_size, initializer=Xavier()),
                    Softmax())
    print("model built\n")
    model.describe()
    model.train(X_train, y_train)
    print("model trained")
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print("model saved in ", filepath)
    
    training_accuracy, training_loss = model.score(X_train, y_train)
    validation_accuracy, validation_loss = model.score(X_validation, y_validation)
    validation_macro_f1 = model.macro_f1(X_validation, y_validation)

    print("training accuracy:\t", training_accuracy*100, "%")
    print("training loss:\t", training_loss)
    print("validation accuracy:\t", validation_accuracy*100, "%")
    print("validation loss:\t", validation_loss)
    print("validation macro f1:\t", validation_macro_f1)

 

    
        