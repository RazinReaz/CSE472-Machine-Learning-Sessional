import torchvision.datasets as ds
from torchvision import transforms
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle



def softmax(x):
    """
    x: (classes, batch_size)
    """
    x_max = np.max(x, axis=0, keepdims=True)
    x = np.exp(x - x_max)
    probabilities = x / np.sum(x, axis=0, keepdims=True)
    return probabilities

def softmax_prime(x):   #! this is concerning as it assumes  that i==j
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

def confusion_heatmap(confusion_matrix, labels, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, cmap='viridis', fmt='d', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.title(title)
    plt.show()


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
        self.weight_optimizer = Adam(input_size, output_size)
        self.bias_optimizer = Adam(1, output_size)
        
        
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
        self.weights = self.weight_optimizer.update(self.weights, self.delta_weights, learning_rate)
        self.bias = self.bias_optimizer.update(self.bias, self.delta_bias, learning_rate)
        # self.weights -= learning_rate * self.delta_weights
        # self.bias -= learning_rate * self.delta_bias

        return np.dot(weights_copy.T, delta_output)


class Adam():
    def __init__(self, input_size, output_size, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros((output_size, input_size))
        self.v = np.zeros((output_size, input_size))
        self.t = 1
    def __str__(self) -> str:
        return "Adam: " + str(self.beta1) + " " + str(self.beta2) + " " + str(self.epsilon)
    def update(self, weights, delta_weights, learning_rate):
        self.m = self.beta1 * self.m + (1 - self.beta1) * delta_weights
        self.v = self.beta2 * self.v + (1 - self.beta2) * delta_weights**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        weights -= learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        self.t += 1
        return weights
        



class Activation():
    def __init__(self, activation, derivative, name):
        self.activation = activation
        self.derivative = derivative
        self.name = name
    def set_size(self, size):
        self.input_size = size
        self.output_size = size
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

class Dropout():
    def __init__(self, keep_prob):
        self.keep_prob = keep_prob
    def set_size(self, size):
        self.input_size = size
        self.output_size = size
    def __str__(self) -> str:
        return "Dropout: " + str(self.keep_prob)
    def __call__(self, input):
        self.input = input
        self.mask = np.random.binomial(1, self.keep_prob, size=input.shape)
        self.output = (input * self.mask) / self.keep_prob
        return self.output
    def backward(self, delta_output, learning_rate):
        return (delta_output * self.mask) / self.keep_prob


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
        print("output shape:", output.shape)
        print("y shape:", y.shape)
        # confusion matrix
        confusion = np.zeros((self.output_size, self.output_size))
        for i in range(len(y)):
            confusion[output[i], y[i]] += 1
        confusion = confusion.astype(int)
        return accuracy, loss, confusion
    
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
    
    def save(self, filepath):
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
        
def show_image(image, label):
    plt.title('Label is {label}'.format(label=label))
    plt.imshow(image.reshape(28,28).T, cmap='gray')
    plt.show()

def save_graph(filepath, model, X, y):
    training_accuracy, training_loss, training_confusion = model.score(X, y)
    validation_accuracy, validation_loss, validation_confusion = model.score(X, y)
    validation_macro_f1 = model.macro_f1(X, y)

    

if __name__ == "__main__":
    filepath = 'offline-3-fnn/trained-models/letter-model.pkl'
    modelpath = 'offline-3-fnn/trained-models/letter-model-3.pkl'
    train_validation_dataset = ds.EMNIST(root='./offline-3-fnn/data', split='letters',
                              train=True,
                              transform=transforms.ToTensor(),
                              download = True)
    print("dataset loaded")
    train_dataset, validation_dataset = train_test_split(train_validation_dataset, test_size=0.15, random_state=42)
    print("dataset split")
    X_train = np.array([sample[0].numpy().flatten() for sample in train_dataset])
    y_train = np.array([sample[1] for sample in train_dataset]) - 1
    X_validation = np.array([sample[0].numpy().flatten() for sample in validation_dataset])
    y_validation = np.array([sample[1] for sample in validation_dataset]) -1
    print("dataset converted to numpy arrays")
    input_size = 784        # 28 * 28
    output_size = 26        # 26 letters

    # X, y = load_digits(return_X_y=True)
    # X = X / X.max()
    # X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.15, random_state=42)
    # input_size = 64
    # output_size = 10

    learning_rate = 5e-4
    batch_size = 1000
    epochs = 200
    
    # model = FNN(input_size, output_size, learning_rate, batch_size, epochs)
    # model.sequential(Dense_layer(input_size, 256, initializer=He()),
    #                 ReLU(),
    #                 Dropout(0.7),
    #                 Dense_layer(256, output_size, initializer=Xavier()),
    #                 Softmax())
    
    # print("model built\n")
    # model.describe()
    # model.train(X_train, y_train)
    # print("model trained")
    # model.save(filepath)
    # print("model saved in ", filepath)

    # load from file
    with open(modelpath, 'rb') as f:
        model = pickle.load(f)
    print("model loaded from ", modelpath)
    model.describe()
    
    training_accuracy, training_loss, training_confusion = model.score(X_train, y_train)
    validation_accuracy, validation_loss, validation_confusion = model.score(X_validation, y_validation)
    validation_macro_f1 = model.macro_f1(X_validation, y_validation)

    print("training accuracy:\t", training_accuracy*100, "%")
    print("validation accuracy:\t", validation_accuracy*100, "%")
    print("training loss:\t\t", training_loss)
    print("validation loss:\t", validation_loss)
    print("validation macro f1:\t", validation_macro_f1)

    characters = [chr(i+97) for i in range(26)]
    confusion_heatmap(training_confusion, labels=characters, title="Training Confusion Matrix")
    confusion_heatmap(validation_confusion, labels=characters, title="Validation Confusion Matrix")

 

    
        