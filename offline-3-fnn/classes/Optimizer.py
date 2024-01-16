import numpy as np

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