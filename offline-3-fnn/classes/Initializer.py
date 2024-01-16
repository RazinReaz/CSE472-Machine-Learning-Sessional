import numpy as np

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
        
