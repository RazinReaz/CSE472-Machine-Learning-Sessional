import numpy as np

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