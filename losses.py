import numpy as np

class lossFunction:
    def mse(self, yTrue, yPred):
        return np.mean(np.power(yTrue-yPred, 2))
    
    def errorPartial(self, yTrue, yPred):
        return (yTrue-yPred)