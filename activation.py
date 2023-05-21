import numpy as np
class activation:
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoidDerivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))