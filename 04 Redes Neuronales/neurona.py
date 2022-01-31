import numpy as np
import math

class Neurona:
    def __init__(self, input1, input2):
        self.weights = np.random.random((1, input1+1))
        self.act = input2

    def weighted_sum(self, input):
        input = np.insert(input, 0, 1)
        suma = (self.weights*input).sum()
        return suma

    def step_function(self, input):
        suma = self.weighted_sum(input)
        if self.act == 1:
            return self.sf_threshold(suma)
        elif self.act == 2:
            return self.sf_sigmoid(suma)
        elif self.act == 3:
            return self.sf_hyperbolic_tangent(suma)
        elif self.act == 4:
            return self.sf_relu(suma)
        else:
            return "Ha habido un error."

    def sf_threshold(self, suma):
        if suma <= 0: return 0
        else: return 1

    def sf_sigmoid(self, suma):
        return 1/(1+math.exp(-suma))

    def sf_hyperbolic_tangent(self, suma):
        return (math.exp(suma) - math.exp(-suma)) / (math.exp(suma) + math.exp(-suma))

    def sf_relu(self, suma):
        return max(0, suma)