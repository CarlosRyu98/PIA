import numpy as np

class Capa:
    def __init__(self, num_neuronas, num_entradas, fun_act):
        self.peso_input = np.random.random((num_entradas+1, num_neuronas))
        self.fun_act = fun_act

    # Entra una matriz de 1 x n y sale una matriz de 1 x n
    def get_output(self, input):
        input = np.insert(input, 0, 1)
        if (input.shape[0] == self.peso_input.shape[0]):
            return self.step_function(np.matmul(input, self.peso_input))
        else: return "Ha habido un error: No coinciden el número de neuronas"

    # Decide qué hacer según la función de activación
    def step_function(self, input):
        if self.fun_act == 1:
            return self.sf_threshold(input)
        elif self.fun_act == 2:
            return self.sf_sigmoid(input)
        elif self.fun_act == 3:
            return self.sf_hyperbolic_tangent(input)
        elif self.fun_act == 4:
            return self.sf_relu(input)
        else:
            return "Ha habido un error: Función de activación incorrecta."

    def sf_threshold(self, input):
        output = input.copy()
        output[output<=0] = 0
        output[output>0] = 1
        return output

    def sf_sigmoid(self, input):
        return 1/(1+np.exp(-input))

    def sf_hyperbolic_tangent(self, input):
        return (np.exp(input) - np.exp(-input)) / (np.exp(input) + np.exp(-input))

    def sf_relu(self, input):
        return np.maximum([0], input)

    # Función para mostrar los datos de la capa
    def print(self):
        print("Inputs: \n", self.peso_input)
        print("Input shape: \n", self.peso_input.shape)
        print("F Act: \n", self.fun_act)