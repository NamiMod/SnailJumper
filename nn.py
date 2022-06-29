import numpy as np

class NeuralNetwork:
    
    w = []
    b = []

    def __init__(self, layer_sizes):
        """
        Neural Network initialization.
        Given layer_sizes as an input, you have to design a Fully Connected Neural Network architecture here.
        :param layer_sizes: A list containing neuron numbers in each layers. For example [3, 10, 2] means that there are
        3 neurons in the input layer, 10 neurons in the hidden layer, and 2 neurons in the output layer.
        """
        # initialize w and b with random numbers and 0

        self.activation = np.vectorize(self.activation)

        for i in range(0,len(layer_sizes) - 1):
            NeuralNetwork.b.append(np.zeros((layer_sizes[i + 1], 1))) # [len(leyer) * 1]
            NeuralNetwork.w.append(np.random.normal(0, 1, size=(layer_sizes[i + 1], layer_sizes[i]))) # [len(leyer[i+1]) * len(layer[i])]
            

    def activation(self, x):
        """
        The activation function of our neural network, e.g., Sigmoid, ReLU.
        :param x: Vector of a layer in our network.
        :return: Vector after applying activation function.
        """
        # Sigmoid :
        return 1/(1 + np.exp(-x))

    def forward(self, x):
        """
        Receives input vector as a parameter and calculates the output vector based on weights and biases.
        :param x: Input vector which is a numpy array.
        :return: Output vector
        """
        # output = w * x + b

        output = x
        for i in range(0,len(self.w)):
            output = self.activation(NeuralNetwork.w[i] @ output + NeuralNetwork.b[i])
            
        return output
