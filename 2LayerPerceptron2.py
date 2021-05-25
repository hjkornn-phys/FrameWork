from Model import Layer, Perceptron, Sigmoid
from Loss import MSE
import numpy as np


class Model(Layer):
    def __init__(self):
        super().__init__()
        self.layers = []

    def add(self, layer):
        self.layers.extend(layer)

    def _forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def set_loss(self, loss):
        self.loss = loss

    def evaluate(self, x, y):
        if self.loss is None:
            return None
        return self.loss.forward(y, self.forward(x))


W1 = np.array([[1, -1], [1, -1]])  # 첫 번째 항은 첫번째 output node로 들어오는 weights에 대응
W2 = np.array([-1, -1])
b1 = np.array([-1.5, 0.5])
b2 = np.array([0.5])

p1 = Perceptron(W1, b1)
p2 = Perceptron(W2, b2)
sig = Sigmoid()


if __name__ == "__main__":
    model = Model()
    model.add([p1, sig, p2, sig])
    model.set_loss(MSE())
    print(model.evaluate(np.array([0, 1]), np.array([[0, 0], [0, 1]])))
