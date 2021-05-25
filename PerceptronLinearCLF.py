from Model import Model, Perceptron
import numpy as np


model = Model()
# AND Gate 기능을 할 수 있는 W값을 제시하세요.
W = np.array([1, 1])
# AND Gate 기능을 할 수 있는 b값을 제시하세요.
b = np.array([-1.5])

p = Perceptron(W, b)

model.add([p])
