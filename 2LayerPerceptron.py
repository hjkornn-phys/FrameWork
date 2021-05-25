from Model import Model, Perceptron, StepFunction
import numpy as np


xor_model = Model()
W1 = np.transpose([[1, 1], [-1, -1]])
W2 = np.array([-1, -1])
b1 = np.array([-1.5, 0.5])
b2 = np.array([0.5])

p1 = Perceptron(W1, b1)
p2 = Perceptron(W2, b2)
step = StepFunction()

xor_model.add([p1, step, p2, step])
