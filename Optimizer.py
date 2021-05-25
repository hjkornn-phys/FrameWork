import numpy as np
from numpy.lib.type_check import _getmaxmin


class Optimizer:
    def __init__(self, lr):
        self.lr = lr

    def update(self, grads):
        raise NotImplementedError


class GradientDescent(Optimizer):
    def update(self, grads):
        for idx, grad_layer in enumerate(grads):
            layer, grad = grad_layer
            grad_w, grad_b = grad
            update_grad = (grad_w * self.lr, grad_b * self.lr)
            layer.update(update_grad)


class Momentum(Optimizer):
    def __init__(self, lr, gamma=0.9):
        super().__init__(lr)
        self.gamma = gamma
        self.prev_grads = None

    def update(self, grads):
        if self.prev_grads == None:
            self.prev_grads = [(0, 0)] * len(grads)
        for idx, grad_layer in enumerate(grads):
            layer, grad = grad_layer
            prev_grad_w, prev_grad_b = self.prev_grads[idx]
            grad_w, grad_b = grad
            update_grad = np.sum(
                (
                    (self.gamma * prev_grad_w, self.gamma * prev_grad_b),
                    (grad_w * self.lr, grad_b * self.lr),
                ),
                axis=0,
            )
            self.prev_grads[idx] = update_grad
            layer.update(update_grad)


class Initializer:
    def create(self, shape):
        raise NotImplementedError


class XavierNormal(Initializer):
    def create(self, shape):
        return np.random.normal(scale=np.sqrt(2 / (shape[0] + shape[1])), size=shape)
