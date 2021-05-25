import numpy as np


class Loss:
    def forward(self, y_true, y_pred):
        raise NotImplementedError

    def _forward(self, y_true, y_pred):
        raise NotImplementedError

    def backward(self, y_true, y_pred):
        raise NotImplementedError


class MSE(Loss):
    def forward(self, y_true, y_pred):
        loss = np.mean(np.square(y_true.reshape(y_pred.shape[0], -1) - y_pred), axis=0)
        return loss

    def backward(self, y_true, y_pred):
        diff = y_pred - y_true
        mean_diff = np.mean(diff, axis=0, keepdims=True)
        return mean_diff


class CrossEntropy(Loss):
    def forward(self, y_true, y_pred):
        y_pred = np.where(y_pred < 1e-15, 1e-15, y_pred)
        cce = -np.sum(y_true * np.log(y_pred).reshape(y_true.shape), axis=0)
        return cce

    def backward(self, y_true, y_pred):
        grad = np.where(y_true == 1, -1.0 / y_pred, 0)
        return grad