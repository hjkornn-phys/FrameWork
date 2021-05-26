import numpy as np
from Loss import MSE, CrossEntropy
from Optimizer import GradientDescent, Momentum, XavierNormal


class Layer:
    def __init__(self):
        self.inputs = None
        self.outputs = None

    def forward(self, x, training=True):
        self.inputs = x
        self.outputs = self._forward(x, training=training)
        return self.outputs

    def _forward(self, x, training=True):
        raise NotImplementedError

    def backward(self, grad=1.0):
        if self.inputs is None or self.outputs is None:
            return None
        return self._backward(grad)

    def _backward(self, grad):
        raise NotImplementedError

    def update(self, grads):
        raise NotImplementedError


class StepFunction(Layer):
    def __init__(self):
        super().__init__()

    def _forward(self, x, training):
        y = np.where(x > 0, True, False)
        return y.astype(np.int)


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def _forward(self, x, training):
        return 1 / (1 + np.exp(-x))

    def _backward(self, grad):
        return grad * self.outputs * (1 - self.outputs)  # (N, out)


class Linear(Layer):
    """
    >>> l = Linear()
    >>> x = np.random.normal(size=(4,5))
    >>> y = l.forward(x)
    >>> np.array_equal(x, y)
    True
    >>> np.array_equal(np.ones_like(x), l.backward())
    True
    """

    def _forward(self, x, training):
        return x

    def _backward(self, grad):
        return grad * np.ones(self.inputs.shape)


class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def _forward(self, x, training):
        return np.where(x > 0, x, 0)

    def _backward(self, grad):
        return grad * np.where(self.outputs > 0, 1, 0)


class Softmax(Layer):
    def __init__(self):
        super().__init__()

    def _forward(self, x, training):
        ex = np.exp(x)
        partition = np.sum(ex, axis=1).reshape(x.shape[0], -1)
        out = ex / partition
        # cache = (ex, partition)
        return out

    def _backward(self, grad):
        grad = grad.reshape(grad.shape[0], -1)
        dipart = np.sum(self.outputs * grad, axis=1, keepdims=True)
        dex = np.ones(self.outputs.shape[1]) * dipart
        dex = grad - dex
        dx = self.outputs * dex
        return dx


class Regularizer:
    def update(self, weights):
        raise NotImplementedError


class WeightDecay(Regularizer):
    def update(self, weights, decay_rate=0.01):
        red = 1 - decay_rate
        return weights * red


class Dropout(Layer):
    def __init__(self, theta=0.5):
        super().__init__()
        self.theta = theta

    def _forward(self, x, training):
        if training:
            rng = np.random.default_rng()
            self.live = rng.binomial(1, self.theta, size=x.shape)
            return x * self.live
        return x * (1 - self.theta)

    def _backward(self, grad):
        return self.outputs * grad


class BatchNormalization(Layer):
    def __init__(self, gamma, beta, momentum=0.9):
        super().__init__()
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.mean = None
        self.var = None

    def _forward(self, x, training=True):
        epsilon = 1e-7
        if training:
            batch_mean = np.mean(x, axis=0, keepdims=False)
            batch_var = np.var(x, axis=0, keepdims=False)
            sqrtvar = (batch_var + epsilon) ** (1 / 2)
            xmu = x - batch_mean
            xhat = xmu / sqrtvar
            y = self.gamma * xhat + self.beta
            self.cache_update = (batch_mean, batch_var)
            self.cache_backward = (xhat, xmu, sqrtvar)
            return y
        xmu = x - self.mean
        sqrtvar = (self.var + epsilon) ** (1 / 2)
        xhat = xmu / sqrtvar
        y = np.mean(self.gamma, axis=0) * xhat + np.mean(self.beta, axis=0)
        return y

    def _backward(self, grad):
        N, D = self.inputs.shape
        xhat, xmu, sqrtvar = self.cache_backward
        grad_beta = grad  # (N, D)
        # grad_beta = np.sum(grad, axis=0, keepdims=True)  # (D,)
        g_gammaxhat = grad  # (N, D)
        grad_gamma = xhat * g_gammaxhat  # (N, D)
        # grad_gamma = np.sum(xhat * g_gammaxhat, axis=0, keepdims=True)  # (D,)
        g_xhat = self.gamma * g_gammaxhat  # (N, D)
        g_ivar = np.sum(xmu * g_xhat, axis=0, keepdims=True)  # (D,)
        g_sqrtvar = -1.0 / sqrtvar ** 2 * g_ivar  # (D,)
        g_var = 1.0 / (2 * sqrtvar) * g_sqrtvar  # (D,)
        g_sq = 1.0 / N * np.ones((N, D)) * g_var  # (N, D)
        g_xmu1 = g_xhat / sqrtvar  # (N, D)
        g_xmu2 = 2 * xmu * g_sq  # (N, D)
        g_xmu = g_xmu1 + g_xmu2  # (N, D)
        g_mu = -1.0 * np.sum(g_xmu, axis=0, keepdims=True)  # (D,)
        g_x1 = 1.0 / N * np.ones((N, D)) * g_mu  # (N, D)
        g_x2 = g_xmu  # (N, D)
        grad_next = g_x1 + g_x2  # (N, D)
        return (grad_gamma, grad_beta), grad_next

    def update(self, grads):
        grad_gamma, grad_beta = grads
        self.gamma -= grad_gamma
        self.beta -= grad_beta

        N, D = self.inputs.shape
        if (not isinstance(self.mean, np.ndarray)) or (
            not isinstance(self.var, np.ndarray)
        ):
            self.mean = np.zeros(D)  # input shape = (N, D).  D dim
            self.var = np.zeros(D)
        assert self.mean.shape == self.var.shape
        batch_mean, batch_var = self.cache_update
        self.mean = self.mean * self.momentum + (1 - self.momentum) * batch_mean
        self.var = (
            self.var * self.momentum + (1 - self.momentum) * N / (N + 1) * batch_var
        )


class Perceptron(Layer):
    def __init__(self, weights, bias, activation, regularizer):
        super().__init__()
        self.weights = weights  # (in, out)
        self.bias = bias  # (1, out)
        self.activation = activation
        self.regularizer = regularizer
        # print(self.weights,self.bias)

    def _forward(self, x, training):
        y = np.matmul(x, self.weights) + self.bias  # (N, out)
        if self.activation is None:
            return y
        y = self.activation.forward(y)
        return y

    def _backward(self, grad):
        if self.activation is None:
            grad_w = (
                np.matmul(np.transpose(self.inputs), grad) / self.inputs.shape[0]
            )  # (in, out)
            grad_b = np.mean(grad, axis=0, keepdims=True)  # (1, out)
            return grad_w, grad_b
        grad_a = self.activation.backward(grad)  # (N, out)
        grad_w = (
            np.matmul(np.transpose(self.inputs), grad_a) / self.inputs.shape[0]
        )  # (in, out)
        grad_b = np.mean(grad_a, axis=0, keepdims=True)  # (1, out)
        grad_next = np.matmul(grad_a, np.transpose(self.weights))  # (N, in)
        return (grad_w, grad_b), grad_next

    def update(self, grads):
        grad_w, grad_b = grads
        self.weights -= grad_w
        self.bias -= grad_b
        if self.regularizer is not None:
            (self.weights, self.bias) = self.regularizer.update(
                (self.weights, self.bias)
            )


class Model(Layer):
    def __init__(self):
        super().__init__()
        self.layers = []
        self.loss = None
        self.optimizer = None

    def add(self, layer):
        self.layers.extend(layer)

    def _forward(self, x, training=True):
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x

    def set_loss(self, loss):
        self.loss = loss

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def evaluate(self, x, y):
        if self.loss is None:
            return None
        return np.sum(self.loss.forward(y, self.predict(x))) / x.shape[0]

    def _backward(self, grad):
        grads = []
        layer_reversed = self.layers.copy()
        layer_reversed.reverse()
        for layer in layer_reversed:
            layer_grad, next_grad = layer.backward(grad)
            grad = next_grad
            grads.append((layer, layer_grad))
        return grads

    def update(self, grads):
        self.optimizer.update(grads)

    def train_once(self, x, y):
        logit = self.forward(x)
        dout = self.loss.backward(y, logit)
        grads = self.backward(dout)
        self.optimizer.update(grads)

    def train(
        self, x, y, batch_size=1, epochs=100000, interval=1000, validation_data=None
    ):
        def batch_generator(batch_size):
            i = 0
            while i < len(x):
                batch = x[i : i + batch_size]
                y_batch = y[i : i + batch_size]
                batch = batch.reshape(batch_size, -1)
                y_batch = y_batch.reshape(batch_size, -1)
                yield batch, y_batch
                i += batch_size

        epoch = 0
        while epoch < epochs:
            gen = batch_generator(batch_size)
            try:
                while 1:
                    batch, y_batch = next(gen)
                    logit_batch = self.forward(batch)
                    dout = self.loss.backward(y_batch, logit_batch)
                    grads = self.backward(dout)
                    self.optimizer.update(grads)

            except StopIteration:
                if not (epoch % interval):
                    print(
                        f"Epoch: {epoch} Train_loss: {self.evaluate(x, y)}, batch_size :{batch.shape}"
                    )
                    if validation_data is not None:
                        x_val, y_val = validation_data
                        print(
                            f"Epoch: {epoch} Validation_loss: {self.evaluate(x_val, y_val)}"
                        )
                epoch += 1
                # print(epoch)

    def predict(self, x):
        return self.forward(x, training=False)


def create_xor_dataset():
    return np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 1, 1, 0])


def to_one_hot(x, n_class):
    return np.eye(n_class, dtype="uint8")[x]
