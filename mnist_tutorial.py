import os
import numpy as np


def load_data(file_name, one_hot=False, dtype=np.float32):
    data = np.loadtxt(file_name, delimiter=',', dtype=np.int32)
    x = data[:, 1:].astype(dtype) / 255  # normalize data
    y = data[:, 0]

    if one_hot:
        _y = np.zeros(shape=[y.shape[0], 10], dtype=dtype)
        for i in range(y.shape[0]):
            _y[i, y[i]] = 1
        y = _y
    y = y.astype(dtype)
    return x, y


class Affine:
    def __init__(self, inputs, outputs, dtype=np.float32, initializer=np.random.randn):
        self.inputs = inputs
        self.outputs = outputs
        self.dtype = dtype
        self.initializer = initializer

        self.trainable = True

        self.params = dict()
        self.params['W'] = initializer(inputs, outputs)
        self.params['b'] = np.ones(shape=[outputs], dtype=dtype)

        self.x = None
        self.grad = dict()

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.x = x
        return np.matmul(x, self.params['W']) + self.params['b']

    def backward(self, dout):
        self.grad['W'] = np.matmul(self.x.T, dout)
        self.grad['b'] = np.sum(dout, axis=0)

        dx = np.matmul(dout, self.params['W'].T)
        return dx

    def update(self, learning_rate):
        self.params['W'] -= learning_rate * self.grad['W']
        self.params['b'] -= learning_rate * self.grad['b']


class Sigmoid:
    def __init__(self, inputs, dtype=np.float32):
        self.inputs = inputs
        self.dtype = dtype

        self.trainable = False
        self.out = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dout):
        return dout * (1.0 - self.out) * self.out


class ReLU:
    def __init__(self, inputs, dtype=np.float32):
        self.inputs = inputs
        self.dtype = dtype

        self.trainable = False
        self.mask = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        return dout


class Softmax:
    def __init__(self, inputs, dtype=np.float32):
        self.inputs = inputs
        self.dtype = dtype
        self.out = None
        self.dS = None
        self.trainable = False

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x = x - np.expand_dims(np.max(x, axis=1), 1)
        exp_x = np.exp(x)
        sum_exp = np.expand_dims(np.sum(exp_x, axis=1), 1)
        self.out =  exp_x / sum_exp
        return self.out

    def backward(self, dout):
        batch_size = dout.shape[0]
        dx = np.zeros(shape=[batch_size, self.inputs], dtype=self.dtype)
        dS = np.zeros(shape=[self.inputs, self.inputs], dtype=self.dtype)

        for b in range(batch_size):
            for i in range(self.inputs):
                for j in range(self.inputs):
                    delta_i_j = int(i == j)
                    dS[i, j] = self.out[b, i] * (delta_i_j - self.out[b, j])

            dx[b, :] = np.matmul(dout[b], dS)

        return dx


class MeanSquareError:
    def __init__(self):
        self.y = None
        self.y_hat = None

    def __call__(self, y, y_hat):
        self.y = y
        self.y_hat = y_hat
        return np.mean(np.sum((y - y_hat) ** 2, axis=1))

    def backward(self):
        return 2 * (self.y - self.y_hat)


class CrossEntropyError:
    def __init__(self, epsilon=1e-10):
        self.y = None
        self.y_hat = None
        self.epsilon = epsilon

    def __call__(self, y, y_hat):
        self.y = np.clip(y, self.epsilon, 1 - self.epsilon)
        self.y_hat = y_hat
        loss = np.sum(-1 * y_hat * np.log(self.y), axis=1)
        return np.mean(loss)

    def backward(self):
        return -1 * self.y_hat / self.y


def xavier_initializer(inputs, outputs):
    scale = np.sqrt(1.0 / inputs)
    initialized_weight = scale * np.random.randn(inputs, outputs)
    return initialized_weight


""" Local Functions """
def accuracy(y_pred, y_true):
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)
    correct_num = np.sum(np.equal(y_pred, y_true).astype(np.int32))
    return correct_num / len(y_pred)


def get_batch(x, y, i, batch_size=100):
    if i + batch_size >= len(x):
        i = 0

    return x[i:i+batch_size], y[i:i+batch_size], i + batch_size


if __name__ == "__main__":
    # Set hyper-parameters
    input_size = 28 * 28
    n_hid = 100
    output_size = 10  # number of class

    batch_size = 100
    learning_rate = 0.01
    train_epoch_num = 100

    # Read MNIST data
    train_x, train_y = load_data("mnist_train.csv", one_hot=True)
    test_x, test_y = load_data("mnist_test.csv", one_hot=True)

    # create model
    A1 = Affine(inputs=input_size, outputs=n_hid, initializer=xavier_initializer)
    Z1 = Sigmoid(inputs=n_hid)
    A2 = Affine(inputs=n_hid, outputs=output_size, initializer=xavier_initializer)
    Z2 = Softmax(inputs=output_size)
    loss_fn = MeanSquareError()

    for epoch in range(1, train_epoch_num+1):
        # get batch
        i = 0
        batch_x, batch_y, i = get_batch(train_x, train_y, i, batch_size=batch_size)

        # feed forward
        y_pred = batch_x
        for layer in [A1, Z1, A2, Z2]:
            y_pred = layer.forward(y_pred)

        train_loss = loss_fn(y_pred, batch_y)
        train_acc = accuracy(y_pred, batch_y)
        print("Epoch : {}, Train Loss : {}, Train Accuracy : {}".format(epoch, train_loss, train_acc))

        # feed backward
        dout = loss_fn.backward()
        for layer in [Z2, A2, Z1, A1]:
            dout = layer.backward(dout)

        # update parameters
        for layer in [A1, A2]:
            layer.update(learning_rate=learning_rate)

        # validate
        if epoch % 10 == 0:
            y_pred = test_x
            for layer in [A1, Z1, A2, Z2]:
                y_pred = layer.forward(y_pred)
            test_loss = loss_fn(y_pred, test_y)
            test_acc = accuracy(y_pred, test_y)
            print("Epoch : {}, Test Loss : {}, Test Accuracy : {}".format(epoch, test_loss, test_acc))


