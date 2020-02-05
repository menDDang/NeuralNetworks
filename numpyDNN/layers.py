import numpy as np
from numpyDNN.initializers import *

class Affine:
	def __init__(self, inputs, outputs, dtype=np.float32, initializer=xavier_initializer):
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


	

















