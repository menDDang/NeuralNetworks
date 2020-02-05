import numpy as np

class SoftmaxWithCrossEntropyLoss:
	def __init__(self):
		self.loss = None
		self.y = None
		self.y_hat = None

	def forward(self, x, y_hat):
		self.y_hat = y_hat
		self.y = softmax(x)
		self.loss = cross_entropy_error(self.y, self.y_hat)
		return self.loss

	def backward(self):
		dx = (self.y - self.y_hat) / self.y.shape[0]
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
		self.y = np.clip(y, self.epsilon, 1-self.epsilon)
		self.y_hat = y_hat
		loss = np.sum(-1 * y_hat * np.log(self.y), axis=1)
		return np.mean(loss)

	def backward(self):
		return -1 * self.y_hat / self.y		
