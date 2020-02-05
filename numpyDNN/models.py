import numpy as np
from numpyDNN.layers import *
from numpyDNN.optimizers import *

class Model:
	def __init__(self, layer_list):
		self.layers = list()
		for layer in layer_list:
			self.layers.append(layer)

		self.loss_fn = None
		self.optimizer = None

	def __call__(self, x):
		for layer in self.layers:
			x = layer(x)
		return x

	def compile(self, loss_fn, optimizer):
		self.loss_fn = loss_fn
		self.optimizer = optimizer

	def evaluate(self, x, y_hat):
		if self.loss_fn is None:
			raise TypeError("model.loss_fn must be compiled!")

		y = self.__call__(x)
		loss = self.loss_fn(y, y_hat)
		return loss

	def fit(self, train_data, epochs=100, valid_data=None, batch_size=100):
		if self.loss_fn is None:
			raise TypeError("model.loss_fn must be compiled!")

		if self.optimizer is None:
			raise TypeError("model.optimizer must be compiled!")

		i = 0
		for epoch in range(epochs):
			# get batch
			if i + batch_size < train_data[0].shape[0]:
				batch_x, batch_y = train_data[0][i:i+batch_size], train_data[1][i:i+batch_size]
				i += batch_size
			else:
				batch_x, batch_y = train_data[0][i:], train_data[1][i:]
				i = 0

			# train on batch
			train_loss = self.train_on_batch(batch_x, batch_y)

			# validate
			if valid_data is not None:
				y = self.__call__(valid_data[0])
				valid_loss = self.loss_fn(y, valid_data[1])
				print("Epoch : {}, Train Loss : {}, Valid Loss : {}".format(epoch, train_loss, valid_loss))
			else:
				print("Epoch : {}, Train Loss : {}".format(epoch, train_loss))


	def train_on_batch(self, batch_x, batch_y):
		if self.loss_fn is None:
			raise TypeError("model.loss_fn must be compiled!")

		if self.optimizer is None:
			raise TypeError("model.optimizer must be compiled!")

		# feed-forward
		y = self.__call__(batch_x)

		# get loss
		train_loss = self.loss_fn(y, batch_y)

		# feed-backward
		dout = self.loss_fn.backward()
		for layer in reversed(self.layers):
			dout = layer.backward(dout)

		# update parameters
		for layer in self.layers:
			if layer.trainable:
				self.optimizer(layer)

		return train_loss



