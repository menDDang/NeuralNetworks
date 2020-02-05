import numpy as np

class GradientDecentOptimizer:
	def __init__(self, learning_rate=0.01):
		self.learning_rate = learning_rate

	def __call__(self, layer):
		for key, param in layer.params.items():
			param -= self.learning_rate * layer.grad[key]
