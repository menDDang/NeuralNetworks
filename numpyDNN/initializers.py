import numpy as np

def He_initializer(inputs, outputs):
	scale = np.sqrt(2.0 - inputs)
	initialized_weight = scale * np.random.randn(inputs, outputs)
	return initialized_weight

def xavier_initializer(inputs, outputs):
	scale = np.sqrt(1.0 / inputs)
	initialized_weight = scale * np.random.randn(inputs, outputs)
	return initialized_weight

