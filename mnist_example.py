import mnist
from numpyDNN.layers import *
from numpyDNN.models import *
from numpyDNN.optimizers import *
from numpyDNN.losses import *

# set hyper parameters
input_size = 28 * 28
output_size = 10

n_hid1 = 100
n_hid2 = 100
n_hid3 = 100

batch_size = 100
learning_rate = 0.01
training_epoch_num = 50

# load mnist data
train_x, train_y = mnist.load_data("mnist_train.csv", one_hot=True)
test_x, test_y = mnist.load_data("mnist_test.csv", one_hot=True)

# build model
model = Model([
	Affine(inputs=input_size, outputs=n_hid1, dtype=np.float32),
	Sigmoid(inputs=n_hid1, dtype=np.float32),

	Affine(inputs=n_hid1, outputs=n_hid2, dtype=np.float32),
	Sigmoid(inputs=n_hid1, dtype=np.float32),

	Affine(inputs=n_hid2, outputs=n_hid3, dtype=np.float32),
	Sigmoid(inputs=n_hid3, dtype=np.float32),

	Affine(inputs=n_hid3, outputs=output_size, dtype=np.float32),
	Softmax(inputs=output_size, dtype=np.float32)
	])


# compile model
loss_fn = CrossEntropyError()
optimizer = GradientDecentOptimizer(learning_rate)
model.compile(loss_fn=loss_fn, optimizer=optimizer)

# do training
model.fit(
	train_data=[train_x, train_y], 
	valid_data=[test_x, test_y],
	epochs=training_epoch_num
	)
