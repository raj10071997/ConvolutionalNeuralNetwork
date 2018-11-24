import keras
import numpy as np
import argparse
from keras.datasets import cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras import regularizers, optimizers
from collections import Counter


parser = argparse.ArgumentParser()
parser.add_argument('-b','--batch_size', type=int, default=128, metavar='NUMBER',
				help='batch size(default: 128)')
parser.add_argument('-e','--epochs', type=int, default=200, metavar='NUMBER',
				help='epochs(default: 200)')
parser.add_argument('-n','--stack_n', type=int, default=0, metavar='NUMBER',
				help='stack number n, total layers = 6 * n + 2 (default: 5)')
parser.add_argument('-d','--dataset', type=str, default="cifar10", metavar='STRING',
				help='dataset. (default: cifar10)')

args = parser.parse_args()
n_stack = args.stack_n
num_classes = 10
seed = 7
np.random.seed(seed)
weight_decay_lambda = 1e-2  # try in range (1e-6,1e-4)

def simpleMultiLayerPerceptron(img_input, X_train_shape, n_stack=0):
	model = Sequential()
	global num_classes
	num_pixels_per_example = X_train_shape[1]*X_train_shape[2]*X_train_shape[3]
	print("num_pixels_per_example = ",num_pixels_per_example)
	# model.add(BatchNormalization(momentum=0.9, epsilon=1e-5))
	model.add(Dense(32, input_shape=(num_pixels_per_example,), kernel_initializer='he_normal', 
				kernel_regularizer=regularizers.l2(weight_decay_lambda), activation='relu'))
	model.add(BatchNormalization(momentum=0.9, epsilon=1e-5))

	for i in range(n_stack):
		model.add(Dense(32,kernel_initializer='he_normal', 
				kernel_regularizer=regularizers.l2(weight_decay_lambda), activation='relu'))
		model.add(BatchNormalization(momentum=0.9, epsilon=1e-5))


	model.add(Dense(num_classes, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay_lambda), activation='softmax'))
	model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()
	return model


#def simpleModel(n_stack, img_input):


def preprocessing(x_train, x_validation, x_test):
	x_train = x_train.astype('float32')
	x_validation = x_validation.astype('float32')
	x_test = x_test.astype('float32')
	# mean_channel_0 = np.mean(x_train[:,0,:,:])
	# mean_channel_1 = np.mean(x_train[:,1,:,:])
	# mean_channel_2 = np.mean(x_train[:,2,:,:])
	# x_train[:,0,:,:] = x_train[:,0,:,:] - mean_channel_0
	# x_train[:,1,:,:] = x_train[:,1,:,:] - mean_channel_1
	# x_train[:,2,:,:] = x_train[:,2,:,:] - mean_channel_2

	# num_pixels_per_channel = x_train.shape[0]*x_train.shape[2]*x_train.shape[3]

	# # for images standard deviation is generally not required
	# std_channel_0 = np.sqrt(np.sum(x_train[:,0,:,:]*x_train[:,0,:,:])/num_pixels_per_channel)
	# std_channel_1 = np.sqrt(np.sum(x_train[:,1,:,:]*x_train[:,1,:,:])/num_pixels_per_channel)
	# std_channel_2 = np.sqrt(np.sum(x_train[:,2,:,:]*x_train[:,2,:,:])/num_pixels_per_channel)

	# x_train[:,0,:,:] /= std_channel_0 
	# x_train[:,1,:,:] /= std_channel_1
	# x_train[:,2,:,:] /= std_channel_2

	# # mean subraction and normalization for validation set
	# x_validation[:,0,:,:] -= mean_channel_0
	# x_validation[:,1,:,:] -= mean_channel_1
	# x_validation[:,2,:,:] -= mean_channel_2

	# x_validation[:,0,:,:] /= std_channel_0 
	# x_validation[:,1,:,:] /= std_channel_1
	# x_validation[:,2,:,:] /= std_channel_2

	# # mean subraction and normalization for test set
	# x_test[:,0,:,:] -= mean_channel_0
	# x_test[:,1,:,:] -= mean_channel_1
	# x_test[:,2,:,:] -= mean_channel_2

	# x_test[:,0,:,:] /= std_channel_0 
	# x_test[:,1,:,:] /= std_channel_1
	# x_test[:,2,:,:] /= std_channel_2	

	mean_of_entire_training_data = np.mean(x_train)
	x_train -= mean_of_entire_training_data
	num_pixels = x_train.shape[0]*x_train.shape[1]*x_train.shape[2]*x_train.shape[3]
	std_of_entire_training_data = np.sqrt(np.sum(x_train*x_train)/num_pixels)
	x_train /= std_of_entire_training_data

	x_validation -= mean_of_entire_training_data
	x_validation /= std_of_entire_training_data

	x_test -= mean_of_entire_training_data
	x_test /= std_of_entire_training_data

	return x_train, x_validation, x_test


if __name__ == '__main__':
	global num_classes
	if args.dataset == "cifar10":
		(x_train, y_train), (x_test, y_test) = cifar10.load_data()
		# print("x_train = ",x_train)
		# print("y_train = ",y_train)
		# print("xshape = ",x_train.shape)

	x_validation, y_validation = x_train[49000:], y_train[49000:]
	x_train, y_train = x_train[:20000], y_train[:20000]
	x_test, y_test = x_test[:1000], y_test[:1000]
	# b_train = y_train.reshape(1,10000)
	# b_validation = y_validation.reshape(1,1000)
	# b_test = y_test.reshape(1,1000)

	# # b_train, b_test, b_validation = Counter(b_train), Counter(b_test), Counter(b_validation)

	# unique, counts = np.unique(b_train, return_counts=True)
	# b_train = dict(zip(unique, counts))

	# unique, counts = np.unique(b_validation, return_counts=True)
	# b_validation = dict(zip(unique, counts))

	# unique, counts = np.unique(b_test, return_counts=True)
	# b_test = dict(zip(unique, counts))

	
	# print("b_train = ", b_train)
	# print("b_validation = ", b_validation)

	x_train, x_validation, x_test = preprocessing(x_train,x_validation,x_test)

	y_train = keras.utils.to_categorical(y_train,num_classes)
	y_test = keras.utils.to_categorical(y_test,num_classes)
	y_validation = keras.utils.to_categorical(y_validation, num_classes)


	img_input = Input(shape=(3,32,32))
	simpleMultiLayerPerceptronModel = simpleMultiLayerPerceptron(img_input, x_train.shape)
	#simpleCNNModel = Model(img_input,output)

	#change the shape for simpleMultiLayerPerceptronModel
	x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2]*x_train.shape[3]).astype('float32')
	x_validation = x_validation.reshape(x_validation.shape[0],x_validation.shape[1]*x_validation.shape[2]*x_validation.shape[3]).astype('float32')
	x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2]*x_test.shape[3]).astype('float32')

	print("x_train.shape x_validation.shape x_test.shape = ",x_train.shape, x_validation.shape, x_test.shape)

	simpleMultiLayerPerceptronModel.fit(x=x_train, y=y_train, validation_data = (x_validation, y_validation),
			epochs=40, steps_per_epoch=int(x_train.shape[0]/35), verbose=2, validation_steps = int(x_validation.shape[0]/35))

	score = simpleMultiLayerPerceptronModel.evaluate(x_test, y_test, verbose=2)
	print("score = {}".format(score))


	
