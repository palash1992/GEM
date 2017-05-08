import numpy as np

from keras.layers import Input, Dense
from keras.models import Model, model_from_json
import keras.regularizers as Reg

import pdb

def get_encoder(node_num, d, K, n_units, nu1, nu2, activation_fn):
	# Input
	x = Input(shape=(node_num,))
	# Encoder layers
	y = [None]*(K+1)
	y[0] = x # y[0] is assigned the input, but there are K other actual hidden layers.
	for i in range(K - 1):
		y[i+1] = Dense(n_units[i], activation=activation_fn, W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))(y[i])
	y[K] = Dense(d, activation=activation_fn, W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))(y[K - 1]) # K-th hidden layer is also the embedding layer
	# Encoder model
	encoder = Model(input=x, output=y[K])
	return encoder

def get_decoder(node_num, d, K, n_units, nu1, nu2, activation_fn):
	# Input
	y = Input(shape=(d,))
	# Decoder layers
	y_hat = [None]*(K+1)
	y_hat[K] = y # decoder's K+1-th layer is also its input (and also the embedding)
	for i in range(K - 1, 0, -1):
		y_hat[i] = Dense(n_units[i-1], activation=activation_fn, W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))(y_hat[i+1])
	y_hat[0] = Dense(node_num, activation=activation_fn, W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))(y_hat[1]) 
	# Output
	x_hat = y_hat[0] # decoder's output is also the actual output
	# Decoder Model
	decoder = Model(input=y, output=x_hat)
	return decoder

def get_autoencoder(encoder, decoder):
	# Input
	x = Input(shape=(encoder.layers[0].input_shape[1],))
	# Generate embedding
	y = encoder(x)
	# Generate reconstruction
	x_hat = decoder(y)
	# Autoencoder Model
	autoencoder = Model(input=x, output=[x_hat, y])
	return autoencoder

def graphify(reconstruction):
	[n1,n2] = reconstruction.shape
	n = min(n1,n2)
	reconstruction = np.copy(reconstruction[0:n, 0:n])
	reconstruction = (reconstruction + reconstruction.T)/2
	reconstruction -= np.diag(np.diag(reconstruction))
	return reconstruction

def loadmodel(filename):
	try:
		model = model_from_json(open(filename).read())
	except:
		print('Error reading file: {0}. Cannot load previous model'.format(filename))
		exit()
	return model

def loadweights(model, filename):
	try:
		model.load_weights(filename)
	except:
		print('Error reading file: {0}. Cannot load previous weights'.format(filename))
		exit()

def savemodel(model, filename):
	json_string = model.to_json()
	open(filename, 'w').write(json_string)

def saveweights(model, filename):
	model.save_weights(filename, overwrite=True)