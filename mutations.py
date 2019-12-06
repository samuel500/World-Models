import numpy as np



def mutate_lstm_hw(weights, new_size):

	new_shape = (new_size, 4*new_size)
	new_weights = np.zeros(new_shape)
	mask = np.ones(new_shape)
	S, S4 = weights.shape[0], weights.shape[1] 

	for i in range(4):
		new_weights[:S,i*new_shape[0]:i*new_shape[0]+int(S4/4)] = weights[:,i*int(S4/4):(i+1)*int(S4/4)]
		mask[:S,i*new_shape[0]:i*new_shape[0]+int(S4/4)] = 0
	#mask?
	return new_weights, mask


def mutate_lstm_xw(weights, new_size):

	new_shape = (weights.shape[0], 4*new_size)
	new_weights = np.zeros(new_shape)
	mask = np.ones(new_shape)
	S, S4 = weights.shape[0], weights.shape[1] 


	for i in range(4):
		new_weights[:, i*new_size:i*new_size+int(S4/4)] = weights[:, i*int(S4/4): (i+1)*int(S4/4)]
		mask[:, i*new_size:i*new_size+int(S4/4)] = 0

	return new_weights, mask


def mutate_lstm_biases(weights, new_size):

	new_weights = np.zeros(new_size*4)
	new_weights[new_size:2*new_size] = np.ones(new_size) # different in TF 1.x? f comes 3rd?
	mask = np.ones(new_size*4)
	S, S4 = len(weights)/4, len(weights)

	for i in range(4):
		new_weights[i*new_size:i*new_size+int(S)] = weights[i*int(S4/4): (i+1)*int(S4/4)]
		mask[i*new_size:i*new_size+int(S)] = 0
	return new_weights, mask


def mutate_lstm(weights, new_size):

	assert len(weights)==3
	w1, m1 = mutate_lstm_xw(weights[0], new_size)
	w2, m2 = mutate_lstm_hw(weights[1], new_size) 
	w3, m3 = mutate_lstm_biases(weights[2], new_size)
	new_weights = [w1, w2, w3]
	masks = [m1, m2, m3]

	return new_weights, masks


def mutate_controller_w(weights, new_rnn_input, new_size, latent=32):
	new_shape = (latent+new_rnn_input, new_size)
	new_weights = np.zeros(new_shape)
	mask = np.ones(new_shape)

	new_weights[:weights.shape[0], :weights.shape[1]] = weights
	mask[:weights.shape[0], :weights.shape[1]] = 0

	return new_weights, mask


def mutate_controller_biases(weights, new_size):
	new_weights = np.zeros(new_size)
	mask = np.ones(new_size)
	new_weights[:len(weights)] = weights
	mask[:len(weights)] = 0
	return new_weights, mask


def mutate_controller(weights, new_rnn_input, new_size, latent=32):
	w1, m1 = mutate_controller_w(weights[0], new_rnn_input, new_size)
	w2, m2 = mutate_controller_biases(weights[1], new_size)
	new_weights = [w1, w2]
	masks = [m1, m2]
	return new_weights, masks


def mutate_output_w(weights, new_input):
	assert weights.shape[1] == 3
	new_shape = (new_input, 3)
	new_weights = np.zeros(new_shape)
	mask = np.ones(new_shape)

	new_weights[:weights.shape[0], :weights.shape[1]] = weights
	mask[:weights.shape[0], :weights.shape[1]] = 0

	return new_weights, mask


def mutate_output_biases(weights): #???
	assert len(weights) == 3
	mask = 0
	return weights, mask


def mutate_output(weights, new_output_input, new_size):
	w1, m1 = mutate_output_w(weights[0], new_output_input)
	w2, m2 =  mutate_output_biases(weights[1])
	new_weights = [w1, w2]
	masks = [m1, m2]
	return new_weights, masks