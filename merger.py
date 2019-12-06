

import numpy as np


def merge_diagonal(weights):
	#print(weights)
	new_shape = (sum([w.shape[0] for w in weights]), sum(w.shape[1] for w in weights))
	new_weights = np.zeros(new_shape)
	x, y = 0, 0
	for w in weights:
		new_weights[x:x+w.shape[0], y:y+w.shape[1]] = w
		x += w.shape[0]
		y += w.shape[1]
	#print(new_weights)

	return new_weights


def merge_lstm_hw(weights):
	#print(weights[0].shape[1])
	#print(weights[0][:,:int(weights[0].shape[1]/4)].shape)
	ifog = [[w[:,:int(w.shape[1]/4)] for w in weights], [w[:,int(w.shape[1]/4):2*int(w.shape[1]/4)] for w in weights],
			 [w[:,2*int(w.shape[1]/4): 3*int(w.shape[1]/4)] for w in weights], [w[:,3*int(w.shape[1]/4):] for w in weights]]
	ret = np.hstack([merge_diagonal(w) for w in ifog])
	#print(ret.shape)
	#ret = merge_diagonal(weights)
	return ret


def merge_lstm_xw(weights):
	#print('w0', weights[0])
	
	#print(weights[0].shape)
	#print('w1',weights[0][:,:int(weights[0].shape[1]/4)])
	ifog = [[w[:,:int(w.shape[1]/4)] for w in weights], [w[:,int(w.shape[1]/4):2*int(w.shape[1]/4)] for w in weights],
			 [w[:,2*int(w.shape[1]/4): 3*int(w.shape[1]/4)] for w in weights], [w[:,3*int(w.shape[1]/4):] for w in weights]]
	#print('dd', ifog[0][0])

	ret = np.hstack([np.hstack(w) for w in ifog])
	#print(ret.shape)
	return ret

#return np.hstack(weights)

def merge_controller(weights, latent=32):
	#print('ggg',len(weights[0]))
	latent_w = [w[:latent] for w in weights]
	rnn_w = [w[latent:] for w in weights]
	latent_w = np.hstack(latent_w)
	
	#print('rer', latent_w.shape)
	rnn_w = merge_diagonal(rnn_w)
	#print('rer2', rnn_w.shape)
	ret = np.vstack((latent_w, rnn_w))
	#ret = np.vstack((rnn_w, latent_w))
	#print(ret.shape)
	return ret


def merge_output(weights):
	ret = merge_diagonal(weights)
	return ret


def merge_lstm_biases(weights):
	#print(len(weights[0]))
	ifog = [[w[:int(len(w)/4)] for w in weights], [w[int(len(w)/4):2*int(len(w)/4)] for w in weights], 
			[w[2*int(len(w)/4): 3*int(len(w)/4)] for w in weights], [w[3*int(len(w)/4):] for w in weights]]
	ret = np.hstack([np.hstack(w) for w in ifog])

	#ret = np.zeros_like(ret)
	#ret = np.hstack(weights)
	return ret

def merge_controller_biases(weights):
	ret = np.hstack(weights)
	return ret


def merge_output_biases(weights):
	ret = np.hstack(weights)
	return ret


def merge_all(weights):
	new_weights = []
	for i in range(len(weights[0])):
		ws = [w[i] for w in weights]
		
		if i==0:
			new_weights.append(merge_lstm_xw(ws))
		elif i==1:
			new_weights.append(merge_lstm_hw(ws))
		elif i==2:
			new_weights.append(merge_lstm_biases(ws))
		elif i==3:
			new_weights.append(merge_controller(ws))
		elif i==4:
			new_weights.append(merge_controller_biases(ws))
		elif i==5:
			new_weights.append(merge_output(ws))
		elif i==6:
			new_weights.append(merge_output_biases(ws))
	return new_weights
