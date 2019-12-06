import numpy as np
import time
import tensorflow as tf



from model import Model 
import gc
from memory_profiler import profile


@profile
def test():
	tf.keras.backend.clear_session()
	print('hello')
	models = []
	models += [ 

		tf.keras.Sequential(
			[
			    tf.keras.layers.InputLayer(input_shape=(512)),
			    tf.keras.layers.Dense(units=1024, activation=tf.nn.relu),
			    tf.keras.layers.Dense(units=1024, activation=None)
			],
			name='controller'
			)

		for _ in range(100)
	]

	#tf.keras.backend.clear_session()

	new_m = []
	for model in models:
		model = None
		new_m.append(model)
	del models
	models = new_m



	gc.collect()
	models2 = [ 

		tf.keras.Sequential(
			[
			    tf.keras.layers.InputLayer(input_shape=(512)),
			    tf.keras.layers.Dense(units=1024, activation=tf.nn.relu),
			    tf.keras.layers.Dense(units=1024, activation=None)
			],
			name='controller'
			)

		for _ in range(100)
	]
	tf.keras.backend.clear_session()

	new_m2 = []

	for model in models2:
		model=None
		new_m2.append(model)

	
	del models2
	models2 = new_m2
	gc.collect()

	tf.keras.backend.clear_session()


if __name__=='__main__':
	test()
	test()
	test()
	test()