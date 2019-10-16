try:		# Tensorflow 2 stuff
	import tensorflow as tf2
	import tensorflow.compat.v1 as tf
	tf.disable_eager_execution()
	tf.set_random_seed = lambda seed: tf2.random.set_seed(seed)
except:
	import tensorflow as tf