try:		# Tensorflow 2 stuff
	import tensorflow.compat.v1 as tf
	import tensorflow.compat.v2 as tf2
	tf.disable_eager_execution()
	if tf.__version__.startswith('2.'):
		tf.set_random_seed = lambda seed: tf2.random.set_seed(seed)
except:
	import tensorflow as tf