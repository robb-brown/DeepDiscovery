from setuptools import setup,find_packages

setup(name='DeepDiscovery',
	version='0.1',
	description='Medical data analysis package using deep learning.',
	url='http://github.com/robb-brown/DeepDiscovery',
	author='Robert A. Brown',
	author_email='robert.brown@mcgill.ca',
	license='MIT',
	packages=find_packages(),
	install_requires=[
		#'tensorflow-gpu',
		'matplotlib',
		'nibabel',
		'numpy',
		'pandas',
		'six',
		'tinydb',
		'tinymongo',
		'dill',
		'colormath',
		'scipy',
	],
	zip_safe=False)


# patch Tensorflow
try:
	import tensorflow as tf
	import os,shutil

	print('Patching TensorFlow convolutional layers...')
	path = os.path.dirname(tf.layers.__file__)
	src = os.path.join(os.path.dirname(__file__),'patches','convolutional.py')
	dst = os.path.join(path,'convolutional.py')
	shutil.copyfile(src, dst)
	print('Path complete')
except:
	pass