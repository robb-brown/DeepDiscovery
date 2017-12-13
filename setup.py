from setuptools import setup

setup(name='DeepDiscovery',
	version='0.1',
	description='Medical data analysis package using deep learning.',
	url='http://github.com/robb-brown/DeepDiscovery',
	author='Robert A. Brown',
	author_email='robert.brown@mcgill.ca',
	license='MIT',
	packages=['DeepDiscovery'],
	install_requires=[
		'tensorflow-gpu',
		'matplotlib',
		'nibabel',
		'numpy',
		'pandas',
		'six',
		'tinydb',
		'tinymongo',
		'dill',
	],
	zip_safe=False)

