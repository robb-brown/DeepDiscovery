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
