import time,os,copy,sys
import math,numpy, random
from collections import OrderedDict
import dill
import tensorflow as tf
from . import Net, ImageNet
from .. import Data
from ..utility import *

import logging

logger = logging.getLogger(__name__)

def upsample(x,factor,dataFormat='channels_last',name='repeat'):
	if dataFormat == 'channels_first':
		startingAxis = 2
	elif dataFormat == 'channels_last':
		startingAxis = 1
	else:
		raise ValueError('Invalid dataFormat:',dataFormat)
		return
	with tf.variable_scope(name):
		for axis in range(startingAxis,startingAxis+len(x.shape)-2):
			x = tf.keras.backend.repeat_elements(x,factor,axis=axis)
	return x



def uNet(input,filterPlan,filterSize=(5,5),maxpool=False,layerThickness=1,dropout=None,normalization=None,activation=tf.nn.relu,dimensions=3,skip=1.0):
	init = tf.contrib.layers.xavier_initializer()
	if dimensions == 2:
		convDown = tf.layers.conv2d; convUp = tf.layers.conv2d_transpose; maxpoolF = tf.layers.max_pooling2d
	else:
		convDown = tf.layers.conv3d; convUp = tf.layers.conv3d_transpose; maxpoolF = tf.layers.max_pooling3d

	downLayers = []; layers = OrderedDict()
	stride = 1 if maxpool else 2
	layerThickness = 1 if layerThickness < 1 else layerThickness
	net = input
	# Robb DEBUG: see if this helps eye fat segmentation
	# net = convDown(	inputs=net,
	# 				filters=filterPlan[0],
	# 				kernel_size=filterSize,
	# 				strides = 1,
	# 				padding='same',
	# 				activation = activation,
	# 				kernel_initializer = init,
	# 				data_format='channels_last',
	# 				name = 'UNet-Initial')

	downLayers.append(net)
	for level,nFilters in enumerate(filterPlan):
		net = convDown(	inputs=net,
						filters=nFilters,
						kernel_size=filterSize,
						strides = stride,
						padding='same',
						activation = activation,
						kernel_initializer = init,
						data_format='channels_last',
						name = 'UNet-DownConv{}'.format(level+1)); layers[net.name] = net
		if normalization is not None:
			pass							# ROBB - add normalization sometime?
		if not dropout is None:
			net = tf.layers.dropout(net,rate=dropout,name='UNet-DownDropout{}'.format(level+1)); layers[net.name] = net
		if maxpool:
			net = maxpoolF(	inputs=net,
							strides = 1,
							pool_size = 2,
							padding='same',
							data_format='channels_last',
							name='UNet-DownMaxpool{}'.format(level+1),); layers[net.name] = net
		for extraIndex in range(layerThickness-1):
			net = convDown(	inputs=net,
							filters=nFilters,
							kernel_size=filterSize,
							padding='same',
							data_format='channels_last',
							activation = activation,
							kernel_initializer = init,
							name = 'UNet-DownExtraConv{}-{}'.format(level+1,extraIndex+1)); layers[net.name] = net
			if not normalization is None:
				pass
			if not dropout is None:
				net = tf.layers.dropout(net,rate=dropout,name='UNet-DownDropout{}'.format(level+1)); layers[net.name] = net
		downLayers.append(net)

	bottom = net

	logger.debug('U Up')
	for index in range(0,len(filterPlan)):
		nFilters = filterPlan[-1-index]
		# Upsample
		net = upsample(net,stride); layers[net.name] = net
		# Concatenation with skip layers
		if not skip is None:
			skipChannels = downLayers[::-1][index+1]
			#print('net is: {}; Merging {} from layer {}'.format(net.shape,skipChannels.shape,len(downLayers)-(index+1)))
			if isinstance(skip,float):
				localSkip = int(math.ceil(skipChannels.shape[-1].value*skip))
			else:
				localSkip = skip
			if localSkip > 0:
				skipChannels = skipChannels[...,0:localSkip]
				net = tf.concat([net,skipChannels],axis=-1,name='UNet-UpConcat{}'.format(index+1)); layers[net.name] = net

		net = convDown(	inputs=net,
						filters=nFilters,
						kernel_size=filterSize,
						strides = 1,
						padding='same',
						data_format='channels_last',
						activation = activation,
						kernel_initializer = init,
						name = 'UNet-UpConv{}'.format(index+1)); layers[net.name] = net
		if not normalization is None:
			pass
		if not dropout is None:
			net = tf.layers.dropout(net,rate=dropout,name='UNet-UpDropout{}'.format(index+1)); layers[net.name] = net
		if not index == len(filterPlan)-1:
			for extraIndex in range(layerThickness-1):
				net = convDown(	inputs=net,
								filters=nFilters,
								kernel_size=filterSize,
								padding='same',
								data_format='channels_last',
								activation = activation,
								kernel_initializer = init,
								name = 'UNet-UpExtraConv{}-{}'.format(index+1,extraIndex+1)); layers[net.name] = net
				if normalization is not None:
					pass
				if not dropout is None:
					net = tf.layers.dropout(net,rate=dropout,name='UNet-UpDropout{}'.format(index+1)); layers[net.name] = net

	return net,layers





class UNet2D(Net):

	def __init__(self,dimensions=None,inputChannels=1,filterPlan=[10,20,30,40,50],filterSize=(5,5),layerThickness=1,postUDepth=2,maxpool=False,normalization=None,nonlinearity=tf.nn.relu,inputDropout=False,inputNoise=False,internalDropout=False,gentleCoding=0.9,standardize=None,name=None,fname=None,skipChannels=1.0,**args):
		dimensions = (None,None,inputChannels) if dimensions is None else dimensions
		self.hyperParameters.update(dict(dimensions=[None] + list(dimensions),
										inputChannels=inputChannels,
										filterPlan = filterPlan,
										filterSize = filterSize,
										layerThickness = layerThickness,
										postUDepth = postUDepth,
										maxpool = maxpool,
										normalization = normalization,
										nonlinearity = nonlinearity,
										inputDropout = inputDropout,
										internalDropout = internalDropout,
										inputNoise = inputNoise,
										gentleCoding = gentleCoding,
										skipChannels = skipChannels,
										**args
										))
		self.hyperParameters['standardize'] = Data.SpotStandardization() if standardize == True else standardize
		self.hyperParameters['preprocessor'] = \
							Data.ImagePreprocessor(	requiredDimensionOrder = ['b','y','x','c'],
												crop = args.get('crop',None),
												standardize = self.standardize,
												pad = len(self.filterPlan),
												mode = '2d',
												)
		super().__init__(fname=fname,name=name)


	def create(self):
		"""Override this method to customize your model"""
		init = tf.contrib.layers.xavier_initializer()

		with tf.variable_scope(self.name):
			self.x = tf.placeholder('float',shape=self.dimensions,name='input')

			self.yp = tf.placeholder('float',shape=[None,None,None,None],name='truth')
			self.modelParameters['attentionWeight'] = tf.placeholder('float',shape=[None,None,None,None],name='attention')

			if self.inputDropout:
				self.modelParameters['inputDropoutProbability'] = tf.placeholder_with_default(0.99,shape=(),name='inputDropout')
			else:
				self.modelParameters['inputDropoutProbability'] = None
			if self.internalDropout:
				self.modelParameters['internalDropoutProbability'] = tf.placeholder('float',name='internalDropout')
			else:
				self.modelParameters['internalDropoutProbability'] = None
			if self.inputNoise:
				self.modelParameters['inputNoiseSigma'] = tf.placeholder('float',name='inputNoise')
			else:
				self.modelParameters['inputNoiseSigma'] = None

			net = inLayer = self.x
			if self.inputDropout:
				net = self.addLayer(tf.layers.Dropout(rate=self.inputDropoutProbability,name='InputDropout')).apply(net,training=True);
			if self.inputNoise:
				net = self.addLayer(tf.reshape(net + tf.random_normal(shape=tf.shape(net), mean=0.0, stddev=self.inputNoiseSigma, dtype='float'),tf.shape(net),name='InputNoise'));

			net,ulayers = uNet(net,filterPlan = self.filterPlan,filterSize = self.filterSize,maxpool=self.maxpool,layerThickness=self.layerThickness,normalization=self.normalization,dimensions=2,skip=self.skipChannels);
			self.layers.update(ulayers)
			for i in range(self.postUDepth):
				net = self.addLayer(tf.layers.conv2d(	inputs=net,
										filters=self.filterPlan[0],
										kernel_size=self.filterSize,
										padding='same',
										activation = self.nonlinearity,
										kernel_initializer = init,
										data_format='channels_last',
										name = 'PostU{}'.format(i+1)));

				if self.normalization is not None:
					pass
				if self.internalDropout:
					net = self.addLayer(tf.layers.dropout(net,rate=self.internalDropoutProbability,name='PostUDropout{}'.format(i+1)))
			self.requiredInputs = [self.x]
			if self.inputDropoutProbability is not None:
				self.requiredInputs += [self.inputDropoutProbability]
			if self.internalDropoutProbability is not None:
				self.requiredInputs += [self.internalDropoutProbability]
			if self.inputNoiseSigma is not None:
				self.requiredInputs.append(self.inputNoiseSigma)
			self.net = net
			self.y = self.output = self.net


	def preprocessInput(self,example,dimensionOrder=None):
		dimensionOrder = example.get('dimensionOrder',None) if dimensionOrder is None else dimensionOrder
		ret = dict(); ret.update(example);
		ret['input'] = self.preprocessor.process(example['input'],dimensionOrder=dimensionOrder)
		if 'truth' in example:
			ret['truth'] = self.preprocessor.process(example['truth'],dimensionOrder=dimensionOrder)
		return ret






class UNet3D(Net):

	def __init__(self,dimensions=None,inputChannels=1,filterPlan=[10,20,30,40,50],filterSize=5,layerThickness=1,postUDepth=2,maxpool=False,normalization=None,nonlinearity=tf.nn.relu,inputDropout=False,inputNoise=False,internalDropout=False,gentleCoding=0.9,standardize=None,name=None,fname=None,skipChannels=1.0,**args):
		dimensions = (None,None,None,inputChannels) if dimensions is None else dimensions
		self.hyperParameters.update(dict(dropoutProbability=None,
										dimensions=[None] + list(dimensions),
										inputChannels = inputChannels,
										filterPlan = filterPlan,
										filterSize = filterSize,
										layerThickness = layerThickness,
										postUDepth = postUDepth,
										maxpool = maxpool,
										normalization = normalization,
										nonlinearity = nonlinearity,
										inputDropout = inputDropout,
										internalDropout = internalDropout,
										inputNoise = inputNoise,
										gentleCoding = gentleCoding,
										skipChannels = skipChannels,
										**args
										))
		self.hyperParameters['standardize'] = Data.SpotStandardization() if standardize == True else standardize
		self.hyperParameters['preprocessor'] = \
							Data.ImagePreprocessor(	requiredDimensionOrder = ['b','z','y','x','c'],
												crop = args.get('crop',None),
												standardize = self.standardize,
												pad = len(self.filterPlan),
												mode = '3d',
												)
		super().__init__(fname=fname,name=name)


	def create(self):
		"""Override this method to customize your model"""
		init = tf.contrib.layers.xavier_initializer()

		with tf.variable_scope(self.name):
			self.x = tf.placeholder('float',shape=self.dimensions,name='input')

			self.yp = tf.placeholder('float',shape=[None,None,None,None,None],name='truth')
			self.modelParameters['attentionWeight'] = tf.placeholder('float',shape=[None,None,None,None,None],name='attention')

			if self.inputDropout:
				self.modelParameters['inputDropoutProbability'] = tf.placeholder_with_default(0.99,shape=(),name='inputDropout')
			else:
				self.modelParameters['inputDropoutProbability'] = None
			if self.internalDropout:
				self.modelParameters['internalDropoutProbability'] = tf.placeholder('float',name='internalDropout')
			else:
				self.modelParameters['internalDropoutProbability'] = None
			if self.inputNoise:
				self.modelParameters['inputNoiseSigma'] = tf.placeholder('float',name='inputNoise')
			else:
				self.modelParameters['inputNoiseSigma'] = None

			net = inLayer = self.x
			if self.inputDropout:
				net = self.addLayer(tf.layers.Dropout(rate=self.inputDropoutProbability,name='InputDropout')).apply(net,training=True);
			if self.inputNoise:
				#logger.info('Using input noise')
				net = self.addLayer(tf.reshape(net + tf.random_normal(shape=tf.shape(net), mean=0.0, stddev=self.inputNoiseSigma, dtype='float'),tf.shape(net),name='inputNoise'))

			net,ulayers = uNet(net,filterPlan = self.filterPlan,filterSize = self.filterSize,maxpool=self.maxpool,layerThickness=self.layerThickness,normalization=self.normalization,dimensions=3,skip=self.skipChannels)
			self.layers.update(ulayers)
			for i in range(self.postUDepth):
				net = self.addLayer(tf.layers.conv3d(	inputs=net,
										filters=self.filterPlan[0],
										kernel_size=self.filterSize,
										padding='same',
										activation = self.nonlinearity,
										kernel_initializer = init,
										data_format='channels_last',
										name = 'PostU{}'.format(i+1)))

				if self.normalization is not None:
					pass
				if self.internalDropout:
					net = self.addLayer(tf.layers.dropout(net,rate=self.internalDropoutProbability,name='PostUDropout{}'.format(i+1)))
			self.requiredInputs = [self.x]
			if self.inputDropoutProbability is not None:
				self.requiredInputs += [self.inputDropoutProbability]
			if self.internalDropoutProbability is not None:
				self.requiredInputs += [self.internalDropoutProbability]
			if self.inputNoiseSigma is not None:
				self.requiredInputs.append(self.inputNoiseSigma)
			self.net = net
			self.y = self.output = self.net


	def preprocessInput(self,example,dimensionOrder=None):
		dimensionOrder = example.get('dimensionOrder',None) if dimensionOrder is None else dimensionOrder
		ret = dict(); ret.update(example);
		ret['input'] = self.preprocessor.process(example['input'],dimensionOrder=dimensionOrder)
		if 'truth' in example:
			ret['truth'] = self.preprocessor.process(example['truth'],dimensionOrder=dimensionOrder)
		return ret
