import time,os,copy,sys
import math,numpy, random
import os.path as pth
import dill
import tensorflow as tf
from . import Net
from collections import OrderedDict
from ..utility import *

import logging

logger = logging.getLogger(__name__)


def uNet(input,filterPlan,filterSize=(5,5),maxpool=False,layerThickness=1,dropout=None,normalization=None,activation=tf.nn.relu,dimensions=3):
	init = tf.contrib.layers.xavier_initializer()
	if dimensions == 2:
		convDown = tf.layers.conv2d; convUp = tf.layers.conv2d_transpose; maxpoolF = tf.layers.max_pooling2d
	else:
		convDown = tf.layers.conv3d; convUp = tf.layers.conv3d_transpose; maxpoolF = tf.layers.max_pooling3d
	
	downLayers = [];
	stride = 1 if maxpool else 2
	layerThickness = 1 if layerThickness < 1 else layerThickness
	net = input
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
						name = 'UNet-DownConv{}'.format(level+1))
		if normalization is not None:
			pass							# ROBB - add normalization sometime?
		if not dropout is None:
			net = tf.layers.dropout(net,rate=dropout,name='UNet-DownDropout{}'.format(level+1),data_format='channels_last')
		if maxpool:
			net = maxpoolF(	inputs=net,
							strides = 1,
							pool_size = 2,
							padding='same',
							data_format='channels_last',
							name='UNet-DownMaxpool{}'.format(level+1),)
		for extraIndex in range(layerThickness-1):
			net = convDown(	inputs=net,
							filters=nFilters,
							kernel_size=filterSize,
							padding='same',
							data_format='channels_last',
							activation = activation,
							kernel_initializer = init,
							name = 'UNet-DownExtraConv{}-{}'.format(level+1,extraIndex+1))
			if not normalization is None:
				pass
			if not dropout is None:
				net = tf.layers.dropout(net,rate=dropout,name='UNet-DownDropout{}'.format(level+1),data_format='channels_last')
		downLayers.append(net)	

	bottom = net
	
	logger.debug('U Up')
	for index in range(0,len(filterPlan)):
		nFilters = filterPlan[-1-index]
	
		net = convUp(	inputs=net,
						filters=nFilters,
						kernel_size=filterSize,
						strides = stride,
						padding='same',
						data_format='channels_last',
						activation = activation,
						kernel_initializer = init,
						name = 'UNet-UpConv{}'.format(index+1))
		if not index == len(filterPlan):
			net = tf.concat([net,downLayers[::-1][index+1]],axis=-1,name='UNet-UpConcat{}'.format(index+1))
		if not normalization is None:
			pass
		if not dropout is None:
			net = tf.layers.dropout(net,rate=dropout,name='UNet-UpDropout{}'.format(index+1),data_format='channels_last')
		if not index == len(filterPlan)-1:
			for extraIndex in range(layerThickness-1):
				net = convDown(	inputs=net,
								filters=nFilters,
								kernel_size=filterSize,
								padding='same',
								data_format='channels_last',
								activation = activation,
								kernel_initializer = init,
								name = 'UNet-UpExtraConv{}-{}'.format(index+1,extraIndex+1))
				if normalization is not None:
					pass
				if not dropout is None:
					net = tf.layers.dropout(net,rate=dropout,name='UNet-UpDropout{}'.format(index+1),data_format='channels_last')
			
	return net





class UNet2D(Net):

	def __init__(self,dimensions=(None,None,1),filterPlan=[10,20,30,40,50],filterSize=(5,5),layerThickness=1,postUDepth=2,maxpool=False,normalization=None,nonlinearity=tf.nn.relu,inputDropout=False,inputNoise=False,internalDropout=False,gentleCoding=0.9,name=None,**args):
		self.hyperParameters.update(dict(dropoutProbability=None,
										dimensions=[None] + list(dimensions),
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
										**args,
										))
		super().__init__()


	def createModel(self):
		"""Override this method to customize your model"""
		init = tf.contrib.layers.xavier_initializer()
		
		with tf.variable_scope(self.name):
			self.x = tf.placeholder('float',shape=self.dimensions,name='input')
		
			self.yp = tf.placeholder('float',shape=[None,None,None,None],name='truth')
			self.hyperParameters['attentionWeight'] = tf.placeholder('float',shape=[None,None,None,None],name='attention')

			if self.inputDropout:
				self.hyperParameters['inputDropoutProbability'] = tf.placeholder('float','inputDropout')
			else:
				self.hyperParameters['inputDropoutProbability'] = None
			if self.internalDropout:
				self.hyperParameters['internalDropoutProbability'] = tf.placeholder('float',name='internalDropout')
			else:
				self.hyperParameters['internalDropoutProbability'] = None
			if self.inputNoise:
				self.hyperParameters['inputNoiseSigma'] = tf.placeholder('float',name='inputNoiseSigma')
			else:
				self.hyperParameters['inputNoiseSigma'] = None

			net = inLayer = self.x
			if self.inputDropout:
				net = tf.layers.dropout(net,rate=self.inputDropoutProbability,name='InputDropout',data_format='channels_last')
			if self.inputNoise:
				net = net + tf.random_normal(shape=tf.shape(net), mean=0.0, stddev=self.inputNoiseSigma, dtype='float')

			net = uNet(net,filterPlan = self.filterPlan,filterSize = self.filterSize,maxpool=self.maxpool,layerThickness=self.layerThickness,normalization=self.normalization,dimensions=2)
			for i in range(self.postUDepth):
				net = tf.layers.conv2d(	inputs=net,
										filters=self.filterPlan[0],
										kernel_size=self.filterSize,
										padding='same',
										activation = self.nonlinearity,
										kernel_initializer = init,
										data_format='channels_last',
										name = 'PostU{}'.format(i+1))
			
				if self.normalization is not None:
					pass
				if self.internalDropout:
					net = tf.layers.dropout(net,rate=self.internalDropoutProbability,name='PostUDropout{}'.format(i+1),data_format='channels_last')
			self.requiredInputs = [self.x]
			if self.inputDropoutProbability is not None:
				self.requiredInputs += [self.inputDropoutProbability]
			if self.internalDropoutProbability is not None:
				self.requiredInputs += [self.internalDropoutProbability]
			if self.inputNoiseSigma is not None:
				self.requiredInputs.append(self.inputNoiseSigma)
			self.trainingInputs = [self.yp]
			self.net = net
			self.y = self.output = self.net


	@property
	def model(self):
		return self.net

	def preprocessInput(self,example):
		ret = dict(); ret.update(example)
		if example['input'].ndim == 4:     # this is single channel data
			x = numpy.expand_dims(example['input'],1)
		else:
			x = example['input']

		x = numpy.transpose(x,[0,2,1,3,4])		# move z axis to second position (batch,z,channel,y,x)
		x = numpy.vstack(x)							# stack z slices into batch dimension
		x = x.astype(numpy.float32);
		ret['input'] = x

		y = example['truth'] if example.get('truth',None) is not None else None
		if y is not None:
			y = numpy.transpose(y,[0,2,1,3,4])		# move z axis to second position (batch,z,channel,y,x)
			y = numpy.vstack(y)							# stack z slices into batch dimension
			y = y.astype(numpy.float32)
			ret['truth'] = y

		attention = example.get('attention',None)
		if attention is not None:
			attention = numpy.expand_dims(attention,1)
			attention = numpy.transpose(attention,[0,2,1,3,4])
			attention = numpy.vstack(attention)
			attention = attention.astype(numpy.float32)
			ret['attention'] = attention

		return ret


class UNet3D(UNet2D):

	def __init__(self,dimensions=(1,None,None,None),filterPlan=[15,20,30,40,50],filterSize=(5,5,5),layerThickness=None,postUDepth=2,maxpool=False,normalization=None,nonlinearity=tf.nn.relu,inputDropout=False,inputNoise=False,internalDropout=False,gentleCoding=0.9,name=None):
		super().__init__(dimensions,filterPlan,filterSize,layerThickness=layerThickness,postUDepth=postUDepth,maxpool=maxpool,normalization=normalization,nonlinearity=nonlinearity,inputDropout=inputDropout,inputNoise=inputNoise,internalDropout=internalDropout,gentleCoding=gentleCoding,name=name)


	def createModel(self,dimensions=(None,1,None,None,None),filterPlan=[10,20,30,40,50],filterSize=(5,5,5),layerThickness=None,postUDepth=2,maxpool=False,dropout=False,inputDropout=False,inputNoise=False,internalDropout=False,**extras):
		self.x = T.tensor5(name='input')
		self.yp = T.tensor5(name='truth')
		self.attentionWeight = T.tensor5(name='attention')
		if self.inputDropout:
			self.inputDropoutProbability = T.scalar('inputDropout')
		else:
			self.inputDropoutProbability = None;
		if self.internalDropout:
			self.internalDropoutProbability = T.scalar('internalDropout')
		else:
			 self.internalDropoutProbability = None
		if self.inputNoise:
			self.inputNoiseSigma = T.scalar('inputNoiseSigma',dtype=theano.config.floatX)
		else:
			self.inputNoiseSigma = None
		inLayer = net = lasagne.layers.InputLayer(shape=dimensions, input_var=self.x)
		if self.inputDropout:
			net = lasagne.layers.DropoutLayer(net,p=self.inputDropoutProbability,rescale=True)
		if self.inputNoise:
			net = lasagne.layers.GaussianNoiseLayer(net,sigma=self.inputNoiseSigma)

		net = uNet3D(net,filterPlan = filterPlan,filterSize = filterSize,maxpool=maxpool,layerThickness=layerThickness,normalization=self.normalization,nonlinearity=self.nonlinearity)

		for i in range(postUDepth):
			net = lasagne.layers.Conv3DLayer(net, filters=filterPlan[0], filter_size = filterSize, pad='same',nonlinearity=self.nonlinearity,kernel_initializer = init)
			net.name = 'postU%d' % i
			if self.normalization is not None:
				net = self.normalization(net)
				net.name = 'Normalization'
			if self.internalDropout:
				net = lasagne.layers.DropoutLayer(net,p=self.internalDropoutProbability,rescale=True)
		self.requiredInputs = [self.x]
		if self.inputDropout:
			self.requiredInputs += [self.inputDropoutProbability]
		if self.internalDropout:
			self.requiredInputs += [self.internalDropoutProbability]
		if self.inputNoise:
			self.requiredInputs.append(self.inputNoiseSigma)
		self.trainingInputs = [self.yp]
		self.net = net
		self.output = lasagne.layers.get_output(self.net)
		self.forwardFunction = theano.function(inputs=self.requiredInputs,outputs=self.output)


	def makeCost(self,withAttention=False,**args):
		#if self.costFragment is None:			# Need to fix this to detect changes in arguments.  Maybe dictionary of cost fragments?
		permuterX = PermuteLayer3D(self.net)
		permutedX = lasagne.layers.get_output(permuterX)
		permuterY = PermuteLayer3D(lasagne.layers.InputLayer(shape=self.dimensions,input_var=self.yp))
		permutedY = lasagne.layers.get_output(permuterY)
		safeOut = theano.tensor.clip(permutedX, 0.00001, 0.99999)

		if not withAttention:
			self.costFragment = ((self.y-self.x)**2).mean()
			self.costFunction = theano.function(self.requiredInputs + self.trainingInputs,outputs=self.costFragment)
		else:
			permuterAttention = PermuteLayer3D(lasagne.layers.InputLayer(shape=self.dimensions,input_var=self.attentionWeight))
			permutedAttention = lasagne.layers.get_output(permuterAttention)
			self.costFragment = ((self.y-self.x)**2*T.transpose(permutedAttention)).mean()
			self.costFunction = theano.function(self.requiredInputs + self.trainingInputs + [self.attentionWeight],outputs=self.costFragment)
		return self.costFragment


	def preprocessInput(self,example):
		ret = OrderedDict(); ret.update(example)
		if example['input'].ndim == 4:     # this is single channel data
			x = numpy.expand_dims(example['input'],1)
		else:
			x = example['input']

		x = x.astype(numpy.float32);
		ret['input']=x

		y = example['truth'] if example.get('truth',None) is not None else None
		if y is not None:
			y = y.astype(numpy.float32)
			ret['truth'] = y

		attention = example.get('attention',None)
		if attention is not None:
			attention = numpy.expand_dims(attention,1)
			attention = attention.astype(numpy.float32)
			ret['attention'] = attention
		return ret
