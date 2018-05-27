from .UNet import UNet2D
import numpy, random, math, time, dill, os, sys, copy
import tensorflow as tf
from collections import OrderedDict

from .. import utility



class Segmenter2D(UNet2D):

	def __init__(self,dimensions=None,inputChannels=1,filterPlan=[10,20,30,40,50],filterSize=(5,5),layerThickness=1,postUDepth=2,outputValues=2,maxpool=False,normalization=None,nonlinearity=tf.nn.relu,inputDropout=False,inputNoise=False,internalDropout=False,gentleCoding=0.9,standardize=True,skipChannels=1.0,name=None,fname=None,**args):
		"""Segmenter expects images and dimensions in (batch,z,y,x,channels) order.
			dimensions is optional, but should be [y, x, channels] if set
		"""
		self.hyperParameters['outputValues'] = outputValues
		super().__init__(dimensions=dimensions,inputChannels=inputChannels,filterPlan=filterPlan,filterSize=filterSize,layerThickness=layerThickness,postUDepth=postUDepth,outputValues=outputValues,maxpool=maxpool,normalization=normalization,nonlinearity=nonlinearity,inputDropout=inputDropout,inputNoise=inputNoise,internalDropout=internalDropout,gentleCoding=gentleCoding,skipChannels=skipChannels,standardize=standardize,name=name,fname=fname,**args)


	def create(self):
		super().create()
		with tf.variable_scope(self.name):
			self.modelParameters['logits'] = None
			init = tf.contrib.layers.xavier_initializer()
			self.net = self.addLayer(tf.layers.conv2d(self.net, filters=self.outputValues, kernel_size = 1, padding='same', activation=None,kernel_initializer = None,data_format='channels_last'))
			self.logits = self.net
			self.y = self.output = self.net = self.addLayer(tf.nn.softmax(self.logits,-1,name='softmax'))


	def forwardPass(self,examples,**args):
		feed,missing = utility.buildFeed(self.requiredInputs,examples,**args)
		if len(missing) > 0:
			logger.error('Missing input values: {}'.format(missing))
		ret = tf.get_default_session().run(self.getOutput(),feed_dict=feed)
		return ret

	def segment(self,image,dimensionOrder=['z','y','x']):
		originalShape = dict([(dimensionOrder[i],image.shape[i]) for i in range(len(image.shape))])
		example = dict(input=image)
		example = self.preprocessInput(example,dimensionOrder=dimensionOrder)
		segmentation = self.forwardPass(example,inputDropout=0.,internalDropout=0.,inputNoise=0.)
		requiredDimensionOrder = dimensionOrder + ['c'] if not 'c' in dimensionOrder else dimensionOrder
		segmentation = self.preprocessor.restore(segmentation,originalShape=originalShape,dimensionOrder=requiredDimensionOrder,requiredDimensionOrder=requiredDimensionOrder)
		return segmentation

	def preprocessInput(self,example,dimensionOrder=None):
		dimensionOrder = example.get('dimensionOrder',None) if dimensionOrder is None else dimensionOrder
		ret = dict(); ret.update(example);
		ret['input'] = self.preprocessor.process(example['input'],dimensionOrder=dimensionOrder)
		if 'truth' in example:
			ret['truth'] = self.preprocessor.process(example['truth'],dimensionOrder=dimensionOrder,oneHot=True)
		if 'attention' in example:
			ret['attention'] = self.preprocessor.process(example['attention'],standardize=False)
		return ret
