from .UNet import UNet2D
import numpy, random, math, time, dill, os, sys, copy
import tensorflow as tf
from collections import OrderedDict

from ..utility import *



class Segmenter2D(UNet2D):

	def __init__(self,dimensions=(None,None,1),filterPlan=[10,20,30,40,50],filterSize=(5,5),layerThickness=1,postUDepth=2,outputValues=2,maxpool=False,normalization=None,nonlinearity=tf.nn.relu,inputDropout=False,inputNoise=False,internalDropout=False,gentleCoding=0.9,name=None,**args):
		"""Segmenter expects images and dimensions in (batch,channels,y,x) order.
			dimensions is optional, but should be [channels, y, x] if set
		"""
		self.hyperParameters['outputValues'] = outputValues
		self.modelParameters['logits'] = None
		super().__init__(dimensions=dimensions,filterPlan=filterPlan,filterSize=filterSize,layerThickness=layerThickness,postUDepth=postUDepth,outputValues=outputValues,maxpool=maxpool,normalization=normalization,nonlinearity=nonlinearity,inputDropout=inputDropout,inputNoise=inputNoise,internalDropout=internalDropout,gentleCoding=gentleCoding,name=name,**args)


	def createModel(self,dimensions=(None,None,None,1),outputValues=2,filterPlan=[10,20,30,40,50],filterSize=(5,5),layerThickness=None,postUDepth=2,postUFilterSize=None,maxpool=False,inputDropout=False,inputNoise=False,internalDropout=False,**extras):
		super().createModel(dimensions=dimensions,outputValues=outputValues,filterPlan=filterPlan,filterSize=filterSize,layerThickness=layerThickness,postUDepth=postUDepth,postUFilterSize=postUFilterSize,maxpool=maxpool,inputDropout=inputDropout,inputNoise=inputNoise,internalDropout=internalDropout,**extras)
		self.net = tf.layers.conv2d(self.net, filters=outputValues, kernel_size = 5, padding='same', activation=tf.nn.relu,data_format='channels_last')
		self.logits = self.net
		self.y = self.output = self.net = tf.nn.softmax(self.logits,-1)
		
		self.modelParameters['cost'] = tf.losses.softmax_cross_entropy(onehot_labels=self.yp, logits=self.logits)
	

	def segment(self,image):
		# output = lasagne.layers.get_output(self.net)
		# return output.eval({self.x:image})
		example = dict()
		example['input'] = padImage(image,depth=len(self.filterPlan),mode='2d')
		example['input'] = numpy.expand_dims(example['input'],1)
		example = self.preprocessInput(example)
		segmentation = self.forwardFunction(example['input'])
		segmentation = padImage(segmentation,depth=1,mode='2d',shape=image.shape)
		return segmentation.transpose(1,0,2,3)


	def preprocessInput(self,example):
		ret = dict(); ret.update(example)
		if example['input'].ndim == 4:     # this is single channel data
			x = numpy.expand_dims(example['input'],-1)
		else:
			x = example['input']
		
		# Squish z into the batch dimension
		x = x.reshape([-1]+list(numpy.array(x.shape)[2:]))
		ret['input'] = x
		if not example.get('truth',None) is None:
			ret['truth'] = example['truth'].reshape([-1]+list(numpy.array(example['truth'].shape)[2:]))
		if not example.get('attention',None) is None:
			ret['attention'] = example['attention'].reshape([-1]+list(numpy.array(example['attention'].shape)[2:]))

		return ret
