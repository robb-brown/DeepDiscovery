import numpy
import math
import copy
import nibabel as nib
from collections import OrderedDict
import dill
import sys
import time
import datetime
import os

from ..utility import *

import logging

logger = logging.getLogger(__name__)


class TrainingData(object):

	def __init__(self,examples,reserveForValidation=0.1,reserveForTest=0.1,balanceClasses=False):
		self.balanceClasses = balanceClasses
		self.examples = numpy.array(examples)
		self.reserveForValidation = reserveForValidation
		self.reserveForTest = reserveForTest
		if self.examples[0].__class__ == dict:
			# We have a list of dictionaries
			if not 'status' in examples[0]:
				#  we need to allocate
				self.trainSet,self.testSet,self.validationSet,self.classes = self.allocateExamples(self.examples,reserveForValidation=reserveForValidation,reserveForTest=reserveForTest,balanceClasses=balanceClasses)
				_ = [example.update({'status' : 'train' if i in self.trainSet else 'test' if i in self.testSet else 'validation'}) for i,example in enumerate(self.examples)]
			else:
				allocation = numpy.array([1 if ex['status'] == 'validation' else 2 if ex['status'] == 'test' else 0 for ex in self.examples])
				self.trainSet = numpy.where(allocation==0)[0]; self.validationSet = numpy.where(allocation==1)[0]; self.testSet = numpy.where(allocation==2)[0]
		else:
			# We've got tuples or lists, not dictionaries
			self.trainSet,self.testSet,self.validationSet,self.classes = self.allocateExamples(self.examples,reserveForValidation=reserveForValidation,reserveForTest=reserveForTest,balanceClasses=balanceClasses)
			self.examples = [dict(input=i,truth=t) for i,t in self.examples]

	@classmethod
	def allocateExamples(self,examples,reserveForValidation=0.1,reserveForTest=0.1,balanceClasses=False,classes=None):
		if classes is None:
			if issubclass(examples[0].__class__,dict):
				classes = [numpy.argmax(i['truth']) for i in examples]
			else:
				examples = numpy.array(examples)
				classes = [numpy.argmax(i) for i in examples[:,1]]
			classes = dict([(c,set(numpy.where(classes==c)[0])) for c in numpy.unique(classes)])
			classN = len(classes.keys())

		allocation = set(range(0,len(examples)))
		toSelect = reserveForTest if reserveForTest >= 1 else int(math.ceil(len(allocation)*reserveForTest))
		if balanceClasses:
			testSet = list(numpy.array([numpy.random.choice(list(set(allocation).intersection(set(classes[c]))),toSelect/classN,replace=False) for c in classes.keys()]).ravel())
			if len(testSet) < toSelect: testSet += list(numpy.random.choice(list(set(allocation) - set(testSet)),toSelect-len(testSet),replace=False))
		else:
			testSet = numpy.random.choice(list(allocation),toSelect,replace=False)
		allocation = allocation.difference(testSet)
		toSelect = reserveForValidation if reserveForValidation >= 1 else int(math.ceil(len(allocation)*reserveForValidation))
		if balanceClasses:
			validationSet = list(numpy.array([numpy.random.choice(list(set(allocation).intersection(set(classes[c]))),toSelect/classN,replace=False) for c in classes.keys()]).ravel())
			if len(validationSet) < toSelect: validationSet += list(numpy.random.choice(list(set(allocation) - set(validationSet)),toSelect-len(validationSet),replace=False))
		else:
			validationSet = numpy.random.choice(list(allocation),toSelect,replace=False)
		allocation = allocation.difference(validationSet)
		trainSet = allocation
		return trainSet,testSet,validationSet,classes


	def getExamples(self,dataset,N=1,balanceClasses=None):
		balanceClasses = balanceClasses if balanceClasses is not None else self.balanceClasses
		if balanceClasses:
			classN = len(self.classes.keys())
			examples = list(numpy.array([numpy.random.choice(list(set(dataset).intersection(set(self.classes[c]))),N/classN,replace=False) for c in self.classes.keys()]).ravel())
			if len(examples) < N: examples += list(numpy.random.choice(list(set(dataset) - set(examples)),N-len(examples),replace=False))
			examples = self.examples[examples]
		else:
			examples = self.examples[numpy.random.choice(list(dataset),N,replace=False)]
		return self.preprocessExamples(copy.deepcopy(examples))
	
	def getTrainingExamples(self,N=1,balanceClasses=None):
		return self.getExamples(self.trainSet,N=N,balanceClasses=balanceClasses)

	def getValidationExamples(self,N=1,balanceClasses=None):
		return self.getExamples(self.validationSet,N=N,balanceClasses=balanceClasses)

	def getTestingExamples(self,N=1,balanceClasses=None):
		return self.getExamples(self.testSet,N=N,balanceClasses=balanceClasses)


	def getAllTestingExamples(self):
		examples = self.examples[self.testSet]
		return self.preprocessExamples(copy.deepcopy(examples))


	def preprocessExamples(self,examples):
		return examples


	def save(self,fname=None):
		with open(fname,'wb') as f:
			dill.dump(self,f,protocol=dill.HIGHEST_PROTOCOL)
		return fname

	@classmethod
	def load(self,fname):
		with open(fname,'rb') as f:
			data = dill.load(f)
			data.fileName = fname
			return data




class ImageTrainingData(TrainingData):

	def __init__(self,examples,reserveForValidation=0.1,reserveForTest=0.1,depth=1,mode='2d',padShape=None,cropTo=None,truthComponents=None,gentleCoding=0.9,attention=None,balanceClasses=False,standardize=True,basepath='.',**attentionArguments):
		self.padShape = padShape
		super(ImageTrainingData,self).__init__(examples,reserveForValidation,reserveForTest)
		self.basepath = basepath
		self.depth = depth
		self.mode = mode
		self.attention = attention
		self.attentionArguments = attentionArguments
		self.cropTo = cropTo
		self.slice = [slice(ax[0],ax[1]) for ax in self.cropTo] if self.cropTo is not None else None
		self.standardize = standardize
		self.truthComponents = truthComponents
		self.gentleCoding = gentleCoding


	def preprocessExamples(self,examples):
		basepath = self.__dict__.get('basepath',os.path.dirname(self.__dict__.get('fileName','')))
		xs = []; ys = []; attentions = []
		for example in examples:
			t1 = time.time()
			if 'truthChannels' in example:
				y = numpy.array([nib.load(os.path.join(basepath,example[channel])).get_data() for channel in example['truthChannels']])
			else:
				truth = example.get(example['truth'],example['truth'])
				y = nib.load(os.path.join(basepath,truth)).get_data()
				y = numpy.expand_dims(y,-1)
			
			if 'inputChannels' in example:
				x = numpy.array([nib.load(os.path.join(basepath,example[channel])).get_data() for channel in example['inputChannels']])
			else:
				x = nib.load(os.path.join(basepath,example['input'])).get_data()
				x = numpy.expand_dims(x,-1)

			t2 = time.time()
			logger.debug("Loading example {} took {:0.2f} ms".format(os.path.abspath(example['input']),(t2-t1)*1000))

			if self.standardize:
				x = x - numpy.average(x) / numpy.std(x)

			# Cropping
			if self.slice is not None:
				slc = [slice(0,x.shape[i]) for i in range(len(x.shape)-len(self.slice))] + self.slice
				x = x[slc];

				if y is not None:
					slc = [slice(None) for i in range(len(y.shape)-len(self.slice))] + self.slice
					y = y[slc]

			# ROBB - move padding into network preprocessing
			x = padImage(x,depth=self.depth,mode=self.mode,shape=self.padShape)
			y = padImage(y,depth=self.depth,mode=self.mode,shape=self.padShape) if y is not None else y

			if not self.attention is None and y is not None:
				attention = example.get('attention',None)
				if attention is None:
					attention = self.attention(y,**self.attentionArguments) if callable(self.attention) else self.attention
				attention = padImage(attention,depth=self.depth,mode=self.mode)
				attentions.append(attention)

			xs.append(x); ys.append(y)

		x = numpy.array(xs); y = numpy.array(ys)

		# Convert y to one hot
		if not self.truthComponents is None:
			y.shape = y.shape[0:-1]
			y = convertToOneHot(y,coding=self.truthComponents,gentleCoding=self.gentleCoding)
		
		# Everything should be multichannel, channel last at this point ROBB fix attention
		
		example = OrderedDict(input=x,truth=y)
		
		if not self.attention is None:
			example['attention'] = numpy.array(attentions)
		
		# this is the responsibility of the network
		# if self.mode == '2d':
		# 	example['input'] = example['input'].reshape([-1]+list(numpy.array(example['input'].shape)[2:]))
		# 	example['truth'] = example['truth'].reshape([-1]+list(numpy.array(example['truth'].shape)[2:]))
		# 	if 'attention' in example:
		# 		example['attention'] = numpy.transpose(example['attention'],[0,2,1,3,4]).reshape([-1]+list(numpy.array(example['attention'].shape)[[1,3,4]]))

		return example


class ThresholdingImageTrainingData(ImageTrainingData):

	def preprocessExamples(self,examples):
		examples = super(ThresholdingImageTrainingData,self).preprocessExamples(examples)
		examples['truth'] = numpy.where(examples['truth']>0.5,1,0)
		return examples


class PandasData(TrainingData):

	def __init__(self,data,truth,predictors=None,categoricalOutput=True,reserveForValidation=0.1,reserveForTest=0.1,balanceClasses=False,gentleCoding=False,extraArgs={}):
		self.data = data
		if categoricalOutput:
			self.data[truth] = self.data[truth].astype('category')
			self.truthCoding = list(self.data[truth].cat.categories)
			truthVector = convertToOneHot(self.data[truth],gentleCoding=gentleCoding)
		else:
			self.truthCoding = None
			truthVector = self.data[truth]

		self.extraArgs = extraArgs
		self.predictors = [c for c in self.data.columns if not c == truth] if predictors is None else predictors
		predictorData = numpy.array(self.data[self.predictors])
		self.examples = numpy.array([(predictorData[i],truthVector[i]) for i in range(len(predictorData))])
		self.reserveForValidation = reserveForValidation
		self.reserveForTest = reserveForTest
		self.balanceClasses = balanceClasses
		self.allocateExamples()

	def preprocessExamples(self,examples):
		examples = OrderedDict(input=numpy.array(list(examples[:,0])).astype(theano.config.floatX),truth=numpy.array(list(examples[:,1])).astype(theano.config.floatX))
		examples.update(self.extraArgs)
		return examples
