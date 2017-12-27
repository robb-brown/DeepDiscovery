import numpy,scipy
import math, copy, sys, time, datetime, os
import nibabel as nib
from collections import OrderedDict
import dill

from .. import utility

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
			data.fname = fname
			return data




class ImageTrainingData(TrainingData):

	def __init__(self,examples,reserveForValidation=0.1,reserveForTest=0.1,truthComponents=None,gentleCoding=0.9,balanceClasses=False,attention=False,basepath='.'):
		super(ImageTrainingData,self).__init__(examples,reserveForValidation,reserveForTest)
		self.basepath = basepath
		self.truthComponents = truthComponents
		self.gentleCoding = gentleCoding
		self.attention = attention


	def preprocessExamples(self,examples):
		basepath = self.__dict__.get('basepath',os.path.dirname(self.__dict__.get('fname','')))
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

			xs.append(x); ys.append(y)

		x = numpy.array(xs); y = numpy.array(ys)

		# Convert y to one hot
		if not self.truthComponents is None:
			y.shape = y.shape[0:-1]
			y = utility.convertToOneHot(y,coding=self.truthComponents,gentleCoding=self.gentleCoding)
		
		example = dict(input=x,truth=y,dimensionOrder=['b','z','y','x','c'])
		
		if not self.__dict__.get('attention',None) is None:
			example['attention'] = self.attention.generate(example)

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
		examples = dict(input=numpy.array(list(examples[:,0])).astype(theano.config.floatX),truth=numpy.array(list(examples[:,1])).astype(theano.config.floatX))
		examples.update(self.extraArgs)
		return examples




class ImagePreprocessor(object):

	def __init__(self,dimensionOrder=['b','z','y','x','c'],requiredDimensionOrder=None,crop=None,standardize=None,pad=None,mode='3d'):
		"""
			crop is a dictionary with keys that are single character dimension codes and values are 
				arguments to slice() (start, end, [step]).

			standardize is a function that takes the data and dimensionOrder as arguments and 
				returns a standardized version.

			pad is False or a depth for standard CNN hierarchy.  

		"""
		self.dimensionOrder = dimensionOrder; self.requiredDimensionOrder = requiredDimensionOrder; 
		self.crop = crop; self.standardize=standardize; self.pad = pad; self.mode = mode


	def process(self,x,dimensionOrder=None,requiredDimensionOrder=None,standardize=None,oneHot=False):
		requiredDimensionOrder = requiredDimensionOrder if not requiredDimensionOrder is None else self.requiredDimensionOrder
		dimensionOrder = dimensionOrder if not dimensionOrder is None else self.dimensionOrder
		standardize = False if oneHot else self.standardize if standardize is None else standardize
		

		# Cropping
		if not (self.crop is None or self.crop == False):
			slices = [slice(*self.crop.get(dimension,[None])) for dimension in dimensionOrder]
			x = x[slices]

		# standardization
		if not (standardize is None or standardize == False):
			if hasattr(self.standardize,'standardize'):
				x = self.standardize.standardize(x,dimensionOrder)
			else:
				x = self.standardize(x,dimensionOrder)

		# padding
		if not (self.pad is None or self.pad == False):
			spatialAxes = [dimensionOrder.index(d) for d in (['z','y','x'] if self.mode == '2d' else ['z','y','x'])]
			if isinstance(self.pad,int):
				x = utility.padImage(x,depth=self.pad,mode=self.mode,spatialAxes=spatialAxes,oneHot=oneHot)
			else:
				x = utility.padImage(x,mode=self.mode,shape=self.pad,spatialAxes=spatialAxes,oneHot=oneHot)

		if (not requiredDimensionOrder is None) and (not dimensionOrder == requiredDimensionOrder):
			# Check for missing dimensions
			for d,dim in enumerate(requiredDimensionOrder):
				if not dim in dimensionOrder:
					x = numpy.expand_dims(x,-1)
					dimensionOrder.append(dim)
			
			# roll any extra dimensions into the batch dimension
			extraDimensions = [dim for dim in dimensionOrder if not dim in requiredDimensionOrder]
			if len(extraDimensions) > 0:
				tempDimensions = ['b'] + extraDimensions + [dim for dim in requiredDimensionOrder if not dim == 'b']
				twiddler = [dimensionOrder.index(dim) for dim in tempDimensions]
				x = x.transpose(twiddler)
				x.shape = [-1] + [dim for d,dim in enumerate(x.shape) if tempDimensions[d] in requiredDimensionOrder and not tempDimensions[d] == 'b']
				dimensionOrder = [dim for dim in dimensionOrder if not dim in extraDimensions]
			
			twiddler = [dimensionOrder.index(dim) for dim in requiredDimensionOrder]
			x = x.transpose(twiddler)

		return x


	def restore(self,x,originalShape=None,dimensionOrder=None,requiredDimensionOrder=None):
		requiredDimensionOrder = requiredDimensionOrder if not requiredDimensionOrder is None else self.dimensionOrder
		dimensionOrder = dimensionOrder if not dimensionOrder is None else self.requiredDimensionOrder if not self.requiredDimensionOrder is None else requiredDimensionOrder

		# padding
		if not (self.pad is None or self.pad == False):
			if self.crop:
				oShape = list(originalShape)
				for d in self.crop.keys():
					oShape[dimensionOrder.index(d)] = self.crop[d][1]-self.crop[d][0]
			else:
				oShape = originalShape

			spatialAxes = [dimensionOrder.index(d) for d in (['y','x'] if self.mode == '2d' else ['z','y','x'])]
			if isinstance(self.pad,int):
				x = utility.depadImage(x,oShape,spatialAxes=spatialAxes)
			else:
				x = utility.depadImage(x,oShape,spatialAxes=spatialAxes)

		# Cropping
		if not (self.crop is None or self.crop == False):
			slices = [slice(*self.crop.get(dimension,[None])) for dimension in dimensionOrder]
			x2 = numpy.zeros(originalShape,x.dtype)
			x2[slices] = x
			x = x2

		# robb - need to fix this so that it unrolls dimensions out of batch instead of inserting empty ones
		if (not requiredDimensionOrder is None) and (not dimensionOrder == requiredDimensionOrder):
			# Check for missing dimensions
			for d,dim in enumerate(requiredDimensionOrder):
				if not dim in dimensionOrder:
					x = numpy.expand_dims(x,-1)
					dimensionOrder.append(dim)

			# roll any extra dimensions into the batch dimension
			extraDimensions = [dim for dim in dimensionOrder if not dim in requiredDimensionOrder]
			if len(extraDimensions) > 0:
				tempDimensions = ['b'] + extraDimensions + [dim for dim in requiredDimensionOrder if not dim == 'b']
				twiddler = [dimensionOrder.index(dim) for dim in tempDimensions]
				x = x.transpose(twiddler)
				x.shape = [-1] + [dim for d,dim in enumerate(x.shape) if tempDimensions[d] in requiredDimensionOrder and not tempDimensions[d] == 'b']
				dimensionOrder = [dim for dim in dimensionOrder if not dim in extraDimensions]

			twiddler = [dimensionOrder.index(dim) for dim in requiredDimensionOrder]
			x = x.transpose(twiddler)

		return x


class SpotStandardization(object):

	def __init__(self,axes=['z','y','x'],centre=0.5,method='median'):
		self.axes = axes; self.centre = 0.5; self.method = method

	def standardize(self,x,dimensionOrder):
		slices = [slice(int(round(x.shape[d]*self.centre/2)),int(round(x.shape[d]*self.centre/2*3))) if dim in self.axes else slice(None) for d,dim in enumerate(dimensionOrder)]
		centre = x[slices]; 
		standardizerShape = [1 if dimensionOrder[d] in self.axes else dim for d,dim in enumerate(x.shape)]
		overAxes = tuple([dimensionOrder.index(d) for d in self.axes])

		if self.method == 'mean':
			average = numpy.reshape(numpy.average(centre,axis=overAxes),standardizerShape)
			std = numpy.reshape(numpy.std(centre,axis=overAxes),standardizerShape)
		else:
			average = numpy.reshape(numpy.median(centre,axis=overAxes),standardizerShape)
			std = numpy.reshape(numpy.percentile(centre, 75,axis=overAxes) - numpy.percentile(centre, 25,axis=overAxes),standardizerShape)			
		x = (x - average) / std
		return x




class EdgeBiasedAttention(object):
	
	def __init__(self,edgeFalloff=10,background=0.01,approximate=True,balanceClasses=True,gentleBalance=10.0):
		self.edgeFalloff = edgeFalloff; self.background = background; 
		self.approximate = approximate
		self.balanceClasses = balanceClasses; self.gentleBalance = gentleBalance
	
	def generate(self,example):
		y = numpy.around(example['truth'][...,0]).astype(numpy.uint8)
		if self.approximate:
			dist1 = scipy.ndimage.distance_transform_cdt(y[0])
			dist2 = scipy.ndimage.distance_transform_cdt(numpy.where(y>0,0,1))
		else:
			dist1 = scipy.ndimage.distance_transform_edt(y, sampling=[1,1,1])
			dist2 = scipy.ndimage.distance_transform_edt(numpy.where(y>0,0,1), sampling=[1,1,1])
		dist = dist1+dist2
		attention = math.e**(1-dist/float(self.edgeFalloff)) + self.background
		if self.balanceClasses:
			classBalance = numpy.sum(attention*(1.0-y)) / numpy.sum(attention*y)
			if classBalance > 1 and classBalance > self.gentleBalance:
				classBalance /= self.gentleBalance
			elif classBalance < 1 and classBalance < 1./self.gentleBalance:
				classBalance *= self.gentleBalance
			attention = attention * (1+(y*classBalance))
		attention /= numpy.average(attention)
		#print "achieved class balance is %0.2f" % (numpy.sum(attention*(1.0-y)) / numpy.sum(attention*y))
		attention = numpy.reshape(attention,y.shape)
		attention = numpy.expand_dims(attention,-1)
		return attention
		