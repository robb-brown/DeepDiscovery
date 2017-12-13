import numpy
import dill
import time
import os.path
from .. import DeepRoot
from ..utility import onlyMyArguments
import sys

sys.setrecursionlimit(5000)

class Net(DeepRoot.DeepRoot):
	""" Abstract superclass representing an artificial neural network model. """

	# class super1 (object):
	#     def __new__(typ, *args, **kwargs):
	#         obj = object.__new__(typ, *args, **kwargs)
	#         obj.attr1 = []
	#         return obj

	def __new__(cls,*args,**kwargs):
		self = super().__new__(cls)
		self.__dict__['modelParameters'] = dict(output = None, y = None, 
									yp = None, x = None, 
									net=None, cost=None, update = None)
		self.__dict__['hyperParameters'] = dict()
		self.__dict__['lockedParameters'] = set()
		self.__dict__['requiredInputs'] = []
		self.__dict__['trainingInputs'] = []
		self.__dict__['name'] = None
		#self.excludeFromPickle = []		
		return self

	def __init__(self,name=None,**args):
		self.name = name if name is not None else 'Net'
		self.hyperParameters.update(**args)
		self.createModel(**self.hyperParameters)
		super().__init__()

	def __getattr__(self,attr):
		if attr in self.__dict__['modelParameters']:
			return self.__dict__['modelParameters'].get(attr)
		elif attr in self.__dict__['hyperParameters']:
			return self.__dict__['hyperParameters'].get(attr)
		else:
			raise AttributeError(attr)

	def __setattr__(self,attr,value):
		if attr in self.__dict__:
			self.__dict__[attr] = value
		elif attr in self.modelParameters:
			self.modelParameters[attr] = value
		elif attr in self.hyperParameters:
			self.hyperParameters[attr] = value
		else:
			raise AttributeError(attr)

	@property
	def parameters(self):
		raise NotImplementedError
		
	@property
	def trainableParameters(self):
		raise NotImplementedError

	def createModel(self):
		"""Override this method to customize your model"""
		raise NotImplementedError

	def preprocessInput(self,example):
		return example

	def getOutput(self,deterministic=False,**args):
		return self.output

	def forwardPass(self,examples,**args):
		if not isinstance(examples,dict):
			examples = self.preprocessInput(dict(input=examples))
		else:
			examples = self.preprocessInput(examples)
		feed = {self.x:examples['input']}; feed.update(**args)
		return self.y.eval(feed)


	def evaluateCost(self,examples,**args):
		examples = self.preprocessInput(examples)
		feed = {self.x:examples['input'],self.yp:examples['truth']}; feed.update(**args)
		return self.cost.eval(feed)


	def getLayers(self):
		return lasagne.layers.get_all_layers(self.net)

	def lockLayers(self,layers):
		for layer in layers:
			[self.lockedParameters.add(param[0]) for param in layer.params.items() if 'trainable' in param[1]]

	def unlockLayers(self,layers):
		for layer in layers:
			[self.lockedParameters.remove(param[0]) for param in layer.params.items() if 'trainable' in param[1]]


	# Pickling support
	def __getstate__(self):
		pickleDict = dict()
		for k,v in self.__dict__.items():
			if not issubclass(v.__class__,theano.compile.function_module.Function) \
				and not k == 'net' \
				and not k in self.excludeFromPickle:
					pickleDict[k] = v
		netParameters = numpy.array(lasagne.layers.get_all_param_values(self.net))
		return (pickleDict,netParameters)

	def __setstate__(self,params):
		self.__dict__.update(params[0])
		# Upgrade old objects
		try:
			if not hasattr(self,'inputDropout'):
				self.inputDropout = self.dropout
				self.internalDropout = False
		except:
			pass
		self.createModel(**onlyMyArguments(self.createModel,self.__dict__))
		lasagne.layers.set_all_param_values(self.net,params[1])


	def save(self,path='.',fname=None,fast=False):
		""" Save a network to the file fname at path.  If fname is not given, the network
			will be saved as name.net"""
		self.saveTime = time.time()
		fname = fname if fname is not None else self.name + '.net'
		with open(os.path.join(path,fname),'wb') as f:
			dill.dump(self,f,protocol=dill.HIGHEST_PROTOCOL)
		return os.path.join(path,fname)

	@classmethod
	def load(self,fname):
		with open(fname,'rb') as f:
			return dill.load(f)
