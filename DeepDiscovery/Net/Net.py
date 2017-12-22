import numpy
import dill
import time
import os.path
from .. import DeepRoot
from ..utility import onlyMyArguments
import sys
import tensorflow as tf

import logging

logger = logging.getLogger(__name__)

sys.setrecursionlimit(5000)

class Net(DeepRoot.DeepRoot):
	""" Abstract superclass representing an artificial neural network model. """

	def __new__(cls,*args,**kwargs):
		self = super().__new__(cls)
		self.__dict__['modelParameters'].update(dict(output = None, y = None, 
									yp = None, x = None, 
									net=None, cost=None, update = None))
		self.__dict__['lockedParameters'] = set()
		self.__dict__['requiredInputs'] = []
		self.__dict__['trainingInputs'] = []
		self.__dict__['excludeFromPickle'] = ['modelParameters']
		return self

	def __init__(self,name=None,**args):
		self.name = name if name is not None else self.__class__.__name__
		self.hyperParameters.update(**args)
		self.createModel()
		super().__init__()

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
	def saveCheckpoint(self,label=None,write_meta_graph=False):
		netName = self.name + ('-{}'.format(label) if not label is None else '')
		self.tfSaver.save(tf.get_default_session(),os.path.join(self.fname,netName),write_meta_graph=write_meta_graph)
		
	def loadCheckpoint(self,label=None,latest=True):
		netName = self.name + ('-{}'.format(label) if not label is None else '')
		if label is None and latest:
			fname = tf.train.latest_checkpoint(self.fname)
		else:
			netName = self.name + ('-{}'.format(label) if not label is None else '')
			fname = os.path.join(self.fname,netName)
		logger.info("Loading checkpoint {}".format(fname))
		self.tfSaver.restore(tf.get_default_session(),fname)

	def save(self,fname=None):
		""" Save a network to the directory fname at path.  If fname is not given, the network
			will be saved as name.net/"""
		netName = self.name + '.net'
		defaultFname = self.__dict__.get('fname',netName)
		fname = defaultFname if fname is None else os.path.join(fname,netName) if fname.endswith(os.path.sep) else fname
		self.__dict__['fname'] = fname
		self.__dict__['saveTime'] = time.time()
		os.makedirs(fname,mode=0o777,exist_ok=True)
		with open(os.path.join(fname,netName),'wb') as f:
			dill.dump(self,f,protocol=dill.HIGHEST_PROTOCOL)
		
		variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
		self.modelParameters['tfSaver'] = tf.train.Saver(var_list=variables)
		self.saveCheckpoint(write_meta_graph=True)
		return os.path.join(fname,netName)

	@classmethod
	def load(self,fname,path='',**args):
		fname = fname[:-1] if fname.endswith(os.path.sep) else fname
		netName = os.path.basename(fname)
		with open(os.path.join(fname,netName),'rb') as f:
			obj = dill.load(f)
		obj.__dict__['fname'] = fname
		obj.createModel()
		variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=obj.name)
		obj.modelParameters['tfSaver'] = tf.train.Saver(var_list=variables)
		obj.loadCheckpoint()
		return obj
