import tensorflow as tf
import time, datetime, os, dill, numpy, sys
from collections import OrderedDict
from .. import DeepRoot, Net, utility
import logging
logger = logging.getLogger(__name__)

class CostFunction(DeepRoot.DeepRoot):
	
	def __new__(cls,*args,**kwargs):
		self = super().__new__(cls)
		self.__dict__['modelParameters'].update(dict(requiredInputs=[]))
		return self
	
	def create(self):
		raise NotImplementedError


class CrossEntropyCost(CostFunction):
	
	def __init__(self,y,yp,attention=False):
		self.modelParameters.update(dict(
			y = y,
			yp = yp
		))
		self.__dict__['attention'] = attention
	
	def create(self):
		cost = tf.keras.losses.categorical_crossentropy(self.yp,self.y)
		if self.attention:
			self.modelParameters['attentionOp'] = tf.placeholder('float',shape=self.yp.get_shape(),name='attention')
			cost *= self.attentionOp[...,0]
			self.requiredInputs.append(self.attentionOp)
		cost = tf.reduce_mean(cost)
		self.requiredInputs += [self.yp]
		return cost



class Trainer(DeepRoot.DeepRoot):
	
	def __new__(cls,*args,**kwargs):
		self = super().__new__(cls)
		self.__dict__['modelParameters'].update(dict(session = None,
													net = None,
													costOp = None,
													updates = None,
													metrics=None))
		return self
	

	def __init__(self,net,cost,examples,progressTracker=None,name=None,metrics=dict(cost=None),**trainingArguments):
		self.__dict__['examples'] = examples
		self.__dict__['progressTracker'] = progressTracker
		self.modelParameters.update(dict(
			net = net,
			updates = None,
			metrics = None,
		))
		self.hyperParameters.update(dict(
			trainingArguments = trainingArguments,
			elapsed = 0.0,
			previouslyElapsed = 0.0,
			epoch = 0,
			startTime = None,
			params = None,
			netFile = None,
			trackerFile = None,
			cost = cost,
			metricNames = metrics,
			netClass = net.__class__,
			trackerClass = progressTracker.__class__,
		))
		self.name = name if not name is None else 'Training-' + self.net.name
		self.initializeTraining()
		self.setupMetrics()


	def initializeTraining(self,):
		with tf.variable_scope(self.name):
			self.modelParameters['costOp'] = self.cost.create()
			self.modelParameters['trainingStep'] = tf.train.AdamOptimizer(**self.trainingArguments).minimize(self.costOp)


	def setupMetrics(self):
		with tf.variable_scope(self.name):
			if not isinstance(self.metricNames,dict):
				metrics = dict([(metric,None) for metric in self.metricNames])
			else:
				metrics = dict(self.metricNames)
			net = self.net
			for metric in metrics:
				if metrics[metric] is None:
					if metric == 'cost':
						metrics[metric] = self.costOp
					elif metric == 'output':
						metrics[metric] = [net.x,net.y,net.yp]
					elif metric == 'accuracy':
						correctPrediction = tf.equal(tf.argmax(net.y,axis=-1),tf.argmax(net.yp,axis=-1))
						accuracy = tf.reduce_mean(tf.cast(correctPrediction,'float'))
						metrics[metric] = accuracy
					elif metric == 'jaccard':
						output = tf.cast(tf.argmax(net.y,axis=-1), dtype=tf.float32)
						truth = tf.cast(tf.argmax(net.yp,axis=-1), dtype=tf.float32)
						intersection = tf.reduce_sum(tf.reduce_sum(tf.multiply(output, truth), axis=-1),axis=-1)
						union = tf.reduce_sum(tf.reduce_sum(tf.cast(tf.add(output, truth)>= 1, dtype=tf.float32), axis=-1),axis=-1)
						jaccard = tf.reduce_mean(intersection / union)
						metrics[metric] = jaccard
			self.metrics = metrics
		


	def trainOne(self,examples=None,N=1,**args):
		if examples is None:
			examples = self.examples.getTrainingExamples(N)
		examples = self.net.preprocessInput(examples)
		feed,missing = utility.buildFeed(self.net.requiredInputs+self.cost.requiredInputs,examples,**args)
		if len(missing) > 0:
			logger.error('Missing training values: {}'.format(missing))
		_,ret = tf.get_default_session().run([self.trainingStep,self.metrics],feed_dict=feed)
		return ret


	def validate(self,examples=None,N=1,**args):
		if examples is None:
			examples = self.examples.getValidationExamples(N)
		examples = self.net.preprocessInput(examples)
		feed,missing = utility.buildFeed(self.net.requiredInputs+self.cost.requiredInputs,examples,**args)
		if len(missing) > 0:
			logger.error('Missing training values: {}'.format(missing))
		ret = tf.get_default_session().run(self.metrics,feed_dict=feed)
		return ret


	def train(self,epochs=1,trainTime=None,examplesPerEpoch=5,trainingExamplesPerBatch=1,validationExamplesPerBatch=1,saveEvery=False,debug=False,trainArgs=dict(),validateArgs=dict()):
		if trainTime is not None:
			logger.info("Training for {:0.2f} hours using {} examples per epoch.".format(trainTime,examplesPerEpoch))
		else:
			logger.info("Training for {} epochs using {} examples per epoch.".format(epochs,examplesPerEpoch))

		if validateArgs is None or len(validateArgs) == 0:
			validateArgs = trainArgs
		self.previouslyElapsed += self.elapsed; self.elapsed = 0
		trainTime *= 3600;
		self.startTime = time.time()
		lastSave = self.elapsed
		epochT1 = time.time()

		while (self.epoch < epochs) or self.elapsed < trainTime:
			# -------------------  Inner training loop --------------------
			for iteration in range(0,examplesPerEpoch):
				example = self.examples.getTrainingExamples(trainingExamplesPerBatch)
				result = self.trainOne(example,**trainArgs)

				t2 = time.time(); self.elapsed = t2-self.startTime;
				if self.progressTracker is not None:
					self.progressTracker.update(example=example,epoch=self.epoch,iteration=iteration,elapsed=self.previouslyElapsed+self.elapsed,totalIterations=self.epoch*examplesPerEpoch+iteration,kind='Training',metrics=result)
				if self.elapsed > trainTime:
					break

			# -------------------  Validation --------------------
			example = self.examples.getValidationExamples(validationExamplesPerBatch)
			result = self.validate(example,**validateArgs)
			
			if self.progressTracker is not None:
				self.progressTracker.update(example=example,epoch=self.epoch,iteration=iteration,elapsed=self.previouslyElapsed+self.elapsed,totalIterations=self.epoch*examplesPerEpoch+iteration,kind='Validation',metrics=result)
			self.epoch += 1; t2 = time.time(); elapsed = t2-self.startTime;
			epochTime = time.time() - epochT1
			epochT1 = time.time()
			try:
				logger.info("After epoch {} validation cost is {:0.2f} ({:0.1f} s / iteration)".format(self.epoch,result['cost'],epochTime/examplesPerEpoch))
			except:
				logger.info("After epoch {} validation cost is {} ({:0.1f} s / iteration)".format(self.epoch,result['cost'],epochTime/examplesPerEpoch))

			if saveEvery and self.elapsed-lastSave > saveEvery*3600:
				totalElapsed = self.previouslyElapsed + self.elapsed
				elapsedModifier,elapsedUnits = (1.,'s') if totalElapsed < 60*5 \
					else (1./60,'m') if totalElapsed < 3600*2 \
					else (1./3600,'h')
				checkpointLabel = '{:0.2fs}'.format(totalElapsed*elapsedModifier,elapsedUnits)
				self.net.saveCheckpoint(label=checkpointLabel)
				logger.info("************ Checkpoint {} saved ************".format(checkpointLabel))
				lastSave = self.elapsed

		example = self.examples.getValidationExamples(validationExamplesPerBatch)
		result = self.validate(example,**validateArgs)
		if self.progressTracker is not None:
			self.progressTracker.update(example=example,epoch=self.epoch,iteration=iteration,elapsed=self.previouslyElapsed+self.elapsed,totalIterations=self.epoch*examplesPerEpoch+iteration,kind='Validation',metrics=result)


	# Pickling support
	def __setstate__(self,params):
		super().__setstate__(params)
		self.previouslyElapsed += self.elapsed; self.elapsed = 0.0
		self.epoch = 0

	def saveCheckpoint(self,label=None,write_meta_graph=False):
		trainerName = self.name + '.trainer'
		name = self.name + ('-{}'.format(label) if not label is None else '')
		self.tfSaver.save(tf.get_default_session(),os.path.join(self.fname,trainerName,name),write_meta_graph=write_meta_graph)
		self.net.saveCheckpoint(label=label,write_meta_graph=write_meta_graph)

	def loadCheckpoint(self,label=None,latest=True):
		trainerName = self.name + '.trainer'
		if label is None and latest:
			fname = tf.train.latest_checkpoint(os.path.join(self.fname,trainerName))
		else:
			name = self.name + ('-{}'.format(label) if not label is None else '')
			fname = os.path.join(self.fname,trainerName,name)
		logger.info("Loading trainer checkpoint {}".format(fname))
		self.tfSaver.restore(tf.get_default_session(),fname)
		self.net.loadCheckpoint(label=label,latest=latest)


	def save(self,fname=None):
		""" Save a trainer to disk.  The trainer creates it's directory and
			saves it's net into that directory. The trainer is saved into
			a .trainer subdirectory.
			The return value is the path to the DD directory."""
		# create the net directory
		netName = self.net.name + '.net'
		trainerName = self.name + '.trainer'
		defaultFname = self.__dict__.get('fname',netName)
		fname = defaultFname if fname is None else os.path.join(fname,netName) if fname.endswith(os.path.sep) else fname
		self.__dict__['fname'] = fname
		self.__dict__['saveTime'] = time.time()

		os.makedirs(os.path.join(fname,trainerName),mode=0o777,exist_ok=True)
		
		# save the net and tracker and replace their references in the trainer with file names
		self.netClass = self.net.__class__
		self.trackerClass = self.progressTracker.__class__
		self.netFile = self.net.save(fname)
		self.trackerFile = self.progressTracker.save(fname)
		
		with open(os.path.join(fname,trainerName,trainerName),'wb') as f:
			dill.dump(self,f,protocol=dill.HIGHEST_PROTOCOL)

		variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
		self.modelParameters['tfSaver'] = tf.train.Saver(var_list=variables)
		self.saveCheckpoint(write_meta_graph=True)
		
		return os.path.join(fname,trainerName,trainerName)


	# Load not updated yet

	@classmethod
	def load(self,fname):
		""""fname is the full path to a trainer dir.  The trainer will load it's
		accompanying net file, which should be in the same directory."""
		fname = fname[:-1] if fname.endswith(os.path.sep) else fname
		trainerName = os.path.basename(fname)
		fname = os.path.dirname(fname)
		with open(os.path.join(fname,trainerName,trainerName),'rb') as f:
			trainer = dill.load(f)
		trainer.__dict__['fname'] = fname
		try:
			trainer.net = trainer.netClass.load(os.path.join(fname))
			trainer.progressTracker = trainer.trackerClass.load(os.path.join(fname,os.path.basename(trainer.trackerFile)))
			trainer.initializeTraining()
			trainer.setupMetrics()
			variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=trainer.name)
			trainer.modelParameters['tfSaver'] = tf.train.Saver(var_list=variables)
			trainer.loadCheckpoint()
		except:
			logger.exception('Failed to load net')
			logger.warning('*** Could not locate saved network. Trainer will be nonfunctional. ***')
		return trainer

