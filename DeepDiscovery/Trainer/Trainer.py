import tensorflow as tf
import time, datetime, os, dill, numpy, sys
from collections import OrderedDict
from .. import DeepRoot
import logging
logger = logging.getLogger(__name__)

class Trainer(DeepRoot.DeepRoot):

	def __init__(self,session,net,examples,useAttention=False,progressTracker=None,name=None,metrics=dict(cost=None),**trainingArguments):
		self.trainingParameters = dict(
			session = session,
			net = net,
			updates = None,
			metrics = metrics,
		)
		self.name = name if not name is None else 'Training-' + self.trainingParameters['net'].name
		self.examples = examples
		self.progressTracker = progressTracker
		self.trainingArguments = trainingArguments
		self.useAttention = useAttention
		self.elapsed = 0.0; self.previouslyElapsed = 0.0
		self.epoch = 0
		self.startTime = None
		self.params = None
		self.excludeFromPickle = ['metricFragments','costFragment','params','updateFragment','trainingFunction','validationFunction']
		self.initializeTraining(**trainingArguments)
		self.setupMetrics(metrics)


	def initializeTraining(self,**trainingArguments):
		self.trainingParameters['trainingStep'] = tf.train.AdamOptimizer(**trainingArguments).minimize(self.trainingParameters['net'].cost)
		self.trainingArguments = trainingArguments


	def setupMetrics(self,metrics=None):
		metrics = self.trainingParameters['metrics'] if metrics is None else metrics
		if not isinstance(metrics,dict):
			metrics = dict([(metric,None) for metric in metrics])
		net = self.trainingParameters['net']
		for metric in metrics:
			if metrics[metric] is None:
				if metric == 'cost':
					metrics[metric] = net.cost
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
		self.trainingParameters['metrics'] = metrics


	def trainOne(self,examples=None,N=1):
		tp = self.trainingParameters
		if examples is None:
			examples = self.examples.getTrainingExamples(N)
		examples = tp['net'].preprocessInput(examples)
		if not self.useAttention and 'attention' in examples:
			examples.pop('attention')

		# ROBB - update this with a utility function to match dictionary keys to their tensor flow named variables
		# also, use requiredTrainingArguments
		feed = {tp['net'].x:examples['input'],tp['net'].yp:examples['truth']};
		_,ret = tp['session'].run([tp['trainingStep'],tp['metrics']],feed_dict=feed)
		return ret


	def validate(self,examples=None,N=1):
		tp = self.trainingParameters
		if examples is None:
			examples = self.examples.getValidationExamples(N)
		examples = tp['net'].preprocessInput(examples)
		if not self.useAttention and 'attention' in examples:
			examples.pop('attention')

		# ROBB - update this with a utility function to match dictionary keys to their tensor flow named variables
		# also, use requiredTrainingArguments
		feed = {tp['net'].x:examples['input'],tp['net'].yp:examples['truth']};
		ret = tp['session'].run(tp['metrics'],feed_dict=feed)
		return ret


	def train(self,epochs=1,trainTime=None,examplesPerEpoch=5,trainingExamplesPerBatch=1,validationExamplesPerBatch=1,saveEvery=False,debug=False,**args):
		if trainTime is not None:
			logger.info("Training for {:0.2f} hours using {} examples per epoch.".format(trainTime,examplesPerEpoch))
		else:
			logger.info("Training for {} epochs using {} examples per epoch.".format(epochs,examplesPerEpoch))

		self.previouslyElapsed += self.elapsed; self.elapsed = 0
		trainTime *= 3600;
		self.startTime = time.time()
		lastSave = self.elapsed
		epochT1 = time.time()

		while (self.epoch < epochs) or self.elapsed < trainTime:
			# -------------------  Inner training loop --------------------
			for iteration in range(0,examplesPerEpoch):
				example = self.examples.getTrainingExamples(trainingExamplesPerBatch)
				example.update(args)

				result = self.trainOne(example)

				t2 = time.time(); self.elapsed = t2-self.startTime;
				if self.progressTracker is not None:
					self.progressTracker.update(example=example,epoch=self.epoch,iteration=iteration,elapsed=self.previouslyElapsed+self.elapsed,totalIterations=self.epoch*examplesPerEpoch+iteration,kind='Training',metrics=result)
				if self.elapsed > trainTime:
					break

			# -------------------  Validation --------------------
			example = self.examples.getValidationExamples(validationExamplesPerBatch)
			example.update(args)
			result = self.validate(example)
			
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
				oldName = self.name; self.name = '{}_{:0.2fs}'.format(oldName,totalElapsed*elapsedModifier,elapsedUnits)
				fname = self.save(path='./checkpoint')
				logger.info("************ Checkpoint saved as {} ************".format(fname))
				self.name = oldName
				lastSave = self.elapsed

		example = self.examples.getValidationExamples(validationExamplesPerBatch)
		example.update(args)
		result = self.validate(example)
		if self.progressTracker is not None:
			self.progressTracker.update(example=example,epoch=self.epoch,iteration=iteration,elapsed=self.previouslyElapsed+self.elapsed,totalIterations=self.epoch*examplesPerEpoch+iteration,kind='Validation',metrics=result)


	# Pickling support
	def __getstate__(self):
		pickleDict = dict()
		for k,v in self.__dict__.items():
			if not v in [self.params] \
				and not issubclass(v.__class__,theano.compile.function_module.Function) \
				and not (k in self.excludeFromPickle or v in self.excludeFromPickle):
					pickleDict[k] = v
		return pickleDict

	def __setstate__(self,params):
		self.__dict__.update(params)
		self.previouslyElapsed += self.elapsed; self.elapsed = 0.0
		self.epoch = 0

		# DEBUG - jump start for old saved objects
		# self.trainingArguments = dict(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-08)
		# self.examples.mode = '2d'
		# self.examples.attention = None
		# self.initializeTraining(**self.trainingArguments)


	def save(self,path='.',fname=None):
		""" Save a trainer to disk.  The trainer creates the directory [path]/[fname] and
			saves itself and it's net into that directory in separate files.
			The return value is the path to the trainer file."""
		self.saveTime = time.time()
		fname = fname if fname is not None else self.name
		if not os.path.exists(os.path.join(path,fname)):
			os.makedirs(os.path.join(path,fname))
		netname = self.net.name + '.net'
		self.net.save(path=os.path.join(path,fname),fname=netname)
		net = self.net
		self.netClass = self.net.__class__
		self.net = netname
		with open(os.path.join(path,fname,fname+'.trainer'),'wb') as f:
			dill.dump(self,f,protocol=dill.HIGHEST_PROTOCOL)
		self.net = net
		return os.path.join(path,fname,fname+'.trainer')

	@classmethod
	def load(self,fname):
		""""fname is the full path to a trainer file.  The trainer will load it's
		accompanying net file, which should be in the same directory."""
		with open(fname,'rb') as f:
			trainer = dill.load(f)
		path = os.path.dirname(fname)
		if os.path.exists(os.path.join(path,trainer.net)):
			trainer.net = trainer.netClass.load(os.path.join(path,trainer.net))
			trainer.initializeTraining(**trainer.trainingArguments)
		else:
			logger.warning('*** Could not locate saved network. Trainer will be nonfunctional. ***')
		return trainer
