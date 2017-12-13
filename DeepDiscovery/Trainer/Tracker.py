import pylab
import matplotlib
import numpy,dill,os
from collections import OrderedDict
from .. import DeepRoot

class ProgressTracker(DeepRoot.DeepRoot):
	
	def __init__(self,figures={},logPlots=True,plotEvery=1,basePath='./tracker',fileType='pdf',**plotArgs):
		self.trainingRecord = OrderedDict({'Training':[],'Validation':[]})
		self.performanceRecord = OrderedDict({'Training':{'elapsed':[]},'Validation':{'elapsed':[]}})
		self.figures = figures
		self.logPlots = logPlots
		self.plotArgs = dict(
			alpha = 0.5,
			marker='o',
		)
		self.plotArgs.update(plotArgs)
		self.plotEvery = plotEvery
		self.counter = 0
		self.excludeFromPickle = ['figures']
		self.basePath = basePath
		self.fileType = fileType

	def update(self,forcePlot=False,**args):
		metrics = args.get('metrics',{})
		example = args.pop('example'); output = metrics.pop('output',None)
		self.trainingRecord[args['kind']].append(args)
		self.performanceRecord[args['kind']]['elapsed'].append(args['elapsed'])
		for metric in metrics.keys():
			self.performanceRecord[args['kind']][metric] = self.performanceRecord[args['kind']].get(metric,[])
			self.performanceRecord[args['kind']][metric].append(metrics[metric])
		display = self.counter % self.plotEvery == 0 or forcePlot
		if display or args['kind'] == 'Validation':
			self.plotMetrics(example,output,**args)
		self.counter +=1

	def plotMetrics(self,example,output,display=True,**args):
		if self.figures is None:
			self.figures = dict()
		if not os.path.exists(self.basePath):
			os.makedirs(self.basePath,mode=0x777)
		if not output is None:
			self.plotOutput(example,output,display=display,**args)
		metrics = args.get('metrics',{})
		for metric in metrics:
			self.plotMetric(example,metric,display=display,**args)
		pylab.pause(0.0000000001)


	def plotMetric(self,example,metric,**args):
		# Make sure the figures are set up
		self.figures[metric] = self.figures.get(metric,pylab.figure(metric.title()))
		self.figures[metric+'Detail'] = self.figures.get(metric+'Detail',pylab.figure('{}-Detail'.format(metric.title())))

		# Plot progress graph
		trainMetric = numpy.array(self.performanceRecord['Training'].get(metric,[])); trainElapsed = numpy.array(self.performanceRecord['Training']['elapsed'])
		validateMetric = numpy.array(self.performanceRecord['Validation'].get(metric,[])); validateElapsed = numpy.array(self.performanceRecord['Validation'].get('elapsed',[]))
		
		# Plot the full record for the metric
		self.figures[metric].clf(); axis = self.figures[metric].add_subplot(111); plotFunction = axis.semilogy if args.get('logPlots',self.logPlots) else axis.plot
		if len(trainElapsed) > 0:
			elapsedModifier,elapsedUnits = (1.,'s') if trainElapsed[-1] < 60*5 \
				else (1./60,'m') if trainElapsed[-1] < 3600*2 \
				else (1./3600,'h')
			plotFunction(trainElapsed*elapsedModifier,trainMetric,color='b',label='Training',**self.plotArgs);

		if len(validateElapsed) > 0:
			plotFunction(validateElapsed*elapsedModifier,validateMetric,color='g',label='Validation',**self.plotArgs);

		axis.set_xlabel('Elapsed Time (%s)' % elapsedUnits)
		axis.set_ylabel(metric.title())
		axis.legend(loc='best')

		# Plot the most recent record of the metric
		self.figures[metric+'Detail'].clf(); axis = self.figures[metric+'Detail'].add_subplot(111); plotFunction = axis.semilogy if args.get('logPlots',self.logPlots) else axis.plot
		if len(trainElapsed) > 0:
			start = 0 if len(trainElapsed) <= 100 else -100;
			plotFunction(trainElapsed[start:]*elapsedModifier,trainMetric[start:],color='b',label='Training',**self.plotArgs);
		if len(validateElapsed) > 0:
			start = numpy.argmin(numpy.abs(validateElapsed-trainElapsed[start]))
			plotFunction(validateElapsed[start:]*elapsedModifier,validateMetric[start:],color='g',label='Validation',**self.plotArgs);
		axis.set_xlabel('Elapsed Time (%s)' % elapsedUnits)
		axis.set_ylabel(metric.title())
		axis.legend(loc='best')

		try:
			self.figures[metric].savefig(os.path.join(self.basePath,'{}.{}'.format(metric,self.fileType)),transparent=False,bbox_inches='tight')
			self.figures[metric+'Detail'].savefig(os.path.join(self.basePath,'{}Detail.{}'.format(metric,self.fileType)),transparent=False,bbox_inches='tight')
		except:
			pass



	def plotOutput(self,example,result,**args):
		# Plot mask
		self.figures['output'] = self.figures.get('output',pylab.figure('Output'))
		self.figures['output'].clf(); cmap = pylab.cm.RdBu; #pylab.cm.viridis

		# ROBB make this squish the channels later.  For now just take channel 1
		inputImage,output,truth = result
		if len(output.shape) == 4:				# batch/z,y,x,channel
			inputImage = inputImage[...,0]
			output = output[...,1]
			truth = numpy.where(truth[...,1]>0.5,1,0)
		else:									# batch, z,y,x,channel
			inputImage = inputImage[0,...,0]
			output = output[0,...,1]
			truth = where(truth[0,...,1]>0.5,1,0)
			
		#  Axial
		axis = self.figures['output'].add_subplot(2,2,1);
		if (truth>0.5).any():
			midSlice = numpy.argmax(numpy.sum(truth,axis=(1,2)))
		else:
			midSlice = truth.shape[0] / 2
		axis.imshow(inputImage[midSlice],cmap=pylab.cm.gray,origin='lower')
		axis.imshow(output[midSlice],cmap=cmap,origin='lower',alpha=0.3,vmin=0,vmax=1.0)
		axis.contour(truth[midSlice],colors=['b'],alpha=0.1,linewidths=1,origin='lower')

		# Coronal
		axis = self.figures['output'].add_subplot(2,2,2);
		if (truth>0.5).any():
			midSlice = numpy.argmax(numpy.sum(truth,axis=(0,2)))
		else:
			midSlice = truth.shape[1] / 2
		axis.imshow(inputImage[:,midSlice],cmap=pylab.cm.gray,origin='lower')
		axis.imshow(output[:,midSlice],cmap=cmap,origin='lower',alpha=0.3,vmin=0,vmax=1.0)
		axis.contour(truth[:,midSlice],colors=['b'],alpha=0.1,linewidths=1,origin='lower')

		# Sagittal
		axis = self.figures['output'].add_subplot(2,2,3);
		if (truth>0.5).any():
			midSlice = numpy.argmax(numpy.sum(truth,axis=(0,1)))
		else:
			midSlice = truth.shape[2] / 2
		axis.imshow(inputImage[:,:,midSlice],cmap=pylab.cm.gray,origin='lower')
		axis.imshow(output[:,:,midSlice],cmap=cmap,origin='lower',alpha=0.3,vmin=0,vmax=1.0)
		axis.contour(truth[:,:,midSlice],colors=['b'],alpha=0.1,linewidths=1,origin='lower')

		try:
			self.figures['output'].savefig(os.path.join(self.basePath,'output.%s'%self.fileType),transparent=True,bbox_inches='tight')
		except:
			pass



	# Pickling support
	def __getstate__(self):
		pickleDict = dict()
		for k,v in self.__dict__.items():
			if not issubclass(k.__class__,matplotlib.figure.Figure)\
				and not (k in self.excludeFromPickle or v in self.excludeFromPickle)\
				and not k == 'excludeFromPickle':
					pickleDict[k] = v
			else:
				pickleDict[k] = None
		pickleDict['excludeFromPickle'] = [k for k in self.__dict__.get('excludeFromPickle',['figures'])]
		return pickleDict

	def __setstate__(self,params):
		#params.pop('excludeFromPickle')
		params['excludeFromPickle'] = params.get('excludeFromPickle',['figures'])
		self.__dict__.update(params)
