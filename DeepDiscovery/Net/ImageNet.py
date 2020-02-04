import time,os,copy,sys
import math,numpy, random
from collections import OrderedDict
import dill
from tfextended.tensorflowCompat import tf
from . import Net
from .. import Data
from ..utility import *

import logging

logger = logging.getLogger(__name__)


class ImageNet(Net):

	def __init__(self,dimensions=None,inputChannels=1,filterSize=None,maxpool=False,normalization=None,nonlinearity=tf.nn.relu,inputDropout=False,inputNoise=False,internalDropout=False,standardize=None,name=None,fname=None,**args):
		self.hyperParameters.update(dict(dimensions = [None]+list(dimensions),
										inputChannels=inputChannels,
										filterSize = filterSize,
										maxpool = maxpool,
										normalization = normalization,
										nonlinearity = nonlinearity,
										inputDropout = inputDropout,
										internalDropout = internalDropout,
										inputNoise = inputNoise,
										mode = '2d' if len(dimensions) == 3 else '3d' if len(dimensions) == 4 else None,
										**args
										))
		self.hyperParameters['standardize'] = Data.SpotStandardization() if standardize == True else standardize
		self.hyperParameters['preprocessor'] = \
							Data.ImagePreprocessor(	requiredDimensionOrder = ['b','y','x','c'] if self.mode == '2d' else ['b','z','y','x','c'],
												crop = args.get('crop',None),
												standardize = self.standardize,
												pad = len(self.filterPlan),
												mode = self.mode,
												)
		super().__init__(fname=fname,name=name)




	def preprocessInput(self,example,dimensionOrder=None):
		dimensionOrder = example.get('dimensionOrder',None) if dimensionOrder is None else dimensionOrder
		ret = dict(); ret.update(example);
		ret['input'] = self.preprocessor.process(example['input'],dimensionOrder=dimensionOrder)
		if 'truth' in example:
			ret['truth'] = self.preprocessor.process(example['truth'],dimensionOrder=dimensionOrder)
		return ret
