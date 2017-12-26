import numpy
import logging

logger = logging.getLogger(__name__)


# ----------------------------   Utility Functions  -----------------------------------
def onlyMyArguments(function,args):
	goodArgs = function.func_code.co_varnames[:function.func_code.co_argcount]
	return dict([i for i in args.items() if i[0] in goodArgs])

def createPath(path):
	if not p.exists(path):
		os.umask(0000)
		os.makedirs(path,mode=0x777)

def computePad(dims,depth,mode='2d',shape=None):
	z1=z2=y1=y2=x1=x2=0
	if not shape is None:
		if mode == '2d':
			y,x = list(shape[[1,2]]); z = dims[0]
		else:
			z,y,x = list(shape[[0,1,2]])
	else:
		z,y,x = [numpy.ceil(dims[i]/float(2**depth)) * (2**depth) for i in range(0,3)]
	x = float(x); y = float(y); z = float(z)

	if mode=='2d':
		y1 = int(numpy.floor((y - dims[1])/2)); y2 = int(numpy.ceil((y - dims[1])/2))
		x1 = int(numpy.floor((x - dims[2])/2)); x2 = int(numpy.ceil((x - dims[2])/2))
	elif mode=='3d':
		z1 = int(numpy.floor((z - dims[0])/2)); z2 = int(numpy.ceil((z - dims[0])/2))
		y1 = int(numpy.floor((y - dims[1])/2)); y2 = int(numpy.ceil((y - dims[1])/2))
		x1 = int(numpy.floor((x - dims[2])/2)); x2 = int(numpy.ceil((x - dims[2])/2))
	return z,y,x,z1,z2,y1,y2,x1,x2


def padImage(img,depth,mode='2d',spatialAxes=[0,1,2],shape=None,oneHot=False):
	"""Pads (or crops) an image so it is evenly divisible by 2**depth.  If mode == '2d' then z is not padded"""
	if not shape is None:
		for axis in spatialAxes[-3:] if mode == '3d' else spatialAxes[-2:]:
			if img.shape[axis] > shape[axis]:
				start = int(numpy.trunc((img.shape[axis]-shape[axis]) / 2));
				img = numpy.rollaxis(img,axis)[start:start+shape[axis]]
				img = numpy.rollaxis(img,0,len(img.shape)+axis+1)

	z,y,x,z1,z2,y1,y2,x1,x2 = computePad(numpy.array(img.shape)[[spatialAxes]],depth,mode,shape)
	dims = [(0,0) for i in img.shape]
	dims[spatialAxes[0]] = (z1,z2); dims[spatialAxes[1]] = (y1,y2); dims[spatialAxes[2]] = (x1,x2)
	if oneHot:
		ret = numpy.pad(img,dims,'edge')
	else:
		ret = numpy.pad(img,dims,'constant')
	return ret
	


def depadImage(img,originalShape,spatialAxes=[0,1,2]):
	z1=z2=y1=y2=x1=x2=0
	z,y,x = numpy.array(img.shape)[spatialAxes]
	z1 = int(numpy.floor((z - originalShape[spatialAxes[0]])/2)); z2 = int(numpy.ceil((z - originalShape[spatialAxes[0]])/2))
	y1 = int(numpy.floor((y - originalShape[spatialAxes[1]])/2)); y2 = int(numpy.ceil((y - originalShape[spatialAxes[1]])/2))
	x1 = int(numpy.floor((x - originalShape[spatialAxes[2]])/2)); x2 = int(numpy.ceil((x - originalShape[spatialAxes[2]])/2))
	slices = list(numpy.repeat(slice(None),len(img.shape)))
	slices[spatialAxes[0]] = slice(z1,z-z2); slices[spatialAxes[1]] = slice(y1,y-y2); slices[spatialAxes[2]] = slice(x1,x-x2);  
	ret = img[slices]
	return ret
	
	
def buildFeed(requiredInputs,example,**args):
	args.update(example)
	return dict([(requiredInput,args.get(requiredInput.name.split(':')[0].split('_')[0].split('/')[-1],None)) for requiredInput in requiredInputs])


def reportLayerSize(outputShape,annotation=''):
	args = dict(
		annotation = annotation,
		outputShape = outputShape,
		size = numpy.prod([i if i is not None else 1 for i in outputShape])*4 / 1024**2
	)
	logger.info(' {annotation} {outputShape} {size} MB'.format(args))


def convertToOneHotVec(a):
	b = numpy.zeros((len(a),len(numpy.unique(a))),dtype=numpy.int64)
	b[numpy.arange(len(a)),a] = 1
	return b

def convertToOneHot(y,coding=[0,1],gentleCoding=True):
	"""Channel will be the last dimension"""
	y = numpy.around(y).astype(numpy.int16);

	# y = numpy.rollaxis(numpy.array([numpy.where(y==code,1,0) for c,code in enumerate(coding)]),0,len(y.shape)+1).astype(numpy.float32)
	# gentleCoding = 0.9 if gentleCoding is True else gentleCoding
	# if gentleCoding:
	# 	y *= gentleCoding
	# 	y += (y == 0) * (1-gentleCoding)

	y = sum([numpy.where(y==code,c,0) for c,code in enumerate(coding)])
	codingMatrix = numpy.eye(len(coding))
	gentleCoding = 0.9 if gentleCoding is True else gentleCoding
	if gentleCoding:
		codingMatrix *= gentleCoding
		codingMatrix += (codingMatrix == 0) * (1-gentleCoding)
	y = codingMatrix[y]
	return y


# ----------------------------   END Utility Functions  -----------------------------------
