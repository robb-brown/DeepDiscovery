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


def padImage(img,depth,mode='2d',shape=None):
	"""Pads (or crops) an image so it is evenly divisible by 2**depth.  If mode == '2d' then z is not padded"""
	if not shape is None:
		for axis in [0,1,2] if mode == '3d' else [1,2]:
			if img.shape[axis] > shape[axis]:
				start = int(numpy.trunc((img.shape[axis]-shape[axis]) / 2));
				img = numpy.rollaxis(img,axis)[start:start+shape[axis]]
				img = numpy.rollaxis(img,0,len(img.shape)+axis+1)

	z,y,x,z1,z2,y1,y2,x1,x2 = computePad(img.shape,depth,mode,shape)
	dims = [(0,0) for i in img.shape]
	dims[0] = (z1,z2); dims[1] = (y1,y2); dims[2] = (x1,x2)
	return numpy.pad(img,dims,'constant')


def depadImage(img,originalShape):
	""" This isn't quite right.  One pixel difference """
	z1=z2=y1=y2=x1=x2=0
	z,y,x = img.shape
	z1 = int(numpy.floor((z - originalShape[0])/2)); z2 = int(numpy.ceil((z - originalShape[0])/2))
	y1 = int(numpy.floor((y - originalShape[1])/2)); y2 = int(numpy.ceil((y - originalShape[1])/2))
	x1 = int(numpy.floor((x - originalShape[2])/2)); x2 = int(numpy.ceil((x - originalShape[2])/2))
	ret = img[z1:z-z2,y1:y-y2,x1:x-x2]
	return ret


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
