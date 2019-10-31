import matplotlib;
#matplotlib.use('Agg')
matplotlib.interactive(True)

import DeepDiscovery as dd
import glob, os
import numpy

# It's useful to import tf from DeepDiscovery. DD provides a compatability layer
# that's required to support different versions of tensorflow.
from DeepDiscovery import tf


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',level=logging.INFO)

# Initialize everything with specific random seeds for repeatability
tf.set_random_seed(1); numpy.random.seed(1)


# -----------------------  Creating a Training Data Object ------------------------
# set the data path to where the included example data is
dataPath = './data'
images = glob.glob(os.path.join(dataPath,'?????.nii.gz'))

# We create a list of dictionaries.  Every dictionary is an example, and should
# have at least the key 'input' equal to an image.  We also have 'truth' equal to
# our true image (the manual CC segmentation).  At this point
# we could manually specify which examples should be used for training and
# which should be reserved for testing or validation, but we can also let
# the DeepDiscovery data object do that for us.
examples = [dict(input=i,truth=i.replace('.nii.gz','_cc.nii.gz')) for i in images]

# Now create a DeepDiscovery ImageTrainingData object using our examples dictionaries.
# reserveForValidation 0.1 keeps 10% of the data in the validation group.  We could also
# give an integer value (like 1) that would reserve that many examples.
# We're training a segmenter, so we need our truth image converted into one-hot (one channel per class).
# We have two classes (not corpus callosum = 0 and corpus callosum = 1) so we set truthComponents to
# [0,1].  The first channel will be the positive mask.
trainingData = dd.Data.ImageTrainingData(examples,reserveForValidation=0.1,reserveForTest=0,truthComponents=[0,1],gentleCoding=0.9)


# Let's save this training data.  Since we asked DeepDiscovery to do the train/test
# allocation, for consistency we might want to use this same object for training
# different networks.
trainingData.save('corpusCallosum.data')

# We can use a spatially weighted cost function (called 'attention', but not to be confused with
# other forms of attention in deep learning).  This can help with tasks that involve segmentation
# of a small structure with lots of background, or where edges are very important and seem to be
# getting overlooked.  Our data object is in charge of generating the spatial weighting functions,
# and we pass it an object for this purpose.  In this case, we'll use one that emphasizes edges and
# puts more weight on the corpus callosum to make up for it's small size.
attention = dd.Data.EdgeBiasedAttention(); trainingData.attention = attention

# You can test out your trainingData object by asking for an example:
example = trainingData.getTrainingExamples(1)

# ---------------------------------------------------------------------------------

session = tf.InteractiveSession()

if not os.path.exists('CorpusCallosum2D.net'):
	# -----------------------  Creating a Network ------------------------
	# This is where we create our actual model.  Let's use a Segmenter2D, which implements something
	# similar to a U or V net. filterPlan is the number of filters at each downsampling layer.
	# filterSize is the size of the kernel, and postUDepth lets you add extra layers after
	# the U net. We give our network a name so that it is distinct from others we might load or create
	# (it puts its tensorflow variables into a variable scope based on the name) and will also
	# default to saving itself under that name.
	filterPlan = [10,20,30,40]; filterSize = 5; postUDepth = 1; standardize=True; inputDropout = True
	segmenter = dd.Net.Segmenter2D(filterPlan=filterPlan,filterSize=filterSize,postUDepth=postUDepth,standardize=standardize,inputDropout=inputDropout,name='CorpusCallosum2D')
	# ---------------------------------------------------------------------

	# -----------------------  Creating a Trainer and Tracker------------------------
	# The trainer object takes care of training our model.  The tracker keeps realtime stats on
	# the performance of the network as it is trained, creates graphs, and dumps these to files
	# on disk so we can look at them or serve them with a webserver.
	tracker = dd.Trainer.ProgressTracker(logPlots=False,plotEvery=50,basePath='./tracker')
	cost = dd.Trainer.CrossEntropyCost(net=segmenter,attention=True)
	metrics = ['output','cost','jaccard']; learning_rate = 1e-3
	trainer = dd.Trainer.Trainer(net=segmenter,cost=cost,examples=trainingData,progressTracker=tracker,metrics=metrics,learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)
	# ---------------------------------------------------------------------

	# Initialize all the newly created variables
	session.run(tf.global_variables_initializer())
	
	# Save the trainer (and all it's components) before we use them
	segmenter.save()
	trainer.save()

else:
	# -------------- Load the trainer, tracker and network -----------------
	trainer = dd.Trainer.Trainer.load('CorpusCallosum2D.net/Training-CorpusCallosum2D.trainer')
	segmenter = trainer.net
	# ---------------------------------------------------------------------

print('\n\n\n')

# ------------------------- Train -----------------------------------
# We can set separate arguments for the training and validation parts.  Here we set the dropout
# on the input data to 0.5 for training but turn it off for validation.
trainArgs=dict(inputDropout=0.5); validateArgs = dict(inputDropout=0.0)

# Train for 10 minutes (10./60. hours)
trainer.train(trainTime=10./60.,examplesPerEpoch=5,trainingExamplesPerBatch=1,trainArgs=trainArgs,validateArgs=validateArgs)
# ---------------------------------------------------------------------

# ------------------------- Save the trainer, tracker and network -----------------------------------
trainer.save()
# ---------------------------------------------------------------------------------------------
