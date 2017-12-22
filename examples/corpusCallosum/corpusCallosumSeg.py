import DeepDiscovery as dd
import glob, os
import tensorflow as tf
import numpy

import matplotlib; matplotlib.interactive(True)

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
# [1,0].  The first channel will be the positive mask.
trainingData = dd.Data.ImageTrainingData(examples,reserveForValidation=0.1,reserveForTest=0,truthComponents=[0,1],gentleCoding=False)

# You can test out your trainingData object by asking for an example:
example = trainingData.getTrainingExamples(1)

# Let's save this training data.  Since we asked DeepDiscovery to do the train/test
# allocation, for consistency we might want to use this same object for training
# different networks.
trainingData.save('corpusCallosum.data')

# Our network downsamples a certain number of times, and the input image needs to be a
# multiple of 2**depth where depth is the number of downsampling steps.  For now, this
# is done by the data object.  This will be moved to the network preprocessing though.
# If we're in 2d mode we only need to pad x and y.
trainingData.mode = '2d'
trainingData.depth = 3
# ---------------------------------------------------------------------------------


session = tf.InteractiveSession()

if 0:
	# -----------------------  Creating a Network ------------------------
	# This is where we create our actual model.  Let's use a Segmenter2D, which implements something
	# similar to a U or V net. filterPlan is the number of filters at each downsampling layer.
	# filterSize is the size of the kernel, and postUDepth lets you add extra layers after
	# the U net. We give our network a name so that it is distinct from others we might load or create
	# (it puts its tensorflow variables into a variable scope based on the name) and will also
	# default to saving itself under that name.
	filterPlan = [10,20,30]; filterSize = 5; postUDepth = 1
	segmenter = dd.Net.Segmenter2D(filterPlan=filterPlan,filterSize=filterSize,postUDepth=postUDepth,name='CorpusCallosum2D')
	# ---------------------------------------------------------------------

	# -----------------------  Creating a Trainer and Tracker------------------------
	# The trainer object takes care of training our model.  The tracker keeps realtime stats on
	# the performance of the network as it is trained, creates graphs, and dumps these to files
	# on disk so we can look at them or serve them with a webserver.
	tracker = dd.Trainer.ProgressTracker(logPlots=False,plotEvery=50,basePath='./tracker')
	metrics = ['output','cost','jaccard']; learning_rate = 1e-3
	trainer = dd.Trainer.Trainer(segmenter,trainingData,progressTracker=tracker,metrics=metrics,learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)
	# ---------------------------------------------------------------------

	# Initialize all the newly created variables
	session.run(tf.global_variables_initializer())
	
	# Save the trainer (and all it's components) before we use them
	trainer.save()

else:
	# -------------- Load the trainer, tracker and network -----------------
	trainer = dd.Trainer.Trainer.load('Segmenter2D.net/Training-Segmenter2D.trainer')
	segmenter = trainer.net
	# ---------------------------------------------------------------------


print('\n\n\n')

# ------------------------- Train -----------------------------------
trainer.train(trainTime=0.1,examplesPerEpoch=5,trainingExamplesPerBatch=1)
# ---------------------------------------------------------------------

# ------------------------- Save the trainer, tracker and network -----------------------------------
trainer.save()
# ---------------------------------------------------------------------------------------------


ex = trainingData.getTrainingExamples()
print("Executing forward pass.  Resulting shape: {}".format(segmenter.forwardPass(ex).shape))
print("Cost = {}".format(segmenter.evaluateCost(ex)))
