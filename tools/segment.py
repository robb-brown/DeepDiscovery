import DeepDiscovery as dd
from DeepDiscovery import tf
import nibabel as nib
import numpy
import sys

session = tf.InteractiveSession()

modelPath = sys.argv[1]
dataPath = sys.argv[2]

net = dd.Net.Net.load(modelPath)
image = nib.load(dataPath)

mask = net.segment(image.get_data())[...,1]

from pylab import *
slc = [slice(None),slice(None),slice(None)]; slc[2] = 100
ion(); figure('Mask'); clf(); imshow(image.get_data()[slc],cmap=cm.gray,origin='lower'); contour(mask[slc],[0.5],colors='g')
