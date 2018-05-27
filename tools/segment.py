import DeepDiscovery as dd
import tensorflow as tf
import nibabel as nib
import numpy

session = tf.InteractiveSession()

modelPath = 'models/BrainExtraction2d.net'
#modelPath = 'models/OcularFat2d.net'
dataPath = 'tal_sub-A00028185_ses-NFB3_t1w.mnc.004.mnc'

net = dd.Net.Net.load(modelPath)
image = nib.load(dataPath)

mask = net.segment(image.get_data())[...,1]

from pylab import *
slc = [slice(None),slice(None),slice(None)]; slc[2] = 100
ion(); figure('Mask'); clf(); imshow(image.get_data()[slc],cmap=cm.gray,origin='lower'); contour(mask[slc],[0.5],colors='g')
