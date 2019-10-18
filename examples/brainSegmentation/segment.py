import DeepDiscovery as dd
from DeepDiscovery import tf
import nibabel as nib
import pylab

session = tf.InteractiveSession()

modelPath = 'BrainExtraction2d.net'
dataPath = 'tal_sub-A00028185_ses-NFB3_t1w.mnc.004.mnc'

net = dd.Net.Net.load(modelPath)
image = nib.load(dataPath)

brainProb = net.segment(image.get_data())[...,1]

output = nib.Nifti2Image(brainProb,image.affine,header=image.header)
nib.save(output,'brainProb.nii')

slc = [slice(None),slice(None),slice(None)]; slc[1] = 100
pylab.ion(); pylab.figure('Brain Probability'); pylab.clf()
pylab.imshow(image.get_data()[slc],cmap=pylab.cm.gray,origin='lower')
pylab.contour(brainProb[slc],[0.5],colors='g')
