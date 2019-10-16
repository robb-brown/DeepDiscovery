---
title: "Brain Segmentation with Deep Discovery"
date: 2018-06-02T11:15:29-04:00
draft: false
beta: true
---

This tutorial shows how to load a trained model, use it to segment an image, and save the resulting brain probability map.

Example data and a trained 2d brain segmentation model can be found [here](https://github.com/robb-brown/DeepDiscovery/tree/master/examples/brainSegmentation). For this tutorial, download both and put them in a working directory. The directory also contains ```segment.py``` if you'd rather not type out the tutorial code.

Create a new python program in your working directory and open it in your favourite code editor; for this tutorial we'll assume you called it ```segment.py```. You can type (or copy and paste) code into this file as we go.  You may also wish to open a terminal, start Python, and copy and paste code into the interactive interpreter as well.

First we need our imports:

``` python
import DeepDiscovery as dd
import tensorflow as tf
import nibabel as nib
import pylab
```

We import Deep Discovery with the name ```dd``` for convenience, tensorflow (named ```tf```) to create a session and nibable (as ```nib```) for reading images. You can also import ```pylab``` for displaying the result.

In order to actually run a TensorFlow model, we need to have a session.  We might as well create it now:

``` python
session = tf.InteractiveSession()
```

Now, to give our program some generalizability, let's define a couple of variables holding the paths to our image and model.  Later we'll convert these into command line arguments:

``` python
modelPath = 'BrainExtraction2d.net'
dataPath = 'tal_sub-A00028185_ses-NFB3_t1w.mnc.004.mnc'
```

Okay, that takes care of the preliminaries.  Now it's time to actually load our trained model and an input image:

```python
net = dd.Net.Net.load(modelPath)
image = nib.load(dataPath)
```

Note that Nibabel can read and write both MINC and NIFTI images, but it can only write NIFTI.  If you'd like MINC output, you will need some way to save a numpy array as a MINC file.

Time to segment.  Ready?

``` python
brainProb = net.segment(image.get_data())[...,1]
```

If you're pasting code into an interactive interpreter, you'll notice that this command takes a while (10 seconds or so) to run.  Tensorflow is running the model.  The object ```net``` is a ```DeepDiscovery.Net.Segmenter2D``` object, which implements a U-Net type deep learning network. The ```segment``` method, by default, expects an image with dimensions ```[z,y,x]```, which is conveniently what nibabel provides.  Since we're doing 2D segmentation, ```Segmenter2D.segment``` reshapes the input to have dimensions ```[batch,y,x,channel]``` where each z slice gets converted into a "batch" and a dummy channel dimension is added on the end.  If we were doing multi-contrast segmentation, we might pack multiple images into different channels. 

Finally, the segmenter maps the TensorFlow output back into a ```[z,y,x,channels]``` array. The channels store the probability of each class. Since we only have two classes (background a brain), we only need the second channel; we select this using the numpy slice ```[...,1]```.

Next, let's save our brain probability map.  We can create a new NIFTI image and save it using Nibabel:

```python
output = nib.Nifti2Image(brainProb,image.affine,header=image.header)
nib.save(output,'brainProb.nii')
```

If you wanted to save a binary brain mask instead, you would threshold brainProb at 0.5 before saving.

Finally, let's see how our segmentation did.  A nice way to do that is to display our original image with the 50% probability contour line displayed on top:

```python
slc = [slice(None),slice(None),slice(None)]; slc[0] = 100
pylab.ion(); pylab.figure('Brain Probability'); pylab.clf()
pylab.imshow(image.get_data()[slc],cmap=pylab.cm.gray,origin='lower')
pylab.contour(brainProb[slc],[0.5],colors='g')
```

By changing the ```slc[0] = 100``` you can control the orientation and location of the slice.  ```slc[0]``` gives an axial slice, with ```slc[1]``` for coronal and ```slc[2]``` for sagittal. Slice 100 is approximately in the centre.

If you haven't been following in the interactive interpreter, save your file and run it now, using:

```python
python -i segment.py
```

in a terminal (don't forget to source your virtual environment). You should see an axial slice of your test image, with a green contour line around the brain. If you select another slice, you can see sagittal and coronal views too:

<div class=".content-center .cf:after">
<img src="/tutorials/images/brainSegmentationTutorialAxial.jpeg", alt="Axial Brain Slice" style="width:150px; float:left"/>
<img src="/tutorials/images/brainSegmentationTutorialCoronal.jpeg", alt="Coronal Brain Slice" style="width:201px; float:left"/>
<img src="/tutorials/images/brainSegmentationTutorialSagittal.jpeg", alt="Sagittal Brain Slice" style="width:214px; float:left"/>
</div>

