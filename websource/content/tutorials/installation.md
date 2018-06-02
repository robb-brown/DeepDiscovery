---
title: "Installation"
date: 2018-05-27T15:59:22-04:00
draft: false
beta: true
---

*If you would like to use anaconda, miniconda, or some other packaging system, or you have difficulty with these instructions, you can follow the [TensorFlow](https://www.tensorflow.org/install/) installation instructions for your system, then skip to step 4. Currently Deep Discovery works with TensorFlow version 1.4 and Python 3.X.*

Installation Steps:

1. Install Python

	If your system doesn't have a pre-installed Python 3, or if you'd rather not use the system Python, download Python 3 from [python.org](https://www.python.org/downloads/) and follow the installation instructions.
    
	*Note: it's important to get Python 3.X, which should be highlighted at the top of the page as the latest version.*

2. Set up a virtual environment

	This step is optional, but recommended.  A virtual environment holds a copy of the Python interpreter and a self-contained library of installed packages. This will isolate your Deep Discovery environment from any other Python environments you may have.

	1. Open a terminal.  On a Mac open Spotlight Search and type terminal, then click on Terminal.app.

	2. Change directory to where you would like your virtual environment stored. It's important to put this somewhere convenient, e.g.: 
	<p align="center">```cd ~/projects/```</p>

	3. Create the virtual environment:  
		<p align="center">```python3 -m venv ddenv``` </p>
		Where you can replace ```ddenv``` with whatever name you like. Python will create a directory called ```ddenv``` 

	4. Source your virtual environment:
	<p align="center">```source ddenv/bin/activate``` </p>
	You should now see an indicator such as ```(ddenv)``` in your command prompt.  This indicates that you're using your virtual environment, and any Python related commands will apply to that environment.
		
3. Install TensorFlow
	
	Deep Discovery can't reliably install TensorFlow on all systems. You can follow the install instructions for TensorFlow, remembering that you already have a functioning virtual environment, or try:	
	<p align="center">```pip install tensorflow==1.4.0```</p>
	If this fails, on a Mac try:
	<p align="center">```pip install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.4.0-py3-none-any.whl```</p>
	If you have a compatible NVIDIA GPU you might also want to install TensorFlow support for it:
	<p align="center">```pip install tensorflow-gpu==1.4.0```</p>
	
4. Install Deep Discovery

	We'll use the Python program ```pip``` to install the Deep Discovery package and all its dependencies.  Making sure you're in your virtual environment, type:  
		<p align="center">```pip install git+git://github.com/robb-brown/DeepDiscovery.git```</p>
	Pip will report that it is collecting Deep Discovery from git, and will begin installing packages, such as matplotlib, numpy, and pandas. Depending on the speed of your computer and Internet connection, this may take some time.

5. Test your installation
	
	1. Start Python by typing  
	<p align="center">```python```</p>
	at the command prompt. You should see a Python prompt that looks like ```>>>```.

	2. Import Deep Discovery.  For convenience, we usually give it the short name 'dd':
	<p align="center">```import DeepDiscovery as dd```</p>
	If you didn't get any errors, your installation is complete!





