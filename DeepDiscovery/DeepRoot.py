class DeepRoot(object):

	def __new__(cls,*args,**kwargs):
		self = super().__new__(cls)
		self.__dict__['modelParameters'] = dict()
		self.__dict__['hyperParameters'] = dict()
		self.__dict__['name'] = None
		self.__dict__['excludeFromPickle'] = ['modelParameters']
		return self

	def __init__(self):
		pass
	
	def __getattr__(self,attr):
		if attr in self.__dict__['modelParameters']:
			return self.__dict__['modelParameters'].get(attr)
		elif attr in self.__dict__['hyperParameters']:
			return self.__dict__['hyperParameters'].get(attr)
		else:
			raise AttributeError(attr)

	def __setattr__(self,attr,value):
		if attr in self.__dict__:
			self.__dict__[attr] = value
		elif attr in self.modelParameters:
			self.modelParameters[attr] = value
		elif attr in self.hyperParameters:
			self.hyperParameters[attr] = value
		else:
			raise AttributeError(attr)

	# Pickling support
	def __getstate__(self):
		pickleDict = dict()
		for k,v in self.__dict__.items():
			if not k in self.excludeFromPickle:
					pickleDict[k] = v
		return pickleDict

	def __setstate__(self,params):
		self.__dict__.update(params)

