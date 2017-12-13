# import Layers
# import DeepNet
# import VectorNet
# import DeepSegmenter
# import DeepSegmenter3D
# import DeepData
# import DeepTracker
# import DeepTrainer

from . import utility
from . import Data
from . import Net
from . import Trainer

import sys

# Check whether we're running interactively.
try:
	if sys.ps1: interactive = True
except AttributeError:
	interactive = False
	if sys.flags.interactive: interactive = True


