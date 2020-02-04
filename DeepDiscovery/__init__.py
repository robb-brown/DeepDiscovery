from tfextended import tensorflowCompat as tensorflowCompat
from tfextended.tensorflowCompat import tf
import sys,traceback
from . import utility
from . import Data
try:
	from . import Net
	from . import Trainer
except:
	print("Deep Discovery could not import Tensorflow components.")
	traceback.print_exc()


# Check whether we're running interactively.
try:
	if sys.ps1: interactive = True
except AttributeError:
	interactive = False
	if sys.flags.interactive: interactive = True


