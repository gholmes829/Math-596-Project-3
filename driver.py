"""

"""
import numpy as np
# import stuff from core here
#from core.module import ...

class Driver:

	def __init__(self) -> None:
		"""
		Has attributes like number of oberserved vars and stuf like that
		"""
		pass

	def run(self) -> None:
		"""
		Runs program with pre configured parameters
		"""
		pass

	@staticmethod
	def RMSE(observed: np.ndarray, targets: np.ndarray) -> np.float64:
		"""
		Returns root mean squared error.
		"""
		return np.sqrt(np.mean((observed-targets)**2))
