"""

"""
import numpy as np
from itertools import product
from core.rungekutta import rk
# import stuff from core here
#from core.module import ...

class Driver:

	def __init__(self) -> None:
		"""
		Has attributes like number of oberserved vars and stuf like that
		"""
		self.observations = np.array(list(product([0, 1], repeat=3)))[1:]  # combinations of [1, 0] excluding [0, 0, 0]

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


