"""

"""
import numpy as np
from itertools import product
from core.kalman import KalmanEnsemble
from core.rungekutta import rungeKutta

class Driver:

	def __init__(self) -> None:
		"""
		Has attributes like number of oberserved vars and stuf like that
		"""	
		rho, sigma, beta = 28.0, 10.0, 8.0 / 3.0
		self.lorenz = np.array([
			lambda t, state: sigma * (state[1]-state[0]),
			lambda t, state: state[0] * (rho - state[2]) - state[1],
			lambda t, state: state[0] * state[1] - beta * state[2],
			])

		self.observations = np.array(list(product([0, 1], repeat=3)))[1:]  # 3d combinations of [1, 0] excluding [0, 0, 0]
		self.filter = KalmanEnsemble(self.lorenz, rungeKutta, self.observations[4])

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


