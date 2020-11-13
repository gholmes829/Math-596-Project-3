"""

"""
import numpy as np
from itertools import product
from core.kalman import EnKF
from core.rungekutta import LorenzSolver
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Driver:
	def __init__(self) -> None:
		"""
		Has attributes like number of oberserved vars and stuf like that
		"""	
		# initializing
		self.n = 10
		self.step = 0.01
		self.iterations = int(self.n/self.step)-1

		# noise
		self.alpha, self.beta = 0.01, 0.01  # can switch bt 0.01, 0.1, and 1
	
		# model
		self.lorenz = LorenzSolver(self.step)
		self.y0 = np.array([1., 1., 1.])
		#print(self.iterations)
		self.t, self.truth = self.lorenz(self.y0, self.iterations)
		#print("Truth: ",self.truth)
		# observations
		self.observationProduct = np.array(list(product([0, 1], repeat=3)))[1:]  # 3d combinations of [1, 0] excluding [0, 0, 0]

		self.members = 5
		self.kalmanFilter = EnKF(self.lorenz, self.y0, self.beta, members=self.members)

		# logging
		self.states = np.zeros((self.iterations, 3))  # mean
		self.covariances = np.zeros((self.iterations, 3, 3))  # C

	def run(self) -> None:
		"""
		Runs program with pre configured parameters
		"""
		observation = [1, 1, 1]  # example for observing x and z
		H = np.array([[int(n == m) for n in range(3)] for m in range(3) if observation[m]])
		
		for t in range(self.iterations):
			#print(H, "\n\n", self.truth[0])
			y = H @ self.truth[t]
			self.kalmanFilter.predict(self.beta)
			self.kalmanFilter.update(y, H, self.alpha)

			self.states[t] = self.kalmanFilter.m_hat
			self.covariances[t] = self.kalmanFilter.C_hat

		
		fig = plt.figure()
		ax = fig.gca(projection="3d")
		ax.plot(*self.states.T)
		plt.draw()

		fig = plt.figure()
		ax = fig.gca(projection="3d")
		ax.plot(*self.truth.T, c="red")

		plt.show()

	@staticmethod
	def RMSE(observed: np.ndarray, targets: np.ndarray) -> np.float64:
		"""
		Returns root mean squared error.
		"""
		return np.sqrt(np.mean((observed-targets)**2))

