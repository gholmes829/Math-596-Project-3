"""

"""
import numpy as np
from itertools import product
from core.kalman import EnKF
from core.rungekutta import RK4
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Driver:

	def __init__(self) -> None:
		"""
		Has attributes like number of oberserved vars and stuf like that
		"""	
		# initializing
		self.trange = (0., 40.)
		self.frequency = 0.01
		self.n = int((self.trange[1]-self.trange[0])/self.frequency)-1

		# noise
		self.alpha, self.beta = 1, 1  # can switch bt 0.01, 0.1, and 1
	
		# model
		rho, sigma, beta = 28.0, 10.0, 8.0 / 3.0
		self.lorenz = np.array([
			lambda t, state: sigma * (state[1]-state[0]),
			lambda t, state: state[0] * (rho - state[2]) - state[1],
			lambda t, state: state[0] * state[1] - beta * state[2],
			])
		self.y0 = np.array([1., 1., 1.])
	
		self.t, self.model = RK4(self.lorenz, self.y0, self.trange, self.frequency)

		# observations
		self.observations = np.array(list(product([0, 1], repeat=3)))[1:]  # 3d combinations of [1, 0] excluding [0, 0, 0]
		self.H = None

		self.members = 5
		self.kalmanFilter = EnKF(members=self.members)

		# logging
		self.states = np.zeros((self.n, 3))  # mean
		self.uncertainty = np.zeros((self.n, 3, 3))  # C		

		# testing

		#print(self.trange, self.n, len(self.truth), self.t)
			

	def run(self) -> None:
		"""
		Runs program with pre configured parameters
		"""
		observation = [1, 0, 1]  # example for observing x and z
		self.H = np.array([[int(n == m) for n in range(3)] for m in range(3) if observation[m]])
		#print(self.model.shape)
		for t in range(self.n):
			y = self.H @ self.model[t]
			self.kalmanFilter.predict(self.model[t], self.beta)
			self.kalmanFilter.update(y, self.H, self.alpha)

			self.states[t] = self.kalmanFilter.m_hat
			self.uncertainty[t] = self.kalmanFilter.C_hat

		fig = plt.figure()
		ax = fig.gca(projection="3d")
		ax.plot(*self.states.T)
		plt.draw()

		fig = plt.figure()
		ax = fig.gca(projection="3d")
		ax.plot(*self.model.T, c="red")

		plt.show()

	@staticmethod
	def RMSE(observed: np.ndarray, targets: np.ndarray) -> np.float64:
		"""
		Returns root mean squared error.
		"""
		return np.sqrt(np.mean((observed-targets)**2))

