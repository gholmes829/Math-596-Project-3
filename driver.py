"""
Driver and helper functions.
"""

import numpy as np
from random import choice
from itertools import product
from core.kalman import EnKF
from core.rungekutta import LorenzSolver
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def RMSE(observed: np.ndarray, targets: np.ndarray) -> np.float64:
	"""
	Returns root mean squared error.
	"""
	return np.sqrt(np.mean((observed-targets)**2))

def dist(pt1: np.ndarray, pt2: np.ndarray) -> np.float64:
	"""
	Euclidean distance.
	"""
	return np.sqrt(np.sum((pt2-pt1)**2))

def init():
	plt.style.use(["dark_background"])
	plt.rc("grid", linestyle="dashed", color="white", alpha=0.25)

class Driver:
	def __init__(self) -> None:
		"""
		Runs main code
		"""	
		# initializing
		init()		
	
		self.n = 15
		self.step = 0.01
		self.iterations = int(self.n/self.step)-1

		# noise
		self.alpha, self.beta, self.gamma = 2., 2., 2.  # observation, model, and initial condition error
	
		# model
		self.lorenz = LorenzSolver(self.step)
		self.y0 = np.array([1., 1., 1.])
		
		# generate truth and combinations of test inputs
		self.t, self.truth = self.lorenz(self.y0, self.iterations)
		self.observationProduct = np.array(list(product([0, 1], repeat=3)))[1:]  # 3d combinations of [1, 0] excluding [0, 0, 0]

		# filter
		self.members = 10
		self.kalmanFilter = EnKF(self.lorenz, self.y0, self.gamma, members=self.members) 

		# logging
		self.states = np.zeros((self.iterations, 3))  # mean
		self.covariances = np.zeros((self.iterations, 3, 3))  # C

	def run(self) -> None:
		"""
		Runs program with sample settings.
		"""
		for observation in self.observationProduct:
			H = np.array([[int(n == m) for n in range(3)] for m in range(3) if observation[m]])
			printed = -1		
			print("Generating with observation: " + str(observation))

			for t in range(self.iterations):
				if int(100*(t+1)/self.iterations)%5==0 and int(100*(t+1)/self.iterations) != printed:
					print(str(t+1) + "/" + str(self.iterations) + " " + str(int(100*(t+1)/self.iterations)) + "%")
					printed = int(100*(t+1)/self.iterations)

				y = H @ self.truth[t]
				self.kalmanFilter.predict(self.beta)
				self.kalmanFilter.update(y, H, self.alpha)

				self.states[t] = self.kalmanFilter.m_hat
				self.covariances[t] = self.kalmanFilter.C_hat
		
			print("\nDone!")
		
			# plotting
			varMap = {0: "x", 1: "y", 2: "z"}			

			fig, axes = plt.subplots(4, sharex=True)
			for i in range(3):
				axes[i].plot(self.t, self.truth[:,i], c="blue", ms=5)
				axes[i].plot(self.t, self.states[:,i], ".", ms=2.5, c="red", alpha=0.5)
				axes[i].title.set_text("Coord: " + varMap[i])
				axes[i].grid()	
			
			axes[3].plot(self.t, [dist(self.states[i], self.truth[i]) for i in range(self.iterations)], c="green")
			axes[3].set_xlabel("Time")
			axes[3].set_ylabel("2-Norm")
			plt.tight_layout()
			plt.grid()	
			fig.set_size_inches(12, 6)
			plt.savefig("figures/" + str(observation[0]) + str(observation[1]) + str(observation[2]), bbox_inches="tight")		
			#self.plot()
			#plt.show()
	
			self.kalmanFilter = EnKF(self.lorenz, self.y0, self.gamma, members=self.members) 

	def plot(self):
		# kalman produced graph
		fig = plt.figure()
		ax = fig.gca(projection="3d")
		ax.plot(*self.states.T)
		plt.draw()

