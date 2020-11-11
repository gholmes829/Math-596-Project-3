"""

"""
import numpy as np

class KalmanEnsemble:

	def __init__(self, model, solver, observations):
		self.model = model
		self.observations = observations
		self.numObservations = sum(self.observations)
		self.H = np.array([[int(n == m) for n in range(3)] for m in range(3) if observations[m]])

