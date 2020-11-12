"""

"""
import numpy as np

class KalmanEnsemble:

	def __init__(self, agents: int, model: np.ndarray, solver: callable, observations: np.ndarray) -> None:
		self.agents = agents
		self.model = model
		self.solver = solver
		self.observations = observations

		
		# SOMEONE PLEASE CHECK, IDK IF THESE ARE RIGHT
		#initialize
		self.H = np.array([[int(n == m) for n in range(3)] for m in range(3) if observations[m]])	
		self.v = np.zeros((2, self.model.shape[0]))
		self.y = np.zeros((2, self.H.shape[0]))		

		# prediction
		self.m = np.zeros(self.v.shape[0])
		self.C = np.zeros((self.v.shape[0], self.model.shape[0]))
		
		# analysis
		self.S = np.zeros((self.model.shape[0], self.model.shape[0]))
		self.K = np.zeros((self.v.shape[0],  self.model.shape[0]))

	def predict(self):
		pass

	def analyze(self):
		pass

	@staticmethod
	def noise(mean: np.ndarray, std: np.ndarray, size: tuple) -> np.ndarray:
		return np.random.normal(mean, std, size)
	
