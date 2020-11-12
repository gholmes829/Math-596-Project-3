"""

"""
import numpy as np

class KalmanEnsemble:

	def __init__(self, agents: int, model: np.ndarray, solver: callable, observations: np.ndarray) -> None:
		self.agents = agents
		self.model = model
		self.solver = solver
		self.observations = observations
	
		self.state = None
		self.H = np.array([[int(n == m) for n in range(3)] for m in range(3) if observations[m]])	

	def predict(self):
		pass

	def analyze(self):
		pass

	@staticmethod
	def noise(mean: np.ndarray, std: np.ndarray, size: tuple) -> np.ndarray:
		return np.random.normal(mean, std, size)
	
