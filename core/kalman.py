"""

"""
import numpy as np

class EnKF:
	def __init__(self, model, v0, beta, members: int=100) -> None:
		"""
		agents: number of ensemble members
		"""
		self.model = model
		self.members = members

		# initialize
		self.v = np.array([v0 + np.random.normal(0, beta, v0.shape[0]) for _ in range(self.members)])
		self.v_hat = None
		self.m_hat = None
		self.C_hat = None
		self.H_hat = None

	def predict(self, beta):
		self.v_hat = np.array([self.model(self.v[i])[1] + np.random.normal(0, beta, self.v.shape[1]) for i in range(self.members)])
		self.m_hat = self.v_hat.mean(axis=0)
		self.C_hat = np.array([(self.v_hat[i]-self.m_hat)[np.newaxis].T @ (self.v_hat[i]-self.m_hat)[np.newaxis] for i in range(self.members)]).sum(axis=0)/(self.members-1)
	
	def update(self, y, H, alpha):
		self.S = H @ self.C_hat @ H.T + alpha*np.eye(H.shape[0])
		self.K = self.C_hat @ H.T @ np.linalg.inv(self.S)
		self.y = np.array([y + np.random.normal(0, alpha, H.shape[0]) for i in range(self.members)]) 
		self.v = np.array([(np.eye(self.v_hat.shape[1]) - self.K @ H) @ self.v_hat[i] + self.K @ self.y[i] for i in range(self.members)])
		
