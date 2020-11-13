"""

"""
import numpy as np

class EnKF:
	def __init__(self, members: int=100) -> None:
		"""
		agents: number of ensemble members
		"""
		self.members = members

		# initialize
		self.v_hat = None
		self.m_hat = None
		self.C_hat = None
		self.H_hat = None
		#self.H = np.array([[int(n == m) for n in range(3)] for m in range(3) if observations[m]])
		#self.v = np.zeros(self.model.shape[0])
		#self.y = np.zeros(self.H.shape[0])		

		# prediction
		#self.m = np.zeros(self.model.shape[0])
		#self.C = np.zeros((self.model.shape[0], self.model.shape[0]))
		
		# analysis
		#self.S = np.zeros((self.H.shape[0], self.H.shape[0]))
		#self.K = np.zeros((self.model.shape[0], self.H.shape[0]))
	
		# model data
		#t, f = solver(model, v0, self.trange, self.frequency)	
		#self.v_hat = f + KalmanEnsemble.noise(0, self.alpha, (len(f), 3))	

	def predict(self, V, beta):
		self.v_hat = np.array([V + np.random.normal(0, beta, V.shape[0]) for _ in range(self.members)])
		#print("v_hat: ", self.v_hat.shape)
		self.m_hat = self.v_hat.mean(axis=0)
		#print("m_hat: ", self.m_hat.shape)
		# print(((self.v_hat-self.m_hat)@(self.v_hat-self.m_hat).T).shape)

		#c
		#c = np.array([(self.v_hat[i]-self.m_hat) @ (self.v_hat[i]-self.m_hat) for i in range(len(self.v_hat))])
		#print(c)
		##print(c.shape)
		#print((self.v_hat[0]-self.m_hat)[np.newaxis].T @ (self.v_hat[0]-self.m_hat)[np.newaxis])
		self.C_hat = np.array([(self.v_hat[i]-self.m_hat)[np.newaxis].T @ (self.v_hat[i]-self.m_hat)[np.newaxis] for i in range(self.members)]).sum(axis=0)/(self.members-1)
		#print("c_hat: ", self.C_hat.shape)
	
	def update(self, y, H, alpha):
		#print(self.C_hat)
		#print("H: ", H.shape)
		self.S = H @ self.C_hat @ H.T + alpha*np.eye(H.shape[0])
		#print("S: ", self.S.shape)
		self.K = self.C_hat @ H.T @ np.linalg.inv(self.S)
		#print("K: ", self.K.shape)
		self.y = np.array([y + np.random.normal(0, alpha, H.shape[0]) for i in range(self.members)]) 
		#print("y: ", self.y.shape)
		#print(self.v_hat.shape)
		#print(((np.eye(self.v_hat.shape[1]) - self.K @ H) @ self.v_hat[0]).shape)
		#print(self.K.shape, y[
		#print((self.K @ y[np.newaxis]).shape)
		self.v = np.array([(np.eye(self.v_hat.shape[1]) - self.K @ H) @ self.v_hat[i] + self.K @ self.y[i][np.newaxis].T for i in range(self.members)])
		#print("v: ", self.v.shape)
		
