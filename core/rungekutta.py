"""
Using RK4 to solve ODEs.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class LorenzSolver:
	"""
	Solve '63 Lorenz equation
	"""
	def __init__(self, h: int=0.01) -> None:
		self.h = h
		rho, sigma, beta = 28.0, 10.0, 8.0/3.0
		self.df = np.array([
			lambda t, state: sigma * (state[1]-state[0]),
			lambda t, state: state[0] * (rho - state[2]) - state[1],
			lambda t, state: state[0] * state[1] - beta * state[2],
			])
	
	def __call__(self, state: np.ndarray, n: int=1) -> any:
		"""
		Example:
		solver = LorenzSolver()
		t, f = solver([1., 1., 1.], 100) 
		"""
		if n==1:
			return RK4(self.df, state, self.h)
		else:
			return RK4(self.df, state, self.h, (0, n*self.h))

def RK4(df: np.ndarray, y0: np.ndarray, h: float, trange: tuple=()) -> any:
	"""
	4th order Runge Kutta to solve ODEs.

	df: array of ODE equations for each dimension.
	y0: initial conditions.
	trange: time range to take samples from. If tuple, returns range, if float returns single value.
	h: time step

	returns: time steps, f(steps)
	"""
	n = int((trange[1]-trange[0])/h)-1 if trange != () else 1
	dimensions = y0.shape[0]
	k = [[0 for dimension in range(dimensions)] for order in range(4)]
	t, f = np.zeros(n+1), np.zeros((n+1, dimensions))
	t[0], f[0] = trange[0] if trange != () else 0, y0
	
	for i in range(n):
		for j in range(dimensions):
			k[0][j] = h*df[j](t[i], f[i,:])
			k[1][j] = h*df[j](t[i] + 0.5*h, f[i,:] + 0.5*k[0][j])
			k[2][j] = h*df[j](t[i] + 0.5*h, f[i,:] + 0.5*k[1][j])
			k[3][j] = h*df[j](t[i] + h, f[i,:] + k[2][j])
			f[i+1, j] = f[i, j] + (1/6)*(k[0][j]+2*k[1][j]+2*k[2][j]+k[3][j])
		t[i+1] = t[i] + h
	return (t, f) if trange!=() else (0, f[1])

def plotLorenzAttractor() -> None:
	"""
	Plots solutions to Lorenz Attractor using 4th order Runge Kutta.
	"""
	plt.style.use(["dark_background"])
	plt.rc("grid", linestyle="dashed", color="white", alpha=0.25)

	rho, sigma, beta = 28.0, 10.0, 8.0/3.0
	lorenz = np.array([
			lambda t, state: sigma * (state[1]-state[0]),
			lambda t, state: state[0] * (rho - state[2]) - state[1],
			lambda t, state: state[0] * state[1] - beta * state[2],
		])

	y0 = np.array([1., 1., 1.])
	trange = (0., 30.)
	t, f = RK4(lorenz, y0, 0.01, trange)
	
	fig = plt.figure()
	ax = fig.gca(projection="3d")
	ax.plot(*f.T, c="red")
	plt.draw()
	plt.show()

