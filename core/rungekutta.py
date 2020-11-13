import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def RK4(df: np.ndarray, y0: np.ndarray, trange: tuple, h: float) -> (np.ndarray, np.ndarray):
	"""
	4th order Runge Kutta to solve ODEs.

	df: array of ODE equations for each dimension.
	y0: initial conditions.
	trange: time range to take samples from. If tuple, returns range, if float returns single value.
	h: time step

	returns: time steps, f(steps)
	"""
	n = int((trange[1]-trange[0])/h)-1
	dimensions = y0.shape[0]

	k = [[0 for dimension in range(dimensions)] for order in range(4)]
	t, f = np.zeros(n+1), np.zeros((n+1, dimensions))
	t[0], f[0] = trange[0], y0

	for i in range(n):
		for j in range(dimensions):
			k[0][j] = h*df[j](t[i], f[i,:])
			k[1][j] = h*df[j](t[i] + 0.5*h, f[i,:] + 0.5*k[0][j])
			k[2][j] = h*df[j](t[i] + 0.5*h, f[i,:] + 0.5*k[1][j])
			k[3][j] = h*df[j](t[i] + h, f[i,:] + k[2][j])
			f[i+1, j] = f[i, j] + (1/6)*(k[0][j]+k[1][j]+2*k[2][j]+2*k[3][j])
		t[i+1] = t[i] + h

	return t, f

def plotLorenzAttractor() -> None:
	"""
	Plots solutions to Lorenz Attractor using 4th order Runge Kutta.
	"""
	rho, sigma, beta = 28.0, 10.0, 8.0/3.0
	lorenz = np.array([
			lambda t, state: sigma * (state[1]-state[0]),
			lambda t, state: state[0] * (rho - state[2]) - state[1],
			lambda t, state: state[0] * state[1] - beta * state[2],
		])

	y0 = np.array([1., 1., 1.])
	trange = (0., 50.)
	t, f = RK4(lorenz, y0, trange, 0.01)
	
	fig = plt.figure()
	ax = fig.gca(projection="3d")
	ax.plot(*f.T)
	plt.draw()
	plt.show()

