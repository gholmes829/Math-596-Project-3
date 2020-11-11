import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rungeKutta(df: np.ndarray, y0: np.ndarray, trange: tuple, h: float) -> (np.ndarray, np.ndarray):
	"""
	4th order Runge Kutta to solve ODEs.

	df: array of ODE equations for each dimension.
	y0: initial conditions.
	trange: time range to take samples from.
	h: time step

	returns: time steps, f(steps)
	"""
	dimensions = y0.shape[0]
	k = [[0 for dimension in range(dimensions)] for order in range(4)]
	n = int((trange[1]-trange[0])/h)
	t, f = np.zeros(n+1), np.zeros((n+1, dimensions))

	t[0], f[0] = trange[0], y0

	for i in range(n):
		for order in range(3):
			k[order] = [h*df[j](t[i], f[i,:]) for j in range(dimensions)]
			for j in range(dimensions):
				f[i, j] += k[order][j]/2

		k[3] = [h*df[j](t[i], f[i,:]) for j in range(dimensions)]
		t[i+1] = t[i] + h
	
		for j in range(dimensions):
			f[i+1, j] = f[i, j] + h*(1/6)*sum([order[j] for order in k])
	return t, f

def plotLorenzAttractor() -> None:
	"""
	Plots solutions to Lorenz Attractor using 4th order Runge Kutta.
	"""
	rho, sigma, beta = 28.0, 10.0, 8.0 / 3.0
	lorenz = np.array([
			lambda t, state: sigma * (state[1]-state[0]),
			lambda t, state: state[0] * (rho - state[2]) - state[1],
			lambda t, state: state[0] * state[1] - beta * state[2],
		])

	y0 = np.array([1.0, 1.0, 1.0])
	trange = (0., 50.)


	t, f = rungeKutta(lorenz, y0, trange, 0.01)

	fig = plt.figure()
	ax = fig.gca(projection="3d")
	ax.plot(*f.T)
	plt.draw()
	plt.show()

