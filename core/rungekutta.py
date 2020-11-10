import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint

rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0

def rk(trange, y0, h):   
    i= int((trange[0]-trange[len(trange)-1])/h * -1)
    time_val = np.zeros(i+1)

    val = np.zeros((i+1, len(y0)))

    #initialize the time_val and s solutions
    val[0,0] = y0[0]
    val[0,1] = y0[1]
    val[0,2] = y0[2]

    time_val[0] = trange[0]

    for t in range(0, i) :
        k1x = h*fx(time_val[t], val[t,:])
        k1y = h*fy(time_val[t], val[t,:])
        k1z = h*fz(time_val[t], val[t,:])

        val[t, 0] = val[t,0] + (k1x/2)
        val[t, 1] = val[t,1] + (k1y/2)
        val[t, 2] = val[t,2] + (k1z/2)

        k2x = h*fx(time_val[t]+ h, val[t,:])
        k2y = h*fy(time_val[t]+ h, val[t,:])
        k2z = h*fz(time_val[t]+ h, val[t,:])

        val[t, 0] = val[t,0] + (k2x/2)
        val[t, 1] = val[t,1] + (k2y/2)
        val[t, 2] = val[t,2] + (k2z/2)

        k3x = h*fx(time_val[t]+ h, val[t,:])
        k3y = h*fy(time_val[t]+ h, val[t,:])
        k3z = h*fz(time_val[t]+ h, val[t,:])

        val[t, 0] = val[t,0] + (k3x/2)
        val[t, 1] = val[t,1] + (k3y/2)
        val[t, 2] = val[t,2] + (k3z/2)

        k4x = h*fx(time_val[t]+ h, val[t,:])
        k4y = h*fy(time_val[t]+ h, val[t,:])
        k4z = h*fz(time_val[t]+ h, val[t,:])


        time_val[t+1] = time_val[t] + h

        val[t+1, 0] = val[t, 0] + h*(1/6)*(k1x + k2x + k3x + k4x)
        val[t+1, 1] = val[t, 1] + h*(1/6)*(k1y + k2y + k3y + k4y)
        val[t+1, 2] = val[t, 2] + h*(1/6)*(k1z + k2z + k3z + k4z)


    return time_val, val


def fx(t, state):
    x, y, z = state  # Unpack the state vector
    return sigma * (y - x) # Derivatives

def fy(t, state):
    x, y, z = state  # Unpack the state vector
    return  x * (rho - z) - y  # Derivatives

def fz(t, state):
    x, y, z = state  # Unpack the state vector
    return x * y - beta * z  # Derivatives

state0 = [1.0, 1.0, 1.0] #initial constraints 
t = np.arange(0.0, 40.0, 0.1) #sets time to intervals of .01 from 0-40

times, val = rk(t, state0, 0.01)

fig = plt.figure()
ax = fig.gca(projection="3d")
ax.plot(val[:, 0], val[:, 1], val[:, 2])
plt.draw()
plt.show()

