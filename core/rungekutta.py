import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint

rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0

def f(state, t):
    x, y, z = state  # Unpack the state vector
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivatives

state0 = [1.0, 1.0, 1.0] #initial constraints 
t = np.arange(0.0, 40.0, 0.01) #sets time to intervals of .01 from 0-40

#diff eq solver. func f subject to state0 initialization at time int t

states = odeint(f, state0, t) 

# fig = plt.figure()
# ax = fig.gca(projection="3d")
# ax.plot(states[:, 0], states[:, 1], states[:, 2])
# plt.draw()
# plt.show()


#4th order runge kutta is (cred to https://mathworld.wolfram.com/Runge-KuttaMethod.html)
# k_1	=	hf(x_n,y_n)	
# k_2	=	hf(x_n+1/2h,y_n+1/2k_1)	
# k_3	=	hf(x_n+1/2h,y_n+1/2k_2)	
# k_4	=	hf(x_n+h,y_n+k_3)	
# y_(n+1)	=	y_n+1/6k_1+1/3k_2+1/3k_3+1/6k_4+O(h^5)


#https://lpsa.swarthmore.edu/NumInt/NumIntFourth.html


#need to deal with fn, how to make all soln based on fn using x,y,z not just one var
def rk(fn, trange, y0, h): 
    i= int((trange[0]-trange[1])/h * -1)
    time_val = np.zeros(i+1)
    y = np.zeros((i+1, len(y0)))

    #initialize the time_val and y solutions
    y[0,:] = y0
    time_val[0] = trange[0]

    for t in range(0, i) :
        k1 = fn(time_val[t], y[t,:])
        k2 = fn(time_val[t]+ (h*k1/2), y[t,:] + (h/2)  )
        k3 = fn(time_val[t]+ (h*k2/2), y[t,:] + (h/2)  )
        k4 = fn(time_val[t]+ (h*k3), y[t,:] + h)

        time_val[t+1] = time_val[t] + h
        y[t+1] = y[t] + h*(k1 + 2*k2 + 2*k3 + k4)/6

    return time_val


