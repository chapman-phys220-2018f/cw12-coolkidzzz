###
# Name:Gabriella Nutt
#Student ID: 2307512
#Email: nutt@chapman.edu
# Name:Raha Pirzadeh
#Student ID: 2290732
#Email: pirzadeh@chapman.edu
#Course: PHYS220/MATH220/CPSC220 Fall 2018
#Assignment: CW12
###

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import numba as nb

@nb.jit
def dxdt(x,y,t):
    """"""
    return y

@nb.jit
def dydt(x,y,t,F,v,w):
    """creates a function of the duffing oscillator"""
    
    dy = -x**3+x-v*y + F*(np.cos(w*t))
    return dy

@nb.jit
def r4(x0,y0,F,v,w,t):
    """function for 4th-order Runge-Kutta Method of Duffing Oscillator"""
    dt = t[1]-t[0]
    #t = np.arange(0,(2*np.pi),dt)
    x = np.zeros_like(t)
    x[0] = x0
    y = np.zeros_like(t)
    y[0] = y0
    for i in range(0,len(t)-1):
        K1x= dt*dxdt(x[i],y[i],t[i])
        K1y= dt*dydt(x[i],y[i],t[i],F,v,w)

        K2x = dt*dxdt(x[i]+K1x/2, y[i]+K1y/2, t[i] + dt/2)
        K2y = dt*dydt(x[i]+K1x/2, y[i]+K1y/2, t[i] + dt/2,F,v,w)

        K3x = dt*dxdt(x[i]+K2x/2, y[i]+K2y/2, t[i] + dt/2)
        K3y = dt*dydt(x[i]+K2x/2, y[i]+K2y/2, t[i] + dt/2,F,v,w)

        K4x = dt*dxdt(x[i]+K3x, y[i]+K3y, t[i] + dt)
        K4y = dt*dydt(x[i]+K3x, y[i]+K3y, t[i] + dt,F,v,w)
        x[i+1]= x[i]+(K1x+2*K2x+2*K3x+K4x)/6
        y[i+1]= y[i]+(K1y+2*K2y+2*K3y+K4y)/6
    return x,y

def r4plot (x,y,t):
    """"this plots the 4th-order Runge-Kutta Method of the Duffing oscillator!"""

    plt.plot(t,x,color="k",label='r4 vs x') # x is the 0 row of r4
    plt.plot(t,r4[1,:],color="blue",label='r4 vs y') # y is the 1st row of r4
    plt.plot(t,sin,'r--',color="orange",label='-sinx')
    plt.legend(loc='upper left') #puts legend in upper left corner
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Plot of Duffing Oscillator')
    plt.ylim((-5,5))
    plt.show()
