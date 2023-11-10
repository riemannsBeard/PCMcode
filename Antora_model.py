#%% IMPORTS

from scipy import signal, fftpack
from scipy.integrate import odeint, solve_ivp, cumtrapz, RK45
from scipy.interpolate import interp1d
from scipy.optimize import fsolve, newton, least_squares, root, minimize
from scipy import interpolate
import matplotlib.pyplot as plt
import math
import scipy as sp
import numpy as np
import csv
import matplotlib_inline
import warnings

from functions import *


warnings.filterwarnings('ignore')
matplotlib_inline.backend_inline.set_matplotlib_formats('svg', 'pdf')


#%% Graphite Cp
def cpG(T):
    
    return 4184*(0.54 + 9.11e-6*T - 90.27/T - 43449.3/T**2 + 1.593e7/T**3 -\
                 1.437e9/T**4)


#%% PROBLEM

L = 44
eps = 0.22
dx = 2.2e-2
xf = L
x = np.arange(0, xf + dx, dx)
dx = x[1] - x[0]

Nx = len(x) - 1

tf = 24*3600
dt = 3.6
t = np.arange(0, tf + dt, dt)/3600
Nt = len(t)

x = np.arange(0, xf + dx, dx)

d = 0.015
mDot = 0.04

rhos = 1800
ks = 120

Tin = 800 + 273

Tf = np.zeros((Nt, Nx+1))
Ts = np.zeros((Nt, Nx+1))

# BCs
Tf[:,0] = Tin
bc1 = np.zeros(Nx)

# ICs
T00 = 300
Tf[0,:] = T00
Ts[0,:] = T00


for i in range(1,len(t)):
    
    # ODE solution -- THERMAL BUFFER
       
    
    # --------------------------------------------------------------------------------------------------- #
    # --------------------------------------------------------------------------------------------------- #
    # --------------------------------------------------------------------------------------------------- #
    
    # Thermocline TANK -- fluid properties changing with temperature
    
    alpha = eps*rhoA(Tf[i-1,:])*cp(Tf[i-1,:])
    beta = eps*kair(Tf[i-1,:])
    
    gamma = (1-eps)*rhos*cpG(Tf[i-1,:])
    betas = ks*(1 - eps)
    
    u = mDot/(rhoA(Tf[i-1,:])*eps*np.pi*L**2)

    Rep = rhoA(Tf[i-1,:])*d*u/mu(Tf[i-1,:])
    Pr = cp(Tf[i-1,:])*mu(Tf[i-1,:])/kair(Tf[i-1,:])
    Nu = 0.664*Rep**0.5*Pr**0.5
    a = Nu*kair(Tf[i-1,:])/d
    h = 6*(1-eps)*beta*(2 + 1.1*Rep**(0.6)*Pr**(1/3))/(d**2)

    p = dt*u/dx
    q = 2*beta/alpha*0.5*dt/(dx**2)
    r = h*dt/alpha

    rs = dt*h/gamma
    qs = 2*betas*0.5*dt/(gamma*dx**2)
    
    diagonals = [np.ones(Nx)*(1 + 2*q + r + p), np.ones(Nx)*(-q), np.ones(Nx)*(-p-q)]
    offsets = [0, 1, -1]
    A.setdiag(diagonals[0], offsets[0])
    A.setdiag(diagonals[1], offsets[1])
    A.setdiag(diagonals[2], offsets[2])
    A[-1,-2] = -2*q - p
    
    diagonals = [np.ones(Nx + 1)*(1 + 2*qs + rs), np.ones(Nx + 1)*(-qs), np.ones(Nx + 1)*(-qs)]
    offsets = [0, 1, -1]
    As.setdiag(diagonals[0], offsets[0])
    As.setdiag(diagonals[1], offsets[1])
    As.setdiag(diagonals[2], offsets[2])
    As[-1,-2] = -2*qs
    As[0,1] = -2*qs  
        
    # PDE solution -- TANK
    
    bc1[0] = Tf[i-1,0]*T0

    Ts[i,:] = sp.sparse.linalg.spsolve(As, Ts[i-1,:] + rs*Tf[i-1,:])
    Tf[i,1:] = sp.sparse.linalg.spsolve(A, Tf[i-1,1:] + (q + p)*bc1 + \
                r*Ts[i-1,1:])
    
    
    # Heat transfer accross molten salts
    qf[i] = 0.25*u*np.pi*d**2*eps*rhof*(Tf[i,15] - Tf[i,-15])*cpf
    
    # theta6[i,k] = Tf[i,-1]/T0