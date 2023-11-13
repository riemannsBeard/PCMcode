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

#%% DEFAULTS

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Times'
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['lines.markersize'] = 3


#%% Graphite Cp
def cpG(T):
    
    return 4184*(0.54 + 9.11e-6*T - 90.27/T - 43449.3/T**2 + 1.593e7/T**3 -\
                 1.437e9/T**4)


#%% PROBLEM

L = 1
eps = 0.22
dx = 2.2e-2
xf = L
x = np.arange(0, xf + dx, dx)
dx = x[1] - x[0]

Nx = len(x) - 1

tf = 24*3600
dt = 0.36 #3.6
t = np.arange(0, tf + dt, dt)/3600
Nt = len(t)

x = np.arange(0, xf + dx, dx)

d = 0.015
mDot = 0.04

rhos = 1800
ks = 120

Tin = 51 + 273

Tf = np.zeros((Nt, Nx+1))
Ts = np.zeros((Nt, Nx+1))

# BCs
Tf[:,0] = Tin
bc1 = np.zeros(Nx)

# ICs
T00 = 800 + 273
Tf[0,:] = T00
Ts[0,:] = T00


#%% MATRIX ASSEMBLY

rhof = 5*101325/(287*T00)

alpha = eps*rhof*cp(T00)
beta = eps*kair(T00)
gamma = (1-eps)*rhos*cpG(T00)
betas = ks*(1 - eps)

u = mDot/(rhof*eps*np.pi*L**2)

Rep = rhof*d*u/mu(T00)
Pr = cp(T00)*mu(T00)/kair(T00)
Nu = 0.664*Rep**0.5*Pr**0.5
a = Nu*kair(T00)/d
#h = 6*(1-eps)*beta*(2 + 1.1*Rep**(0.6)*Pr**(1/3))/(d**2)

h = a*6*(1-eps)/d

p = dt*u/dx
q = 2*beta/alpha*0.5*dt/(dx**2)
r = h*dt/alpha

rs = dt*h/gamma
qs = 2*betas*0.5*dt/(gamma*dx**2)



# Fluid
diagonals = [np.ones(Nx)*(1 + 2*q + r + p), np.ones(Nx)*(-q), np.ones(Nx)*(-p-q)]
offsets = [0, 1, -1]
A = sp.sparse.diags(diagonals, offsets).toarray()
A[-1,-2] = -2*q - p

A = sp.sparse.csr_matrix(A)


# Solid
diagonals = [np.ones(Nx + 1)*(1 + 2*qs + rs), np.ones(Nx + 1)*(-qs), np.ones(Nx + 1)*(-qs)]
offsets = [0, 1, -1]
As = sp.sparse.diags(diagonals, offsets).toarray()
As[-1,-2] = -2*qs
As[0,1] = -2*qs

As = sp.sparse.csr_matrix(As)



#%% SOLUTION

for i in range(1,len(t)):
    
    # ODE solution -- ANTORA
       
    
    # --------------------------------------------------------------------------------------------------- #
    # --------------------------------------------------------------------------------------------------- #
    # --------------------------------------------------------------------------------------------------- #
    
    # ANTORA TANK -- fluid properties changing with temperature
    
    # Air density at 5 bar
    rhof = 5*101325/(287*T00)
    
    alpha = eps*rhof*cp(T00)
    beta = eps*kair(T00)
    
    gamma = (1-eps)*rhos*cpG(T00)
    betas = ks*(1 - eps)
    
    u = mDot/(rhof*eps*np.pi*L**2)

    Rep = rhof*d*u/mu(T00)
    Pr = cp(T00)*mu(T00)/kair(T00)
    Nu = 0.664*Rep**0.5*Pr**0.5
    a = Nu*kair(T00)/d
    h = 6*(1-eps)*beta*(2 + 1.1*Rep**(0.6)*Pr**(1/3))/(d**2)

    p = dt*u/dx
    q = 2*beta/alpha*0.5*dt/(dx**2)
    r = h*dt/alpha

    rs = dt*h/gamma
    qs = 2*betas*0.5*dt/(gamma*dx**2)
    
    diagonals = [np.ones(Nx)*(1 + 2*q + r + p), np.ones(Nx)*(-q),
                 np.ones(Nx)*(-p-q)]
    offsets = [0, 1, -1]
    A.setdiag(diagonals[0], offsets[0])
    A.setdiag(diagonals[1], offsets[1])
    A.setdiag(diagonals[2], offsets[2])
    A[-1,-2] = -2*q - p
    
    diagonals = [np.ones(Nx + 1)*(1 + 2*qs + rs), np.ones(Nx + 1)*(-qs),
                 np.ones(Nx + 1)*(-qs)]
    offsets = [0, 1, -1]
    As.setdiag(diagonals[0], offsets[0])
    As.setdiag(diagonals[1], offsets[1])
    As.setdiag(diagonals[2], offsets[2])
    As[-1,-2] = -2*qs
    As[0,1] = -2*qs  
        
    # PDE solution -- TANK
    
    bc1[0] = Tin #Tf[i-1,0]
    

    Ts[i,:] = sp.sparse.linalg.spsolve(As, Ts[i-1,:] + rs*Tf[i-1,:])
    Tf[i,1:] = sp.sparse.linalg.spsolve(A, Tf[i-1,1:] + (q + p)*bc1 + \
                r*Ts[i-1,1:])
    
    
    # Heat transfer accross molten salts
    # qf[i] = 0.25*u*np.pi*d**2*eps*rhof*(Tf[i,15] - Tf[i,-15])*cpf
    
    # theta6[i,k] = Tf[i,-1]/T0
    
    
    
#%% PLOTS

fig, ax = plt.subplots()

cs1 = ax.contourf(x, t, Tf-273, 32, extend='both')
plt.tight_layout()
cbar1 = plt.colorbar(cs1)
cbar1.set_label(r'$T_f$ ($^\circ C$)', fontsize=16)
ax.set_ylabel(r'$t$ (h)')
ax.set_xlabel(r'$L$ (m)')
ax.set_box_aspect(1)

plt.show()


#%% PLOTS
fig, ax = plt.subplots()

plt.plot(t, Tf[:,int(len(x)/4)]-273, t, Tf[:,int(len(x)/2)]-273,
         t, Tf[:,int(3*len(x)/4)]-273, t, Tf[:,-1]-273)
plt.ylabel(r'$T_f$ ($^\circ C$)')
plt.xlabel(r'$t$ (h)')
plt.legend([r'$x = 1/4$', r'$x = 1/2$', r'$x = 3/4$', r'$x = 1$'])

plt.show()


#%% PLOTS
fig, ax = plt.subplots()

plt.plot(x, Tf[int(len(x)/6),:]-273, x, Tf[int(2*len(x)/6),:]-273,
         x, Tf[int(3*len(x)/6),:]-273, x, Tf[int(4*len(x)/6),:]-273,
         x, Tf[int(5*len(x)/6),:]-273)
plt.ylim([780, 810])
plt.ylabel(r'$T_f$ ($^\circ C$)')
plt.xlabel(r'$x$ (m)')
plt.legend([r'$t = 4$ h', r'$t = 8$ h', r'$t = 12$ h', r'$t = 16$ h',
            r'$t = 20$ h', r'$t = 24$ h'])

plt.show()


#%% PLOTS
ff = 10*np.exp(-0.5*(x - L/2)**2/(0.025**2))
plt.plot(x, ff)

plt.show()
