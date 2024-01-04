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
from dishLib import *

warnings.filterwarnings('ignore')
# matplotlib_inline.backend_inline.set_matplotlib_formats('svg', 'pdf')

#%% DEFAULTS

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ["Computer Modern Roman"]
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['lines.markersize'] = 3


#%% DISH STUFF

fig, ax1 = plt.subplots()
ax1.plot(t, Ib_march*1e-3, t, Ib_june*1e-3, t, Ib_dec*1e-3)
plt.legend([r'March', r'June', r'Dec.'], loc='upper left')
ax1.set_xlabel(r'$t$ (h)')
ax1.set_ylabel(r'$I_b$ (kW)')

fig, ax1 = plt.subplots()
ax1.plot(t, Tamb_march - 273, t, Tamb_june - 273, t, Tamb_dec - 273)
plt.legend([r'March', r'June', r'Dec.'], loc='upper left')
ax1.set_xlabel(r'$t$ (h)')
ax1.set_ylabel(r'$T$ ($^\circ$C)')


#%% Graphite Cp
def cpG(T):
    
    return 4184*(0.54 + 9.11e-6*T - 90.27/T - 43449.3/T**2 + 1.593e7/T**3 -\
                 1.437e9/T**4)

#%% PROBLEM

L = 10 #21.25 #6.1
eps = 0.22
dx = 2.2e-2
xf = L
x = np.arange(0, xf + dx, dx)
dx = x[1] - x[0]

Nx = len(x) - 1

tf = 24*3600
dt = 3.6
t = np.arange(0, tf + dt, dt)
Nt = len(t)

x = np.arange(0, xf + dx, dx)

d = 0.015
mDot = 0.04

rhos = 1800
ks = 120

#Tin = 100 + 273
Tin = np.interp(t, ts*3600, To[:, 0])
Tin[Tin != 0] += 273
Tin[Tin < 500] = 500

Tf = np.zeros((Nt, Nx+1))
Ts = np.zeros((Nt, Nx+1))

# ICs
T00 = 800 + 273 #Tin[0] #500 + 273
Tf[0,:] = T00
Ts[0,:] = T00

# BCs
Tf[:,0] = Tin
bc1 = np.zeros(Nx)

#%% MATRIX ASSEMBLY

rhof = cpr.PropsSI('D', 'T', T00, 'P', 5e5, 'Air')
cp = cpr.PropsSI('C', 'T', T00, 'P', 5e5, 'Air')
kair = cpr.PropsSI('L', 'T', T00, 'P', 5e5, 'Air')
mu = cpr.PropsSI('V', 'T', T00, 'P', 5e5, 'Air')

alpha = eps*rhof*cp
beta = eps*kair
gamma = (1-eps)*rhos*cpG(T00)
betas = ks*(1 - eps)

D = 1.67*2 #3
R = 0.5*D
u = 4*mDot/(rhof*eps*np.pi*D**2)

Rep = rhof*d*u/mu
Pr = cp*mu/kair
Nu = 0.664*Rep**0.5*Pr**0.5
a = Nu*kair/d
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


# Heat addition in the solid
qv = 1e-6
Qv = 0*x
Qv[3::8] = 2.5e-5
Qv = sp.sparse.diags(Qv, 0).toarray()

#%% SOLUTION

# PID initialization
err = t*0
erri = 0
qv = Ts*0

Tref = 800 + 273

# Gains
Kp = 0.02
Ki = 1e-5
Kd = 0

# Resistors position (equally spaced)
xqv1 = int(1/dx)
xqv2 = int(0.50*L/dx)
xqv3 = int(5/dx)


for i in range(1,len(t)):
    
    if t[i]/3600 >= 17:
        erri += Tref - Tf[i-1,-1]
    
    # Error update
    err[i] = Tref - Tf[i-1,-1]

    qv[i,xqv1] = Kp*err[i] + Ki*erri*dt +\
                            Kd*(err[i] - err[i-1])/dt
    qv[i,xqv2] = Kp*err[i] + Ki*erri*dt +\
                            Kd*(err[i] - err[i-1])/dt
    qv[i,xqv3] = Kp*err[i] + Ki*erri*dt +\
                            Kd*(err[i] - err[i-1])/dt
    
    # Set 0 to any of nonpositive values
    qv[i, qv[i,:] < 0] = 0
    
    # ODE solution -- ANTORA
       
    
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    
    # ANTORA TANK -- fluid properties changing with temperature
    
    # Air density at 5 bar
    rhof = cpr.PropsSI('D', 'T', np.mean(Tf[i-1,:]), 'P', 5e5, 'Air')
    cp = cpr.PropsSI('C', 'T', np.mean(Tf[i-1,:]), 'P', 5e5, 'Air')
    kair = cpr.PropsSI('L', 'T', np.mean(Tf[i-1,:]), 'P', 5e5, 'Air')
    mu = cpr.PropsSI('V', 'T', np.mean(Tf[i-1,:]), 'P', 5e5, 'Air')
    
    alpha = eps*rhof*cp
    beta = eps*kair
    
    gamma = (1-eps)*rhos*cpG(T00)
    betas = ks*(1 - eps)

    D = 3    
    u = 4*mDot/(rhof*eps*np.pi*D**2)

    Rep = rhof*d*u/mu
    Pr = cp*mu/kair
    Nu = 0.664*Rep**0.5*Pr**0.5
    a = Nu*kair/d
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
    
    bc1[0] = Tin[i] #Tf[i-1,0]

    Ts[i,:] = sp.sparse.linalg.spsolve(As, Ts[i-1,:] + rs*Tf[i-1,:] + 0*qv[i,:])
    Tf[i,1:] = sp.sparse.linalg.spsolve(A, Tf[i-1,1:] + (q + p)*bc1 + \
                r*Ts[i-1,1:])
    
    
    # Heat transfer accross molten salts
    # qf[i] = 0.25*u*np.pi*d**2*eps*rhof*(Tf[i,15] - Tf[i,-15])*cpf
    
    # theta6[i,k] = Tf[i,-1]/T0
    
#%% CALCULO CAIDA PRESIÓN (despreciable)

# p1 = 5e5
# T1 = 1e3
# T2 = 500
# rho1 = cpr.PropsSI('D', 'T', T1, 'P', p1, 'Air')

# p2_guess = 2.5e3  # Supongamos un valor inicial para p2
# p2_solution = fsolve(eqP2, p2_guess, args=(p1, rho1, T1, T2))   
    
    
#%% PLOTS

fig, ax = plt.subplots()

cs1 = ax.contourf(x, t/3600, Tf-273, 32)
plt.tight_layout()
cbar1 = plt.colorbar(cs1)
cbar1.set_label(r'$T_f$ ($^\circ C$)', fontsize=16)
ax.set_ylabel(r'$t$ (h)')
ax.set_xlabel(r'$L$ (m)')
ax.set_box_aspect(1)

plt.show()

#%% PLOTS

fig, ax = plt.subplots()

cs1 = ax.contourf(x, t/3600, Ts-273, 32)
plt.tight_layout()
cbar1 = plt.colorbar(cs1)
cbar1.set_label(r'$T_s$ ($^\circ C$)', fontsize=16)
ax.set_ylabel(r'$t$ (h)')
ax.set_xlabel(r'$L$ (m)')
ax.set_box_aspect(1)

plt.show()


#%% PLOTS
fig, ax = plt.subplots()

plt.plot(t/3600, Tf[:,0], 'k--')
plt.plot(t/3600, Tf[:,int(len(x)/4)]-273, t/3600, Tf[:,int(len(x)/2)]-273,
         t/3600, Tf[:,int(3*len(x)/4)]-273, t/3600, Tf[:,-1]-273)
plt.ylabel(r'$T_f$ ($^\circ C$)')
plt.xlabel(r'$t$ (h)')
plt.legend([r'$x = 0$', r'$x = 1/4$', r'$x = 1/2$', r'$x = 3/4$', r'$x = 1$'])

plt.show()


#%% PLOTS
fig, ax = plt.subplots()

plt.plot(t/3600, Ts[:,int(len(x)/4)]-273, t/3600, Ts[:,int(len(x)/2)]-273,
         t/3600, Ts[:,int(3*len(x)/4)]-273, t/3600, Ts[:,-1]-273)
plt.ylabel(r'$T_s$ ($^\circ C$)')
plt.xlabel(r'$t$ (h)')
plt.legend([r'$x = 1/4$', r'$x = 1/2$', r'$x = 3/4$', r'$x = 1$'])

plt.show()


#%% PLOTS
fig, ax = plt.subplots()

plt.plot(t[1:]/3600, np.sum(qv[1:,:], axis=1))
plt.ylabel(r'$\dot{q}_t$ (W)')
plt.xlabel(r'$t$ (h)')

plt.show()

#%% PLOTS
fig, ax = plt.subplots()

plt.plot(t[1:]/3600, err[1:]/Tref)
plt.ylabel(r'$\epsilon$')
plt.xlabel(r'$t$ (h)')

plt.show()

#%% PLOTS
fig, ax = plt.subplots()

plt.plot(t/3600, Tf[:,-1]-273, t/3600, Tin-273, t/3600, t*0 + Tref-273, 'k--')
plt.ylabel(r'$T_f$ ($^\circ$C)')
plt.xlabel(r'$t$ (h)')

plt.show()



#%% PLOTS
# fig, ax = plt.subplots()

# ax.plot(x, Tf[int(len(x)/6),:]-273, x, Tf[int(2*len(x)/6),:]-273,
#          x, Tf[int(3*len(x)/6),:]-273, x, Tf[int(4*len(x)/6),:]-273,
#          x, Tf[int(5*len(x)/6),:]-273)
# # ax.set_ylim(780, 810)
# plt.ylabel(r'$T_f$ ($^\circ C$)')
# plt.xlabel(r'$x$ (m)')
# plt.legend([r'$t = 4$ h', r'$t = 8$ h', r'$t = 12$ h', r'$t = 16$ h',
#             r'$t = 20$ h', r'$t = 24$ h'])

# plt.show()

tc = L/u

Lambdas = L/u*h/(rhos*cpG(1000))/(1 - eps)
Lambdaf = L/u*h/(rhof*cp)/eps
a = Lambdas*Lambdaf
b = 0
Delta = np.sqrt(a**2 - 4*a*b)

solf = T00*np.exp(-x*0/L) + T00 + 0.5*T00*((1 - a/Delta)*np.exp(-0.5*t[:,np.newaxis]/tc*(Delta + a)) +\
                (1 + a/Delta)*np.exp(0.5*t[:,np.newaxis]/tc*(Delta - a)))