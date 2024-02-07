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
# from dishLib import *

warnings.filterwarnings('ignore')
# matplotlib_inline.backend_inline.set_matplotlib_formats('svg', 'pdf')

#%% DEFAULTS

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ["Computer Modern Roman"]
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['lines.markersize'] = 1

cm = 1/2.54 


#%% DISH STUFF

# fig, ax1 = plt.subplots()
# ax1.plot(t, Ib_march*1e-3, t, Ib_june*1e-3, t, Ib_dec*1e-3)
# plt.legend([r'March', r'June', r'Dec.'], loc='upper left')
# ax1.set_xlabel(r'$t$ (h)')
# ax1.set_ylabel(r'$I_b$ (kW)')

# fig, ax1 = plt.subplots()
# ax1.plot(t, Tamb_march - 273, t, Tamb_june - 273, t, Tamb_dec - 273)
# plt.legend([r'March', r'June', r'Dec.'], loc='upper left')
# ax1.set_xlabel(r'$t$ (h)')
# ax1.set_ylabel(r'$T$ ($^\circ$C)')


#%% Graphite Cp
def cpG(T):
    
    return 4184*(0.54 + 9.11e-6*T - 90.27/T - 43449.3/T**2 + 1.593e7/T**3 -\
                 1.437e9/T**4)

#%% PROBLEM

L = 1.5 #1.25 #0.5 #2 #4.5
D = 1 #L/2 #1.67*2 #3

eps = 0.22 #0.22
dx = 2.2e-2
xf = L
x = np.arange(0, xf + dx, dx)
dx = x[1] - x[0]

volR = 2*(1-eps)*dx*0.25*np.pi*D**2

Nx = len(x) - 1

tf = 24*3600
dt = 3.6
t = np.arange(0, tf + dt, dt)
Nt = len(t)

x = np.arange(0, xf + dx, dx)

d = 0.015
mDot = 0.043

rhos = 1800
ks = 120

##

em = np.zeros((3, len(t)))
emSolar = np.zeros((3, len(t)))

QQ = np.zeros((3, len(t)))
QQsolar = np.zeros((3, len(t)))

pp = np.zeros((3, len(t)))
ppSolar = np.zeros((3, len(t)))

CO2 = np.zeros(3)
CO2Solar = np.zeros(3)

EE = 0
EEsolar = 0

E = np.zeros(3)
Esolar = np.zeros(3)
CO2Theo = np.zeros(3)


charge = np.zeros((3, len(t)))

priceTheo = np.zeros((3, len(t)))
price = np.zeros((3, len(t)))
priceSolar = np.zeros((3, len(t)))

priceTheoDay = np.zeros(3)
priceDay = np.zeros(3)
priceDaySolar = np.zeros(3)

priceTheoCumDay = np.zeros((3, len(t)))
priceCumDay = np.zeros((3, len(t)))
priceCumDaySolar = np.zeros((3, len(t)))

##


G0 = 0.043
N = 2

To, ts, month, loc_ = computeDish(G0/N, 0)

##

#%% ENERGY PRICE

# Price €/MWh
march_pelec = np.loadtxt('./March_pelec.csv', delimiter=";",
                       dtype=float)
june_pelec = np.loadtxt('./June_pelec.csv', delimiter=";",
                      dtype=float)
dec_pelec = np.loadtxt('./Dec_pelec.csv', delimiter=";",
                     dtype=float)

march_pelec = np.vstack([march_pelec, [286.25, 24.]])
june_pelec = np.vstack([june_pelec, [354.52, 24.]])
dec_pelec = np.vstack([dec_pelec, [67., 24.]])


# Interpolacion
pp[0,:] = np.interp(t/3600, march_pelec[:,1], march_pelec[:,0])
pp[1,:] = np.interp(t/3600, june_pelec[:,1], june_pelec[:,0])
pp[2,:] = np.interp(t/3600, dec_pelec[:,1], dec_pelec[:,0])


#%% EMISIONS

# eq CO2 kg/MWh
march_em = np.loadtxt('./March_em.csv', delimiter=";",
                       dtype=float)
june_em = np.loadtxt('./June_em.csv', delimiter=";",
                      dtype=float)
dec_em = np.loadtxt('./Dec_em.csv', delimiter=";",
                     dtype=float)

# Interpolacion
em[0,:] = np.interp(t/3600, march_em[:,0], march_em[:,1])
em[1,:] = np.interp(t/3600, june_em[:,0], june_em[:,1])
em[2,:] = np.interp(t/3600, dec_em[:,0], dec_em[:,1])


#%% MONTHS LOOP


# LL = np.arange(0.25, 2.25, 0.25)
# DD = np.arange(0.25, 2.25, 0.25)


# for kk in range(0, len(LL)):
# for jj in range(0, len(DD)):


for ii in range(0, 3):

        # L = 1 #1.25 #0.5 #2 #4.5
        # D = 1 #L/2 #1.67*2 #3

        # eps = 0.22 #0.22
        # dx = 2.2e-2
        # xf = L
        # x = np.arange(0, xf + dx, dx)
        # dx = x[1] - x[0]
        
        # volR = 2*(1-eps)*dx*0.25*np.pi*D**2
        
        # Nx = len(x) - 1
        
        # tf = 24*3600
        # dt = 3.6
        # t = np.arange(0, tf + dt, dt)
        # Nt = len(t)
        
        # x = np.arange(0, xf + dx, dx)
        
        # d = 0.015
        # mDot = 0.043
        
        # rhos = 1800
        # ks = 120
        
        # # Tin = 100 + 273
        
        nM = ii
        
        Tin = np.interp(t, ts*3600, To[nM,:])
        Tin[Tin != 0] += 273
        Tin[Tin < 500] = 500
        
        Tf = np.zeros((Nt, Nx+1))
        Ts = np.zeros((Nt, Nx+1))
        
        # ICs
        T00 = 227 + 273
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
        
        
        # Other initializations        
        Tf0 = t*0
        Tf0[0] = T00
        
        #%% SOLUTION
        
        # Setpoint        
        Tref = 800 + 273

        for i in range(1, len(t)):
        
            # # ODE solution -- TES
               
            
            # # --------------------------------------------------------------- #
            # # --------------------------------------------------------------- #
            # # --------------------------------------------------------------- #
            
            # # TES TANK -- fluid properties changing with temperature
            
            # # Air density at 5 bar
            # rhof = cpr.PropsSI('D', 'T', np.mean(Tf[i-1,:]), 'P', 5e5, 'Air')
            # cp = cpr.PropsSI('C', 'T', np.mean(Tf[i-1,:]), 'P', 5e5, 'Air')
            # kair = cpr.PropsSI('L', 'T', np.mean(Tf[i-1,:]), 'P', 5e5, 'Air')
            # mu = cpr.PropsSI('V', 'T', np.mean(Tf[i-1,:]), 'P', 5e5, 'Air')
            
            # alpha = eps*rhof*cp
            # beta = eps*kair
            
            # gamma = (1-eps)*rhos*cpG(np.mean(Ts[i-1,:]))
            # betas = ks*(1 - eps)
        
            # u = 4*mDot/(rhof*eps*np.pi*D**2)
        
            # Rep = rhof*d*u/mu
            # Pr = cp*mu/kair
            # Nu = 0.664*Rep**0.5*Pr**0.5
            # a = Nu*kair/d
            # h = 6*(1-eps)*beta*(2 + 1.1*Rep**(0.6)*Pr**(1/3))/(d**2)
        
            # p = dt*u/dx
            # q = 2*beta/alpha*0.5*dt/(dx**2)
            # r = h*dt/alpha
        
            # rs = dt*h/gamma
            # qs = 2*betas*0.5*dt/(gamma*dx**2)
            
            
            # diagonals = [np.ones(Nx)*(1 + 2*q + r + p), np.ones(Nx)*(-q),
            #              np.ones(Nx)*(-p-q)]
            # offsets = [0, 1, -1]
            # A.setdiag(diagonals[0], offsets[0])
            # A.setdiag(diagonals[1], offsets[1])
            # A.setdiag(diagonals[2], offsets[2])
            # A[-1,-2] = -2*q - p
            
            # diagonals = [np.ones(Nx + 1)*(1 + 2*qs + rs), np.ones(Nx + 1)*(-qs),
            #              np.ones(Nx + 1)*(-qs)]
            # offsets = [0, 1, -1]
            # As.setdiag(diagonals[0], offsets[0])
            # As.setdiag(diagonals[1], offsets[1])
            # As.setdiag(diagonals[2], offsets[2])
            # As[-1,-2] = -2*qs
            # As[0,1] = -2*qs  
                
            # # PDE solution -- TANK
            
            # bc1[0] = Tin[i] #Tf[i-1,0]
        
            # # Ts[i,:] = sp.sparse.linalg.spsolve(As, Ts[i-1,:] + rs*Tf[i-1,:] +
            # #                                    (1-eps)*qv[i,:])
            # # Tf[i,1:] = sp.sparse.linalg.spsolve(A, Tf[i-1,1:] + (q + p)*bc1 + \
            # #             r*Ts[i-1,1:] + eps*qv[i,1:])
            
            # Ts[i,:] = sp.sparse.linalg.spsolve(As, Ts[i-1,:] + rs*Tf[i-1,:])
            # Tf[i,1:] = sp.sparse.linalg.spsolve(A, Tf[i-1,1:] + (q + p)*bc1 + \
            #             r*Ts[i-1,1:])

            
            # if Tin[i] >= Tref:
                
            #     # TSS charge
    
            #     Tf0[i] = np.maximum(Tref, Tf[i,-1])
            #     QQ[ii,i] = mDot*cp*(Tref - Tf0[i])
                
            #     if QQ[ii,i] < 0:
            #         QQ[ii,i] = 0
                
            # else:
                
            #     if Tin[i] >= Tf[i,-1]:
                    
            #         Tf[i,:] = Tf[i-1,:]
            #         Ts[i,:] = Ts[i-1,:]
                    
            #         Tf0[i] = Tref
            #         QQ[ii,i] = mDot*cp*(Tref - Tin[i])
                    
            #         if QQ[ii,i] < 0:
            #             QQ[ii,i] = 0
                        
            #     else:
                
            #         Tf0[i] = Tref
            #         QQ[ii,i] = mDot*cp*(Tref - Tf[i-1,-1])
                    
            #         if QQ[ii,i] < 0:
            #             QQ[ii,i] = 0
                
        
            
            if Tin[i] >= Tref and Tin[i] >= Tf[i-1,-1]:
                # TSS charge
                
                charge[ii,i] = 1
                
                # ODE solution -- TES
                   
                # --------------------------------------------------------------- #
                # --------------------------------------------------------------- #
                # --------------------------------------------------------------- #
                
                # TES TANK -- fluid properties changing with temperature
                
                # Air density at 5 bar
                rhof = cpr.PropsSI('D', 'T',0.5*(Tref + np.maximum(Tref, Tf[i-1,-1])), 'P', 5e5, 'Air')
                cp = cpr.PropsSI('C', 'T', 0.5*(Tref + np.maximum(Tref, Tf[i-1,-1])), 'P', 5e5, 'Air')
                kair = cpr.PropsSI('L', 'T', 0.5*(Tref + np.maximum(Tref, Tf[i-1,-1])), 'P', 5e5, 'Air')
                mu = cpr.PropsSI('V', 'T', 0.5*(Tref + np.maximum(Tref, Tf[i-1,-1])), 'P', 5e5, 'Air')
                
                alpha = eps*rhof*cp
                beta = eps*kair
                
                gamma = (1-eps)*rhos*cpG(np.mean(Ts[i-1,:]))
                betas = ks*(1 - eps)
            
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
                
                Ts[i,:] = sp.sparse.linalg.spsolve(As, Ts[i-1,:] + rs*Tf[i-1,:])
                Tf[i,1:] = sp.sparse.linalg.spsolve(A, Tf[i-1,1:] + (q + p)*bc1 + \
                            r*Ts[i-1,1:])
                    
                if Tf[i,-1] >= Tref:
                    Tf0[i] = Tf[i,-1]
                    QQ[ii,i] = 0
                else:
                    Tf0[i] = Tref
                    QQ[ii,i] = mDot*cp*(Tref - Tf0[i])
                    
 
                    
            elif Tin[i] >= Tref and Tin[i] < Tf[i-1,-1]:
                
                # Bypass
                
                # ODE solution -- TES
                   
                # --------------------------------------------------------------- #
                # --------------------------------------------------------------- #
                # --------------------------------------------------------------- #
                
                # TES TANK -- fluid properties changing with temperature
                
                # Air density at 5 bar
                rhof = cpr.PropsSI('D', 'T', 0.5*(Tref + np.maximum(Tref, Tin[i])), 'P', 5e5, 'Air')
                cp = cpr.PropsSI('C', 'T', 0.5*(Tref + np.maximum(Tref, Tin[i])), 'P', 5e5, 'Air')
                kair = cpr.PropsSI('L', 'T', 0.5*(Tref + np.maximum(Tref, Tin[i])), 'P', 5e5, 'Air')
                mu = cpr.PropsSI('V', 'T', 0.5*(Tref + np.maximum(Tref, Tin[i])), 'P', 5e5, 'Air')
                
                alpha = eps*rhof*cp
                beta = eps*kair
                
                gamma = (1-eps)*rhos*cpG(np.mean(Ts[i-1,:]))
                betas = ks*(1 - eps)
            
                u = 0
            
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
                A[-1,-2] = -2*q
                As[0,1] = -2*q
                
                diagonals = [np.ones(Nx + 1)*(1 + 2*qs + rs), np.ones(Nx + 1)*(-qs),
                             np.ones(Nx + 1)*(-qs)]
                offsets = [0, 1, -1]
                As.setdiag(diagonals[0], offsets[0])
                As.setdiag(diagonals[1], offsets[1])
                As.setdiag(diagonals[2], offsets[2])
                As[-1,-2] = -2*qs
                As[0,1] = -2*qs  
                    
                # PDE solution -- TANK
                                
                Ts[i,:] = sp.sparse.linalg.spsolve(As, Ts[i-1,:] + rs*Tf[i-1,:])
                Tf[i,1:] = sp.sparse.linalg.spsolve(A, Tf[i-1,1:] + \
                            r*Ts[i-1,1:])
    
                
                Tf0[i] = Tin[i]
                
                if Tf0[i] < Tref:
                    QQ[ii,i] = mDot*cp*(Tref - Tf0[i])
                else:
                    QQ[ii,i] = 0
                    
        
            elif Tin[i] < Tref and Tin[i] >= Tf[i-1,-1]:
                
                # Bypass
                # ODE solution -- TES
                   
                # --------------------------------------------------------------- #
                # --------------------------------------------------------------- #
                # --------------------------------------------------------------- #
                
                # TES TANK -- fluid properties changing with temperature
                
                # Air density at 5 bar
                rhof = cpr.PropsSI('D', 'T', 0.5*(Tref + Tin[i]), 'P', 5e5, 'Air')
                cp = cpr.PropsSI('C', 'T', 0.5*(Tref + Tin[i]), 'P', 5e5, 'Air')
                kair = cpr.PropsSI('L', 'T', 0.5*(Tref + Tin[i]), 'P', 5e5, 'Air')
                mu = cpr.PropsSI('V', 'T', 0.5*(Tref + Tin[i]), 'P', 5e5, 'Air')
                
                alpha = eps*rhof*cp
                beta = eps*kair
                
                gamma = (1-eps)*rhos*cpG(np.mean(Ts[i-1,:]))
                betas = ks*(1 - eps)
            
                u = 0
            
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
                A[-1,-2] = -2*q
                As[0,1] = -2*q
                
                diagonals = [np.ones(Nx + 1)*(1 + 2*qs + rs), np.ones(Nx + 1)*(-qs),
                             np.ones(Nx + 1)*(-qs)]
                offsets = [0, 1, -1]
                As.setdiag(diagonals[0], offsets[0])
                As.setdiag(diagonals[1], offsets[1])
                As.setdiag(diagonals[2], offsets[2])
                As[-1,-2] = -2*qs
                As[0,1] = -2*qs  
                    
                # PDE solution -- TANK
                                
                Ts[i,:] = sp.sparse.linalg.spsolve(As, Ts[i-1,:] + rs*Tf[i-1,:])
                Tf[i,1:] = sp.sparse.linalg.spsolve(A, Tf[i-1,1:] + \
                            r*Ts[i-1,1:])
                    
                
                Tf0[i] = Tref
                QQ[ii,i] = mDot*cp*(Tref - Tin[i])

                    

            elif Tin[i] < Tref and Tin[i] < Tf[i-1,-1] and Tf[i-1,-1] < Tref \
                and Tf[i-1,-1] <= 500:
                
                # Bypass
                # ODE solution -- TES
                   
                # --------------------------------------------------------------- #
                # --------------------------------------------------------------- #
                # --------------------------------------------------------------- #
                
                # TES TANK -- fluid properties changing with temperature
                
                # Air density at 5 bar
                rhof = cpr.PropsSI('D', 'T', 0.5*(Tref + Tin[i]), 'P', 5e5, 'Air')
                cp = cpr.PropsSI('C', 'T', 0.5*(Tref + Tin[i]), 'P', 5e5, 'Air')
                kair = cpr.PropsSI('L', 'T', 0.5*(Tref + Tin[i]), 'P', 5e5, 'Air')
                mu = cpr.PropsSI('V', 'T', 0.5*(Tref + Tin[i]), 'P', 5e5, 'Air')
                
                alpha = eps*rhof*cp
                beta = eps*kair
                
                gamma = (1-eps)*rhos*cpG(np.mean(Ts[i-1,:]))
                betas = ks*(1 - eps)
            
                u = 0
            
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
                A[-1,-2] = -2*q
                As[0,1] = -2*q
                
                diagonals = [np.ones(Nx + 1)*(1 + 2*qs + rs), np.ones(Nx + 1)*(-qs),
                             np.ones(Nx + 1)*(-qs)]
                offsets = [0, 1, -1]
                As.setdiag(diagonals[0], offsets[0])
                As.setdiag(diagonals[1], offsets[1])
                As.setdiag(diagonals[2], offsets[2])
                As[-1,-2] = -2*qs
                As[0,1] = -2*qs  
                    
                # PDE solution -- TANK
                                
                Ts[i,:] = sp.sparse.linalg.spsolve(As, Ts[i-1,:] + rs*Tf[i-1,:])
                Tf[i,1:] = sp.sparse.linalg.spsolve(A, Tf[i-1,1:] + \
                            r*Ts[i-1,1:])
                
                Tf0[i] = Tref
                QQ[ii,i] = mDot*cp*(Tref - Tin[i])
                
                
            elif Tin[i] < Tref and Tin[i] < Tf[i-1,-1] and Tf[i-1,-1] < Tref \
                and Tf[i-1,-1] > 500:
                
                # OJO: Discharge
                charge[ii,i] = -1
                
                # ODE solution -- TES
                   
                # --------------------------------------------------------------- #
                # --------------------------------------------------------------- #
                # --------------------------------------------------------------- #
                
                # TES TANK -- fluid properties changing with temperature
                
                # Air density at 5 bar
                rhof = cpr.PropsSI('D', 'T', 0.5*(Tref + Tf[i-1,-1]), 'P', 5e5, 'Air')
                cp = cpr.PropsSI('C', 'T', 0.5*(Tref + Tf[i-1,-1]), 'P', 5e5, 'Air')
                kair = cpr.PropsSI('L', 'T', 0.5*(Tref + Tf[i-1,-1]), 'P', 5e5, 'Air')
                mu = cpr.PropsSI('V', 'T', 0.5*(Tref + Tf[i-1,-1]), 'P', 5e5, 'Air')
                
                alpha = eps*rhof*cp
                beta = eps*kair
                
                gamma = (1-eps)*rhos*cpG(np.mean(Ts[i-1,:]))
                betas = ks*(1 - eps)
            
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
                
                Ts[i,:] = sp.sparse.linalg.spsolve(As, Ts[i-1,:] + rs*Tf[i-1,:])
                Tf[i,1:] = sp.sparse.linalg.spsolve(A, Tf[i-1,1:] + (q + p)*bc1 + \
                            r*Ts[i-1,1:])
                
                    
                Tf0[i] = Tref
                QQ[ii,i] = mDot*cp*(Tref - Tf[i-1,-1])
                    

            elif Tin[i] < Tref and Tin[i] < Tf[i-1,-1] and Tf[i-1,-1] >= Tref:
                
                # OJO: Discharge
                charge[ii,i] = -1
                
                # ODE solution -- TES
                   
                # --------------------------------------------------------------- #
                # --------------------------------------------------------------- #
                # --------------------------------------------------------------- #
                
                # TES TANK -- fluid properties changing with temperature
                
                # Air density at 5 bar
                rhof = cpr.PropsSI('D', 'T', 0.5*(Tref + Tin[i]), 'P', 5e5, 'Air')
                cp = cpr.PropsSI('C', 'T', 0.5*(Tref + Tin[i]), 'P', 5e5, 'Air')
                kair = cpr.PropsSI('L', 'T', 0.5*(Tref + Tin[i]), 'P', 5e5, 'Air')
                mu = cpr.PropsSI('V', 'T', 0.5*(Tref + Tin[i]), 'P', 5e5, 'Air')
                
                alpha = eps*rhof*cp
                beta = eps*kair
                
                gamma = (1-eps)*rhos*cpG(np.mean(Ts[i-1,:]))
                betas = ks*(1 - eps)
            
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
                
                Ts[i,:] = sp.sparse.linalg.spsolve(As, Ts[i-1,:] + rs*Tf[i-1,:])
                Tf[i,1:] = sp.sparse.linalg.spsolve(A, Tf[i-1,1:] + (q + p)*bc1 + \
                            r*Ts[i-1,1:])
                
                    
                Tf0[i] = Tf[i-1,-1]
                QQ[ii,i] = 0
                
                    
            # Extra power without TSS (only solar dish)   
                
            QQsolar[ii,i] = mDot*cpr.PropsSI('C', 'T', 0.5*(Tref + Tin[i]),
                                             'P', 5e5, 'Air')*(Tref - Tin[i])
    
            if QQsolar[ii,i] < 0:
                QQsolar[ii,i] = 0            
                    
                
#%% Energy
        
        E[ii] = np.trapz(QQ[ii,:]/1e3, t/3600)
        Esolar[ii] = np.trapz(QQsolar[ii,:]/1e3, t/3600)
        
        display()
        display(str(month[ii]) + ':')
        display('E = ' + str(E[ii]) + ' kWh')
        display('Esolar = ' + str(Esolar[ii]) + ' kWh')
        
        with open('./E_' + 'LbyD_' + str(L/D) + '_N_' + str(int(N)) + '_' +
                  month[nM] + '_' + loc_ + '_T0_' + str(T00 - 273), 'w') as archivo:
        # Escribir el resultado en el archivo
            archivo.write(str(int(np.round(E[ii],0))))
            
        with open('./Esolar_' + 'LbyD_' + str(L/D) + '_N_' + str(int(N)) + '_' +
                  month[nM] + '_' + loc_ + '_T0_' + str(T00 - 273), 'w') as archivo:
        # Escribir el resultado en el archivo
            archivo.write(str(int(np.round(Esolar[ii],0))))
            
            
#%% Price
        
        priceDay[ii] = np.trapz(QQ[ii,:]/1e6*pp[ii,:], t/3600)
        priceDaySolar[ii] = np.trapz(QQsolar[ii,:]/1e6*pp[ii,:], t/3600)
        priceTheoDay[ii] = np.trapz(QQ[ii,:]*0 + mDot*cpr.PropsSI('C', 'T', 0.5*(Tref + 500), 'P', 5e5, 'Air')*
                                    (Tref - 500)*1e-6*pp[ii,:], t/3600)
        
        CO2[ii] = np.trapz(QQ[ii,:]/1e6*em[ii,:], t/3600)
        CO2Solar[ii] = np.trapz(QQsolar[ii,:]/1e6*em[ii,:], t/3600)
        CO2Theo[ii] = np.trapz(QQ[ii,:]*0 + mDot*cpr.PropsSI('C', 'T', 0.5*(Tref + 500), 'P', 5e5, 'Air')*
                                    (Tref - 500)/1e6*em[ii,:], t/3600)
        
        price[ii,:] = QQ[ii,:]/1e6*pp[ii,:]
        priceSolar[ii,:] = QQsolar[ii,:]/1e6*pp[ii,:]
        priceTheo[ii,:] = QQ[ii,:]*0 + mDot*cpr.PropsSI('C', 'T', 0.5*(Tref + 500), 'P', 5e5, 'Air')*\
            (Tref - 500)*1e-6*pp[ii,:]
                                    
       
        priceCumDay[ii,:] = cumtrapz(QQ[ii,:]/1e6*pp[ii,:], t/3600,
                                  initial = 0)
        priceCumDaySolar[ii,:] = cumtrapz(QQsolar[ii,:]/1e6*pp[ii,:], t/3600,
                                  initial = 0)
        priceTheoCumDay[ii,:] = cumtrapz(QQ[ii,:]*0 + mDot*cpr.PropsSI('C', 'T', 0.5*(Tref + 500), 'P', 5e5, 'Air')*
                                         (Tref - 500)*1e-6*pp[ii,:], t/3600, initial = 0) 
       
        display('price/day = ' + str(priceDay[ii]) + ' €')
        display('solar price/day = ' + str(priceDaySolar[ii]) + ' €')
        display('priceTheo/day = ' + str(priceTheoDay[ii]) + ' €')
        
        with open('./pp_' + 'LbyD_' + str(L/D) + '_N_' + str(int(N)) + '_' +
                  month[nM] + '_' + loc_ + '_T0_' + str(T00 - 273), 'w') as archivo:
        # Escribir el resultado en el archivo
            archivo.write('precio = ' + str(int(np.round(priceDay[ii],0))) + '\n')
            archivo.write('precio solar = ' + str(int(np.round(priceDaySolar[ii],0))) + '\n')
            archivo.write('precio teo = ' + str(int(np.round(priceTheoDay[ii],0))))
            
            
        display('CO2/day = ' + str(CO2[ii]) + ' eq. T')
        display('solar CO2/day = ' + str(CO2Solar[ii]) + ' eq. T')
        display('CO2 theo/day = ' + str(CO2Theo[ii]) + ' eq. T')
       
        with open('./pp_' + 'LbyD_' + str(L/D) + '_N_' + str(int(N)) + '_' +
                  month[nM] + '_' + loc_ + '_T0_' + str(T00 - 273), 'w') as archivo:
        # Escribir el resultado en el archivo
            archivo.write('T eq. CO2 = ' + str(int(np.round(CO2[ii],0))) + '\n')
            archivo.write('T eq. CO2 solar = ' + str(int(np.round(CO2Solar[ii],0))) + '\n')
            archivo.write('T eq. CO2 teo = ' + str(int(np.round(CO2Theo[ii],0))))
        

#%% CALCULO CAIDA PRESIÓN TÉRMICA (despreciable)
        
        # p1 = 5e5
        # T1 = 1e3
        # T2 = 500
        # rho1 = cpr.PropsSI('D', 'T', T1, 'P', p1, 'Air')
        
        # p2_guess = 2.5e3  # Supongamos un valor inicial para p2
        # p2_solution = fsolve(eqP2, p2_guess, args=(p1, rho1, T1, T2))   
            
            
#%% PLOTS
        
        # fig, ax = plt.subplots(figsize=(7*cm, 7*cm))
        
        # cs1 = ax.contourf(x, t/3600, Tf-273, 32)
        # plt.tight_layout()
        # cbar1 = plt.colorbar(cs1)
        # cbar1.set_label(r'$T_f$ ($^\circ$C)', fontsize=16)
        # ax.set_ylabel(r'$t$ (h)')
        # ax.set_xlabel(r'$L$ (m)')
        # ax.set_box_aspect(1)
        # plt.savefig('./TfContour_' + month[nM] + '_' + loc_ +'.eps',
        #             bbox_inches='tight', format='eps')
        # plt.show()
        
#%% PLOTS
        
        # fig, ax = plt.subplots(figsize=(7*cm, 7*cm))
        
        # cs1 = ax.contourf(x, t/3600, Ts-273, 32)
        # plt.tight_layout()
        # cbar1 = plt.colorbar(cs1)
        # cbar1.set_label(r'$T_s$ ($^\circ$C)', fontsize=16)
        # ax.set_ylabel(r'$t$ (h)')
        # ax.set_xlabel(r'$L$ (m)')
        # ax.set_box_aspect(1)
        # plt.savefig('./TsContour_' + month[nM] + '_' + loc_ +'.eps',
        #             bbox_inches='tight', format='eps')
        
        # plt.show()
        
        #%% PLOTS
        # fig, ax = plt.subplots(figsize=(7*cm, 7*cm))
        
        # plt.plot(t/3600, Tf[:,0], 'k-')
        # plt.plot(t/3600, Tf[:,int(len(x)/4)]-273, t/3600, Tf[:,int(len(x)/2)]-273,
        #          t/3600, Tf[:,int(3*len(x)/4)]-273, t/3600, Tf[:,-1]-273)
        # # plt.plot(t/3600, Ts[:,int(len(x)/4)]-273, '--',
        # #          t/3600, Ts[:,int(len(x)/2)]-273, '--',
        # #          t/3600, Ts[:,int(3*len(x)/4)]-273, '--',
        # #          t/3600, Ts[:,-1]-273, '--')
        
        # plt.plot(t/3600, t*0 + 800, 'k--')
        # plt.plot(t/3600, t*0 + 800*0.95, 'k--')
        
        # plt.ylabel(r'$T_f$ ($^\circ$C)')
        # plt.xlabel(r'$t$ (h)')
        # plt.ylim([500,1000])
        # plt.legend([r'$x = 0$', r'$x = 1/4$', r'$x = 1/2$', r'$x = 3/4$', r'$x = 1$'])
        # plt.savefig('./Tf_' + month[nM] + '_' + loc_ +'.eps', bbox_inches='tight',
        #             format='eps')
        # plt.show()
        
        
#%% PLOTS
        fig, ax = plt.subplots(figsize=(7*cm, 7*cm))
        
        plt.plot(t/3600, Tf[:,int(len(x)/4)]-273, t/3600, Tf[:,int(len(x)/2)]-273,
                 t/3600, Tf[:,int(3*len(x)/4)]-273, t/3600, Tf[:,-1]-273)
        
        plt.gca().set_prop_cycle(None)
        
        plt.plot(t/3600, Ts[:,int(len(x)/4)]-273, '--',
                 t/3600, Ts[:,int(len(x)/2)]-273, '--',
                 t/3600, Ts[:,int(3*len(x)/4)]-273, '--',
                 t/3600, Ts[:,-1]-273, '--')
        plt.ylabel(r'$T$ ($^\circ$C)')
        plt.xlabel(r'$t$ (h)')
        plt.legend([r'$x = 1/4$', r'$x = 1/2$', r'$x = 3/4$', r'$x = 1$'])
        plt.savefig('./Tf_' + month[nM] + '_' + loc_ +'.eps', bbox_inches='tight',
                    format='eps')
        
        plt.show()
        
#%% PLOTS
        # fig, ax = plt.subplots(figsize=(7*cm, 7*cm))

        # plt.plot(t/3600, Tf0-273)
        # plt.ylabel(r'$T_{f0}$ ($^\circ$C)')
        # plt.xlabel(r'$t$ (h)')
        # # plt.legend([r'$x = 1/4$', r'$x = 1/2$', r'$x = 3/4$', r'$x = 1$'])
        # plt.savefig('./Tf0_' + month[nM] + '_' + loc_ +'.eps',
        #             bbox_inches='tight', format='eps')
        
        # plt.show()        
        
#%% PLOTS
        # fig, ax = plt.subplots(figsize=(7*cm, 7*cm))
        
        # QQtheo = mDot*cpr.PropsSI('C', 'T', 0.5*(T00 + Tref),
        #                           'P', 5e5, 'Air')*(Tref - T00)
        
        # plt.plot(t/3600, QQ[ii,:]/1e3)
        # plt.plot(t/3600, QQ[ii,:]*0 + QQtheo/1e3, 'k--')
        
        # plt.ylabel(r'$\dot{Q}_0$ (kW)')
        # plt.xlabel(r'$t$ (h)')
        
        # plt.savefig('./Q0_' + month[nM] + '_' + loc_ +'.eps',
        #            bbox_inches='tight', format='eps')

        # plt.show()
        
#%% PLOTS
        # fig, ax = plt.subplots()
        
        # plt.plot(t[1:]/3600, 100*err[1:]/Tref)
        # plt.ylabel(r'$\varepsilon_r$ (\%)')
        # plt.xlabel(r'$t$ (h)')
        
        # plt.show()
        
#%% PLOTS
        # fig, ax = plt.subplots(figsize=(7*cm, 7*cm))

        # plt.plot(t/3600, pp)
        # plt.ylabel(r'€/MWh')
        # plt.xlabel(r'$t$ (h)')
        # # plt.legend([r'$x = 1/4$', r'$x = 1/2$', r'$x = 3/4$', r'$x = 1$'])
        # # plt.savefig('./Tf0_' + month[nM] + '_' + loc_ +'.eps',
        # #             bbox_inches='tight', format='eps')
        
        # plt.show()   
        
#%% PLOTS
        fig, ax1 = plt.subplots(figsize=(7*cm, 7*cm))
        
        # ax1.plot(t/3600, Tf[:,-1]-273, t/3600, Tin-273,
        #          t/3600, Tf0-273, 'k-')
        
        ax1.scatter(t/3600, Tf[:,-1]-273, linestyle='solid', linewidths=0.25)
        ax1.scatter(t[charge[ii,:] == 1]/3600, Tf[charge[ii,:] == 1,-1]-273,
                    linestyle='solid', linewidths=0.25, color='purple')
        ax1.set_prop_cycle(None)
        ax1.scatter(t[charge[ii,:] == -1]/3600, Tf[charge[ii,:] == -1,-1]-273,
                    linestyle='dashed', linewidths=0.25, color='magenta')
        ax1.set_prop_cycle(plt.cycler('color', plt.cm.tab10.colors[1:]))
        ax1.plot(t/3600, Tin-273, t/3600, Tf0-273, 'k-')        
        ax1.plot(0, T00-273, 'ro', markersize = 3)
        ax1.set_ylabel(r'$T$ ($^\circ$C)')
        ax1.set_xlabel(r'$t$ (h)')
        ax1.set_ylim([200, 2000])
        ax1.set_yticks(np.linspace(200, 2000, 7))
        if nM == 2:
            ax1.set_ylim([200, 1000])
            ax1.set_yticks(np.linspace(200, 1000, 5))
        
        # qarr = (Tref - Tf[:,-1])*mDot*cpr.PropsSI('C', 'T', (0.5*(Tref - Tf[:,-1])),
        #                                           'P', 5e5, 'Air')
        
        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.plot(t[1:]/3600, QQ[ii,1:]/1e3, color=color)
        ax2.plot(t[1:-1:500]/3600, QQ[ii,1:-1:500]*0 +
                 mDot*cpr.PropsSI('C', 'T', 0.5*(Tref + 500), 'P', 5e5, 'Air')*
                 (Tref - 500)*1e-3, 'o-', markersize = 2, color=color)
        
        # ax2.plot(t[0:-1:500]/3600, qarr[0:-1:500]/1000, 's-',
        #          markersize = 2, color=color)
        
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.tick_params(axis='y', color=color)
        ax2.spines['right'].set_color(color)
        ax2.set_ylabel(r'$\dot{Q}_0$ (kW)', color=color)
        ax2.set_ylim([-0.4, 28]) 
        plt.savefig('./Q0_Tf_' + 'LbyD_' + str(L/D) + '_N_' + str(int(N)) + '_' +
                  month[nM] + '_' + loc_ + '_T0_' + str(T00 - 273) +'.eps', bbox_inches='tight',
                    format='eps')
        
        plt.show()
        
#%% MEAN ENERGY & CO2 EMISSIOS

EE = 0.25*(2*E[0] + E[1] + E[2])
CO2tot = 0.25*(2*CO2[0] + CO2[1] + CO2[2])

display('EE = ' + str(EE) + ' kWh')
display('CO2tot = ' + str(CO2tot) + ' eq. T')

with open('./Emean_' + 'LbyD_' + str(L/D) + '_N_' + str(int(N)) + '_' +
          month[nM] + '_' + loc_ + '_T0_' + str(T00 - 273), 'w') as archivo:
# Escribir el resultado en el archivo
    archivo.write(str(int(np.round(EE,0))))

with open('./CO2mean_' + 'LbyD_' + str(L/D) + '_N_' + str(int(N)) + '_' +
          month[nM] + '_' + loc_ + '_T0_' + str(T00 - 273), 'w') as archivo:
# Escribir el resultado en el archivo
    archivo.write(str(int(np.round(CO2tot,0))))
    
#%% SOLAR ENERGY & EMISSIONS

EEsolar = 0.25*(2*Esolar[0] + Esolar[1] + Esolar[2])
CO2totSolar = 0.25*(2*CO2Solar[0] + CO2Solar[1] + CO2Solar[2])

display('EEsolar = ' + str(EEsolar) + ' kWh')
display('CO2totSolar = ' + str(CO2totSolar) + ' eq. T')


with open('./EmeanSolar_' + 'LbyD_' + str(L/D) + '_N_' + str(int(N)) + '_' +
          month[nM] + '_' + loc_ + '_T0_' + str(T00 - 273), 'w') as archivo:
# Escribir el resultado en el archivo
    archivo.write(str(int(np.round(EEsolar,0))))
    
with open('./CO2meanSolar_' + 'LbyD_' + str(L/D) + '_N_' + str(int(N)) + '_' +
          month[nM] + '_' + loc_ + '_T0_' + str(T00 - 273), 'w') as archivo:
# Escribir el resultado en el archivo
    archivo.write(str(int(np.round(CO2totSolar,0))))
    
#%% ENERGY PRICE

fig, ax = plt.subplots(figsize=(7*cm, 7*cm))

plt.plot(t/3600, pp[0], t/3600, pp[1], t/3600, pp[2])
plt.ylabel(r'€/MWh')
plt.xlabel(r'$t$ (h)')
plt.legend([r'March', r'June', r'Dec.'])
plt.savefig('./Market_energy_price.eps',
            bbox_inches='tight', format='eps')

plt.show()


#%% PLOTS
fig, ax1 = plt.subplots(figsize=(7*cm, 7*cm))

ax1.plot(t/3600, priceCumDay.T, '--')
ax1.set_prop_cycle(None)
ax1.plot(t/3600, priceTheoCumDay.T, '-')
plt.ylabel(r'€')
plt.xlabel(r'$t$ (h)')
plt.legend([r'March', r'June', r'Dec.'])
plt.savefig('./Comp_cum_energy_price' + '_LbyD_' + str(L/D) + '_' + loc_ +
            '_N_' + str(int(N)) + '.eps',
            bbox_inches='tight', format='eps')
plt.show()


#%% PLOTS
fig, ax1 = plt.subplots(figsize=(7*cm, 7*cm))

ax1.plot(t/3600, price.T, '--')
ax1.set_prop_cycle(None)
ax1.plot(t/3600, priceTheo.T, '-')
plt.ylabel(r'€/h')
plt.xlabel(r'$t$ (h)')
plt.legend([r'March', r'June', r'Dec.'])
plt.savefig('./Comp_inst_energy_price' + '_LbyD_' + str(L/D) + '_' + loc_ +
            '_N_' + str(int(N)) + '.eps',
            bbox_inches='tight', format='eps')
plt.show()


