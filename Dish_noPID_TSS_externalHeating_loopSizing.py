#%% IMPORTS

from scipy import signal, fftpack
from scipy.integrate import odeint, solve_ivp, cumtrapz, RK45
from scipy.interpolate import interp1d
from scipy.optimize import fsolve, newton, least_squares, root, minimize
from scipy import interpolate
import matplotlib.pyplot as plt
import math
import json
import scipy as sp
import numpy as np
import csv
import matplotlib_inline
import warnings

from functions import *

warnings.filterwarnings('ignore')
# matplotlib_inline.backend_inline.set_matplotlib_formats('svg', 'pdf')

#%% DEFAULTS

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ["Computer Modern Roman"]
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['figure.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['lines.markersize'] = 1
plt.rc('text', usetex=True)

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

tf = 24*3600
dt = 3.6
t = np.arange(0, tf + dt, dt)
Nt = len(t)

d = 0.015
mDot = 0.043

rhos = 1800
ks = 120

##

QQ = np.zeros((3, len(t)))

##

LbyD = np.arange(0.25, 2.25, 0.25)
G0v = np.array([mDot/4, mDot/3, mDot/2, mDot])

EE = np.zeros((len(LbyD),len(G0v)))
E = np.zeros(3)

##


for mm in range(0, len(G0v)):
    
    G0 = G0v[mm]
    To, ts, month, loc_ = computeDish(G0, 1)

    for jj in range(0, len(LbyD)):
        
        # Month loop
        for ii in range(0, 3):
        
            D = 1
            L = LbyD[jj]*D
            
            eps = 0.22 #0.22
            dx = 2.2e-2
            xf = L
            x = np.arange(0, xf + dx, dx)
            dx = x[1] - x[0]
            Nx = len(x) - 1
            
            x = np.arange(0, xf + dx, dx)
            
            volR = 2*(1-eps)*dx*0.25*np.pi*D**2
    
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
            
                # ODE solution -- TES
                   
                
                # ----------------------------------------------------------------------- #
                # ----------------------------------------------------------------------- #
                # ----------------------------------------------------------------------- #
                
                # TES TANK -- fluid properties changing with temperature
                
                # Air density at 5 bar
                rhof = cpr.PropsSI('D', 'T', np.mean(Tf[i-1,:]), 'P', 5e5, 'Air')
                cp = cpr.PropsSI('C', 'T', np.mean(Tf[i-1,:]), 'P', 5e5, 'Air')
                kair = cpr.PropsSI('L', 'T', np.mean(Tf[i-1,:]), 'P', 5e5, 'Air')
                mu = cpr.PropsSI('V', 'T', np.mean(Tf[i-1,:]), 'P', 5e5, 'Air')
                
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
            
                # Ts[i,:] = sp.sparse.linalg.spsolve(As, Ts[i-1,:] + rs*Tf[i-1,:] +
                #                                    (1-eps)*qv[i,:])
                # Tf[i,1:] = sp.sparse.linalg.spsolve(A, Tf[i-1,1:] + (q + p)*bc1 + \
                #             r*Ts[i-1,1:] + eps*qv[i,1:])
                
                Ts[i,:] = sp.sparse.linalg.spsolve(As, Ts[i-1,:] + rs*Tf[i-1,:])
                Tf[i,1:] = sp.sparse.linalg.spsolve(A, Tf[i-1,1:] + (q + p)*bc1 + \
                            r*Ts[i-1,1:])
    
                
                if Tin[i] >= Tref:
        
                    Tf0[i] = np.maximum(Tref, Tf[i,-1])
                    QQ[ii,i] = mDot*cp*(Tref - Tf0[i])
                    
                    if QQ[ii,i] < 0:
                        QQ[ii,i] = 0
                    
                else:
                    
                    if Tin[i] >= Tf[i,-1]:
                        
                        Tf[i,:] = Tf[i-1,:]
                        Ts[i,:] = Ts[i-1,:]
                        
                        Tf0[i] = Tref
                        QQ[ii,i] = mDot*cp*(Tref - Tin[i])
                        
                        if QQ[ii,i] < 0:
                            QQ[ii,i] = 0
                            
                    else:
                    
                        Tf0[i] = Tref
                        QQ[ii,i] = mDot*cp*(Tref - Tf[i-1,-1])
                        
                        if QQ[ii,i] < 0:
                            QQ[ii,i] = 0
                    
                
                    
            #%% Energy
            
            # E = np.trapz(np.sum(qv[1:,:]*volR/1e3, axis=1), t[1:]/3600)
            E[ii] = np.trapz(QQ[ii,:]/1e3, t/3600)
            
            
            display('E = ' + str(E[ii]) + ' kWh')
            
            with open('./E_' + month[nM] + '_' + loc_ +
                        '_T0_' + str(T00 - 273), 'w') as archivo:
            # Escribir el resultado en el archivo
                archivo.write(str(E[ii]))
            
    
            #%% CALCULO CAIDA PRESIÓN TÉRMICA (despreciable)
            
            # # p1 = 5e5
            # # T1 = 1e3
            # # T2 = 500
            # # rho1 = cpr.PropsSI('D', 'T', T1, 'P', p1, 'Air')
            
            # # p2_guess = 2.5e3  # Supongamos un valor inicial para p2
            # # p2_solution = fsolve(eqP2, p2_guess, args=(p1, rho1, T1, T2))   
                
                
            # #%% PLOTS
            # cm = 1/2.54 
            
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
            
            # #%% PLOTS
            
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
            
            
            # #%% PLOTS
            # # fig, ax = plt.subplots(figsize=(7*cm, 7*cm))
            
            # # plt.plot(t/3600, Tf[:,0], 'k-')
            # # plt.plot(t/3600, Tf[:,int(len(x)/4)]-273, t/3600, Tf[:,int(len(x)/2)]-273,
            # #          t/3600, Tf[:,int(3*len(x)/4)]-273, t/3600, Tf[:,-1]-273)
            # # # plt.plot(t/3600, Ts[:,int(len(x)/4)]-273, '--',
            # # #          t/3600, Ts[:,int(len(x)/2)]-273, '--',
            # # #          t/3600, Ts[:,int(3*len(x)/4)]-273, '--',
            # # #          t/3600, Ts[:,-1]-273, '--')
            
            # # plt.plot(t/3600, t*0 + 800, 'k--')
            # # plt.plot(t/3600, t*0 + 800*0.95, 'k--')
            
            # # plt.ylabel(r'$T_f$ ($^\circ$C)')
            # # plt.xlabel(r'$t$ (h)')
            # # plt.ylim([500,1000])
            # # plt.legend([r'$x = 0$', r'$x = 1/4$', r'$x = 1/2$', r'$x = 3/4$', r'$x = 1$'])
            # # plt.savefig('./Tf_' + month[nM] + '_' + loc_ +'.eps', bbox_inches='tight',
            # #             format='eps')
            # # plt.show()
            
            
            # #%% PLOTS
            # fig, ax = plt.subplots(figsize=(7*cm, 7*cm))
            
            # plt.plot(t/3600, Tf[:,int(len(x)/4)]-273, t/3600, Tf[:,int(len(x)/2)]-273,
            #          t/3600, Tf[:,int(3*len(x)/4)]-273, t/3600, Tf[:,-1]-273)
            
            # plt.gca().set_prop_cycle(None)
            
            # plt.plot(t/3600, Ts[:,int(len(x)/4)]-273, '--',
            #          t/3600, Ts[:,int(len(x)/2)]-273, '--',
            #          t/3600, Ts[:,int(3*len(x)/4)]-273, '--',
            #          t/3600, Ts[:,-1]-273, '--')
            # plt.ylabel(r'$T$ ($^\circ$C)')
            # plt.xlabel(r'$t$ (h)')
            # plt.legend([r'$x = 1/4$', r'$x = 1/2$', r'$x = 3/4$', r'$x = 1$'])
            # plt.savefig('./Tf_' + month[nM] + '_' + loc_ +'.eps', bbox_inches='tight',
            #             format='eps')
            
            # plt.show()
            
            # #%% PLOTS
            # fig, ax = plt.subplots(figsize=(7*cm, 7*cm))
    
            # plt.plot(t/3600, Tf0-273)
            # plt.ylabel(r'$T_{f0}$ ($^\circ$C)')
            # plt.xlabel(r'$t$ (h)')
            # # plt.legend([r'$x = 1/4$', r'$x = 1/2$', r'$x = 3/4$', r'$x = 1$'])
            # plt.savefig('./Tf0_' + month[nM] + '_' + loc_ +'.eps',
            #             bbox_inches='tight', format='eps')
            
            # plt.show()        
            
            # #%% PLOTS
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
            
            # #%% PLOTS
            # # fig, ax = plt.subplots()
            
            # # plt.plot(t[1:]/3600, 100*err[1:]/Tref)
            # # plt.ylabel(r'$\varepsilon_r$ (\%)')
            # # plt.xlabel(r'$t$ (h)')
            
            # # plt.show()
            
            # #%% PLOTS
            # fig, ax1 = plt.subplots(figsize=(7*cm, 7*cm))
            
            # ax1.plot(t/3600, Tf[:,-1]-273, t/3600, Tin-273,
            #          t/3600, Tf0-273, 'k-')
            # ax1.plot(0, T00-273, 'ro', markersize = 3)
            # ax1.set_ylabel(r'$T_f$ ($^\circ$C)')
            # ax1.set_xlabel(r'$t$ (h)')
            # ax1.set_ylim([195, 1055]) 
            
            
            # # qarr = (Tref - Tf[:,-1])*mDot*cpr.PropsSI('C', 'T', (0.5*(Tref - Tf[:,-1])),
            # #                                           'P', 5e5, 'Air')
            
            # ax2 = ax1.twinx()
            # color = 'tab:green'
            # ax2.plot(t[1:]/3600, QQ[ii,1:]/1e3, color=color)
            # ax2.plot(t[1:-1:500]/3600, QQ[1:-1:500]*0 + 27,
            #          'o-', markersize = 2, color=color)
            
            # # ax2.plot(t[0:-1:500]/3600, qarr[0:-1:500]/1000, 's-',
            # #          markersize = 2, color=color)
            
            # ax2.tick_params(axis='y', labelcolor=color)
            # ax2.tick_params(axis='y', color=color)
            # ax2.spines['right'].set_color(color)
            # ax2.set_ylabel(r'$\dot{Q}_0$ (kW)', color=color)
            # ax2.set_ylim([-1, 28]) 
            # plt.savefig('./Q0_Tf_' + month[nM] + '_' + loc_ +
            #             '_T0_' + str(T00 - 273) +'.eps', bbox_inches='tight',
            #             format='eps')
            
            # plt.show()
            
    
    #%% TOTAL ENERGY
    
        EE[jj,mm] = 0.25*(2*E[0] + E[1] + E[2])
        
        display('EE = ' + str(EE[jj,mm]) + ' kWh')
        display('L/D = ' + str(LbyD[jj]))
        display('mDot = ' + str(G0v[mm]))    
    
# fig, ax = plt.subplots(figsize=(7*cm, 7*cm))
# plt.plot(LbyD, (EE/np.max(EE)), 'o-', markersize=2)
# plt.xlabel('$L/D$')
# plt.ylabel(r'$\tilde{Q}_0^*$')
# plt.show()


with open('./EE_vs_LbyD_vs_mDot_' + loc_, 'w') as archivo:
# Escribir el resultado en el archivo
    archivo.write(str(EE))
    
    
with open('./EE_vs_LbyD_vs_mDot_' + loc_ + '.json', 'w') as archivo:
# Escribir el resultado en el archivo
    json.dump(EE.tolist(), archivo)

      #%% TOTAL ENERGY
          
fig, ax = plt.subplots(figsize=(7*cm, 7*cm))
cs1 = ax.contourf(0.043/G0v, LbyD, EE, 16)
plt.tight_layout()
cbar1 = plt.colorbar(cs1)
cbar1.set_label(r'$\overline{Q}_0$ (kWh)', fontsize=12)
ax.set_ylabel(r'$L/D$')
ax.set_xlabel(r'$N$')
ax.set_box_aspect(1)
plt.savefig('./Q0_vs_LbyD_vs_mDot_' + loc_ +'.eps',
            bbox_inches='tight', format='eps')
plt.show()


fig, ax1 = plt.subplots(figsize=(7*cm, 7*cm))
c = ax1.pcolor(0.043/G0v, LbyD, EE, edgecolors='k', linewidths=2)
ax1.set_ylabel(r'$L/D$')
ax1.set_xlabel(r'$N$')
ax1.set_box_aspect(1)
fig.colorbar(c)
ax1.set_title(r'$\overline{Q}_0$ (kWh)', fontsize=12)
plt.savefig('./Q0_vs_LbyD_vs_mDot_pcolor' + loc_ +'.eps',
            bbox_inches='tight', format='eps')
plt.show()

