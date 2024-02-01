from scipy import signal, fftpack
from scipy.integrate import odeint, solve_ivp, cumtrapz, RK45
from scipy.interpolate import interp1d
from scipy.optimize import fsolve, newton, least_squares, root, minimize
from scipy import interpolate
import matplotlib.pyplot as plt
import math
import scipy as sp
import numpy as np
import pandas as pd
import csv
import matplotlib_inline
import warnings

from functions import *

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


#%% LOAD DATA

Ad = 44
tf = 24
dt = 0.5
t = np.arange(0, tf + dt, dt)
ts = t
# G0 = 0.043/2

loc = ['Cordoba', 'Malaga']
loc_ = loc[1]


# AMBIENT TEMPERATURE
Tamb_june = np.loadtxt('./' + loc_ + 'June/' + loc_ + '_June_Tamb.csv',
                       delimiter=",", dtype=float)
Tamb_march = np.loadtxt('./' + loc_ + 'March/' + loc_ + '_March_Tamb.csv',
                        delimiter=",", dtype=float)
Tamb_dec = np.loadtxt('./' + loc_ + 'Dec/' + loc_ + '_Dec_Tamb.csv',
                      delimiter=",", dtype=float)

# Interpolacion
Tamb_june = np.interp(t, Tamb_june[:,0], Tamb_june[:,1]) + 273
Tamb_march = np.interp(t, Tamb_march[:,0], Tamb_march[:,1]) + 273
Tamb_dec = np.interp(t, Tamb_dec[:,0], Tamb_dec[:,1]) + 273


# DNI (W/m^2) and Ib (W)
DNI_march = np.loadtxt('./' + loc_ + 'March/' + loc_ +
                       '_March_DNI.csv', delimiter=",",
                       dtype=float)
DNI_june = np.loadtxt('./' + loc_ + 'June/' + loc_ +
                      '_June_DNI.csv', delimiter=",",
                      dtype=float)
DNI_dec = np.loadtxt('./' + loc_ + 'Dec/' + loc_ +
                     '_Dec_DNI.csv', delimiter=",",
                     dtype=float)

# Interpolacion
Ib_june = Ad*np.interp(t, DNI_june[:,0], DNI_june[:,1])
Ib_march = Ad*np.interp(t, DNI_march[:,0], DNI_march[:,1])
Ib_dec = Ad*np.interp(t, DNI_dec[:,0], DNI_dec[:,1])


#%% SOLAR DISH MODEL


# Ib, G, Ti, Tamb = 28e3, 0.043, 330, 313.0


# From Garcia-Ferrero 2023
Ti0 = 227 + 273

# unks00 = np.array([343.42854746, 363.03218814, 370.13383945, 693.71771782,
#        792.54183421, 654.51447981, 628.93153313, 594.0143274, 671.73647441])

# T1, T2, T3, T4, Tf, Tw, Tgi, Tgo, To
unks00 = np.array([Ti0*1.05, Ti0*1.10, Ti0*1.15, Ti0*1.20,
       Ti0*1.25, Ti0*1.5, Ti0*1.07, Ti0*1.03, Ti0*1.27])

# Object with thermophysical properties of ideal air
# mu = cp.PropsSI('V', 'T', 1273, 'P', 5e5, 'Air')

monthNo = 0, 1, 2
month = 'March', 'June', 'December'

Ibs = np.array([Ib_march, Ib_june, Ib_dec])
Tambs = np.array([Tamb_march, Tamb_june, Tamb_dec])

To = np.zeros((3,len(t)))

for j in range(0, 3):
    
    # Initialize solution arrays
    T1 = np.zeros(len(t))
    T2 = np.zeros(len(t))
    T3 = np.zeros(len(t))
    T4 = np.zeros(len(t))
    Tf = np.zeros(len(t))
    Tw = np.zeros(len(t))
    Tgi = np.zeros(len(t))
    Tgo = np.zeros(len(t))

    Qloss = np.zeros((7, len(t)))
    eta = np.zeros(len(t))

    flag = np.ones((len(month), len(t)))

    # Solar radiation power impining the dish
    ii = np.where(Ibs[j] >= 7e3)
    i0 = ii[0]
    for i in range(i0[0], i0[-1]):
    
        Ib, G, Ti, Tamb = Ibs[j,i], G0, Ti0, Tambs[j,i]
        args0 = (Ib, G, Ti, Tamb)
    
        # T1, T2, T3, T4, Tf, Tw, Tgi, Tgo, To
        # unks0 = np.array([77.0, 91.0, 97.0, 496.0, 300.5, 257.0, 400.0, 233.0, 450.0]) + 273
        
        if i == i0[0]:
            unks0 = unks00
        else:
            unks0 = np.array([T1[i-1], T2[i-1], T3[i-1], T4[i-1], Tf[i-1], Tw[i-1],
                              Tgi[i-1], Tgo[i-1], To[j,i-1]]) + 273
    
        sol = root(dish, unks0, args=args0, method='lm')
        T1[i], T2[i], T3[i], T4[i], Tf[i], Tw[i], Tgi[i], Tgo[i], To[j,i] = sol.x
        flag[j,i] = sol.success
        
        Qloss[:,i] = Qlosses(Ib, Tamb, T1[i], T2[i], T3[i], T4[i], Tf[i], Tw[i],
                        Tgi[i], Tgo[i], To[j,i])
        eta[i] = (Ib - np.sum(Qloss[:,i]))/Ib
        
        T1[i], T2[i], T3[i], T4[i], Tf[i], Tw[i], Tgi[i], Tgo[i], To[j,i] = sol.x - 273
    
        temps = ('Ti', 'T1', 'T2', 'T3', 'T4', 'Tf', 'Tw', 'Tgi', 'Tgo', 'To')
        losses = ('glass emission', 'glass reflectance', 'foam emission',
                  'foam reflectance', 'wall emission', 'convection', 'conduction')
    
        print()
        print(list(zip(temps, np.insert(sol.x - 273, 0, Ti - 273))))
        
        print()
        print('Eta (%)= ', eta[i]*100)
        print()
        print('Losses (%):')
        print(list(zip(losses, Qloss[:,i]/Ib*100)))
        print()
    
    #%% PLOTS
    
    fig, ax1 = plt.subplots()
    ax1.plot(t, (Tambs[j] - 273), t, To[j,:]/10, t, Tf/10, t, Ibs[j]/1e3, 'k-')
    plt.legend([r'$T_a$ ($^\circ$C)',  r'$T_o/10$ ($^\circ$C)', r'$T_f/10$ ($^\circ$C)',
                r'$I_b$ (kW)'], loc='upper left')
    ax1.set_xlabel(r'$t$ (h)')
    ax1.set_ylim([-5, 85])
    ax1.set_title(month[j])
    # ax1.set_ylim([-5, 105])
    
    
    ax2 = ax1.twinx()
    color = 'tab:purple'
    ax2.plot(t, eta*100, color=color)
    ax2.set_ylabel(r'$\eta$ (\%)', color=color)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([-5, 105])
    ax2.set_title(month[j])
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('./' + loc_+ '_' + str(month[j]) + '.eps', bbox_inches='tight',
                format='eps')
    plt.show()



# fig, ax1 = plt.subplots()
# ax1.plot(t, Tgo, t, Tgi, t, Tf)
# plt.legend([r'$T_{go}$ ($^\circ$C)',  r'$T_{gi}$ ($^\circ$C)',
#             r'$T_{f}$ ($^\circ$C)'], loc='upper left')
# ax1.set_xlabel(r'$t$ (h)')

fig, ax1 = plt.subplots()
c = ax1.pcolor(flag, edgecolors='k', linewidths=2)
fig.colorbar(c)
ax1.set_title('convergence')
