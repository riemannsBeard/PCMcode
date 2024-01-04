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
import pandas as pd
import csv
import matplotlib_inline
import warnings

from functions import *

warnings.filterwarnings('ignore')
matplotlib_inline.backend_inline.set_matplotlib_formats('svg', 'pdf')

#%% DEFAULTS

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ["Computer Modern Roman"]
plt.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['lines.markersize'] = 3

#%% LOAD DATA

Ad = 44
tf = 24
dt = 0.5
t = np.arange(0, tf + dt, dt)
G0 = 0.043

# AMBIENT TEMPERATURE
Tamb_june = np.loadtxt(r"./CordobaJune/Cordoba_June_Tamb.csv",
                       delimiter=",", dtype=float)
Tamb_march = np.loadtxt('./CordobaMarch/Cordoba_March_Tamb.csv',
                        delimiter=",", dtype=float)
Tamb_dec = np.loadtxt('./CordobaDec/Cordoba_Dec_Tamb.csv',
                      delimiter=",", dtype=float)

# Interpolacion
Tamb_june = np.interp(t, Tamb_june[:,0], Tamb_june[:,1]) + 273
Tamb_march = np.interp(t, Tamb_march[:,0], Tamb_march[:,1]) + 273
Tamb_dec = np.interp(t, Tamb_dec[:,0], Tamb_dec[:,1]) + 273

fig, ax1 = plt.subplots()
ax1.plot(t, Tamb_march - 273, t, Tamb_june - 273, t, Tamb_dec - 273)
plt.legend([r'March 24th', r'June 24th', 
            r'Dec. 22th'], loc='upper left')
ax1.set_box_aspect(1)
ax1.set_xlabel(r'$t$ (h)')
ax1.set_ylabel(r'$T$ ($^\circ$C)')

# DNI (W/m^2) and Ib (W)
DNI_march = np.loadtxt('./CordobaMarch/Cordoba_March_DNI.csv', delimiter=",",
                       dtype=float)
DNI_june = np.loadtxt('./CordobaJune/Cordoba_June_DNI.csv', delimiter=",",
                      dtype=float)
DNI_dec = np.loadtxt('./CordobaDec/Cordoba_Dec_DNI.csv', delimiter=",",
                     dtype=float)

# Interpolacion
Ib_june = Ad*np.interp(t, DNI_june[:,0], DNI_june[:,1])
Ib_march = Ad*np.interp(t, DNI_march[:,0], DNI_march[:,1])
Ib_dec = Ad*np.interp(t, DNI_dec[:,0], DNI_dec[:,1])


fig, ax1 = plt.subplots()
ax1.plot(t, Ib_march*1e-3, t, Ib_june*1e-3, t, Ib_dec*1e-3)
plt.legend([r'March 24th', r'June 24th', 
            r'Dec. 22th'], loc='upper left')
ax1.set_box_aspect(1)
ax1.set_xlabel(r'$t$ (h)')
ax1.set_ylabel(r'$I_b$ (kW)')


# ELECTRICITY DEMAND (MW)
dem_march = np.loadtxt('./March_demand.csv',
                        delimiter=";", dtype=float)
dem_june = np.loadtxt('./June_demand.csv',
                        delimiter=";", dtype=float)
dem_dec = np.loadtxt('./Dec_demand.csv',
                        delimiter=";", dtype=float)

# Interpolacion
dem_march = np.interp(t, dem_march[:,0], dem_march[:,1])
dem_june = np.interp(t, dem_june[:,0], dem_june[:,1])
dem_dec = np.interp(t, dem_dec[:,0], dem_dec[:,1])

fig, ax1 = plt.subplots()
plt.plot(t, dem_march, t, dem_june, t, dem_dec)
plt.title('Energy Demand')
plt.legend([r'March 24th', r'June 24th', 
            r'Dec. 22th'], loc='upper left')
ax1.set_xlabel(r'$t$ (h)')
ax1.set_ylabel(r'MW')
plt.show()



# ELECTRICITY GENERATION (MW)
cl = plt.rcParams['axes.prop_cycle'].by_key()['color']

rwidx = ['Hora', 'Eólica', 'Hidráulica', 'Solar fotovoltaica', 'Solar térmica',
         'Térmica renovable']
nonrwidx = ['Hora', 'Nuclear', 'Fuel/gas', 'Carbón', 'Ciclo combinado',
       'Intercambios int', 'Enlace balear', 'Cogeneración y residuos']

gen_march = pd.read_csv('./March_gen.csv', delimiter=";", encoding='latin-1')
gen_rw_march = gen_march[rwidx]
gen_nonrw_march = gen_march[nonrwidx]

gen_june = pd.read_csv('./June_gen.csv', delimiter=";", encoding='latin-1')
gen_rw_june = gen_june[rwidx]
gen_nonrw_june = gen_june[nonrwidx]

gen_dec = pd.read_csv('./Dec_gen.csv', delimiter=";", encoding='latin-1')
gen_rw_dec = gen_dec[rwidx]
gen_nonrw_dec = gen_dec[nonrwidx]


# Interpolacion
rw_march_gen = np.interp(t, gen_rw_march.iloc[:,0],
                         gen_rw_march.iloc[:,1:].sum(axis=1))
nonrw_march_gen = np.interp(t, gen_nonrw_march.iloc[:,0],
                         gen_nonrw_march.iloc[:,1:].sum(axis=1))

rw_june_gen = np.interp(t, gen_rw_june.iloc[:,0],
                         gen_rw_june.iloc[:,1:].sum(axis=1))
nonrw_june_gen = np.interp(t, gen_nonrw_june.iloc[:,0],
                         gen_nonrw_june.iloc[:,1:].sum(axis=1))

rw_dec_gen = np.interp(t, gen_rw_dec.iloc[:,0],
                         gen_rw_dec.iloc[:,1:].sum(axis=1))
nonrw_dec_gen = np.interp(t, gen_nonrw_dec.iloc[:,0],
                         gen_nonrw_dec.iloc[:,1:].sum(axis=1))


fig, ax1 = plt.subplots()
plt.title('Renewable (-) vs. Non-Renewable (- -) Energy Generation')
ax1.plot(t, rw_march_gen, '-', color=cl[0], label='March')
ax1.plot(t, nonrw_march_gen, '--', color=cl[0])
ax1.plot(t, rw_june_gen, '-', color=cl[1], label='June')
ax1.plot(t, nonrw_june_gen, '--', color=cl[1])
ax1.plot(t, rw_dec_gen, '-', color=cl[2], label='Dec.')
ax1.plot(t, nonrw_dec_gen, '--', color=cl[2])
plt.legend()
ax1.set_xlabel(r'$t$ (h)')
ax1.set_ylabel(r'MW')
plt.show()


# ELECTRICITY PRICE (€/MWh)
pelec_march = np.loadtxt('./March_pelec.csv',
                        delimiter=";", dtype=float)
pelec_june = np.loadtxt('./June_pelec.csv',
                        delimiter=";", dtype=float)
pelec_dec = np.loadtxt('./Dec_pelec.csv',
                        delimiter=";", dtype=float)

# Interpolacion
pelec_march = np.interp(t, pelec_march[:,1], pelec_march[:,0])
pelec_june = np.interp(t, pelec_june[:,1], pelec_june[:,0])
pelec_dec = np.interp(t, pelec_dec[:,1], pelec_dec[:,0])


fig, ax1 = plt.subplots()
plt.plot(t, pelec_march, t, pelec_june, t, pelec_dec)
plt.legend([r'March 24th', r'June 24th', 
            r'Dec. 22th'], loc='upper left')
ax1.set_xlabel(r'$t$ (h)')
ax1.set_ylabel(r'€/MWh')



# CO2 EMISSIONS (t CO2 eq/MWh)
em_march = np.loadtxt('./March_em.csv',
                        delimiter=";", dtype=float)
em_june = np.loadtxt('./June_em.csv',
                        delimiter=";", dtype=float)
em_dec = np.loadtxt('./Dec_em.csv',
                        delimiter=";", dtype=float)

# Interpolacion
em_march = np.interp(t, em_march[:,0], em_march[:,1])
em_june = np.interp(t, em_june[:,0], em_june[:,1])
em_dec = np.interp(t, em_dec[:,0], em_dec[:,1])


fig, ax1 = plt.subplots()
plt.title(r'Emissions')
plt.plot(t, em_march, t, em_june, t, em_dec)
plt.legend([r'March 24th', r'June 24th', 
            r'Dec. 22th'], loc='upper left')
ax1.set_xlabel(r'$t$ (h)')
ax1.set_ylabel(r'CO$_2$ eq. t/MWh')



#%% SOLAR DISH MODEL


# Ib, G, Ti, Tamb = 28e3, 0.043, 330, 313.0



Ti0 = 400 #500 + 273

# unks00 = np.array([343.42854746, 363.03218814, 370.13383945, 693.71771782,
#        792.54183421, 654.51447981, 628.93153313, 594.0143274, 671.73647441])

# T1, T2, T3, T4, Tf, Tw, Tgi, Tgo, To
unks00 = np.array([Ti0*1.05, Ti0*1.10, Ti0*1.15, Ti0*1.20,
       Ti0*1.25, Ti0*1.5, Ti0*1.07, Ti0*1.03, Ti0*1.27])

# Object with thermophysical properties of ideal air
# mu = cp.PropsSI('V', 'T', 1273, 'P', 5e5, 'Air')

monthNo = 0, 1, 2
month = 'March 24th', 'June 24th', 'Dec. 24th'

Ibs = np.array([Ib_march, Ib_june, Ib_dec])
Tambs = np.array([Tamb_march, Tamb_june, Tamb_dec])

for j in range(0, 3):
    
    # Initialize solution arrays
    To = np.zeros(len(t))
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
                              Tgi[i-1], Tgo[i-1], To[i-1]]) + 273
    
        sol = root(dish, unks0, args=args0, method='lm')
        T1[i], T2[i], T3[i], T4[i], Tf[i], Tw[i], Tgi[i], Tgo[i], To[i] = sol.x
        flag[j,i] = sol.success
        
        Qloss[:,i] = Qlosses(Ib, Tamb, T1[i], T2[i], T3[i], T4[i], Tf[i], Tw[i],
                        Tgi[i], Tgo[i], To[i])
        eta[i] = (Ib - np.sum(Qloss[:,i]))/Ib
        
        T1[i], T2[i], T3[i], T4[i], Tf[i], Tw[i], Tgi[i], Tgo[i], To[i] = sol.x - 273
    
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
    ax1.plot(t, (Tambs[j] - 273), t, To/10, t, Tf/10, t, Ibs[j]/1e3, 'k--')
    plt.legend([r'$T_a$ ($^\circ$C)',  r'$T_o/10$ ($^\circ$C)', r'$T_f/10$ ($^\circ$C)',
                r'$I_b$ (kW)'], loc='upper left')
    ax1.set_xlabel(r'$t$ (h)')
    ax1.set_title(month[j])
    ax1.set_box_aspect(0.8)
    # ax1.set_ylim([-5, 105])
    
    
    ax2 = ax1.twinx()
    color = 'tab:purple'
    ax2.plot(t, eta*100, color=color)
    ax2.set_ylabel(r'$\eta$ (\%)', color=color)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([-5, 105])
    ax2.set_title(month[j])
    ax2.set_box_aspect(0.8)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
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

#%% CAIDA DE PRESION POR EFECTO DE LA TEMPERATURA (despreciable)

# p1 = 5e5
# T1 = 1e3
# T2 = 700
# rho1 = cpr.PropsSI('D', 'T', T1, 'P', p1, 'Air')

# p2, rho2, v2, T2  = pDrop(p1, T1, T2, G, 0.5)

