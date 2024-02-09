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


#%% DATA
month = ['Dec.', 'March', 'June']
N = ['1', '2', '4']
# LbyD_C = [2.0, 1.5, 1.5]
# LbyD_M = [2.0, 1.0, 1.0]

pC = np.array([[523, 355, 206], [434, 228, 196], [625, 600, 545]])
pM = np.array([[552, 452, 241], [451, 232, 196], [633, 615, 577]])

# Configurar colores
colors = ['r', 'g', 'b']

cm = 1/2.54 

# Crear el gráfico de barras
fig, ax = plt.subplots(figsize=(14*cm, 14*cm))
# Para cada mes
for i, m in enumerate(month):
    # Obtener los valores para cada N
    values = pC.T[i]
    # Graficar las barras para cada N, con el mismo color pero diferente ancho
    plt.bar(month, values, bottom=0, color=colors[i], label=m)

# plt.xlabel('N')
plt.ylabel('Precio')
plt.legend([r'$N = 1$', r'$N = 2$', r'$N = 4$'])
ax.set_box_aspect(1)
ax.set_ylim([0, 800])
plt.savefig('./pCbar.eps', bbox_inches='tight',
            format='eps')

plt.show()


# Crear el gráfico de barras
fig, ax = plt.subplots(figsize=(14*cm, 14*cm))

# Para cada mes
for i, m in enumerate(month):
    # Obtener los valores para cada N
    values = pM.T[i]
    # Graficar las barras para cada N, con el mismo color pero diferente ancho
    plt.bar(month, values, bottom=0, color=colors[i], label=m)

# plt.xlabel('N')
plt.ylabel('Precio')
plt.legend([r'$N = 1$', r'$N = 2$', r'$N = 4$'])
ax.set_box_aspect(1)
ax.set_ylim([0, 800])
plt.savefig('./pMbar.eps', bbox_inches='tight',
            format='eps')
        
plt.show()
