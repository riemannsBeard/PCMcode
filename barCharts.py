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
pMonth = ['March', 'June', 'Dec.']
N = ['1', '2', '4']
# LbyD_C = [2.0, 1.5, 1.5]
# LbyD_M = [2.0, 1.0, 1.0]

pC = np.array([[169, 112, 58], [165, 86, 73], [76, 73, 66]])
pM = np.array([[179, 146, 69], [171, 88, 69], [77, 75, 70]])

qC = np.array([[625, 600, 545], [523, 355, 206], [434, 228, 196]])
qM = np.array([[633, 615, 577], [552, 452, 241], [451, 232, 196]])

# Configurar colores
colors = ['r', 'g', 'b']

cm = 1/2.54 

# Crear el gráfico de barras
fig, ax = plt.subplots(figsize=(14*cm, 14*cm))
# Para cada mes
for i, m in enumerate(pMonth):
    # Obtener los valores para cada N
    values = pC[i]
    # Graficar las barras para cada N, con el mismo color pero diferente ancho
    plt.bar(N, values, bottom=0, color=colors[i], label=m)

# plt.xlabel('N')
plt.ylabel('Precio')
plt.legend(month)
ax.set_box_aspect(1)
ax.set_ylim([0, 300])
plt.savefig('./pCbar.eps', bbox_inches='tight',
            format='eps')

plt.show()


# Crear el gráfico de barras
fig, ax = plt.subplots(figsize=(14*cm, 14*cm))

# Para cada mes
for i, m in enumerate(pMonth):
    # Obtener los valores para cada N
    values = pM[i]
    # Graficar las barras para cada N, con el mismo color pero diferente ancho
    plt.bar(N, values, bottom=0, color=colors[i], label=m)

# plt.xlabel('N')
plt.ylabel('Precio')
plt.legend(month)
ax.set_box_aspect(1)
ax.set_ylim([0, 300])
plt.savefig('./pMbar.eps', bbox_inches='tight',
            format='eps')
        
plt.show()



# Crear el gráfico de barras
fig, ax = plt.subplots(figsize=(14*cm, 14*cm))
# Para cada mes
for i, m in enumerate(month):
    # Obtener los valores para cada N
    values = qC[i]
    # Graficar las barras para cada N, con el mismo color pero diferente ancho
    plt.bar(N, values, bottom=0, color=colors[i], label=m)

# plt.xlabel('N')
plt.ylabel('$Q$ (kWh)')
plt.legend(month)
ax.set_box_aspect(1)
ax.set_ylim([0, 800])
plt.savefig('./qCbar.eps', bbox_inches='tight',
            format='eps')

plt.show()


# Crear el gráfico de barras
fig, ax = plt.subplots(figsize=(14*cm, 14*cm))

# Para cada mes
for i, m in enumerate(month):
    # Obtener los valores para cada N
    values = qM[i]
    # Graficar las barras para cada N, con el mismo color pero diferente ancho
    plt.bar(N, values, bottom=0, color=colors[i], label=m)

# plt.xlabel('N')
plt.ylabel('$Q$ (kWh)')
plt.legend(month)
ax.set_box_aspect(1)
ax.set_ylim([0, 800])
plt.savefig('./qMbar.eps', bbox_inches='tight',
            format='eps')
        
plt.show()

