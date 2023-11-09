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

#%% VALIDATION

# Ib, G, Ti, Tamb = 28e3, 0.044, 317, 307.3
Ib, G, Ti, Tamb = 28e3, 0.043, 330, 313.0
args0 = (Ib, G, Ti, Tamb)
    

# T1, T2, T3, T4, Tf, Tw, Tgi, Tgo, To
unks0 = np.array([77.0, 91.0, 97.0, 496.0, 300.5, 257.0, 400.0, 233.0, 450.0]) + 273

sol = root(dish, unks0, args=args0, method='hybr')


temps = ('Ti', 'T1', 'T2', 'T3', 'T4', 'Tf', 'Tw', 'Tgi', 'Tgo', 'To')
losses = ('glass emission', 'glass reflectance', 'foam emission', 'foam reflectance',
          'wall emission', 'convection', 'conduction')

print()
print(sol.x)

T1, T2, T3, T4, Tf, Tw, Tgi, Tgo, To = sol.x

print()
print(list(zip(temps, np.insert(sol.x - 273, 0, Ti - 273))))


# Qlglass, QlglassR, Qlfe, Qlfr, Qlwall, Qlconv
Qloss = Qlosses(Ib, Tamb, T1, T2, T3, T4, Tf, Tw, Tgi, Tgo, To)
eta = (Ib - np.sum(Qloss))/Ib

print()
print('Eta = ', eta)
print()
print('Losses (%):')
print(list(zip(losses, Qloss/Ib*100)))

y_pos = np.arange(len(losses))

plt.bar(y_pos, Qloss/Ib*100, align='center', alpha=0.5)
plt.xticks(y_pos, losses, rotation='vertical')
plt.ylabel('(\%)')
plt.title('Losses')

plt.show()