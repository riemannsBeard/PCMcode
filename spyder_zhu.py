# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

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
warnings.filterwarnings('ignore')

matplotlib_inline.backend_inline.set_matplotlib_formats('svg', 'pdf')


plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Times'
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['lines.markersize'] = 3


def cp(T):

    coeffs = np.array([[1068.53, -0.5252, 1.338e-3, -1.031e-6, 3.208e-10,
                        -2.908e-14],
                       [-4.457e-4, 1.089e-4, -8.1629e-8, 6.323e-11, -2.734e-14,
                        4.944e-18],
                       [2.374e-8, 7.740e-8, -6.885e-11, 5.362e-14, -2.338e-17,
                        4.256e-21]])

    cp_ = coeffs[0, 0] + coeffs[0, 1]*T + coeffs[0, 2]*T**2 + coeffs[0, 3]*T**3 +\
        coeffs[0, 4]*T**4 + coeffs[0, 5]*T**5

    return cp_


def kair(T):

    coeffs = np.array([[1068.53, -0.5252, 1.338e-3, -1.031e-6, 3.208e-10,
                        -2.908e-14],
                       [-4.457e-4, 1.089e-4, -8.1629e-8, 6.323e-11, -2.734e-14,
                        4.944e-18],
                       [2.374e-8, 7.740e-8, -6.885e-11, 5.362e-14, -2.338e-17,
                        4.256e-21]])

    kair_ = coeffs[1, 0] + coeffs[1, 1]*T + coeffs[1, 2]*T**2 + coeffs[1, 3]*T**3 +\
        coeffs[1, 4]*T**4 + coeffs[1, 5]*T**5

    return kair_


def kk(T):

    Temp_ = [300, 400, 600, 800, 1000]
    kk_ = [15.1, 17.3, 20.0, 22.8, 25.4]

    interp = interp1d(Temp_, kk_, fill_value=True, bounds_error=False)

    return interp(T)[()]


def mu(T):

    coeffs = np.array([[1068.53, -0.5252, 1.338e-3, -1.031e-6, 3.208e-10,
                        -2.908e-14],
                       [-4.457e-4, 1.089e-4, -8.1629e-8, 6.323e-11, -2.734e-14,
                        4.944e-18],
                       [2.374e-8, 7.740e-8, -6.885e-11, 5.362e-14, -2.338e-17,
                        4.256e-21]])

    mu_ = coeffs[2, 0] + coeffs[2, 1]*T + coeffs[2, 2]*T**2 + coeffs[2, 3]*T**3 +\
        coeffs[2, 4]*T**4 + coeffs[2, 5]*T**5

    return mu_


def rhoA(T):

    coeffs = np.array([8.78552, -7.54226e-2, 2.69671e-4, -3.42800e-7])

    rhoA_ = coeffs[0] + coeffs[1]*T + coeffs[2]*T**2 + coeffs[3]*T**3

    return rhoA_


def cpMean(Th, Tc):

    coeffs = np.array([[1068.53, -0.5252, 1.338e-3, -1.031e-6, 3.208e-10,
                        -2.908e-14],
                       [-4.457e-4, 1.089e-4, -8.1629e-8, 6.323e-11, -2.734e-14,
                        4.944e-18],
                       [2.374e-8, 7.740e-8, -6.885e-11, 5.362e-14, -2.338e-17,
                        4.256e-21]])

    if Th > Tc:
        cpMean_ = 1/(Th - Tc)*(coeffs[0, 0]*(Th - Tc) +
                               coeffs[0, 1]/2*(Th**2 - Tc**2) +
                               coeffs[0, 2]/3*(Th**3 - Tc**3) +
                               coeffs[0, 3]/4*(Th**4 - Tc**4) +
                               coeffs[0, 4]/5*(Th**5 - Tc**5) +
                               coeffs[0, 5]/6*(Th**6 - Tc**6))
    elif Th < Tc:
        aux = Th
        Th = Tc
        Tc = aux

        cpMean_ = 1/(Th - Tc)*(coeffs[0, 0]*(Th - Tc) +
                               coeffs[0, 1]/2*(Th**2 - Tc**2) +
                               coeffs[0, 2]/3*(Th**3 - Tc**3) +
                               coeffs[0, 3]/4*(Th**4 - Tc**4) +
                               coeffs[0, 4]/5*(Th**5 - Tc**5) +
                               coeffs[0, 5]/6*(Th**6 - Tc**6))
    elif Th == Tc:
        cpMean_ = cp(Th)

    return cpMean_


# def zone1L(unks, *args0):
#     T1, T2, T3, T4, Tf, Tw, Tg, To = unks

#     rg = 0.125
#     rf = 0.182

#     ri = 0.197
#     ro = 0.2

#     L1 = 0.195
#     L2 = 0.1079


#     # Internal losses
#     Dh = 2*(ri - rf)

#     TL1in = 0.5*(Ti + TL1)
    
#     Re = 2*G/(np.pi*(ri + rf)*mu(TL1in))
#     Pr = cp(TL1in)*mu(TL1in)/kair(TL1in)
#     fp = (0.790*np.log(Re) - 1.64)**(-2)

#     if (Re > 3000 and Re <= 5e6) and (Pr > 0.5 and Pr <= 2e3):
#         NuL1in = (fp/8)*(Re - 1e3)*Pr/(1 + 12.7*np.sqrt(fp/8)*(Pr**(2/3) - 1))
#         hL1in = NuL1in*kair(TL1in)/Dh

#     else:
#         NuL1in = 0.664*Re**0.5*Pr**(1/3)
#         hL1in = NuL1in*kair(TL1in)/L1
    
#     ki0 = 0.06 # Aluminum silicate (0.06 W/(m.K))
    
#     AiL1cyl = 2*np.pi*ri*L1
#     UcylL1 = 1/(1/hL1in + ri*np.log(ro/ri)/ki0)

#     rpo = 0.01
#     rpi = 0.042
#     eo = 3e-3

#     AiL1flat = np.pi*(ri**2 - rpo**2 - 3*rpi**2)
#     UflatL1 = 1/(1/hL1in + eo/ki0)
    
#     AiL1UL1 = AiL1cyl*UcylL1 + AiL1flat*UflatL1


#     # Ambient losses
#     TL1amb = 0.5*(TL1 + Tamb)
#     Pr = cp(TL1amb)*mu(TL1amb)/kair(TL1amb)

#     # Radiation losses
#     epsL1 = 0.8
#     sigma = 5.67e-8  # W/(m^2 K^4) (Stephan-Boltzmann constant)

#     hrL1 = epsL1*sigma*(TL1 + Tamb)*(TL1**2 + Tamb**2)
#     Ao1 = 2*np.pi*ro*L1 + np.pi*(ro**2 - rpo**2 - 3*rpi**2)
       
#     # Convection heat transfer coeff horizontal flat plate (Cengel)
#     beta = 1/(0.5*(Tamb + TL1))
#     g = 9.81
#     Pr = cp(1/beta)*mu(1/beta)/kair(1/beta)
#     Ra = Pr*g*beta*(1/beta - Tamb)*(L1)**3/(mu(1/beta)/rhoA(1/beta))**2
    
#     NucL1 = (0.6 + (0.387*Ra**(1/6))/(1 + (0.559/Pr)**(9/16))**(8/27))**2
#     hcL1 = NucL1*kair(1/beta)/(L1)
    
#     NucL1 = (0.68 + (0.67*Ra**0.25)/(1 + (0.492/Pr)**(9/16))**(4/9))


#     QL11 = AiL1UL1*((T1 - TL1) - (Ti - TL1))/np.log((T1 - TL1)/(Ti - TL1))
#     QL12 = Ao1*(hcL1 + hrL1)*(TL1 - Tamb)

#     return QL11, QL12


def zone1(unks, *args0):
    T1, T2, T3, T4, Tf, Tw, Tg, To = unks

    rg = 0.125
    rf = 0.182

    ri = 0.197
    ro = 0.2

    L1 = 0.195
    L2 = 0.1079

    Dh = 2*(ri - rf)

    # Convection heat transfer coeff i-1 (B.1)
    Ti1 = 0.5*(Ti + T1)

    Re = 2*G/(np.pi*(ri + rf)*mu(Ti1))  # 4*G/(np.pi*Dh*mu(Ti1))
    Pr = cp(Ti1)*mu(Ti1)/kair(Ti1)
    fp = (1.8*np.log10(Re) - 1.5)**(-2)

    if (Re > 3000 and Re <= 5e6) and (Pr > 0.5 and Pr <= 2e3):
        ct = (T1/Tw)**0.45
        Nui1 = (fp/8)*(Re - 1e3)*Pr/(1 + 12.7*np.sqrt(fp/8)*(Pr**(2/3) - 1))*\
            (1 + (Dh/L1)**(2/3))*ct
        hi1 = Nui1*kair(Ti1)/Dh

    else:
        Nui1 = 7.54 + 0.03*(Dh/L1)*Re*Pr/(1 + 0.016*((Dh/L1)*Re*Pr)**(2/3))
        hi1 = Nui1*kair(Ti1)/L1

    #
    # Convection heat transfer coeff 4-o (B.5)
    T4o = 0.5*(T4 + To)
    
    Re = 4*G/(2*np.pi*rf*mu(T4o))
    Pr = cp(T4o)*mu(T4o)/kair(T4o)
    fp = (1.8*np.log10(Re) - 1.5)**(-2)
    
    if (Re > 3000 and Re <= 5e6) and (Pr > 0.5 and Pr <= 2e3):
        ct = (T4/Tw)**0.45
        Nu4o = (fp/8)*(Re - 1e3)*Pr/(1 + 12.7*np.sqrt(fp/8)*(Pr**(2/3) - 1))*\
            (1 + (Dh/L1)**(2/3))*ct
        h4o = Nu4o*kair(T4o)/Dh

    else:
        Nu4o = 0.664*Re**0.5*Pr**(1/3)
        h4o = Nu4o*kair(T4o)/L1
    
    # Overall heat transfer
    ew = 1e-3  # inner cylinder wall thickness
    A1 = 2*np.pi*rf*L1
    kw = 0.5*(kk(Ti1) + kk(T4o))  # stainless steel plate (k = 15 W/m·K) Çengel
    U1 = 1/(1/hi1 + ew/kw + 1/h4o)

    Q11 = G*cpMean(T1, Ti)*(T1 - Ti)
    Q12 = G*cpMean(T4, To)*(T4 - To)
    Q13 = U1*A1*((To - Ti) - (T4 - T1))/np.log((To - Ti)/(T4 - T1))

    return Q11, Q12, Q13


# def zone2L(unks, *args0):
#     T1, T2, T3, T4, Tf, Tw, Tg, To = unks

#     rg = 0.125

#     rg = 0.125
#     rf = 0.182

#     ri = 0.197
#     ro = 0.2

#     L1 = 0.195
#     L2 = 0.1079

#     Aw = 0.1788

#     Dh = 2*(ri - rf)

#     TL2in = 0.5*(Ti + TL2)

#     # Convection heat transfer coeff horizontal flat plate (Cengel)
#     beta = 1/(0.5*(Ti + TL2))
#     g = 9.81
#     Pr = cp(1/beta)*mu(1/beta)/kair(1/beta)
#     Ra = Pr*g*beta*(TL2 - Tamb)*L2**3/(mu(1/beta)/rhoA(1/beta))**2

#     NuL2in = 0.27*Ra**(1/4)
#     hL2in = NuL2in*kair(1/beta)/L1
    
    
#     ki0 = 0.06 # Aluminum silicate (0.06 W/(m.K))
#     ki = ki0*((T1 + T2 + TL2)/3)

#     AiL2cyl = 2*np.pi*ri*L2
#     UcylL2 = 1/(1/hL2in + ri*np.log(ro/ri)/ki)

#     rpo = 0.01
#     rpi = 0.042
#     eo = 3e-3
#     AiL2flat = np.pi*(ri**2 - rg**2)
#     UflatL2 = 1/(1/hL2in + eo/ki)

#     AiL2UL2 = AiL2cyl*UcylL2 + AiL2flat*UflatL2

#     epsL2 = 0.8
#     sigma = 5.67e-8  # W/(m^2 K^4) (Stephan-Boltzmann constant)

#     hrL2 = epsL2*sigma*(TL2 + Tamb)*(TL2**2 + Tamb**2)
#     Ao2 = 2*np.pi*ro*L1 + np.pi*(ro**2 - rpo**2 - 3*rpi**2)

#     TL2amb = 0.5*(TL2 + Tamb)

#     Re = 2*G/(np.pi*(ri + rf)*mu(TL2amb))
#     Pr = cp(TL2amb)*mu(TL2amb)/kair(TL2amb)

    
#     # Convection heat transfer insulator L2
#     Re = 2*G/(np.pi*(ri + rf)*mu(TL2amb))
#     Pr = cp(TL2amb)*mu(TL2in)/kair(TL2amb)
#     fp = (0.790*np.log(Re) - 1.64)**(-2)

#     if (Re > 3000 and Re <= 5e6) and (Pr > 0.5 and Pr <= 2e3):
#         NucL2 = (fp/8)*(Re - 1e3)*Pr/(1 + 12.7*np.sqrt(fp/8)*(Pr**(2/3) - 1))

#     else:
#         NucL2 = 7.54 + 0.03*(Dh/L2)*Re*Pr/((Dh/L2)*Re*Pr)

#     hcL2 = NucL2*kair(TL2amb)


#     QL21 = AiL2UL2*((T2 - TL2) - (T1 - TL2))/np.log((T2 - TL2)/(T1 - TL2))
#     QL22 = Ao2*(hcL2 + hrL2)*(TL2 - Tamb)

#     return QL21, QL22


# def zone3B(unks, *args0):
#     T1, T2, T3, T4, Tf, Tw, Tg, To = unks

#     rg = 0.125
#     rf = 0.182

#     ri = 0.197
#     ro = 0.2

#     L1 = 0.195
#     L2 = 0.1079
#     Lf = 0.065

#     Aw = 2*np.pi*rf*(L2 - Lf)

#     Dh = 2*(ri - rf)

#     # Convection heat transfer coeff win (B.5)
#     Twin = 0.5*(Tw + T3B)

#     Re = G/(np.pi*(2*rg)*mu(Twin))
#     Pr = cp(Twin)*mu(Twin)/kair(Twin)
#     fp = (0.790*np.log(Re) - 1.64)**(-2)

#     if (Re > 3000 and Re <= 5e6) or (Pr > 0.5 and Pr <= 2e3):
#         Nuwin = (fp/8)*(Re - 1e3)*Pr/(1 + 12.7*np.sqrt(fp/8)*(Pr**(2/3) - 1))
#         hwin = Nuwin*kair(Twin)/Dh
#     else:
#         Nuwin = 0.664*Re**0.5*Pr**(1/3)
#         hwin = Nuwin*kair(Twin)/(2*rg)

#     Q3B1 = G*cpMean(T3B, T3)*(T3B - T3)
#     Q3B2 = hwin*Aw*((Tw - T3) - (Tw - T3B))/np.log((Tw - T3)/(Tw - T3B))

#     return Q3B1, Q3B2


def zone2(unks, *args0):
    T1, T2, T3, T4, Tf, Tw, Tg, To = unks

    rg = 0.125
    rf = 0.182

    ri = 0.197
    ro = 0.2

    L1 = 0.195
    L2 = 0.1079

    Aw = 0.1788
    
    Af = np.pi*rf**2
    Ag = np.pi*rg**2
    
    etaOpt = 0.84645

    # Ffg = 0.2956
    # Ffw = 0.7044
    # Fgf = 0.6267
    # Fgw = 0.3733
    # Fwf = 0.4110
    # Fwg = 0.1027

    Ffg = 0.3171
    Ffw = 0.4340
    Fwg = 0.1340
    Fgf = Af/Ag*Ffg
    Fwf = Aw/Af*Ffw
    Fgw = Aw/Ag*Fwg

    epsf = 0.95
    epsw = 0.8  # 1e-3

    alphag = 0.013  # Absorptivity at visible wavelengths
    alphagp = 1  # Absorptivity at long wavelengths

    epsgp = 0.88  # Glass emisivity at long wavelengths
    epsg = 0.04  # Glass emisivity at visible wavelengths

    Dh = 2*(ri - rf)

    # Convection heat transfer coeff 1-w (B.2)
    T2w = 0.5*(T2 + Tw)

    Re = 2*G/(np.pi*(ri + rf)*mu(T2w))  # 4*G/(np.pi*Dh*mu(T1w))
    Pr = cp(T2w)*mu(T2w)/kair(T2w)
    fp = (0.790*np.log(Re) - 1.64)**(-2)

    if (Re > 3000 and Re <= 5e6) and (Pr > 0.5 and Pr <= 2e3):
        ct = (T2/Tw)**0.45
        Nu2 = (fp/8)*(Re - 1e3)*Pr/(1 + 12.7*np.sqrt(fp/8)*(Pr**(2/3) - 1))*\
            (1 + (Dh/L2)**(2/3))*ct
        hw = Nu2*kair(T2w)/L2

    else:
        Nu2 = 7.54 + 0.03*(Dh/L2)*Re*Pr/(1 + 0.016*((Dh/L2)*Re*Pr)**(2/3))
        hw = Nu2*kair(T2w)/Dh

    taug = 0.851  # glass transmissivity
    rhof = 0.05  # foam reflectivity at visible wave
    rhow = 0.2  # window reflectivity at visible wave
    sigma = 5.67e-8  # W/(m^2 K^4) (Stephan-Boltzmann constant)

    Tgi = 0.5*(Tg + T3)

    Q21 = G*cpMean(T2, T1)*(T2 - T1)
    Q22 = hw*Aw*((Tw - T1) - (Tw - T2))/np.log((Tw - T1)/(Tw - T2))
    # Q23 = taug*Ib*Fgf*rhof*Ffw + taug*Ib*Fgw*(1 - rhow*Fwf - rhow*Fwg) + \
    #     sigma*(Tf**4 - Tw**4)/((1-epsf)/(Af*epsf) + 1/(Af*Ffw) +
    #                            (1-epsw)/(Aw*epsw)) - \
    #     sigma*(Tw**4 - Tgi**4)/((1-epsw)/(Aw*epsw) + 1/(Aw*Fwg) +
    #                             (1-epsg)/(Ag*epsg))
        
    Q23 = epsw*Ffw*(epsf*Af*sigma*(Tf**4 - T2**4) + rhof*taug*Ib)

    return Q21, Q22, Q23


def zone3(unks, *args0):
    T1, T2, T3, T4, Tf, Tw, Tg, To = unks

    alphag = 0.013  # Absorptivity at visible wavelengths
    alphagp = 1  # Absorptivity at long wavelengths

    epsgp = 0.326  # 0.88 # Glass emisivity at long wavelengths
    epsg = 0.013  # 0.04 # Glass emisivity at visible wavelengths

    epsf = 0.95
    epsw = 0.8  # 1e-3

    rg = 0.125
    rf = 0.182

    ri = 0.197
    ro = 0.2

    L1 = 0.195
    L2 = 0.1079
    Lg = 0.015

    Dh = 2*rg

    Ag = np.pi*rg**2
    Af = np.pi*rf**2
    Aw = np.pi*(rf**2 - rg**2) + 2*np.pi*rf*L2

    taug = 0.851  # glass transmissivity
    rhof = 0.05  # foam reflectivity at visible wave
    rhow = 0.2  # reflectivity at visible wave
    sigma = 5.67e-8  # W/(m^2 K^4) (Stephan-Boltzmann constant)

    # Convection heat transfer coeff 3gi (B.10)
    Tgi = 0.5*(T3 + Tg)

    # Re = 4*G*L2/(np.pi*rg**2*mu(Tgi))
    Re = 4*G*rg/(np.pi*rg**2*mu(Tgi))

    Pr = cp(Tgi)*mu(Tgi)/kair(Tgi)

    if (Re > 5e5 and Re <= 1e7) and (Pr > 0.6):
        Nu3i = 0.037*Re**0.8*Pr**(1/3)
    else:
        Nu3i = 0.664*Re**0.5*Pr**(1/3)

    hgi = Nu3i*kair(Tgi)/rg #L2

    Tgo = 0.5*(Tg + Tamb)

    # Convection heat transfer coeff 3go
    beta = 1/(0.5*(Tgo + Tamb))
    g = 9.81
    Pr = cp(1/beta)*mu(1/beta)/kair(1/beta)
    Ra = Pr*g*beta*(Tgo - Tamb)*(np.sqrt(Ag))**3/(mu(1/beta)/rhoA(1/beta))**2

    # Nu3o = (0.825 + 0.387*Ra**(1/6)/((1 + (0.492/Pr)**(9/16))**(8/27)))**2
    Nu3o = 0.59*Ra**0.25 #0.27*Ra**0.25 # Cengel - horizontal plate
    hgo = Nu3o*kair(Tgo)/np.sqrt(Ag)

    # Ffg = 0.2956
    # Ffw = 0.7044
    # Fgf = 0.6267
    # Fgw = 0.3733
    # Fwf = 0.4110
    # Fwg = 0.1027

    Ffg = 0.3171
    Ffw = 0.4340
    Fwg = 0.1340
    Fgf = Af/Ag*Ffg
    Fwf = Aw/Af*Ffw
    Fgw = Aw/Ag*Fwg

    taug = 0.851  # glass transmissivity
    rhof = 0.05  # foam reflectivity at visible wave
    rhow = 0.20  # reflectivity at visible wave
    sigma = 5.67e-8  # W/(m^2 K^4) (Stephan-Boltzmann constant)


    Q31 = G*cpMean(T3, T2)*(T3 - T2)
    Q32 = hgi*Ag*((Tg - T3) - (Tg - T2))/np.log((Tg - T3)/(Tg - T2))
    Q33 = alphag*Ib + \
        alphagp*Ffg*(rhof*taug*Ib + epsf*Af*sigma*(Tf**4 - Tg**4)) + \
                    alphagp*Fwg*epsw*Aw*sigma*(Tw**4 - Tg**4) - \
        epsgp*Ag*sigma*(Tg**4 - Tamb**4) + hgo*Ag*(Tg - Tamb) - \
            hgi*Ag*((Tg - T3) - (Tg - T2))/np.log((Tg - T3)/(Tg - T2))

    return Q31, Q32, Q33


def zone4(unks, *args0):
    T1, T2, T3, T4, Tf, Tw, Tg, To = unks

    alphag = 0.013  # Absorptivity at visible wavelengths
    alphagp = 1  # Absorptivity at long wavelengths

    epsgp = 0.013  # 0.88 # Glass emisivity at long wavelengths
    epsg = 0.04  # Glass emisivity at visible wavelengths

    epsf = 0.95
    epsw = 0.8  # 1e-3

    rg = 0.125
    rf = 0.182

    ri = 0.197
    ro = 0.2

    L1 = 0.195
    L2 = 0.1079

    Ag = np.pi*rg**2
    Af = np.pi*rf**2
    Aw = np.pi*(rf**2 - rg**2) + 2*np.pi*rf*L2

    phi = 0.792
    dc = 1.86e-3
    dp = 3.40e-4
    Ls = 6.58e-4
    Lf = 0.065
    Af = np.pi*rf**2
    PPI = 75  # Pores Per Inch

    # Ffg = 0.2956
    # Ffw = 0.7044
    # Fgf = 0.6267
    # Fgw = 0.3733
    # Fwf = 0.4110
    # Fwg = 0.1027

    Ffg = 0.3171
    Ffw = 0.4340
    Fwg = 0.1340
    Fgf = Af/Ag*Ffg
    Fwf = Aw/Af*Ffw
    Fgw = Aw/Ag*Fwg


    Tf4 = 0.5*(Tf + T4)

    Dh = 2*rf
    Re = 4*G/(np.pi*Dh*mu(Tf4))
    PPM =29.53*100
    dp = np.sqrt(4*phi/np.pi)/PPM
    dc = 1.86e-3
    df = dp*1.18*np.sqrt((1-phi)/(3*np.pi))/(1 - np.exp(-(1-phi)/0.04))
    dd = (1 - np.exp(-(1-phi)/0.04))*df
    phi = 0.788
    Redc = 4*G*dc/(np.pi**Dh**2*mu(Tf4))
    Ref = 4*G*df/(phi*np.pi**Dh**2*mu(Tf4))
    Rep = 4*G*dp/(np.pi**Dh**2*mu(Tf4))

    Pr = cp(Tf4)*mu(Tf)/kair(Tf4)
    
    # Correlaciones de Zukaukas (ver e.g. Cengel)
    if (Ref >= 0 and Ref < 500):
        Nu4 = 1.04*Ref**0.4*Pr**0.36
        
    elif (Redc >= 500 and Redc <= 1e3):
        Nu4 = 0.71*Ref**0.5*Pr**0.36
        
    hsf4 = Nu4*kair(Tf4)/df
    

    taug = 0.851  # glass transmissivity
    rhof = 0.05  # foam reflectivity at visible wave
    rhow = 0.20  # reflectivity at visible wave
    sigma = 5.67e-8  # W/(m^2 K^4) (Stephan-Boltzmann constant)

    # Q41 = G*cpMean(T4, T3)*(T4 - T3)
    # Q42 = Vf*hvf*((Tf - T3) - (Tf - T4))/np.log((Tf - T3)/(Tf - T4))
    # Q43 = taug*Ib*Fgf*(1-rhof) + taug*Ib*Fgw*Fwf*rhow - \
    #         sigma*(Tf**4 - Tw**4)/((1-epsf)/(Af*epsf) + 1/(Af*Ffw) + \
    #                                (1-epsw)/(Aw*epsw)) - \
    #         sigma*(Tf**4 - Tg**4)/((1-epsf)/(Af*epsf) + 1/(Af*Ffg) + \
    #                                (1-epsg)/(Ag*epsg))

    Tgi = 0.5*(Tg + T3)
    T34 = 0.5*(T3 + T4)

    
    Q41 = G*cpMean(T4, T3)*(T4 - T3)
    Q42 = hsf4*np.pi*rf**2*(Tf - T34)
    Q43 = taug*Ib*Fgf*(1-rhof) + taug*Ib*Fgw*Fwf*rhow - \
        sigma*(Tf**4 - Tw**4)/((1-epsf)/(Af*epsf) + 1/(Af*Ffw) +
                               (1-epsw)/(Aw*epsw)) - \
        sigma*(Tf**4 - Tg**4)/((1-epsf)/(Af*epsf) + 1/(Af*Ffg) +
                                (1-epsg)/(Ag*epsg))

    return Q41, Q42, Q43


def glass(unks, args0):
    To, T1, T2, T3, T3B, T4, Tw, Tf, Tg, TL1, TL2 = unks

    alphag = 0.013  # Absorptivity at visible wavelengths
    alphagp = 1  # Absorptivity at long wavelengths

    epsgp = 0.326  # 0.88 # Glass emisivity at long wavelengths
    epsg = 0.013  # 0.04 # Glass emisivity at visible wavelengths

    epsf = 0.95
    epsw = 0.8  # 1e-3

    rg = 0.125
    rf = 0.182

    ri = 0.197
    ro = 0.2

    L1 = 0.195
    L2 = 0.1079
    Lg = 0.015

    Dh = 2*(ri - rf)

    Ag = np.pi*rg**2
    Af = np.pi*rf**2
    Aw = np.pi*(rf**2 - rg**2) + 2*np.pi*rf*L2

    taug = 0.851  # glass transmissivity
    rhof = 0.05  # foam reflectivity at visible wave
    rhow = 0.2  # reflectivity at visible wave
    sigma = 5.67e-8  # W/(m^2 K^4) (Stephan-Boltzmann constant)

    kg = 1.40
    Lg = 0.015
    alphag = 0.013

    Ffg = 0.2956
    Ffw = 0.7044
    Fgf = 0.6267
    Fgw = 0.3733
    Fwf = 0.4110
    Fwg = 0.1027

    q = (alphag*Ib + taug*Ib*(Fgf*rhof*Ffg + Fgw*Fwg*rhow))/(np.pi*rg**2*Lg)

    c1 = (Tgi - Tgo + q*Lg**2/(2*kg))/Lg
    c2 = Tgo

    qgo = -kg*c1
    qgi = -kg*(-q*Lg/kg + c1)

    # Convection heat transfer coeff 3go
    beta = 1/(0.5*(Tgo + Tamb))
    g = 9.81
    Pr = cp(1/beta)*mu(1/beta)/kair(1/beta)
    Ra = Pr*g*beta*(Tgo - Tamb)*(np.sqrt(Ag))**3/(mu(1/beta)/0.35)**2

    Nu3o = (0.825 + 0.387*Ra**(1/6)/((1 + (0.492/Pr)**(9/16))**(8/27)))**2

    T3go = 0.5*(T3 + Tgo)

    hgo = Nu3o*kair(T3go)/np.sqrt(Ag)

    Q3g1 = qgo + hgo*Ag*(Tgo - Tamb) + epsgp*Ag*sigma*(Tgo**4 - Tamb**4)
    Q3g2 = qgi + sigma*(Tf**4 - Tgi**4)/((1-epsf)/(Af*epsf) + 1/(Af*Ffg) +
                                         (1-epsgp)/(Ag*epsgp)) + \
        sigma*(Tw**4 - Tgi**4)/((1-epsw)/(Aw*epsw) + 1/(Aw*Fwg) +
                                (1-epsgp)/(Ag*epsgp))

    return Q3g1, Q3g2



def Qlosses(Ib, Tamb, T1, T2, T3, T4, Tf, Tw, Tg, To):
    
    kg = 1.46
    
    alphag = 0.013  # Absorptivity at visible wavelengths
    alphagp = 1  # Absorptivity at long wavelengths

    epsgp = 0.326  # 0.88 # Glass emisivity at long wavelengths
    epsg = 0.013  # 0.04 # Glass emisivity at visible wavelengths

    epsf = 0.95
    epsw = 0.8  # 1e-3
    
    taug = 0.851  # glass transmissivity at visible wave
    taugp = 0.549 # glass transmissivity
    
    rhof = 0.05  # foam reflectivity at visible wave
    rhow = 0.20  # wall reflectivity at visible wave
    rhog = 0.136 # glass reflectivity at visible wave
    
    sigma = 5.67e-8  # W/(m^2 K^4) (Stephan-Boltzmann constant)
    
    rg = 0.125
    rf = 0.182

    ri = 0.197
    ro = 0.2

    L1 = 0.195
    L2 = 0.1079
    Lg = 0.015

    Dh = 2*(ri - rf)

    Ag = np.pi*rg**2
    Af = np.pi*rf**2
    Aw = np.pi*(rf**2 - rg**2) + 2*np.pi*rf*L2
    
    Ffg = 0.3171
    Ffw = 0.4340
    Fwg = 0.1340
    Fgf = Af/Ag*Ffg
    Fwf = Aw/Af*Ffw
    Fgw = Aw/Ag*Fwg
    
    
    Tgo = 0.5*(Tg + Tamb)
    Tfo = 0.5*(Tf + Tamb)
    Two = 0.5*(Tw + Tamb)

    
    QlglassE = epsgp*Ag*sigma*(Tg**4 - Tamb**4)
    QlglassR = rhog*Ib
    Qlfe = taugp*Ffg*epsf*Af*sigma*(Tf**4 - Tamb**4)
    Qlfr = taugp*Ffg*rhof*Ib
    Qlwall = Fwg*epsw*Aw*sigma*(Tw**4 - Tamb**4)
    
    
    
    # Convection heat transfer coeff 3gi (B.10)
    Tgi = 0.5*(T3 + Tg)

    Re = 4*G*L2/(np.pi*rg**2*mu(Tgi))
    Pr = cp(Tgi)*mu(Tgi)/kair(Tgi)

    if (Re > 5e5 and Re <= 1e7) and (Pr > 0.6):
        Nu3i = 0.037*Re**0.8*Pr**(1/3)
    else:
        Nu3i = 0.664*Re**0.5*Pr**(1/3)

    hgi = Nu3i*kair(Tgi)/L2


    # Convection heat transfer coeff 3go
    beta = 1/(0.5*(Tgo + Tamb))
    g = 9.81
    Pr = cp(1/beta)*mu(1/beta)/kair(1/beta)
    Ra = Pr*g*beta*(Tgo - Tamb)*(np.sqrt(Ag))**3/(mu(1/beta)/rhoA(1/beta))**2

    # Nu3o = (0.825 + 0.387*Ra**(1/6)/((1 + (0.492/Pr)**(9/16))**(8/27)))**2
    Nu3o = 0.27*Ra**0.25 # Cengel - horizontal plate
    hgo = Nu3o*kair(Tgo)/np.sqrt(Ag)    
    
    Qlconv = Ag*hgo*(Tgo - Tamb)
        
    losses = np.array([QlglassE, QlglassR, Qlfe, Qlfr, Qlwall, Qlconv])
    
    return losses


def dish(unks, *args0):

    # Ib, G, Ti, Tamb = args

    T1, T2, T3, T4, Tf, Tw, Tg, To = unks

    #
    # ZONE 1
    #
    eq1 = zone1(unks, args0)[0] - zone1(unks, args0)[2]
    eq2 = zone1(unks, args0)[0] - zone1(unks, args0)[1]

    # #
    # # ZONE 2
    # #
    eq3 = zone2(unks, args0)[0] - zone2(unks, args0)[1]  # Eq2
    eq4 = zone2(unks, args0)[0] - zone2(unks, args0)[2]  # Eq2

    # #
    # # ZONE 3
    # #
    # # 
    eq5 = zone3(unks, Ib, G, Ti, Tamb)[0] - zone3(unks, Ib, G, Ti, Tamb)[1] # Eq3
    eq6 = zone3(unks, Ib, G, Ti, Tamb)[2]

    # #
    # # ZONE 4
    # #
    eq7 = zone4(unks, Ib, G, Ti, Tamb)[0] - zone4(unks, Ib, G, Ti, Tamb)[1] # Eq5
    eq8 = zone4(unks, Ib, G, Ti, Tamb)[0] - zone4(unks, Ib, G, Ti, Tamb)[2] # Eq5


    res = np.array([eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8])

    print(eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8)
    print()

    return res


# Ib, G, Ti, Tamb = 28e3, 0.044, 317, 307.3
Ib, G, Ti, Tamb = 27e3, 0.043, 334.0, 300.0
args0 = (Ib, G, Ti, Tamb)
    
# To, T3, T4, Tw, Tf, Tg, TL1, TL2 = 1183.0, 716.0, 1195.0, 1043.0, 1245.0,\
#     1174.0, 382.0, 358.0

TL1, TL2 = 382.0, 358.0



# T1, T2, T3, T4, Tf, Tw, Tg, To
# unks0 = [536.0, 600.0, 650.0, 1195.0, 1183.0, 1043.0, 1174.0, 1245.0]

unks0 = np.array([77.0, 91.0, 97.0, 496.0, 300.5, 257.0, 233.0, 450.0]) + 273


# lower_bounds = 0.0*unks0 + 300.0  # Establece límite inferior en 0
# upper_bounds = 0.0*unks0 + 5000.0

#cons = ({'type': 'ineq', 'fun': gg0}, {'type': 'ineq', 'fun': gg1})


# sol = least_squares(dish, unks0, args=args0, method='lm')
sol = root(dish, unks0, args=args0, method='hybr')

# sol = minimize(dish, unks0, args=args0)

temps = ('Ti', 'T1', 'T2', 'T3', 'T4', 'Tf', 'Tw', 'Tg', 'To')
# bounds=list(zip(lower_bounds, upper_bounds)),


# sol = fmin_tnc(func=dish, x0=unks0, approx_grad=True, args=(1e3*44, 0.04, 500, 300),
#                bounds=list(zip(lower_bounds, upper_bounds)))

print()
print(sol.x)

T1, T2, T3, T4, Tf, Tw, Tg, To = sol.x

print()
print(list(zip(temps, [Ti - 273, sol.x - 273])))


# Qlglass, QlglassR, Qlfe, Qlfr, Qlwall, Qlconv
Qloss = Qlosses(Ib, Tamb, T1, T2, T3, T4, Tf, Tw, Tg, To)
eta = (Ib - np.sum(Qloss))/Ib

print()
print('Eta = ', eta)