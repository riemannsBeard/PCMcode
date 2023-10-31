# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import warnings
warnings.filterwarnings('ignore')

import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg','pdf')

import csv
import numpy as np
import scipy as sp
import math


import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import fsolve, newton, least_squares, root, minimize
from scipy.interpolate import interp1d
from scipy.integrate import odeint, solve_ivp, cumtrapz, RK45
from scipy import signal, fftpack

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

    cp_ = coeffs[0,0] + coeffs[0,1]*T + coeffs[0,2]*T**2 + coeffs[0,3]*T**3 +\
        coeffs[0,4]*T**4 + coeffs[0,5]*T**5
    
    return cp_


def kair(T):
        
    coeffs = np.array([[1068.53, -0.5252, 1.338e-3, -1.031e-6, 3.208e-10,
                        -2.908e-14],
             [-4.457e-4, 1.089e-4, -8.1629e-8, 6.323e-11, -2.734e-14,
              4.944e-18],
             [2.374e-8, 7.740e-8, -6.885e-11, 5.362e-14, -2.338e-17,
              4.256e-21]])
    
    kair_ = coeffs[1,0] + coeffs[1,1]*T + coeffs[1,2]*T**2 + coeffs[1,3]*T**3 +\
        coeffs[1,4]*T**4 + coeffs[1,5]*T**5
    
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
    
    mu_ = coeffs[2,0] + coeffs[2,1]*T + coeffs[2,2]*T**2 + coeffs[2,3]*T**3 +\
        coeffs[2,4]*T**4 + coeffs[2,5]*T**5
    
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
        cpMean_ = 1/(Th - Tc)*(coeffs[0,0]*(Th - Tc) + 
                               coeffs[0,1]/2*(Th**2 - Tc**2) + 
                               coeffs[0,2]/3*(Th**3 - Tc**3) + 
                               coeffs[0,3]/4*(Th**4 - Tc**4) + 
                               coeffs[0,4]/5*(Th**5 - Tc**5) + 
                               coeffs[0,5]/6*(Th**6 - Tc**6))
    elif Th < Tc:
        aux = Th
        Th = Tc
        Tc = aux

        cpMean_ = 1/(Th - Tc)*(coeffs[0,0]*(Th - Tc) + 
                               coeffs[0,1]/2*(Th**2 - Tc**2) + 
                               coeffs[0,2]/3*(Th**3 - Tc**3) + 
                               coeffs[0,3]/4*(Th**4 - Tc**4) + 
                               coeffs[0,4]/5*(Th**5 - Tc**5) + 
                               coeffs[0,5]/6*(Th**6 - Tc**6))        
    elif Th == Tc:
        cpMean_ = cp(Th)
        
    return cpMean_


def zone1L(unks, *args0):
    T1, T2 = unks
    
    rg = 0.125
    rf = 0.182
    
    ri = 0.197
    ro = 0.2
    
    L1 = 0.195
    L2 = 0.1079
    
    Dh = 2*(ri - rf)
    
    TL1in = 0.5*(Ti + TL1)
    
    # Convection heat transfer insulator L1    
    Re = 4*G/(np.pi*Dh*mu(TL1in))
    Pr = cp(TL1in)*mu(TL1in)/kair(TL1in)
    fp = (0.790*np.log(Re) - 1.64)**(-2)
    
    if (Re > 3000 and Re <= 5e6) and (Pr > 0.5 and Pr <= 2e3):
        # Gnielinski
        NuL1in = (fp/8)*(Re - 1e3)*Pr/(1 + 12.7*np.sqrt(fp/8)*(Pr**(2/3) - 1))
    else:
        NuL1in = 7.54 + 0.03*(Dh/L1)*Re*Pr/((Dh/L1)*Re*Pr)
    
    hL1in = NuL1in*kair(TL1in)
    ki = kk(0.5*(TL1 + Ti)) #stainless steel plate (k = 15 W/m·K) Çengel
    
    AiL1cyl = 2*np.pi*ri*L1
    UcylL1 = 1/(1/hL1in + ri*np.log(ro/ri)/ki)
        
    rpo = 0.01
    rpi = 0.042
    eo = 3e-3
    
    AiL1flat = np.pi*(ri**2 - rpo**2 - 3*rpi**2)
    UflatL1 = 1/(1/hL1in + eo/ki)
    
    AiL1UL1 = AiL1cyl*UcylL1 + AiL1flat*UflatL1
        

    TL1amb = 0.5*(TL1 + Tamb)

    Re = 4*G/(np.pi*(2*ri + 2*rf)*mu(TL1amb)) #4*G/(np.pi*Dh*mu(TL1amb))
    Pr = cp(TL1amb)*mu(TL1amb)/kair(TL1amb)

    # if (Re > 5e5 and Re <= 1e7) and (Pr > 0.6):
    #     NucL1 = 0.037*Re**0.8*Pr**(1/3)
    #     hcL1 = NucL1*kair(TL1amb)/Dh
        
    # elif (Re <= 5e5) and (Pr > 0.6):
    #     NucL1 = 0.664*Re**0.5*Pr**(1/3)
    #     hcL1 = NucL1*kair(TL1amb)/L1
    

    epsL1 = 0.8
    sigma = 5.67e-8 # W/(m^2 K^4) (Stephan-Boltzmann constant)

    hrL1 = epsL1*sigma*(TL1 + Tamb)*(TL1**2 + Tamb**2)
    Ao1 = 2*np.pi*ro*L1 + np.pi*(ro**2 - rpo**2 - 3*rpi**2)
    
    # Convection heat transfer coeff horizontal flat plate (Cengel)
    beta = 1/(0.5*(TL1 + Tamb))
    g = 9.81
    Pr = cp(1/beta)*mu(1/beta)/kair(1/beta)    
    Ra = Pr*g*beta*(TL1 - Tamb)*L1**3/(mu(1/beta)/rhoA(0.5*(TL1 + Tamb)))**2
    
    NucL1 = 0.27*Ra**(1/4)
    hcL1 = NucL1*kair(TL1amb)/L1
    
    
    QL11 = AiL1UL1*((T1 - TL1) - (Ti - TL1))/np.log((T1 - TL1)/(Ti - TL1))
    QL12 = Ao1*(hcL1 + hrL1)*(TL1 - Tamb)
    
    return QL11, QL12


def zone1(unks, *args0):
    T1, T2 = unks
    
    rg = 0.125
    rf = 0.182
    
    ri = 0.197
    ro = 0.2
    
    L1 = 0.195
    L2 = 0.1079
    
    Dh = 2*(ri - rf)
    
    # Convection heat transfer coeff i-1 (B.1)
    Ti1 = 0.5*(Ti + T1)
    
    Re = 2*G/(np.pi*(ri + rf)*mu(Ti1)) #4*G/(np.pi*Dh*mu(Ti1))
    Pr = cp(Ti1)*mu(Ti1)/kair(Ti1)    
    fp = (0.790*np.log(Re) - 1.64)**(-2)    
    
    if (Re > 3000 and Re<= 5e6) and (Pr > 0.5 and Pr <= 2e3):
        Nui1 = (fp/8)*(Re - 1e3)*Pr/(1 + 12.7*np.sqrt(fp/8)*(Pr**(2/3) - 1))
        
    else:
        Nui1 = 7.54 + 0.03*(Dh/L1)*Re*Pr/((Dh/L1)*Re*Pr)
        
    hi1 = Nui1*kair(Ti1)/Dh
        
    # Convection heat transfer coeff 4-o (B.5)
    T4o = 0.5*(T4 + To)

    Re = 2*G/(np.pi*(ri + rf)*mu(T4o)) #4*G/(np.pi*Dh*mu(T4o))
    Pr = cp(T4o)*mu(T4o)/kair(T4o)    
    fp = (0.790*np.log(Re) - 1.64)**(-2)
    
    if (Re > 3000 and Re <= 5e6) and (Pr > 0.5 and Pr <= 2e3):
        Nu4o = (fp/8)*(Re - 1e3)*Pr/(1 + 12.7*np.sqrt(fp/8)*(Pr**(2/3) - 1))
        h4o = Nu4o*kair(T4o)/Dh
        
    else:
        Nu4o = 0.664*Re**0.5*Pr**(1/3)
        h4o = Nu4o*kair(T4o)/L1

            
    # Overall heat transfer
    ew = 1e-3 # inner cylinder wall thickness
    A1 = 2*np.pi*rf*L1
    kw = 0.5*(kk(Ti1) + kk(T4o)) #stainless steel plate (k = 15 W/m·K) Çengel        
    U1 = 1/(1/hi1 + ew/kw + 1/h4o) 
  
    Q11 = G*cpMean(T1,Ti)*(T1 - Ti)
    Q12 = G*cpMean(T4,To)*(T4 - To)    
    Q13 = U1*A1*((To - Ti) - (T4 - T1))/np.log((To - Ti)/(T4 - T1))
    
    return Q11, Q12, Q13


def zone2L(unks, *args0):
    T1, T2 = unks
    
    rg = 0.125

    rg = 0.125
    rf = 0.182
    
    ri = 0.197
    ro = 0.2
    
    L1 = 0.195
    L2 = 0.1079
    
    Aw = 0.1788
    
    Dh = 2*(ri - rf)
    
    TL2in = 0.5*(Ti + TL2)
    
    # Convection heat transfer insulator L2    
    Re = 2*G/(np.pi*(ri + rf)*mu(TL2in))
    Pr = cp(TL2in)*mu(TL2in)/kair(TL2in)    
    fp = (0.790*np.log(Re) - 1.64)**(-2)    
    
    if (Re > 3000 and Re <= 5e6) and (Pr > 0.5 and Pr <= 2e3):
        NuL2in = (fp/8)*(Re - 1e3)*Pr/(1 + 12.7*np.sqrt(fp/8)*(Pr**(2/3) - 1))
        
    else:
        NuL2in = 7.54 + 0.03*(Dh/L2)*Re*Pr/((Dh/L2)*Re*Pr)
    
    hL2in = NuL2in*kair(TL2in)
    ki = kk(0.5*(Ti + TL2)) #stainless steel plate (k = 15 W/m·K) Çengel    
    
    AiL2cyl = 2*np.pi*ri*L2
    UcylL2 = 1/(1/hL2in + ri*np.log(ro/ri)/ki)
    
    rpo = 0.01
    rpi = 0.042
    eo = 3e-3
    AiL2flat = np.pi*(ri**2 - rg**2)
    UflatL2 = 1/(1/hL2in + eo/ki)
    
    AiL2UL2 = AiL2cyl*UcylL2 + AiL2flat*UflatL2
    
    epsL2 = 0.8
    sigma = 5.67e-8 # W/(m^2 K^4) (Stephan-Boltzmann constant)
    
    
    hrL2 = epsL2*sigma*(TL2 + Tamb)*(TL2**2 + Tamb**2)
    Ao2 = 2*np.pi*ro*L1 + np.pi*(ro**2 - rpo**2 - 3*rpi**2)
    
    TL2amb = 0.5*(TL2 + Tamb)
    
    
    Re = 2*G/(np.pi*(ri + rf)*mu(TL2amb))
    Pr = cp(TL2amb)*mu(TL2amb)/kair(TL2amb)
    
    # if (Re > 5e5 and Re  <= 1e7) and (Pr > 0.6):
    #     NucL2 = 0.037*Re**0.8*Pr**(1/3)
    #     hcL2 = NucL2*kair(TL2amb)/Dh
        
    # elif (Re <= 5e5) and (Pr > 0.6):
    #     NucL2 = 0.664*Re**0.5*Pr**(1/3)
    #     hcL2 = NucL2*kair(TL2amb)/L2
    
    # Convection heat transfer coeff horizontal flat plate (Cengel)
    beta = 1/(0.5*(TL2 + Tamb))
    g = 9.81
    Pr = cp(1/beta)*mu(1/beta)/kair(1/beta)    
    Ra = Pr*g*beta*(TL2 - Tamb)*L2**3/(mu(1/beta)/rhoA(1/beta))**2
    
    NucL2 = 0.27*Ra**(1/4)
    hcL2 = NucL2*kair(1/beta)/L2
        

    QL21 = AiL2UL2*((T2 - TL2) - (T1 - TL2))/np.log((T2 - TL2)/(T1 - TL2))    
    QL22 = Ao2*(hcL2 + hrL2)*(TL2 - Tamb)
    
    return QL21, QL22


def zone3B(unks, *args0):
    T1, T2 = unks

    rg = 0.125
    rf = 0.182
    
    ri = 0.197
    ro = 0.2
    
    L1 = 0.195
    L2 = 0.1079
    
    Aw = 0.1788
    
    Dh = 2*(ri - rf)    

    # Convection heat transfer coeff win (B.5)
    Twin = 0.5*(Tw + T3)
    
    Re = 2*G/(np.pi*(ri + rf)*mu(Twin))
    Pr = cp(Twin)*mu(Twin)/kair(Twin)    
    fp = (0.790*np.log(Re) - 1.64)**(-2)
    
    if (Re > 3000 and Re <= 5e6) or (Pr > 0.5 and Pr <= 2e3):
        Nuwin = (fp/8)*(Re - 1e3)*Pr/(1 + 12.7*np.sqrt(fp/8)*(Pr**(2/3) - 1))
        hwin = Nuwin*kair(Twin)/Dh
    else:
        Nuwin = 0.664*Re**0.5*Pr**(1/3)
        hwin = Nuwin*kair(Twin)/L2
        
    Q3B1 = G*cpMean(T3B, T3)*(T3B - T3)
    Q3B2 = hwin*Aw*((Tw - T3) - (Tw - T3B))/np.log((Tw - T3)/(Tw - T3B))

    
    return Q3B1, Q3B2


def zone2(unks, *args0):
    T1, T2 = unks
    
    rg = 0.125
    rf = 0.182
    
    ri = 0.197
    ro = 0.2
    
    L1 = 0.195
    L2 = 0.1079
    
    Aw = 0.1788
    Af = np.pi*rf**2
    Ag = np.pi*rg**2
    
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
    epsw = 0.8 #1e-3

    alphag = 0.013 # Absorptivity at visible wavelengths
    alphagp = 1 # Absorptivity at long wavelengths
    
    epsgp = 0.88 # Glass emisivity at long wavelengths
    epsg = 0.04 # Glass emisivity at visible wavelengths

    
    Dh = 2*(ri - rf)
    
    # Convection heat transfer coeff 1-w (B.2)
    T1w = 0.5*(T1 + Tw)
    
    Re = 2*G/(np.pi*(ri + rf)*mu(T1w)) #4*G/(np.pi*Dh*mu(T1w))
    Pr = cp(T1w)*mu(T1w)/kair(T1w)
    fp = (0.790*np.log(Re) - 1.64)**(-2)
    
        
    if (Re > 3000 and Re<= 5e6) and (Pr > 0.5 and Pr <= 2e3):
        Nui2 = (fp/8)*(Re - 1e3)*Pr/(1 + 12.7*np.sqrt(fp/8)*(Pr**(2/3) - 1))
        
    else:
        Nui2 = 7.54 + 0.03*(Dh/L2)*Re*Pr/((Dh/L2)*Re*Pr)
        
    h2w = Nui2*kair(T1w)/Dh
    
    
    taug = 0.851 # glass transmissivity
    rhof = 0.05 # foam reflectivity at visible wave
    rhow = 0.2 # window reflectivity at visible wave
    sigma = 5.67e-8 # W/(m^2 K^4) (Stephan-Boltzmann constant)
    
    Tgi = 0.5*(Tg + T3)
        
    Q21 = G*cpMean(T2, T1)*(T2 - T1)
    Q22 = h2w*Aw*((Tw - T1) - (Tw - T2))/np.log((Tw - T1)/(Tw - T2))
    Q23 = taug*Ib*Fgf*rhof*Ffw + taug*Ib*Fgw*(1 - rhow*Fwf - rhow*Fwg) + \
            sigma*(Tf**4 - Tw**4)/((1-epsf)/(Af*epsf) + 1/(Af*Ffw) + \
                                   (1-epsw)/(Aw*epsw)) - \
            sigma*(Tw**4 - Tgi**4)/((1-epsw)/(Aw*epsw) + 1/(Aw*Fwg) + \
                                    (1-epsg)/(Ag*epsg))
    
    return Q21, Q22, Q23


def zone3(unks, *args0):
    T1, T2 = unks

    alphag = 0.013 # Absorptivity at visible wavelengths
    alphagp = 1 # Absorptivity at long wavelengths
    
    epsgp = 0.326 #0.88 # Glass emisivity at long wavelengths
    epsg = 0.013 #0.04 # Glass emisivity at visible wavelengths
    
    epsf = 0.95
    epsw = 0.8 #1e-3
    
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
        
    taug = 0.851 # glass transmissivity
    rhof = 0.05 # foam reflectivity at visible wave
    rhow = 0.2 # reflectivity at visible wave
    sigma = 5.67e-8 # W/(m^2 K^4) (Stephan-Boltzmann constant)    
    
    # Convection heat transfer coeff 3gi (B.10)
    T3gi = 0.5*(T3 + Tg)
    
    Re = 2*G/(np.pi*(ri + rf)*mu(T3gi))
    Pr = cp(T3gi)*mu(T3gi)/kair(T3gi)    
     
    if (Re > 5e5 and Re <= 1e7) and (Pr > 0.6):
        Nu3i = 0.037*Re**0.8*Pr**(1/3)
    else:
        Nu3i = 0.664*Re**0.5*Pr**(1/3)
        
    hgi = Nu3i*kair(T3gi)/rg

    
    Tgo = 0.5*(Tg + Tamb)

    # Convection heat transfer coeff 3go
    beta = 1/(0.5*(Tgo + Tamb))
    g = 9.81
    Pr = cp(1/beta)*mu(1/beta)/kair(1/beta)    
    Ra = Pr*g*beta*(Tgo - Tamb)*(np.sqrt(Ag))**3/(mu(1/beta)/rhoA(1/beta))**2
    
    Nu3o = (0.825 + 0.387*Ra**(1/6)/((1 + (0.492/Pr)**(9/16))**(8/27)))**2
    
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
    
    taug = 0.851 # glass transmissivity
    rhof = 0.05 # foam reflectivity at visible wave
    rhow = 0.20 # reflectivity at visible wave
    sigma = 5.67e-8 # W/(m^2 K^4) (Stephan-Boltzmann constant)
    
    Tgi = 0.5*(Tg + T3)
    
    Q31 = G*cpMean(T3, T2)*(T3 - T2)    
    Q32 = hgi*Ag*((Tgi - T3) - (Tgi - T2))/np.log((Tgi - T3)/(Tgi - T2)) 
    Q33 = alphag*Ib + taug*Ib*(Fgf*rhof*Ffg + Fgw*Fwg*rhow) + \
            sigma*(Tf**4 - Tgi**4)/((1-epsf)/(Af*epsf) + 1/(Af*Ffg) + \
                                   (1-epsgp)/(Ag*epsgp)) + \
            sigma*(Tw**4 - Tgi**4)/((1-epsw)/(Aw*epsw) + 1/(Aw*Fwg) + \
                                   (1-epsgp)/(Ag*epsgp)) - \
            hgo*Ag*(Tgo - Tamb) - epsgp*Ag*sigma*(Tgo**4 - Tamb**4)
    
    return Q31, Q32, Q33



def zone4(unks, *args0):
    T1, T2 = unks
    
    alphag = 0.013 # Absorptivity at visible wavelengths
    alphagp = 1 # Absorptivity at long wavelengths
    
    epsgp = 0.013 #0.88 # Glass emisivity at long wavelengths
    epsg = 0.04 # Glass emisivity at visible wavelengths
    
    epsf = 0.95
    epsw = 0.8 #1e-3
        
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
    PPI = 75 # Pores Per Inch
    
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
    
    
    Re = 2*G/(np.pi*(ri + rf)*mu(Tf))
    Redc = G*phi**2/(np.pi*rf*mu(Tf))
    
    Nuv = (32.504*phi**0.38 - 109.94*phi**1.38 + \
           166.65*phi**2.38 - 86.95*phi**3.38)*Redc**0.438
    hvf = Nuv*kair(Tf)/(2*rf)
    
    Vf = Af*Lf*phi


    taug = 0.851 # glass transmissivity
    rhof = 0.05 # foam reflectivity at visible wave
    rhow = 0.20 # reflectivity at visible wave
    sigma = 5.67e-8 # W/(m^2 K^4) (Stephan-Boltzmann constant)
    
    # Q41 = G*cpMean(T4, T3)*(T4 - T3)
    # Q42 = Vf*hvf*((Tf - T3) - (Tf - T4))/np.log((Tf - T3)/(Tf - T4))
    # Q43 = taug*Ib*Fgf*(1-rhof) + taug*Ib*Fgw*Fwf*rhow - \
    #         sigma*(Tf**4 - Tw**4)/((1-epsf)/(Af*epsf) + 1/(Af*Ffw) + \
    #                                (1-epsw)/(Aw*epsw)) - \
    #         sigma*(Tf**4 - Tg**4)/((1-epsf)/(Af*epsf) + 1/(Af*Ffg) + \
    #                                (1-epsg)/(Ag*epsg))
    
    Tgi = 0.5*(Tg + T3)
    
    Asf = 0.2023
    hsf = 450
    
    Q41 = G*cpMean(T4, T3)*(T4 - T3)
    # Q42 = Vf*hvf*((Tf - T3) - (Tf - T4))/np.log((Tf - T3)/(Tf - T4))
    Q42 = Asf*hsf*(Tf - 0.5*(T3B + T4))
    Q43 = taug*Ib*Fgf*(1-rhof) + taug*Ib*Fgw*Fwf*rhow - \
            sigma*(Tf**4 - Tw**4)/((1-epsf)/(Af*epsf) + 1/(Af*Ffw) + \
                                   (1-epsw)/(Aw*epsw)) - \
            sigma*(Tf**4 - Tgi**4)/((1-epsf)/(Af*epsf) + 1/(Af*Ffg) + \
                                   (1-epsg)/(Ag*epsg))
    
    return Q41, Q42, Q43


def gg0(TT):    
    
    return TT - 290

def gg1(TT):    
    
    return 5000 - TT

def glass(unks, args0):
    To, T1, T2, T3, T3B, T4, Tw, Tf, Tg, TL1, TL2 = unks
    
    alphag = 0.013 # Absorptivity at visible wavelengths
    alphagp = 1 # Absorptivity at long wavelengths
    
    epsgp = 0.326 #0.88 # Glass emisivity at long wavelengths
    epsg = 0.013 #0.04 # Glass emisivity at visible wavelengths
    
    epsf = 0.95
    epsw = 0.8 #1e-3
    
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
        
    taug = 0.851 # glass transmissivity
    rhof = 0.05 # foam reflectivity at visible wave
    rhow = 0.2 # reflectivity at visible wave
    sigma = 5.67e-8 # W/(m^2 K^4) (Stephan-Boltzmann constant)    
    
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
    Q3g2 = qgi + sigma*(Tf**4 - Tgi**4)/((1-epsf)/(Af*epsf) + 1/(Af*Ffg) + \
                                         (1-epsgp)/(Ag*epsgp)) + \
                    sigma*(Tw**4 - Tgi**4)/((1-epsw)/(Aw*epsw) + 1/(Aw*Fwg) + \
                                            (1-epsgp)/(Ag*epsgp))
    
    return Q3g1, Q3g2


def dish(unks, *args0):

    # Ib, G, Ti, Tamb = args
    
    # To, T1, T2, T3, T3B, T4, Tw, Tf, Tg, TL1, TL2 = unks
    T1, T2 = unks
             
    #
    # ZONE 1
    #    
    eq1 = zone1(unks, args0)[0] + zone1L(unks, args0)[0] - \
        zone1(unks, args0)[2] # Eq1
        
    # eq1 = zone1(unks, args0)[1] - zone1(unks, args0)[2] # Eq6
    
    # eq2 = zone1L(unks, Ib, G, Ti, Tamb)[0] - zone1L(unks, Ib, G, Ti, Tamb)[1] # Eq7
    

    #
    # ZONE 2
    #
    eq2 = zone2(unks, args0)[0] + zone2L(unks, args0)[0] - \
            zone2(unks, args0)[1] # Eq2
                        
    # eq2 = zone2L(unks, Ib, G, Ti, Tamb)[0] - zone2L(unks, Ib, G, Ti, Tamb)[1] # Eq8
    

    #
    # ZONE 3
    #    
    # eq3 = zone3(unks, Ib, G, Ti, Tamb)[0] - zone3(unks, Ib, G, Ti, Tamb)[1] # Eq3
    
    # eq7 = zone3B(unks, Ib, G, Ti, Tamb)[1] - zone3B(unks, Ib, G, Ti, Tamb)[0] # Eq4
    
    # eq8 = zone3(unks, Ib, G, Ti, Tamb)[1] - zone3(unks, Ib, G, Ti, Tamb)[2] # Eq9
    
    # eq8 = glass(unks, Ib, G, Ti, Tamb)[0]
    
    # eq9 = zone2(unks, Ib, G, Ti, Tamb)[1] + zone3B(unks, Ib, G, Ti, Tamb)[1] - \
    #         zone2(unks, Ib, G, Ti, Tamb)[2] # Eq10
                              
                              
    #
    # ZONE 4
    #
    # eq10 = zone4(unks, Ib, G, Ti, Tamb)[0] - zone4(unks, Ib, G, Ti, Tamb)[1] # Eq5
    
    # eq11 = zone4(unks, Ib, G, Ti, Tamb)[1] - zone4(unks, Ib, G, Ti, Tamb)[2] # Eq11
    
    # eq12 = glass(unks, Ib, G, Ti, Tamb)[1]
    
    
    
    res = np.array([eq1, eq2])
    
    print(eq1, eq2)
    print()

    return res



Ib, G, Ti, Tamb, To, T3, T3B, T4, Tw, Tf, Tg, TL1, TL2 = \
    33e3, 0.04, 528.66, 307.3, 1183.0, 714.0, 716.0, 1195.0, 1043.0, 1245.0,\
    1174.0, 382.0, 358.0
    
args0 = (Ib, G, Ti, Tamb, To, T3, T3B, T4, Tw, Tf, Tg, TL1, TL2)

# #################  To,  T1,  T2,  T3,  T3B,  T4,   Tw,   Tf,  Tgi,  Tgo, TL1, TL2
# unks0 = np.array([1183, 536, 707, 714, 716, 1195, 1043, 1245, 1174, 794.5, 382, 358])

unks0 = [538, 800]

# lower_bounds = 0.0*unks0 + 300.0  # Establece límite inferior en 0
# upper_bounds = 0.0*unks0 + 5000.0

cons = ({'type': 'ineq', 'fun': gg0}, {'type': 'ineq', 'fun': gg1})


# sol = least_squares(dish, unks0, args=args0, method='lm')
sol = root(dish, unks0, args=args0, method='hybr')

# sol = minimize(dish, unks0, args=args0)

temps = ('T1', 'T2')
# bounds=list(zip(lower_bounds, upper_bounds)),


# sol = fmin_tnc(func=dish, x0=unks0, approx_grad=True, args=(1e3*44, 0.04, 500, 300),
#                bounds=list(zip(lower_bounds, upper_bounds)))

print()
print(sol.x)

print()
list(zip(temps, sol.x))
