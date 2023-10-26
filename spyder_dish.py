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
from autograd import grad


import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import fsolve, newton, least_squares, fmin_tnc, minimize
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
    
    return interp(T)


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


def zone1L(unks, Ib, G, Ti, Tamb):
    To, T1, T2, T3, T3B, T4, Tw, Tf, Tg, TL1, TL2 = unks
    
    ri = 0.136
    ro = 0.2

    L1 = 0.195
    rf = 0.122
    
    Dh = 2*(ri - rf)
    
    TL1in = 0.5*(Ti + TL1)
    
    # Convection heat transfer insulator L1    
    Re = 4*G/(np.pi*(2*ri + 2*rf)*mu(TL1in)) #4*G/(np.pi*Dh*mu(TL1in))
    Pr = cp(TL1in)*mu(TL1in)/kair(TL1in)    
    fp = (0.790*np.log(Re) - 1.64)**(-2)    
    
    if (Re > 3000 and Re <= 5e6) and (Pr > 0.5 and Pr <= 2e3):
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
    
    QL11 = AiL1UL1*((T1 - TL1) - (Ti - TL1))/np.log((T1 - TL1)/(Ti - TL1))
    

    TL1amb = 0.5*(TL1 + Tamb)

    Re = 4*G/(np.pi*(2*ri + 2*rf)*mu(TL1amb)) #4*G/(np.pi*Dh*mu(TL1amb))
    Pr = cp(TL1amb)*mu(TL1amb)/kair(TL1amb)    
    
    if (Re > 5e5 and Re <= 1e7) and (Pr > 0.6):
        NucL1 = 0.037*Re**0.8*Pr**(1/3)
        hcL1 = NucL1*kair(TL1amb)/Dh
    elif (Re <= 5e5) and (Pr > 0.6):
        NucL1 = 0.664*Re**0.5*Pr**(1/3)
        hcL1 = NucL1*kair(TL1amb)/L1

    epsL1 = 0.8
    sigma = 5.67e-8 # W/(m^2 K^4) (Stephan-Boltzmann constant)

    hrL1 = epsL1*sigma*(TL1 + Tamb)*(TL1**2 + Tamb**2)
    Ao1 = 2*np.pi*ro*L1 + np.pi*(ro**2 - rpo**2 - 3*rpi**2)
    
    QL12 = Ao1*(hcL1 + hrL1)*(TL1 - Tamb)
    
    return QL11, QL12


def zone1(unks, Ib, G, Ti, Tamb):
    To, T1, T2, T3, T3B, T4, Tw, Tf, Tg, TL1, TL2 = unks
    
    ri = 0.136
    ro = 0.2

    L1 = 0.195
    rf = 0.122
    
    Dh = 2*(ri - rf)
    
    # Convection heat transfer coeff i-1 (B.1)
    Ti1 = 0.5*(Ti + T1)
    
    Re = 4*G/(np.pi*(2*ri + 2*rf)*mu(Ti1)) #4*G/(np.pi*Dh*mu(Ti1))
    Pr = cp(Ti1)*mu(Ti1)/kair(Ti1)    
    fp = (0.790*np.log(Re) - 1.64)**(-2)    
    
    if (Re > 3000 and Re<= 5e6) and (Pr > 0.5 and Pr <= 2e3):
        Nui1 = (fp/8)*(Re - 1e3)*Pr/(1 + 12.7*np.sqrt(fp/8)*(Pr**(2/3) - 1))
    else:
        Nui1 = 7.54 + 0.03*(Dh/L1)*Re*Pr/((Dh/L1)*Re*Pr)
        
    hi1 = Nui1*kair(Ti1)/Dh
        
    # Convection heat transfer coeff 4-o (B.5)
    T4o = 0.5*(T4 + To)

    Re = 4*G/(np.pi*(2*ri + 2*rf)*mu(T4o)) #4*G/(np.pi*Dh*mu(T4o))
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
    kw =kk( 0.5*(Ti + To)) #stainless steel plate (k = 15 W/m·K) Çengel        
    U1 = 1/(1/hi1+ ew/kw + 1/h4o) 
  
    Q11 = G*cpMean(T1,Ti)*(T1 - Ti)
    Q12 = G*cpMean(T4,To)*(T4 - To)    
    Q13 = U1*A1*((To - Ti) - (T4 - Ti))/np.log((To - Ti)/(T4 - Ti))
    
    return Q11, Q12, Q13


def zone2L(unks, Ib, G, Ti, Tamb):
    To, T1, T2, T3, T3B, T4, Tw, Tf, Tg, TL1, TL2 = unks
    
    ri = 0.136
    ro = 0.2
    rg = 0.125

    L1 = 0.195
    rf = 0.122
    
    L2 = 0.1079
    Aw = 0.1788
    
    Dh = 2*(ri - rf)
    
    TL2in = 0.5*(Ti + TL2)
    
    # Convection heat transfer insulator L2    
    Re = 4*G/(np.pi*(2*ri + 2*rf)*mu(TL2in)) #4*G/(np.pi*Dh*mu(TL2in))
    Pr = cp(TL2in)*mu(TL2in)/kair(TL2in)    
    fp = (0.790*np.log(Re) - 1.64)**(-2)    
    
    if (Re > 3000 and Re <= 5e6) and (Pr > 0.5 and Pr <= 2e3):
        NuL2in = (fp/8)*(Re - 1e3)*Pr/(1 + 12.7*np.sqrt(fp/8)*(Pr**(2/3) - 1))
    else:
        NuL2in = 7.54 + 0.03*(Dh/L1)*Re*Pr/((Dh/L1)*Re*Pr)
    
    hL2in = NuL2in*kair(TL2in)
    ki = kk(0.5*TL2in) #stainless steel plate (k = 15 W/m·K) Çengel    
    
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
    Re = 4*G/(2*np.pi*rg*mu(TL2amb))
    Pr = cp(TL2amb)*mu(TL2amb)/kair(TL2amb)
    
    if (Re > 5e5 and Re<= 1e7) and (Pr > 0.6):
        NucL2 = 0.037*Re**0.8*Pr**(1/3)
        hcL2 = NucL2*kair(TL2amb)/Dh
    elif (Re <= 5e5) and (Pr > 0.6):
        NucL2 = 0.664*Re**0.5*Pr**(1/3)
        hcL2 = NucL2*kair(TL2amb)/L2
        

    QL21 = AiL2UL2*((T2 - TL2) - (T1 - TL2))/np.log((T2 - TL2)/(T1 - TL2))    
    QL22 = Ao2*(hcL2 + hrL2)*(TL2 - Tamb)
    
    return QL21, QL22


def zone3B(unks, Ib, G, Ti, Tamb):
    To, T1, T2, T3, T3B, T4, Tw, Tf, Tg, TL1, TL2 = unks

    ri = 0.136
    rf = 0.122
    ro = 0.2

    L1 = 0.195
    
    L2 = 0.1079
    Aw = 0.1788
    
    Dh = 2*(ri - rf)    

    # Convection heat transfer coeff win (B.5)
    Twin = 0.5*(Tw + T3B)
    
    Re = 4*G/(np.pi*(2*ri + 2*rf)*mu(Twin)) #4*G/(np.pi*Dh*mu(Twin))
    Pr = cp(Twin)*mu(Twin)/kair(Twin)    
    fp = (0.790*np.log(Re) - 1.64)**(-2)
    
    if (Re > 3000 and Re <= 5e6) or (Pr > 0.5 and Pr <= 2e3):
        Nuwin = (fp/8)*(Re - 1e3)*Pr/(1 + 12.7*np.sqrt(fp/8)*(Pr**(2/3) - 1))
        hwin = Nuwin*kair(Twin)/Dh
    else:
        Nuwin = 0.664*Re**0.5*Pr**(1/3)
        hwin = Nuwin*kair(Twin)/L2
        
    Q3B1 = hwin*Aw*((Tw - T3) - (Tw - T3B))/np.log((Tw - T3)/(Tw - T3B))
    Q3B2 = G*cpMean(T3B, T3)*(T3B - T3)
    
    return Q3B1, Q3B2


def zone2(unks, Ib, G, Ti, Tamb):
    To, T1, T2, T3, T3B, T4, Tw, Tf, Tg, TL1, TL2 = unks
    
    Ffg = 0.2956
    Ffw = 0.7044
    Fgf = 0.6267
    Fgw = 0.3733
    Fwf = 0.4110
    Fwg = 0.1027
    
    ri = 0.136
    ro = 0.2

    L1 = 0.195
    rf = 0.122
    rg = 0.125
    
    L2 = 1079
    Aw = 0.1788
    Af = np.pi*rf**2
    Ag = np.pi*rg**2    
    
    epsf = 0.95
    epsw = 1e-3

    alphag = 0.013 # Absorptivity at visible wavelengths
    alphagp = 1 # Absorptivity at long wavelengths
    
    epsgp = 0.88 # Glass emisivity at long wavelengths
    epsg = 0.04 # Glass emisivity at visible wavelengths

    
    Dh = 2*(ri - rf)  
    
    # Convection heat transfer coeff 1-w (B.2)
    T1w = 0.5*(T1 + Tw)
    
    Re = 4*G/(np.pi*(2*ri + 2*rf)*mu(T1w)) #4*G/(np.pi*Dh*mu(T1w))
    Pr = cp(T1w)*mu(T1w)/kair(T1w)
    fp = (0.790*np.log(Re) - 1.64)**(-2)    
    
    if (Re > 3000 and Re<= 5e6) and (Pr > 0.5 and Pr <= 2e3):
        Nu1w = (fp/8)*(Re - 1e3)*Pr/(1 + 12.7*np.sqrt(fp/8)*(Pr**(2/3) - 1))
        h1w = Nu1w*kair(T1w)/Dh
    else:
        Nu1w = 0.664*Re**0.5*Pr**(1/3)
        h1w = Nu1w*kair(T1w)/L2
    
    
    taug = 0.851 # glass transmissivity
    rhof = 0.05 # foam reflectivity at visible wave
    rhow = 0.2 # reflectivity at visible wave
    sigma = 5.67e-8 # W/(m^2 K^4) (Stephan-Boltzmann constant)
    
    Tgi = 0.5*(Tg + Ti)
    
    Q21 = G*cpMean(T2, T1)*(T2 - T1)
    Q22 = h1w*Aw*((Tw - T1) - (Tw - T2))/np.log((Tw - T1)/(Tw - T2))
    Q23 = taug*Ib*Fgf*rhof*Ffw + taug*Ib*Fgw*(1 - rhow*Fwf - rhow*Fwg) + \
            sigma*(Tf**4 - Tw**4)/((1-epsf)/(Af*epsf) + 1/(Af*Ffw) + \
                                   (1-epsw)/(Aw*epsw)) - \
            sigma*(Tw**4 - Tgi**4)/((1-epsw)/(Aw*epsw) + 1/(Aw*Fwg) + \
                                    (1-epsg)/(Ag*epsg))
    
    return Q21, Q22, Q23


def zone3(unks, Ib, G, Ti, Tamb):
    To, T1, T2, T3, T3B, T4, Tw, Tf, Tg, TL1, TL2 = unks

    alphag = 0.013 # Absorptivity at visible wavelengths
    alphagp = 1 # Absorptivity at long wavelengths
    
    epsgp = 0.88 # Glass emisivity at long wavelengths
    epsg = 0.04 # Glass emisivity at visible wavelengths
    
    epsf = 0.95
    epsw = 1e-3
    
    rg = 0.125
    rf = 0.122
    
    ri = 0.136
    ro = 0.2
    
    Dh = 2*(ri - rf)  
    
    Ag = np.pi*rg**2
    Af = np.pi*rf**2
    Aw = 0.1788
    
    L2 = 0.1079
    
    taug = 0.851 # glass transmissivity
    rhof = 0.05 # foam reflectivity at visible wave
    rhow = 0.2 # reflectivity at visible wave
    sigma = 5.67e-8 # W/(m^2 K^4) (Stephan-Boltzmann constant)    
    
    # Convection heat transfer coeff 3gi (B.10)
    T3gi = 0.5*(T3 + Tg)
    
    Re = 4*G/(np.pi*(2*ri + 2*rf)*mu(T3gi)) #4*G/(2*np.pi*rg*mu(T3gi))
    Pr = cp(T3gi)*mu(T3gi)/kair(T3gi)    
    
    if (Re > 5e5 and Re<= 1e7) and (Pr > 0.6):
        Nu3i = 0.037*Re**0.8*Pr**(1/3)
    else:
        Nu3i = 0.664*Re**0.5*Pr**(1/3)
        
    h3gi = Nu3i*kair(T3gi)/rg


    # Convection heat transfer coeff 3go
    beta = 1/(0.5*(Tg + Tamb))
    g = 9.81
    Pr = cp(1/beta)*mu(1/beta)/kair(1/beta)    
    Ra = Pr*g*beta*(Tg - Tamb)*(np.sqrt(Ag))**3/(mu(1/beta/1.225))**2
    
    Nu3o = (0.825 + 0.387*Ra**(1/6)/((1 + (0.492/Pr)**(9/16))**8/27))**2
    
    T3go = 0.5*(T3 + Tg)
    
    hgo = Nu3o*kair(T3go)/rg
    
    Ffg = 0.2956
    Ffw = 0.7044
    Fgf = 0.6267
    Fgw = 0.3733
    Fwf = 0.4110
    Fwg = 0.1027
    
    taug = 0.851 # glass transmissivity
    rhof = 0.05 # foam reflectivity at visible wave
    rhow = 0.2 # reflectivity at visible wave
    sigma = 5.67e-8 # W/(m^2 K^4) (Stephan-Boltzmann constant)
    
    Q31 = G*cpMean(T3, T2)*(T3 - T2)    
    Q32 = h3gi*Ag*((Tg - T3) - (Tg - T2))/np.log((Tg - T3)/(Tg - T2))
    Q33 = alphag*Ib + taug*Ib*(Fgf*rhof*Ffg + Fgw*Fwg*rhow) + \
            sigma*(Tf**4 - Tg**4)/((1-epsf)/(Af*epsf) + 1/(Af*Ffg) + \
                                   (1-epsgp)/(Ag*epsgp)) + \
            sigma*(Tw**4 - Tg**4)/((1-epsw)/(Aw*epsw) + 1/(Aw*Fwg) + \
                                   (1-epsgp)/(Ag*epsgp)) - \
            hgo*Ag*(Tg - Tamb) - epsgp*Ag*sigma*(Tg**4 - Tamb**4)
    
    return Q31, Q32, Q33




def zone4(unks, Ib, G, Ti, Tamb):
    To, T1, T2, T3, T3B, T4, Tw, Tf, Tg, TL1, TL2 = unks
    
    alphag = 0.013 # Absorptivity at visible wavelengths
    alphagp = 1 # Absorptivity at long wavelengths
    
    epsgp = 0.88 # Glass emisivity at long wavelengths
    epsg = 0.04 # Glass emisivity at visible wavelengths
    
    epsf = 0.95
    epsw = 1e-3
        
    rg = 0.125
    rf = 0.122
    
    ri = 0.136
    ro = 0.2    

    Ag = np.pi*rg**2
    Af = np.pi*rf**2
    Aw = 0.1788
    
    phi = 0.792
    dc = 1.86e-3
    dp = 3.40e-4
    Ls = 6.58e-4
    Lf = 0.065
    Af = np.pi*rf**2
    PPI = 75 # Pores Per Inch
    
    Ffg = 0.2956
    Ffw = 0.7044
    Fgf = 0.6267
    Fgw = 0.3733
    Fwf = 0.4110
    Fwg = 0.1027
    
    
    Re = 4*G/(2*np.pi*rf*mu(Tf))
    Redc = Re*dc/(2*rf)/phi
    
    Nuv = (32.504*phi**0.38 - 109.94*phi**1.38 + \
           166.65*phi**2.38 - 86.95*phi**3.38)*Redc**0.438
    hvf = Nuv*kair(Tf)/rf
    
    Vf = Af*Lf*phi


    taug = 0.851 # glass transmissivity
    rhof = 0.05 # foam reflectivity at visible wave
    rhow = 0.2 # reflectivity at visible wave
    sigma = 5.67e-8 # W/(m^2 K^4) (Stephan-Boltzmann constant)
    
    Q41 = G*cpMean(T4, T3)*(T4 - T3)
    Q42 = Vf*hvf*((Tf - T3) - (Tf - T4))/np.log((Tf - T3)/(Tf - T4))
    Q43 = taug*Ib*Fgf*(1-rhof) + taug*Ib*Fgw*Fwf*rhow - \
            sigma*(Tf**4 - Tw**4)/((1-epsf)/(Af*epsf) + 1/(Af*Ffw) + \
                                   (1-epsw)/(Aw*epsw)) - \
            sigma*(Tf**4 - Tg**4)/((1-epsf)/(Af*epsf) + 1/(Af*Ffg) + \
                                   (1-epsg)/(Ag*epsg))
    
    return Q41, Q42, Q43


def gg0(TT):    
    
    return TT - 290

def gg1(TT):    
    
    return 5000 - TT



def dish(unks, Ib, G, Ti, Tamb):

    # Ib, G, Ti, Tamb = args
    To, T1, T2, T3, T3B, T4, Tw, Tf, Tg, TL1, TL2 = unks
             
    #
    # ZONE 1
    #    
    eq1 = zone1(unks, Ib, G, Ti, Tamb)[0] + zone1L(unks, Ib, G, Ti, Tamb)[0] - \
        zone1(unks, Ib, G, Ti, Tamb)[1]
        
    eq2 = zone1(unks, Ib, G, Ti, Tamb)[0] + zone1L(unks, Ib, G, Ti, Tamb)[1] - \
        zone1(unks, Ib, G, Ti, Tamb)[1]
        
    eq3 = zone1(unks, Ib, G, Ti, Tamb)[1] - zone1(unks, Ib, G, Ti, Tamb)[2]
    

    #
    # ZONE 2
    #
    eq4 = zone2(unks, Ib, G, Ti, Tamb)[0] + zone2L(unks, Ib, G, Ti, Tamb)[0] - \
            zone2(unks, Ib, G, Ti, Tamb)[1]
            
    eq5 = zone2(unks, Ib, G, Ti, Tamb)[0] + zone2L(unks, Ib, G, Ti, Tamb)[1] - \
            zone2(unks, Ib, G, Ti, Tamb)[1]
                
    eq6 = zone2(unks, Ib, G, Ti, Tamb)[1] + zone3B(unks, Ib, G, Ti, Tamb)[0] - \
            zone2(unks, Ib, G, Ti, Tamb)[2]
    
    eq7 = zone2(unks, Ib, G, Ti, Tamb)[1] + zone3B(unks, Ib, G, Ti, Tamb)[1] - \
            zone2(unks, Ib, G, Ti, Tamb)[2]
    

    #
    # ZONE 3
    #    
    eq8 = zone3(unks, Ib, G, Ti, Tamb)[0] - zone3(unks, Ib, G, Ti, Tamb)[1]
    
    eq9 = zone3(unks, Ib, G, Ti, Tamb)[1] - zone3(unks, Ib, G, Ti, Tamb)[2]
                              
                              
    #
    # ZONE 4
    #
    eq10 = zone4(unks, Ib, G, Ti, Tamb)[0] - zone4(unks, Ib, G, Ti, Tamb)[1]
    
    eq11 = zone4(unks, Ib, G, Ti, Tamb)[1] - zone4(unks, Ib, G, Ti, Tamb)[2]
    
    
    print(eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11)
    print()

    res = np.array([eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11])
    return np.sqrt(np.sum(res**2))





#################  To,  T1,  T2,  T3,  T3B,  T4,   Tw,   Tf,  Tg,  TL1, TL2
unks0 = np.array([1200, 510, 700, 710, 720, 1250, 1000, 1300, 500, 400, 410])

lower_bounds = 0.0*unks0 + 300.0  # Establece límite inferior en 0
upper_bounds = 0.0*unks0 + 5000.0

# sol = least_squares(dish, unks0, args=(1e3*44, 0.04, 500, 300), bounds=(lower_bounds, upper_bounds))
cons = ({'type': 'ineq', 'fun': gg0}, {'type': 'ineq', 'fun': gg1})
sol = minimize(dish, unks0, args=(1e3*44, 0.04, 500, 300), constraints=cons,
               tol=1e-10, bounds=list(zip(lower_bounds, upper_bounds)))

# bounds=list(zip(lower_bounds, upper_bounds)),


# sol = fmin_tnc(func=dish, x0=unks0, approx_grad=True, args=(1e3*44, 0.04, 500, 300),
#                bounds=list(zip(lower_bounds, upper_bounds)))

print()
print(sol.x)
