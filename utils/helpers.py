#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions
- conversion kg/kg to kg/m3
- compute density of air
- make RACMO precipitation and cloud classificatin
- compute RACMO rain fall speed
- compute ATLID-CPR composite
- compute 500 hpa geopotential

@author: Thirza Feenstra
"""

import numpy as np
from scipy import special
from math import gamma

def kgkg_to_kgm3(conc, temp, hum, pres):
    '''convert water concentration from kg/kg to kg/m3'''
    M = 0.0289652 #average molmass of air in kg/mol
    R = 8.314 #gascostant 
    rair = 287.05
    epsilon = rair/461.51
    zepsm = (1/epsilon) - 1
    t_v = (1+zepsm*hum)*temp
    return conc*pres*M/(R*t_v)

def density(temp, hum, pres):
    '''compute air density in kg/m3'''
    M = 0.0289652 #average molmass of air in kg/mol
    R = 8.314 #gascostant 
    rair = 287.05
    epsilon = rair/461.51
    zepsm = (1/epsilon) - 1
    t_v = (1+zepsm*hum)*temp
    return pres*M/(R*t_v)

def make_RACMO_classification(ice, liq, snow, rain):
    '''make classification aligned with EarthCARE 
    classification for cloud phase plus
    precipitation based on RACMO modeled clouds'''
    clas = np.full((np.shape(ice)), np.nan)
    csnow = np.full((np.shape(ice)), np.nan)
    crain = np.full((np.shape(ice)), np.nan)
    clas[((ice > 1e-7) & (liq < 1e-7)) | ((ice > 1e-7) & (liq > 1e-7) & (liq / ice < 0.1)) \
        | ((ice > 1e-7) & (liq > 1e-7) & (ice / liq < 0.1))] = 0.1
    clas[(ice > 1e-7) & (liq > 1e-7) & (liq / ice > 0.1) & (ice / liq > 0.1)] = 0.5
    clas[(ice < 1e-7) & (liq > 1e-7)] = 0.9
    csnow[snow > 1e-7] = 1
    crain[rain > 1e-7] = 1
    return clas, csnow, crain

def rain_fallspeed(rwc, rho):
    '''
    Compute rain fall speed as in ECMWF IFS cycle47r1
    rwc = rain water content [kg/kg] (height, latitude)
    rho = density of air [kg m-3]
    '''
    fall = np.zeros(np.shape(rwc))
    for i in range(len(fall)):
        if i == 0:
            rwc_i = rwc[i]
        else:
            rwc_i = 0.5*rwc[i] + 0.5*rwc[i-1]
        lam = (1000*np.pi/6*0.22 * special.gamma(4)/(rho[i]*rwc_i))**(1/1.8)
        const = 386.8 * special.gamma(4.67) / special.gamma(4)
        fall[i] = (1/rho[i])**0.4 * const * lam**(-0.67)
    return fall


def blowing_snow_fallspeed(pressure,
                            temperature,
                            qb,
                            height,
                            ustar=0.30,
                            z0=1e-3,
                            zsalt=0.05):
    '''
    Compute blowing snow fallspeed,
    based on offline Python code of 
    Piektuk-D model 
    (https://zenodo.org/records/19252674).
    '''
    ni = 16
    visc = 1.53e-5
    dice = 917.0
    g = 9.80665
    Rd = 287.0
    kappa = 0.40
    alpha = 4.0
    ar = 1e-4
    beta = ar / alpha
    m = 3.0
    pressure = np.asarray(pressure)
    temperature = np.asarray(temperature)
    height = np.asarray(height)
    qb = np.asarray(qb)
    rho = pressure / (Rd * temperature)
    qb0 = qb[0] if qb.ndim > 0 else qb
    rho0 = rho[0]
    rzeta = np.linspace(0.0, 1.0, ni)
    stretch = 0.5 * (1.0 + np.tanh(2.1 * (rzeta - 0.5)) / np.tanh(1.05))
    radlb = 2e-6
    radub = 400e-6
    drn = np.zeros(ni + 1)
    rn = np.zeros(ni + 1)
    for i in range(1, ni + 1):
        drn[i] = stretch[i - 1] * (radub - radlb) + radlb
    for i in range(1, ni + 1):
        rn[i] = 0.5 * (drn[i] - drn[i - 1]) + drn[i - 1]
    rn[1] = radlb
    ri = rn[1:]
    dr = drn[1:] - drn[:-1]
    N0 = (
        3.0 * rho0 * qb0
        * gamma(alpha)
        / (4.0 * np.pi * dice * gamma(alpha + 3.0) * beta**3)
    )
    fi = ri**(alpha - 1) * np.exp(-ri / beta) / beta**alpha
    F0 = N0 * fi / gamma(alpha)
    ri2 = ri[None, :]
    rho2 = rho[:, None]
    vt_2d = (
        -12.0 * visc
        + np.sqrt(
            144.0 * visc**2
            + 20.6336 * ri2**3 * g * dice / rho2
        )
    ) / (3.8688 * ri2)
    zratio = (height[:, None] + z0) / (zsalt + z0)
    exponent = -vt_2d / (kappa * ustar)
    log_height_factor = exponent * np.log(zratio)
    log_height_factor = np.clip(log_height_factor, -20, 20)
    height_factor = np.exp(log_height_factor)
    Fr = F0[None, :] * height_factor
    Fr = np.maximum(Fr, 1e-300) 
    weight = ri**m * dr
    numerator = np.sum(vt_2d * Fr * weight[None, :], axis=1)
    denominator = np.sum(Fr * weight[None, :], axis=1)
    w = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0
    )
    return w

def compute_composite(a_iwc, a_sig_iwc, a_re, a_sig_re, 
                      c_iwc, c_sig_iwc_log, c_sig_re_log):
    ''' Computes water content composite of ATLID 
    (ATL_ICE_2A) and CPR (CPR_CLD_2A) level 2a products.
    Note that CPR water content contains not only ice, 
    but also snow (and rain?)
    Based on: https://doi.org/10.5194/amt-16-4271-2023,
    section 3.2
    Input:
        - a_iwc: ATLID ice water content (ice_water_content)
        - a_sig_iwc: ATLID ice water content standard
            deviation (ice_water_content_error)
        - a_re: ATLID ice effective radius (ice_effective_radius)
        - a_sig_re: ATLID ice effective radius standard
            deviation (ice_effective_radius_error)
        - c_iwc: CPR water content (water_content)
        - c_sig_iwc_log: CPR water content logarithmic 
            standard deviation (water_content_log_error)
        - c_re: CPR particle effective diameter
            (characteristic_diameter)
        - c_sig_re_log: CPR particle effective diameter
            logarithmic standard deviation 
            (characteristic_diameter_log_error)
    Output:
        - comp: water content composite profile
    '''
    try:
        a_iwc[~np.isfinite(a_iwc)] = 0 
        c_iwc[~np.isfinite(c_iwc)] = 0 
        a_iwc[a_iwc < 1e-7] = 0 
        c_iwc[c_iwc < 1e-7] = 0 
        sig_a = np.sqrt((a_sig_iwc/a_iwc)**2+(a_sig_re/a_re)**2)
        sig_a[sig_a > 1e30] = np.nan
        sig_a[~np.isfinite(sig_a)] = 0
        # use error propagation to convert the log stds
        sig_c = np.sqrt((c_sig_iwc_log*np.log(10))**2+(c_sig_re_log*np.log(10))**2)
        sig_c[sig_c > 1e30] = np.nan
        sig_c[~np.isfinite(sig_c)] = 0
        comp = np.zeros(np.shape(a_iwc))
        for i in range(len(comp)):
            for j in range(len(comp[0])):
                if (a_iwc[i,j]==0) & (c_iwc[i,j] > 0):
                    comp[i,j] = c_iwc[i,j]
                elif (c_iwc[i,j]==0) & (a_iwc[i,j] > 0):
                    comp[i,j] = a_iwc[i,j]
                elif (a_iwc[i,j]>0) & (c_iwc[i,j] > 0):
                    if sig_a[i,j] > sig_c[i,j]:
                        comp[i,j] = c_iwc[i,j]
                    else:
                        comp[i,j] = a_iwc[i,j]
    except:
        if a_iwc < 1e-7:
            a_iwc = 0 
        if c_iwc < 1e-7:
            c_iwc = 0 
        sig_a = np.sqrt((a_sig_iwc/a_iwc)**2+(a_sig_re/a_re)**2)
        sig_c = np.sqrt((c_sig_iwc_log*np.log(10))**2+(c_sig_re_log*np.log(10))**2)
        if a_iwc > 1e30:
            a_iwc = 0
        if c_iwc > 1e30:
            c_iwc = 0
        if sig_a > 1e30:
            sig_a = 0
        if sig_c > 1e30: 
            sig_c = 0
        if (a_iwc==0) & (c_iwc > 0):
            comp = c_iwc
        elif (c_iwc==0) & (a_iwc > 0):
            comp = a_iwc
        elif (a_iwc>0) & (c_iwc > 0):
            if sig_a > sig_c:
                comp = c_iwc
            else:
                comp = a_iwc
        else:
            comp = 0
    return comp    


def geopotential_500hpa(pres, gph):
    """
    Compute 500 hPa geopotential height from 3D pressure and geopotential arrays.

    Parameters:
        pres : np.ndarray
            3D array of pressure [lev, lat, lon] in Pa.
        gph : np.ndarray
            3D array of geopotential [lev, lat, lon].

    Returns:
        fi500 : np.ndarray
            2D array [lat, lon] of geopotential at 500 hPa.
            If 500 hPa is outside the vertical range, returns np.nan.
    """
    nlev, nlat, nlon = pres.shape
    fi500 = np.full((nlat, nlon), np.nan)
    mask = (pres[:-1, :, :] < 50000) & (pres[1:, :, :] >= 50000)
    idx = np.argmax(mask, axis=0) 
    valid = mask.any(axis=0)
    lat_idx, lon_idx = np.where(valid)
    lev_idx = idx[lat_idx, lon_idx]
    p1 = pres[lev_idx, lat_idx, lon_idx]
    p2 = pres[lev_idx + 1, lat_idx, lon_idx]
    f1 = gph[lev_idx, lat_idx, lon_idx]
    f2 = gph[lev_idx + 1, lat_idx, lon_idx]
    fi500[lat_idx, lon_idx] = (
        np.log(p2 / 50000.0) * f1 + np.log(50000.0 / p1) * f2
    ) / np.log(p2 / p1)
    return fi500
