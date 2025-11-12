#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple radar simulator.
Reflectivity relationships based on:
- ice and snow: 
    Protat, A., Delanoë, J., Bouniol, D., Heymsfield, A. J., 
    Bansemer, A., & Brown, P. (2007). Evaluation of ice water 
    content retrievals from cloud radar reflectivity and 
    temperature using a large airborne in situ microphysical 
    database. Journal of applied meteorology and climatology, 
    46(5), 557-572.
- liquid and rain:
    Matrosov, S. Y., Uttal, T., & Hazen, D. A. (2004). 
    Evaluation of radar reflectivity–based estimates of water 
    content in stratiform marine clouds. Journal of Applied 
    Meteorology, 43(3), 405-419.
Attenuation relationships based on:
- snowfall:
    Matrosov, S. Y. (2007). Modeling backscatter properties 
    of snowfall at millimeter wavelengths. Journal of the 
    atmospheric sciences, 64(5), 1727-1736.
- rainfall:
    Matrosov, S. Y., Battaglia, A., & Rodriguez, P. (2008). 
    Effects of multiple scattering on attenuation-based 
    retrievals of stratiform rainfall from CloudSat. Journal 
    of Atmospheric and Oceanic Technology, 25(12), 2199-2208.
- liquid water, water vapor, oxygen:
    Matrosov, S. Y., Uttal, T., & Hazen, D. A. (2004). 
    Evaluation of radar reflectivity–based estimates of water 
    content in stratiform marine clouds. Journal of Applied 
    Meteorology, 43(3), 405-419.

@author: Thirza Feenstra
"""
import numpy as np
import helpers

def compute_radar_attenuation(srf_pres, srf_temp, hum, pres, temp, liq, snow, rain, height):
    '''compute radar two-way attenuation for W-band
    Attenuation from rain, snow, liquid water, water vapor and oxygen
    Input:
        - srf_pres: surface pressure (Pa)
        - srf_temp: surface temperature (K)
        - hum: humidity (kg/kg)
        - pres: pressure (Pa)
        - temp: temperature (K)
        - liq: cloud liquid water (kg/kg)
        - snow: cloud snow water (kg/kg)
        - rain: cloud rain water (kg/kg)
        - height: height above surface (m)
    '''
    pa = helpers.density(temp, hum, pres)
    w = helpers.kgkg_to_kgm3(liq, temp, hum, pres) 
    s = helpers.kgkg_to_kgm3(snow, temp, hum, pres)*2*3600
    r = helpers.kgkg_to_kgm3(rain, temp, hum, pres)*helpers.rain_fallspeed(rain, pa)*3600
    zeros_row = np.zeros((1, height.shape[1]))
    height2 = np.vstack([height, zeros_row])
    dz = np.abs(np.diff(height2/1000, axis=0)) 
    att_r = r/(1.32 * pa**-0.45) 
    att_s = 0.12*s**1.1 
    att_w = 7.56 * np.cumsum(w * dz*1000, axis=0) * (1 + (293 - temp)*0.012)
    att_H2O = 0.077 * np.cumsum(hum * dz*1000, axis=0) * srf_pres/1013 * \
        (293/srf_temp)**1.5 * (1 - np.exp(-0.42*(height[0]/1000-height/1000)))
    att_O2 = (srf_pres/1013)**2*(293/srf_temp)**2*((7.02e-2*(height[0]/1000-height/1000) - \
        4.81e-3*(height[0]/1000-height/1000)**2 + 1.22e-4*(height[0]/1000-height/1000)**3))
    att1 = att_r + att_s 
    att2 = att_H2O + att_O2 + att_w 
    att_integral = 2*np.cumsum(att1 * dz, axis=0) + att2 
    att_integral[~np.isfinite(att_integral)] = 0 
    return att_integral 

def radar_sim(ice, liq, snow, rain, temp, hum, pres, height):
    '''compute radar reflectivity using Z-WC relationships
    correction for attenuation by using 
    compute_radar_attenuation function
    Input:
        - ice: cloud ice water (kg/kg)
        - liq: cloud liquid water (kg/kg)
        - snow: cloud snow water (kg/kg)
        - rain: cloud rain water (kg/kg)
        - temp: temperature (K)
        - hum: humidity (kg/kg)
        - pres: pressure (Pa)
        - height: height above surface (m)'''
    ice_g= helpers.kgkg_to_kgm3(ice+snow, temp, hum, pres) * 1000
    ice_g[ice_g < 1e-3] = 0
    ze_ice = 10**((np.log10(ice_g)+0.0023*(temp-273.16)+0.84)/(0.000491*(temp-273.16)+0.0939) / 10) 
    ze_ice[~np.isfinite(ze_ice)] = 0
    liq_g= helpers.kgkg_to_kgm3(liq+rain, temp, hum, pres) * 1000
    liq_g[liq_g < 1e-3] = 0
    ze_liq = (liq_g/2.4)**2 # matrotrov 2004
    ze_liq[~np.isfinite(ze_liq)] = 0
    zliq = 10*np.log10(ze_liq)
    zliq[~np.isfinite(zliq)] = 0
    z = 10*np.log10(ze_liq + ze_ice)
    z = z - compute_radar_attenuation(pres[-1]/100, temp[-1], hum, pres, temp, liq, snow, rain, height)
    z[~np.isfinite(z)] = np.nan 
    z[z==0] = np.nan
    return z 

