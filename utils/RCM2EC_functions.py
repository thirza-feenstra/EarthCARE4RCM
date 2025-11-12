#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for extracting the EarthCARE coordinates 
from the RACMO domain and for interpolation of 
EarthCARE data to RACMO grid. 
Nearest neighbour interpolation is done on the 
rotated lat-lon coordinates of RACMO.
Code to convert to this grid can be found in 
Convert_From_To_RacmoGrid.

@author: Thirza Feenstra
"""

import numpy as np
import Convert_From_To_RacmoGrid as CFTR
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({'font.size': 12, 
                     "legend.fontsize": 14, "axes.labelsize": 14,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, 
                     "hatch.color":'darkgrey', 
                    'mathtext.default': 'regular'})


def get_EC_traj(f, domain, gridfile):
    '''  compute the coordinates of the EarthCARE trajectory on the rotated
    RACMO grid
    input:
        f: the loaded .h5 EarthCARE file
        domain: [lat_min, lat_max, lon_min, lon_max]
        gridfile: needed for the regridding process, loaded for the chosen domain
    returns: 
        (lat_EC_r, lon_EC_r), h_EC, lat_EC, lon_EC 
        lat_EC_r: latitude of EarthCARE trajectory on rotated grid
        lon_EC_r: longitude of EarthCARE trajectory on rotated grid
        h_EC: heights of EarthCARE observations
        lat_EC: latitude of EarthCARE trajectory on EarthCARE grid, within 'domain'
        lon_EC: longitude of EarthCARE trajectory on EarthCARE grid, within 'domain'
    '''
    try:
        lat_EC = f['latitude'].values
        lon_EC = f['longitude'].values
        if len(np.shape(lat_EC)) == 2:
            lat_EC = f['latitude_active'].values
            lon_EC = f['longitude_active'].values
        try:
            h_EC = f['height'].values - np.tile(f['geoid_offset'].values, (len(f['height'].values[0]),1)).T
        except:
            try:
                h_EC = f['binHeight'].values
            except:
                try:
                    h_EC = (f['height_layer'].values - np.tile(f['geoid_offset'].values, (len(f['height_layer'].values),1))).T
                except:
                    h_EC = f['height_layers'].values
    except:
        try:   
            lat_EC = f['ellipsoid_latitude'].values
            lon_EC = f['ellipsoid_longitude'].values
            h_EC = f['sample_altitude'].values - np.tile(f['geoid_offset'].values, (len(f['sample_altitude'].values[0]),1)).T
        except:
            lat_EC = f['barycentre_latitude'].values
            lon_EC = f['barycentre_longitude'].values
            h_EC = np.full((len(lat_EC), 20), np.nan)
    lat_min, lat_max, lon_min, lon_max = domain
    idx = np.where((lat_EC > lat_min) & (lat_EC < lat_max) & \
                   (lon_EC > lon_min) & (lon_EC < lon_max))
    lat_EC = lat_EC[idx]
    lon_EC = lon_EC[idx]
    h_ECn = np.zeros((len(lat_EC), len(h_EC[0])))
    for i in range(len(h_ECn[0])):
        h_ECn[:,i] = h_EC[idx,i]
    return CFTR.RealWorld2RotatedGrid(lat_EC, lon_EC, gridfile), h_ECn, lat_EC, lon_EC

def get_EC_r(f, var, domain, gpt=None, plot=False, return_H=False):
    ''' extract values of EarthCARE variable for a given domain and RACMO height range
    input:
        f: the loaded .h5 EarthCARE file
        var: name of the variable
        domain: [lat_min, lat_max, lon_min, lon_max]
        gpt: RACMO geopotential height (lev, lat, lon)
        plot: True if 2D profile needs to be plotted  (default: False)
        return_H: True if heights corresponding to the levels needs to be returned 
        (default: False)
    returns:
        d_r: EarthCARE values for chosen variable for given domain and RACMO heights
        if return_H == True: height of levels
    '''
    d = f[var].values
    lat_min, lat_max, lon_min, lon_max = domain
    if len(np.shape(d)) == 3: # ACM-COM 
        d = d[1].T 
    if len(np.shape(d)) == 2:
        try:
            lat_EC = f['latitude'].values
            lon_EC = f['longitude'].values
            if len(np.shape(lat_EC)) == 2:
                lat_EC = f['latitude_active'].values
                lon_EC = f['longitude_active'].values
            try:
                h_EC = f['height'].values - np.tile(f['geoid_offset'].values, (len(f['height'].values[0]),1)).T
            except:
                try:
                    h_EC = f['binHeight'].values
                except:
                    try:
                        h_EC = (f['height_layer'].values - np.tile(f['geoid_offset'].values, (len(f['height_layer'].values),1))).T
                    except:
                        h_EC = f['height_layers'].values
        except:
            lat_EC = f['ellipsoid_latitude'].values
            lon_EC = f['ellipsoid_longitude'].values
            h_EC = f['sample_altitude'].values - np.tile(f['geoid_offset'].values, (len(f['sample_altitude'].values[0]),1)).T
        try:
            idx = np.where((np.tile(lat_EC, (len(h_EC[0]),1)).T > lat_min) & (np.tile(lat_EC, (len(h_EC[0]),1)).T < lat_max) & \
                        (np.tile(lon_EC, (len(h_EC[0]),1)).T > lon_min) & (np.tile(lon_EC, (len(h_EC[0]),1)).T < lon_max) & \
                            (h_EC > (np.min(gpt)-10)) & (h_EC < np.max(gpt)))
            idx2 = np.where((lat_EC > lat_min) & (lat_EC < lat_max) & \
                        (lon_EC > lon_min) & (lon_EC < lon_max))
            h_ECn = np.zeros((len(d), len(h_EC[0])))
            for i in range(len(h_ECn[0])):
                h_ECn[:,i] = h_EC[idx2,i]
            d_r = d[idx].reshape(-1, len(h_ECn[0]))
        except:
            idx = np.where((np.tile(lat_EC, (len(h_EC[0]),1)).T > lat_min) & (np.tile(lat_EC, (len(h_EC[0]),1)).T < lat_max) & \
                        (np.tile(lon_EC, (len(h_EC[0]),1)).T > lon_min) & (np.tile(lon_EC, (len(h_EC[0]),1)).T < lon_max))
            try:
                d_r = d[idx].reshape(-1, len(h_EC[0]))
            except:
                d_r = d.T[idx].reshape(-1, len(h_EC[0]))
            idx2 = np.where((lat_EC > lat_min) & (lat_EC < lat_max) & \
                        (lon_EC > lon_min) & (lon_EC < lon_max))
            h_ECn = np.zeros((len(d_r), len(h_EC[0])))
            for i in range(len(h_ECn[0])):
                h_ECn[:,i] = h_EC[idx2,i]
            hidxmin = np.where(np.nanmean(h_ECn, axis=0) >= (np.min(gpt)-10))[0][-1]
            hidxmax = np.where(np.nanmean(h_ECn, axis=0) <= np.max(gpt))[0][0]
            d_r = d_r[:,hidxmax:hidxmin+1]
        if plot:
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax2 = ax1.twiny()
            plot = ax1.pcolormesh(d_r.T, shading='auto', 
                                cmap='Blues')
            fig.colorbar(plot)
            if lat_EC[0] < lat_EC[-1]:
                ax2.set_xlim(lon_EC[0], lon_EC[-1])
            else:
                ax2.set_xlim(lon_EC[-1], lon_EC[0])
    else:
        try:
            lat_EC = f['latitude'].values
            lon_EC = f['longitude'].values
            if len(np.shape(lat_EC)) == 2:
                lat_EC = f['latitude_active'].values
                lon_EC = f['longitude_active'].values
        except:
            try:
                lat_EC = f['ellipsoid_latitude'].values
                lon_EC = f['ellipsoid_longitude'].values
            except:
                lat_EC = f['barycentre_latitude'].values
                lon_EC = f['barycentre_longitude'].values
        idx = np.where((lat_EC > lat_min) & (lat_EC < lat_max) & \
                        (lon_EC > lon_min) & (lon_EC < lon_max))
        d_r = d[idx]
    if return_H:
        idx = np.where((lat_EC > lat_min) & (lat_EC < lat_max) & \
                        (lon_EC > lon_min) & (lon_EC < lon_max))
        h_ECn = np.zeros((len(d_r), len(h_EC[0])))
        idx2 = np.where((lat_EC > lat_min) & (lat_EC < lat_max) & \
                        (lon_EC > lon_min) & (lon_EC < lon_max))
        h_ECn = np.zeros((len(d_r), len(h_EC[0])))
        for i in range(len(h_ECn[0])):
            h_ECn[:,i] = h_EC[idx2,i]
        hidxmin = np.where(np.nanmean(h_ECn, axis=0) >= (np.min(gpt)-10))[0][-1]
        hidxmax = np.where(np.nanmean(h_ECn, axis=0) <= np.max(gpt))[0][0]
        h_ECn = h_ECn[:,hidxmax:hidxmin+1]
        return d_r, h_ECn
    else:
        return d_r

def plev2gph(t, h, ps, gs, return_half=False):
    ''' numbered pressure levels to geopotential height
    For 1 timestep of 1 column, nsteps timesteps for 1 column or for 
    3D (lev, lat, lon) field
    input:
        t: temperature field (lev)/(nsteps, lev)/(lev, lat, lon)
        h: humidity field (lev)/(nsteps, lev)/(lev, lat, lon)
        ps: surface pressure field (1)/(nsteps)/(lat, lon)
        gs: surface geopotential field (1)/(nsteps)/(lat, lon)
    returns:
        geopotential height array (lev)/(lev, nsteps)/(lev, lat, lon)
    '''
    ai = np.array([ 0.0000000,  2174.9854984,  4350.0145016,  6524.9854984,  8700.0145016,
                    10376.1183109, 12077.4598141, 13775.3136234, 15379.8113766, 16819.4620609,
                    18045.1863766, 19027.6964359, 19755.1160641, 20222.1964359, 20429.8660641,
                    20384.4776859, 20097.4051266, 19584.3292484, 18864.7488766, 17961.3604984,
                    16899.4676266, 15706.4542484, 14411.1238766, 13043.2276859, 11632.7566891,
                    10209.5011234,  8802.3582516,  7438.8058109,  6144.3191891,  4941.7745609,
                    3850.9129391,  2887.6974125,  2063.7796383,  1385.9132328,   855.3616695,
                    467.3336430,   210.3938961,    65.8892460,     7.3677425,     0.0000000,
                    0.0000000])
    bi = np.array([0.0000000,     0.0000000,     0.0000000,     0.0000000,     0.0000000,
                    0.0004618,     0.0018147,     0.0050815,     0.0111425,     0.0206783,
                    0.0341208,     0.0516908,     0.0735334,     0.0996752,     0.1300220,
                    0.1643847,     0.2024755,     0.2439334,     0.2883227,     0.3351552,
                    0.3838917,     0.4339633,     0.4847711,     0.5357102,     0.5861682,
                    0.6355478,     0.6832684,     0.7287860,     0.7715964,     0.8112535,
                    0.8473748,     0.8796570,     0.9078838,     0.9319404,     0.9518214,
                    0.9676453,     0.9796627,     0.9882702,     0.9940194,     0.9976301,
                    1.0000000])
    af = np.array([1.08749292e+03, 3.26250000e+03, 5.43750000e+03, 7.61250000e+03,
                    9.53806641e+03, 1.12267891e+04, 1.29263867e+04, 1.45775625e+04,
                    1.60996367e+04, 1.74323242e+04, 1.85364414e+04, 1.93914062e+04,
                    1.99886562e+04, 2.03260312e+04, 2.04071719e+04, 2.02409414e+04,
                    1.98408672e+04, 1.92245391e+04, 1.84130547e+04, 1.74304141e+04,
                    1.63029609e+04, 1.50587891e+04, 1.37271758e+04, 1.23379922e+04,
                    1.09211289e+04, 9.50592969e+03, 8.12058203e+03, 6.79156250e+03,
                    5.54304688e+03, 4.39634375e+03, 3.36930518e+03, 2.47573853e+03,
                    1.72484644e+03, 1.12063745e+03, 6.61347656e+02, 3.38863770e+02,
                    1.38141571e+02, 3.66284943e+01, 3.68387127e+00, 0.00000000e+00])
    bf = np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                    2.30902340e-04, 1.13827549e-03, 3.44813662e-03, 8.11201334e-03,
                    1.59103945e-02, 2.73995213e-02, 4.29057851e-02, 6.26121163e-02,
                    8.66042972e-02, 1.14848614e-01, 1.47203386e-01, 1.83430135e-01,
                    2.23204494e-01, 2.66128063e-01, 3.11738968e-01, 3.59523475e-01,
                    4.08927500e-01, 4.59367216e-01, 5.10240674e-01, 5.60939193e-01,
                    6.10857964e-01, 6.59408092e-01, 7.06027210e-01, 7.50191212e-01,
                    7.91424990e-01, 8.29314172e-01, 8.63515913e-01, 8.93770397e-01,
                    9.19912100e-01, 9.41880941e-01, 9.59733367e-01, 9.73653972e-01,
                    9.83966410e-01, 9.91144776e-01, 9.95824754e-01, 9.98815060e-01])
    rair = 287.05
    epsilo = rair/461.51
    gravit = 9.80665
    zepsm1 = (1/epsilo) - 1
    invGrav = 1/gravit
    if len(np.shape(t)) == 1:
        presi = np.zeros(len(t)+1)
        presf = np.zeros(len(t))
        for i in range(len(t)): # lev
            presi[i] = ai[i] + bi[i] * ps
            presf[i] = af[i] + bf[i] * ps
        geoi = np.zeros(len(t)+1)
        geof = np.zeros(len(t)+1)
        geoi[-1] = gs
        presi[-1] = ps
        for l in range(len(t)-1,-1,-1):
            tv = (1 + zepsm1*h[l])*t[l]
            geoi[l] = geoi[l+1] + np.log(presi[l+1]/presi[l])*rair*tv
            geof[l] = geoi[l+1] + np.log(presi[l+1]/presf[l])*rair*tv
    elif len(np.shape(t)) == 2:
        presi = np.zeros((len(t), len(t[0])+1))
        presf = np.zeros((len(t), len(t[0])))
        for i in range(len(t[0])): # lev
            for n in range(len(t)): #time
                presi[n,i] = ai[i] + bi[i] * ps[n]
                presf[n,i] = af[i] + bf[i] * ps[n]
        geoi = np.zeros((len(t), len(t[0])+1))
        geof = np.zeros((len(t), len(t[0])))
        geoi[:,-1] = np.full(len(t), gs)
        presi[:,-1] = ps
        for l in range(len(t[0])-1,-1,-1):
            tv = (1 + zepsm1*h[:,l])*t[:,l]
            geoi[:,l] = geoi[:,l+1] + np.log(presi[:,l+1]/presi[:,l])*rair*tv
            geof[:,l] = geoi[:,l+1] + np.log(presi[:,l+1]/presf[:,l])*rair*tv
    elif len(np.shape(t)) == 3:
        presi = np.zeros((len(t)+1, len(t[0]), len(t[0][0])))
        presf = np.zeros((len(t), len(t[0]), len(t[0][0])))
        for i in range(len(t)): # lev
            for j in range(len(t[0])): # lat
                for k in range(len(t[0][0])): # lon
                    presi[i,j,k] = ai[i] + bi[i] * ps[j,k]
                    presf[i,j,k] = af[i] + bf[i] * ps[j,k]
        geoi = np.zeros((len(t)+1, len(t[0]), len(t[0][0])))
        geof = np.zeros((len(t), len(t[0]), len(t[0][0])))
        geoi[-1] = gs
        presi[-1] = ps
        for l in range(len(t)-1,-1,-1):
            tv = (1 + zepsm1*h[l])*t[l]
            geoi[l] = geoi[l+1] + np.log(presi[l+1]/presi[l])*rair*tv
            geof[l] = geoi[l+1] + np.log(presi[l+1]/presf[l])*rair*tv
    if return_half:
        return geof * invGrav, geoi * invGrav
    else:
        return geof * invGrav

def intp_NN(d_R, gpt, lat_EC, lon_EC, h_EC, varname, topo, date=None):
    '''
    input:
        d_R: xr dataset containing at least RCM lat, lon and 3D data fields
        gpt: geopotential height 3D field RCM
        lat_EC, lon_EC: lat and lon of EarthCARE trajectory
        h_EC: geopotential height EarthCARE trajectory
        varname: variable name
        topo: topographic heights (2D)
        date: give date if one needs to be selected
    returns:
        profile of varname on EC trajectory, heights of EC trajectory within RCM domain
    '''
    h_EC_m = h_EC[(h_EC > np.min(gpt)) & (h_EC < np.max(gpt))]
    prof = np.full((len(lat_EC),len(h_EC_m)), np.nan)
    if date != None:
        d_val = d_R.sel(time=date, method='nearest')[varname].values
    else:
        d_val = d_R[varname].values
    try:
        lat_R, lon_R = np.meshgrid(d_R.rlat.values, d_R.rlon.values)
        lat_lon_flat = np.column_stack((lat_R.ravel(), lon_R.ravel()))
        lat_lon_EC = np.column_stack((lat_EC, lon_EC))
        tree = KDTree(lat_lon_flat)
        nn_coords = lat_lon_flat[tree.query(lat_lon_EC)[1]]
        lat_indices = {lat: np.where(d_R.rlat.values == lat)[0] for lat in np.unique(nn_coords[:, 0])}
        lon_indices = {lon: np.where(d_R.rlon.values == lon)[0] for lon in np.unique(nn_coords[:, 1])}
    except:
        # the names for lat and lon are different in the L1 datasets
        lat_R, lon_R = np.meshgrid(np.round(d_R.Latitude.values[:,0],2), np.round(d_R.Longitude.values[0],2))
        lat_lon_flat = np.column_stack((lat_R.ravel(), lon_R.ravel()))
        lat_lon_EC = np.column_stack((lat_EC, lon_EC))
        tree = KDTree(lat_lon_flat)
        nn_coords = np.round(lat_lon_flat[tree.query(lat_lon_EC)[1]],2)
        lat_indices = {lat: np.where(np.round(d_R.Latitude.values[:,0],2) == lat)[0] for lat in np.unique(nn_coords[:, 0])}
        lon_indices = {lon: np.where(np.round(d_R.Longitude.values[0],2) == lon)[0] for lon in np.unique(nn_coords[:, 1])}
    for i, (lat, lon) in enumerate(nn_coords):
        lat_idx = lat_indices[lat][0]
        lon_idx = lon_indices[lon][0]
        v = d_val[:, lat_idx, lon_idx]
        g = gpt[:, lat_idx, lon_idx]
        t = topo[lat_idx, lon_idx]
        h_topo = h_EC_m[h_EC_m > t]
        if h_topo.size > 0:
            for j in range(len(h_topo)):
                prof[i, j] = v[np.abs(g - h_topo[j]).argmin()]
    return prof, h_EC_m

def intp_NN_racmogrid(d_R, gpt, lat_EC, lon_EC, varname, gpt_half=None):
    '''
    input:
        d_R = xr dataset containing at least RCM lat, lon and 3D data fields
        lat_EC, lon_EC = lat and lon of EarthCARE trajectory
        varname = variable name
        topo = topographic heights (2D)
        date = give date if one needs to be selected
    returns:
        profile of varname on EC trajectory, but on the RACMO grid points
    '''
    prof = np.full((len(lat_EC),len(gpt)), np.nan)
    height = np.full((len(lat_EC),len(gpt)), np.nan)
    if gpt_half is not None:
        height_half = np.full((len(lat_EC),len(gpt_half)), np.nan)
    nlat = []
    nlon = []
    d_val = d_R[varname].values
    if varname == 'refficlo' or varname == 'refflclo': # in some way this one has 41 level instead of 40...
        d_val = d_val[:-1]
    rlat, rlon = d_R.rlat.values, d_R.rlon.values
    drlat = rlat[1] - rlat[0]
    drlon = rlon[1] - rlon[0]
    rlat = np.append(rlat, np.arange(rlat[-1]+drlat, rlat[-1]+4*drlat, drlat))
    rlat = np.insert(rlat, 0, np.arange(rlat[0]-3*drlat, rlat[0], drlat))
    rlon = np.append(rlon, np.arange(rlon[-1]+drlon, rlon[-1]+4*drlon, drlon))
    rlon = np.insert(rlon, 0, np.arange(rlon[0]-3*drlon, rlon[0], drlon))
    lat_R, lon_R = np.meshgrid(rlat, rlon)
    lat_lon_flat = np.column_stack((lat_R.ravel(), lon_R.ravel()))
    lat_lon_EC = np.column_stack((lat_EC, lon_EC))
    tree = KDTree(lat_lon_flat)
    nn_coords = lat_lon_flat[tree.query(lat_lon_EC)[1]]
    lat_indices = {lat: np.where(d_R.rlat.values == lat)[0] for lat in np.unique(nn_coords[:, 0])}
    lon_indices = {lon: np.where(d_R.rlon.values == lon)[0] for lon in np.unique(nn_coords[:, 1])}
    for i, (lat, lon) in enumerate(nn_coords[np.sort(np.unique(nn_coords, axis=0, return_index=True)[1])]):
        if (lat >= np.min(d_R.rlat.values)) & (lat <= np.max(d_R.rlat.values)) & (lon >= np.min(d_R.rlon.values)) & (lon <= np.max(d_R.rlon.values)):
            lat_idx = lat_indices[lat][0]
            lon_idx = lon_indices[lon][0]
            prof[i, :] = d_val[:, lat_idx, lon_idx]
            height[i,:] = gpt[:, lat_idx, lon_idx]
            nlat.append(lat)
            nlon.append(lon)
            if gpt_half is not None:
                height_half[i,:] = gpt_half[:, lat_idx, lon_idx]
    prof = prof.compress(np.logical_not(np.all(np.isnan(prof), axis=1)), axis=0)
    height = height.compress(np.logical_not(np.all(np.isnan(height), axis=1)), axis=0)
    if gpt_half is not None:
        height_half = height_half.compress(np.logical_not(np.all(np.isnan(height_half), axis=1)), axis=0)
        return prof, np.array(nlat), np.array(nlon), height, height_half
    else:
        return prof, np.array(nlat), np.array(nlon), height

def intp_NN_racmogrid_1D(d_R, lat_EC, lon_EC, varname, idx=None):
    '''
    input:
        d_R = xr dataset containing at least RCM lat, lon and 2D data fields
        lat_EC, lon_EC = lat and lon of EarthCARE trajectory
        varname = variable name
    returns:
        profile of varname on EC trajectory, but on the RACMO grid points
    '''
    prof = np.full((len(lat_EC)), np.nan)
    nlat = []
    nlon = []
    d_val = d_R[varname].values
    if len(np.shape(d_val)) == 3:
        d_val = d_val[idx]
    rlat, rlon = d_R.rlat.values, d_R.rlon.values
    drlat = rlat[1] - rlat[0]
    drlon = rlon[1] - rlon[0]
    rlat = np.append(rlat, np.arange(rlat[-1]+drlat, rlat[-1]+4*drlat, drlat))
    rlat = np.insert(rlat, 0, np.arange(rlat[0]-3*drlat, rlat[0], drlat))
    rlon = np.append(rlon, np.arange(rlon[-1]+drlon, rlon[-1]+4*drlon, drlon))
    rlon = np.insert(rlon, 0, np.arange(rlon[0]-3*drlon, rlon[0], drlon))
    lat_R, lon_R = np.meshgrid(rlat, rlon)
    lat_lon_flat = np.column_stack((lat_R.ravel(), lon_R.ravel()))
    lat_lon_EC = np.column_stack((lat_EC, lon_EC))
    tree = KDTree(lat_lon_flat)
    nn_coords = lat_lon_flat[tree.query(lat_lon_EC)[1]]
    lat_indices = {lat: np.where(d_R.rlat.values == lat)[0] for lat in np.unique(nn_coords[:, 0])}
    lon_indices = {lon: np.where(d_R.rlon.values == lon)[0] for lon in np.unique(nn_coords[:, 1])}
    for i, (lat, lon) in enumerate(nn_coords[np.sort(np.unique(nn_coords, axis=0, return_index=True)[1])]):
        if (lat >= np.min(d_R.rlat.values)) & (lat <= np.max(d_R.rlat.values)) & (lon >= np.min(d_R.rlon.values)) & (lon <= np.max(d_R.rlon.values)):
            lat_idx = lat_indices[lat][0]
            lon_idx = lon_indices[lon][0]
            prof[i] = d_val[lat_idx, lon_idx]
            nlat.append(lat)
            nlon.append(lon)
    return np.compress(~np.isnan(prof), prof), np.array(nlat), np.array(nlon)


def downsample(data, latitudes, longitudes, heights, target_latitudes, 
               target_longitudes, target_heights, sample_type='mean'):
    heights[~np.isfinite(heights)] = -100000
    mask = (target_latitudes >= np.min(latitudes)) & (target_latitudes <= np.max(latitudes))
    filtered_latitudes = target_latitudes[mask]
    filtered_longitudes = target_longitudes[mask]
    downsampled_data = np.zeros((len(target_heights), len(filtered_latitudes)))
    count = np.zeros_like(downsampled_data)
    latitudes = latitudes[:len(data)]
    longitudes = longitudes[:len(data)]
    lat_lon_EC = np.column_stack((latitudes, longitudes))
    lat_lon_flat = np.column_stack((filtered_latitudes, filtered_longitudes))
    tree = KDTree(lat_lon_flat)
    nn_indices = tree.query(lat_lon_EC)[1] 
    for i, idx in enumerate(nn_indices):
        if sample_type == 'mean':
            v = data[i, :]
        elif sample_type == 'error':
            v = data[i, :]**2
        h = heights[i, :]
        if len(np.shape(target_heights)) == 1:
            mask_h = (h >= target_heights[-1]) & (h <= target_heights[0])
            v, h = v[mask_h], h[mask_h]
            h_indices = np.abs(h[:, None] - target_heights).argmin(axis=1)
        elif len(np.shape(target_heights)) == 2:
            mask_h = (h >= target_heights[-1][idx]) & (h <= target_heights[0][idx])
            v, h = v[mask_h], h[mask_h]
            h_indices = np.abs(h[:, None] - target_heights[:,idx]).argmin(axis=1)
        if v.size > 0:
            np.add.at(downsampled_data[:, idx], h_indices, v)
            np.add.at(count[:, idx], h_indices, 1)   
    if sample_type == 'mean':
        downsampled_data = downsampled_data / count
    elif sample_type == 'error':
        downsampled_data = np.sqrt(downsampled_data) / count
    count[count == 0] = np.nan
    nan_mask = np.isnan(count)
    lat_mis = []
    lon_mis = []
    mis_idx = []
    for i in range(downsampled_data.shape[1]):
        if np.all(nan_mask[:, i]):  # Entire column is NaN
            lat_mis.append(filtered_latitudes[i])
            lon_mis.append(filtered_longitudes[i])
            mis_idx.append(i)
    lat_lon_mis = np.column_stack((lat_mis, lon_mis))
    lat_lon_EC = np.column_stack((latitudes, longitudes))
    tree = KDTree(lat_lon_EC)
    nn_indices = tree.query(lat_lon_mis)[1]
    for i, idx in enumerate(nn_indices):
        v = data[(idx-1):(idx+2),:]
        if np.all(np.isnan(v)):
            v = data[(idx-2):(idx+3),:]
            if np.all(np.isnan(v)):
                v = data[(idx-3):(idx+4),:]
        if len(np.shape(target_heights)) == 1:
            h = target_heights[:]
        elif len(np.shape(target_heights)) == 2:     
            h = target_heights[:,mis_idx[i]]
        h_indices = np.abs(heights[idx,:] - h[:, None]).argmin(axis=1)
        for j in range(len(h_indices)):
            downsampled_data[j,mis_idx[i]] = np.nanmean(v[:,(h_indices[j]-1):(h_indices[j]+2)])
    # Fill missing values with nearest available values
    nan_mask = np.isnan(downsampled_data)
    for i in range(downsampled_data.shape[1]):
        col = downsampled_data[:, i] 
        nan_idx = np.where(np.isnan(col))[0]  # Find NaN indices in column
        for j in nan_idx:
            if j < len(target_heights) - 1 and not np.isnan(col[j + 1]):  # Use value below if available
                col[j] = col[j + 1] 
            elif j > 0 and not np.isnan(col[j - 1]):  # Use value above if available
                col[j] = col[j - 1]
    return downsampled_data, (filtered_latitudes, filtered_longitudes)

def downsample_l2b_classification(clas, latitudes, longitudes, heights, 
                                  target_latitudes, target_longitudes, 
                                  target_heights):
    mask = (target_latitudes >= np.min(latitudes)) & (target_latitudes <= np.max(latitudes))
    filtered_latitudes = target_latitudes[mask]
    filtered_longitudes = target_longitudes[mask]
    downsampled_data = np.full((len(target_heights), len(filtered_latitudes)), np.nan)
    downsampled_snow = np.full((len(target_heights), len(filtered_latitudes)), np.nan)
    downsampled_rain = np.full((len(target_heights), len(filtered_latitudes)), np.nan)
    snow = np.zeros((len(target_heights), len(filtered_latitudes)))
    rain = np.zeros((len(target_heights), len(filtered_latitudes)))
    count = np.zeros_like(downsampled_data)
    count_snow = np.zeros_like(downsampled_data)
    count_rain = np.zeros_like(downsampled_data)
    latitudes = latitudes[:len(clas)]
    longitudes = longitudes[:len(clas)]
    lat_lon_EC = np.column_stack((latitudes, longitudes))
    lat_lon_flat = np.column_stack((filtered_latitudes, filtered_longitudes))
    tree = KDTree(lat_lon_flat)
    nn_indices = tree.query(lat_lon_EC)[1] 
    for i, idx in enumerate(nn_indices):
        v = clas[i, :]
        h = heights[i, :]
        mask_h = (h >= (target_heights[0][idx]-10)) & (h <= target_heights[0][idx])
        v, h = v[mask_h], h[mask_h]
        h_indices = np.abs(h[:, None] - target_heights[:,idx]).argmin(axis=1)
        if v.size > 0:
            c = np.full(len(v), np.nan)
            s = np.zeros(len(v))
            r = np.zeros(len(v))
            c[np.isin(v, np.array([3,13,14,15,19,21]))] = 0                       
            c[np.isin(v, np.array([16,17,20]))] = 0.5               
            c[np.isin(v, np.array([8,9,18]))] = 1   
            s[np.isin(v, np.array([3,6,13,14,15,16,17]))] = 1 
            r[np.isin(v, np.array([2,5,6,9,10,11,12]))] = 1 
            for k,h in enumerate(h_indices):
                if np.isfinite(c[k]):
                    if not np.isfinite(downsampled_data[h, idx]):
                        downsampled_data[h, idx] = c[k] 
                        count[h, idx] += 1
                        count_snow[h,idx] += 1
                        count_rain[h,idx] += 1
                    else:
                        downsampled_data[h, idx] += c[k]
                        count[h, idx] += 1
                        count_snow[h,idx] += 1
                        count_rain[h,idx] += 1
                if np.isfinite(s[k]):
                    snow[h, idx] += s[k]
                    count_snow[h,idx] += 1
                    rain[h, idx] += s[k]
                    count_rain[h,idx] += 1
                        
    downsampled_data = downsampled_data / count
    downsampled_snow[(snow >= 0.5*count_snow) & (count_snow > 0)] = 1
    downsampled_rain[(rain >= 0.5*count_rain) & (count_rain > 0)] = 1
    nan_mask = np.isnan(downsampled_data)
    lat_mis = []
    lon_mis = []
    mis_idx = []
    for i in range(downsampled_data.shape[1]):
        if np.all(nan_mask[:, i]):  # Entire column is NaN
            lat_mis.append(filtered_latitudes[i])
            lon_mis.append(filtered_longitudes[i])
            mis_idx.append(i)
    lat_lon_mis = np.column_stack((lat_mis, lon_mis))
    lat_lon_EC = np.column_stack((latitudes, longitudes))
    tree = KDTree(lat_lon_EC)
    nn_indices = tree.query(lat_lon_mis)[1]
    for i, idx in enumerate(nn_indices):
        v = clas[idx,:]
        h = target_heights[:,mis_idx[i]]
        h_indices = np.abs(heights[idx,:] - h[:, None]).argmin(axis=1)
        if v.size > 0:
            c = np.full(len(v), np.nan)
            c[np.isin(v, np.array([3,13,14,15,19,21]))] = 0                       
            c[np.isin(v, np.array([16,17,20]))] = 0.5    
            c[np.isin(v, np.array([8,9,18]))] = 1 
        downsampled_data[:,mis_idx[i]] = c[h_indices]
    nan_mask = np.isnan(downsampled_snow)
    lat_mis = []
    lon_mis = []
    mis_idx = []
    for i in range(downsampled_snow.shape[1]):
        if np.all(nan_mask[:, i]):  # Entire column is NaN
            lat_mis.append(filtered_latitudes[i])
            lon_mis.append(filtered_longitudes[i])
            mis_idx.append(i)
    lat_lon_mis = np.column_stack((lat_mis, lon_mis))
    lat_lon_EC = np.column_stack((latitudes, longitudes))
    tree = KDTree(lat_lon_EC)
    nn_indices = tree.query(lat_lon_mis)[1]
    for i, idx in enumerate(nn_indices):
        v = clas[idx,:]
        h = target_heights[:,mis_idx[i]]
        h_indices = np.abs(heights[idx,:] - h[:, None]).argmin(axis=1)
        if v.size > 0:
            s = np.full(len(v), np.nan)
            r = np.full(len(v), np.nan)
            s[np.isin(v, np.array([3,6,13,14,15,16,17]))] = 1 
            r[np.isin(v, np.array([2,5,6,9,10,11,12]))] = 1 
        downsampled_snow[:,mis_idx[i]] = s[h_indices]
        downsampled_rain[:,mis_idx[i]] = r[h_indices]
    return downsampled_data, downsampled_snow, downsampled_rain, (filtered_latitudes, filtered_longitudes)

def downsample_1D(data, latitudes, longitudes, target_latitudes, target_longitudes):
    mask = (target_latitudes >= np.min(latitudes)) & (target_latitudes <= np.max(latitudes))
    filtered_latitudes = target_latitudes[mask]
    filtered_longitudes = target_longitudes[mask]
    downsampled_data = np.zeros(len(filtered_latitudes))
    count = np.zeros_like(downsampled_data)
    latitudes = latitudes[:len(data)]
    longitudes = longitudes[:len(data)]
    lat_lon_EC = np.column_stack((latitudes, longitudes))
    lat_lon_flat = np.column_stack((filtered_latitudes, filtered_longitudes))
    tree = KDTree(lat_lon_flat)
    nn_indices = tree.query(lat_lon_EC)[1] 
    for i, idx in enumerate(nn_indices):
        v = data[i]
        if v.size > 0:
            downsampled_data[idx] += v
            count[idx] += 1
    downsampled_data = downsampled_data / count
    nan_mask = np.isnan(downsampled_data)
    lat_mis = []
    lon_mis = []
    mis_idx = []
    for i in range(len(downsampled_data)):
        if np.isnan(nan_mask[i]):  
            lat_mis.append(filtered_latitudes[i])
            lon_mis.append(filtered_longitudes[i])
            mis_idx.append(i)
    if len(mis_idx) > 0:
        lat_lon_mis = np.column_stack((lat_mis, lon_mis))
        lat_lon_EC = np.column_stack((latitudes, longitudes))
        tree = KDTree(lat_lon_EC)
        nn_indices = tree.query(lat_lon_mis)[1]
        for j, idx in enumerate(nn_indices):
            v = data[(idx-1):(idx+2)]
        downsampled_data[mis_idx[i]] = np.nanmean(v)
    return downsampled_data, (filtered_latitudes, filtered_longitudes)
