#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code to make files of RACMO coordinates on 
EarthCARE trajectory and input for the 
ATLID simulator.
Works for one month at the time.
Input:
- domain: RACMO domain name
- exp: RACMO experiment name (for location)
- month: month to process (e.g. 202503 for March 2025)

@author: Thirza Feenstra
"""

import numpy as np
import pandas as pd
import sys
sys.path.append('../utils/')
import Convert_From_To_RacmoGrid as CFTR
import RCM_gridSettings as RCMG
import RCM2EC_functions
import xarray as xr
from scipy.interpolate import CubicSpline
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import os
from datetime import datetime
import warnings
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings("ignore")


plt.rcParams.update({'font.size': 12, 
                     "legend.fontsize": 14, "axes.labelsize": 14,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, 
                     "hatch.color":'darkgrey', 
                    'mathtext.default': 'regular'})

user = 'username'

def make_clm_input(path_rcm_in, path_ec, path_out, path_aer, dtop, gridfile, domain='FGRN055'):
    racmo = xr.open_dataset(path_rcm_in, engine='netcdf4').squeeze()
    topo = dtop.Topography.values
    gs = dtop.Geopotential.values
    t = racmo.temp.values
    h = racmo.hum.values
    ps = racmo.ps.values
    gpt = RCM2EC_functions.plev2gph(t, h, ps, gs)
    date = path_rcm_in[-17:-5]
    dtime = datetime.strptime(date, '%Y%m%d%H%M')
    if 'CPR' in path_ec:
        if path_ec[-5] == 'M':
            ec = xr.open_dataset(path_ec, phony_dims='sort', engine='h5netcdf')
        else:
            ec = xr.open_dataset(path_ec, group='ScienceData/Geo', phony_dims='sort', engine='h5netcdf')
    else:
        if path_ec[-5] == 'M':
            ec = xr.open_dataset(path_ec, engine='h5netcdf')
        else:
            ec = xr.open_dataset(path_ec, group='ScienceData', engine='h5netcdf')
    if domain == 'FGRN055':
        (lat_EC_r, lon_EC_r), h_EC_r, lat_EC, lon_EC = RCM2EC_functions.get_EC_traj(ec, domain, [55,90,-105,35], gridfile)
    elif domain == 'PXANT11':
        (lat_EC_r, lon_EC_r), h_EC_r, lat_EC, lon_EC = RCM2EC_functions.get_EC_traj(ec, domain, [-90,-50,-180,180], gridfile)
    lat_rac_r, lon_rac_r, height = RCM2EC_functions.intp_NN_racmogrid(racmo, gpt, lat_EC_r, lon_EC_r, 'hum', topo)[1:]
    lat_rac, lon_rac = CFTR.RotatedGrid2RealWorld(lat_rac_r, lon_rac_r, gridfile)
    hum = RCM2EC_functions.intp_NN_racmogrid(racmo, gpt, lat_EC_r, lon_EC_r,'hum', topo)[0].T
    temp = RCM2EC_functions.intp_NN_racmogrid(racmo, gpt, lat_EC_r, lon_EC_r,'temp', topo)[0].T
    cldi = RCM2EC_functions.intp_NN_racmogrid(racmo, gpt, lat_EC_r, lon_EC_r, 'cldi', topo)[0].T
    clds = RCM2EC_functions.intp_NN_racmogrid(racmo, gpt, lat_EC_r, lon_EC_r,'clds', topo)[0].T
    cldr = RCM2EC_functions.intp_NN_racmogrid(racmo, gpt, lat_EC_r, lon_EC_r, 'cldr', topo)[0].T
    cldw = RCM2EC_functions.intp_NN_racmogrid(racmo, gpt, lat_EC_r, lon_EC_r, 'cldw', topo)[0].T
    refficlo = RCM2EC_functions.intp_NN_racmogrid(racmo, gpt, lat_EC_r, lon_EC_r, 'refficlo', topo)[0].T
    refflclo = RCM2EC_functions.intp_NN_racmogrid(racmo, gpt, lat_EC_r, lon_EC_r, 'refflclo', topo)[0].T
    uwind = RCM2EC_functions.intp_NN_racmogrid(racmo, gpt, lat_EC_r, lon_EC_r, 'uwind', topo)[0].T
    vwind = RCM2EC_functions.intp_NN_racmogrid(racmo, gpt, lat_EC_r, lon_EC_r, 'vwind', topo)[0].T
    p = RCM2EC_functions.intp_NN_racmogrid(racmo, gpt, lat_EC_r, lon_EC_r, 'p', topo)[0].T
    aer = xr.open_dataset(path_aer).sel(month=int(dtime.strftime('%m').lstrip('0')))
    gpt_r = gpt.reshape(40,-1)
    pres_r = racmo.p.values.reshape(40,-1)
    lata, lona = np.meshgrid(aer.lat.values, aer.lon.values)
    lat_lon_flat = np.column_stack((lata.ravel(),lona.ravel()))
    lat_lon_rac = np.column_stack((lat_rac, lon_rac))
    tree = KDTree(lat_lon_flat)
    nn_coords = lat_lon_flat[tree.query(lat_lon_rac)[1]]    
    lat_indices = {lat: np.where(aer.lat.values == lat)[0] for lat in np.unique(nn_coords[:, 0])}
    lon_indices = {lon: np.where(aer.lon.values == lon)[0] for lon in np.unique(nn_coords[:, 1])}
    try:
        SeaSalt1 = np.zeros((len(height[0]),len(lat_rac_r)))
        SeaSalt2 = np.zeros((len(height[0]),len(lat_rac_r)))
        SeaSalt3 = np.zeros((len(height[0]),len(lat_rac_r)))
        MineralDust1 = np.zeros((len(height[0]),len(lat_rac_r)))
        MineralDust2 = np.zeros((len(height[0]),len(lat_rac_r)))
        MineralDust3 = np.zeros((len(height[0]),len(lat_rac_r)))
        OrganicMatterHydrophilic = np.zeros((len(height[0]),len(lat_rac_r)))
        OrganicMatterHydrophobic = np.zeros((len(height[0]),len(lat_rac_r)))
        BlackCarbonHydrophilic = np.zeros((len(height[0]),len(lat_rac_r)))
        BlackCarbonHydrophobic = np.zeros((len(height[0]),len(lat_rac_r)))
        Sulfate = np.zeros((len(height[0]),len(lat_rac_r)))
        for i, (lat, lon) in enumerate(nn_coords):
            lat_idx = lat_indices[lat][0]
            lon_idx = lon_indices[lon][0]
            h_topo = height[i,:]
            pres_aer = aer.half_level_pressure.values[:,lat_idx,lon_idx] + \
                        aer.half_level_delta_pressure.values[:,lat_idx,lon_idx]/2
            pres_rac = p[:,i]
            spl = CubicSpline(np.sort(np.flip(pres_rac)), np.flip(h_topo)[np.argsort(np.flip(pres_rac))])
            nn_gpt = spl(pres_aer)
            nn_gpt[nn_gpt < 0] = 0
            dh = np.zeros(len(nn_gpt))
            dh[0:-1] = np.abs(nn_gpt[:-1] - nn_gpt[1:])
            dh[-1] = nn_gpt[-1] 
            SS1 = aer.Sea_Salt_bin1.values[:, lat_idx, lon_idx]/dh
            SS2 = aer.Sea_Salt_bin2.values[:, lat_idx, lon_idx]/dh
            SS3 = aer.Sea_Salt_bin3.values[:, lat_idx, lon_idx]/dh
            MD1 = aer.Mineral_Dust_bin1.values[:, lat_idx, lon_idx]/dh
            MD2 = aer.Mineral_Dust_bin2.values[:, lat_idx, lon_idx]/dh
            MD3 = aer.Mineral_Dust_bin3.values[:, lat_idx, lon_idx]/dh
            OMphil = aer.Organic_Matter_hydrophilic.values[:, lat_idx, lon_idx]/dh
            OMphob = aer.Organic_Matter_hydrophobic.values[:, lat_idx, lon_idx]/dh
            BCphil = aer.Black_Carbon_hydrophilic.values[:, lat_idx, lon_idx]/dh
            BCphob = aer.Black_Carbon_hydrophobic.values[:, lat_idx, lon_idx]/dh
            sulf = aer.Sulfates.values[:, lat_idx, lon_idx]/dh
            if h_topo.size > 0:
                for j in range(len(SeaSalt1)):
                    SeaSalt1[j,i] = SS1[np.abs(pres_aer - pres_rac[j]).argmin()]
                    SeaSalt2[j,i] = SS2[np.abs(pres_aer - pres_rac[j]).argmin()]
                    SeaSalt3[j,i] = SS3[np.abs(pres_aer - pres_rac[j]).argmin()]
                    MineralDust1[j,i] = MD1[np.abs(pres_aer - pres_rac[j]).argmin()]
                    MineralDust2[j,i] = MD2[np.abs(pres_aer - pres_rac[j]).argmin()]
                    MineralDust3[j,i] = MD3[np.abs(pres_aer - pres_rac[j]).argmin()]
                    OrganicMatterHydrophilic[j,i] = OMphil[np.abs(pres_aer - pres_rac[j]).argmin()]
                    OrganicMatterHydrophobic[j,i] = OMphob[np.abs(pres_aer - pres_rac[j]).argmin()]
                    BlackCarbonHydrophilic[j,i] = BCphil[np.abs(pres_aer - pres_rac[j]).argmin()]
                    BlackCarbonHydrophobic[j,i] = BCphob[np.abs(pres_aer - pres_rac[j]).argmin()]
                    Sulfate[j,i]= sulf[np.abs(pres_aer - pres_rac[j]).argmin()]
                    SeaSalt1[j,i] = SS1[np.abs(nn_gpt - h_topo[j]).argmin()]
                    SeaSalt2[j,i] = SS2[np.abs(nn_gpt - h_topo[j]).argmin()]
                    SeaSalt3[j,i] = SS3[np.abs(nn_gpt - h_topo[j]).argmin()]
                    MineralDust1[j,i] = MD1[np.abs(nn_gpt - h_topo[j]).argmin()]
                    MineralDust2[j,i] = MD2[np.abs(nn_gpt - h_topo[j]).argmin()]
                    MineralDust3[j,i] = MD3[np.abs(nn_gpt - h_topo[j]).argmin()]
                    OrganicMatterHydrophilic[j,i] = OMphil[np.abs(nn_gpt - h_topo[j]).argmin()]
                    OrganicMatterHydrophobic[j,i] = OMphob[np.abs(nn_gpt - h_topo[j]).argmin()]
                    BlackCarbonHydrophilic[j,i] = BCphil[np.abs(nn_gpt - h_topo[j]).argmin()]
                    BlackCarbonHydrophobic[j,i] = BCphob[np.abs(nn_gpt - h_topo[j]).argmin()]
                    Sulfate[j,i]= sulf[np.abs(nn_gpt - h_topo[j]).argmin()]
            SeaSalt1[np.where(h_topo < np.nanmin(nn_gpt)), i] = 0
            SeaSalt2[np.where(h_topo < np.nanmin(nn_gpt)), i] = 0
            SeaSalt3[np.where(h_topo < np.nanmin(nn_gpt)), i] = 0
            MineralDust1[np.where(h_topo < np.nanmin(nn_gpt)), i] = 0
            MineralDust2[np.where(h_topo < np.nanmin(nn_gpt)), i] = 0
            MineralDust3[np.where(h_topo < np.nanmin(nn_gpt)), i] = 0
            OrganicMatterHydrophilic[np.where(h_topo < np.nanmin(nn_gpt)), i] = 0
            OrganicMatterHydrophobic[np.where(h_topo < np.nanmin(nn_gpt)), i] = 0
            BlackCarbonHydrophilic[np.where(h_topo < np.nanmin(nn_gpt)), i] = 0
            BlackCarbonHydrophobic[np.where(h_topo < np.nanmin(nn_gpt)), i] = 0
            Sulfate[np.where(h_topo < np.nanmin(nn_gpt)), i] = 0
        fmwa = 0.01*2*(SeaSalt1+SeaSalt2+SeaSalt3) + 0.95*2*Sulfate
        fmsa = 0.97*2*(BlackCarbonHydrophilic+BlackCarbonHydrophobic) + 0.97*2*(OrganicMatterHydrophilic+OrganicMatterHydrophobic)
        cms = MineralDust1 + MineralDust2 + MineralDust3 + 2*0.09*(SeaSalt1+SeaSalt2+SeaSalt3) + 2*0.05*Sulfate + \
                2*0.03*(BlackCarbonHydrophilic+BlackCarbonHydrophobic) + 2*0.03*(OrganicMatterHydrophilic+OrganicMatterHydrophobic)
        cmns = 0.90*2*(SeaSalt1+SeaSalt2+SeaSalt3)
        fmwa[~np.isfinite(fmwa)] = 0
        fmsa[~np.isfinite(fmsa)] = 0
        cms[~np.isfinite(cms)] = 0
        cmns[~np.isfinite(cmns)] = 0
        lev = racmo.lev.values
        var_dict = dict(height=(["lev","rlat"], height.T),
                                        hum=(["lev","rlat"], hum),
                                        temp=(["lev","rlat"], temp),
                                        cldi=(["lev","rlat"], cldi),
                                        clds=(["lev","rlat"], clds),
                                        cldr=(["lev","rlat"], cldr),
                                        cldw=(["lev","rlat"], cldw),
                                        refficlo=(["lev","rlat"], refficlo),
                                        refflclo=(["lev","rlat"], refflclo),
                                        uwind=(["lev","rlat"], uwind),
                                        vwind=(["lev","rlat"], vwind),
                                        p=(["lev","rlat"], p),
                                        fmwa=(["lev","rlat"], fmwa),
                                        fmsa=(["lev","rlat"], fmsa),
                                        cms=(["lev","rlat"], cms),
                                        cmns=(["lev","rlat"], cmns),
                                        rlon=(["rlon"], lon_rac_r),
                                        lat=(["lat"], lat_rac),
                                        lon=(["lon"], lon_rac))
        ds = xr.Dataset(data_vars=var_dict, coords=dict(lev=("lev", lev),
                                                    rlat=("rlat", lat_rac_r)))
        encoding = {}
        for var, da in var_dict.items():
            size = da[1].size  
            # Compress only if the variable is big enough
            if size > 10000:
                encoding[var] = {"zlib": True, "complevel": 3}
            else:
                encoding[var] = {} 
            if path_ec[-5] == 'M':
                ds.to_netcdf(path_out+'clm_in'+date+path_ec[-5:-3]+'.nc', encoding=encoding)
            else:
                ds.to_netcdf(path_out+'clm_in'+date+path_ec[-4]+'.nc', encoding=encoding)
    except:
        ds = xr.Dataset()
        if path_ec[-5] == 'M':
            ds.to_netcdf(path_out+'clm_in'+date+path_ec[-5:-3]+'.nc')
        else:
            ds.to_netcdf(path_out+'clm_in'+date+path_ec[-4]+'.nc')
    return ds

def make_grid_file(lat_grid, lon_grid, lat_ec, lon_ec, savepath, date):
    grid_bin = np.zeros((len(lat_grid), len(lat_grid[0])))
    for lat, lon in zip(lat_ec, lon_ec):
        idx = np.where((np.round(lat_grid,3)==np.round(lat,3)) & (np.round(lon_grid,3)==np.round(lon,3)))
        grid_bin[idx] = 1
    var_dict = {}
    var_dict['lat']=(["lat_ix","lon_ix"], lat_grid)
    var_dict['lon']=(["lat_ix","lon_ix"], lon_grid)
    var_dict['grid_bin']=(["lat_ix","lon_ix"], grid_bin)
    lat_ix = np.arange(0,len(lat_grid),1)
    lon_ix = np.arange(0,len(lat_grid[0]),1)
    ds = xr.Dataset(data_vars=var_dict,
                    coords=dict(lat_ix=("lat_ix", lat_ix),
                                lon_ix=("lon_ix", lon_ix)))

    ds.to_netcdf(savepath+'grid_file_'+date+'.nc')
    return ds
    

def make_ec_traj(path_rcm_in, path_ec, dtop, gridfile, date, savepath, savepath_grid, domain='FGRN055'):
    racmo = xr.open_dataset(path_rcm_in, engine='netcdf4').squeeze()
    topo = dtop.Topography.values
    gs = dtop.Geopotential.values
    t = racmo.temp.values
    h = racmo.hum.values
    ps = racmo.ps.values
    gpt, gpt_half = RCM2EC_functions.plev2gph(t, h, ps, gs, return_half=True)
    if 'CPR' in path_ec:
        if path_ec[-5] == 'M':
            ec = xr.open_dataset(path_ec, phony_dims='sort', engine='h5netcdf')
        else:
            ec = xr.open_dataset(path_ec, group='ScienceData/Geo', phony_dims='sort', engine='h5netcdf')
    else:
        if path_ec[-5] == 'M':
            ec = xr.open_dataset(path_ec, engine='h5netcdf')
        else:
            ec = xr.open_dataset(path_ec, group='ScienceData', engine='h5netcdf')
    if domain == 'FGRN055':
        (lat_EC_r, lon_EC_r), h_EC_r, lat_EC, lon_EC = RCM2EC_functions.get_EC_traj(ec, domain, [55,90,-105,35], gridfile)
    elif domain == 'PXANT11':
        (lat_EC_r, lon_EC_r), h_EC_r, lat_EC, lon_EC = RCM2EC_functions.get_EC_traj(ec, domain, [-90,-35,-180,180], gridfile)
    lat_rac_r, lon_rac_r, height, height_half = RCM2EC_functions.intp_NN_racmogrid(racmo, gpt, lat_EC_r, lon_EC_r, 'hum', gpt_half=gpt_half)[1:]
    lat_rac, lon_rac = CFTR.RotatedGrid2RealWorld(lat_rac_r, lon_rac_r, gridfile)
    var_dict = {}
    lev41 = ['lwupa', 'lwupc', 'lwdna', 'lwdnc', 'swupa', 'swupc', 'swdna', 'swdnc',
             'lwnet', 'swnet']
    lev40 = ['lnc', 'ccn', 'p', 'swhra', 'swhrc', 'lwhra', 'lwhrc', 'cldr', 'clds',
             'temp', 'uwind', 'vwind', 'hum', 'cldw', 'cldi', 'cldf', 'refflclo',
             'refficlo', 'zwind']
    tilevars = ['tskintiled', 'tilefraction']
    lev1 =  ['ps', 'hfls', 'hfss', 'tskin', 'swnttcs', 'lwnttcs', 'lwntscs', 'swntscs',
             'rlds', 'rsds']
    for var_name, values in racmo.items():
        if var_name in lev41:
            data = RCM2EC_functions.intp_NN_racmogrid(racmo, gpt_half, lat_EC_r, lon_EC_r, var_name)[0].T
            var = xr.DataArray(data, dims=["lev_half", "rlat"], attrs=values.attrs)
            var_dict[var_name] = var
        elif var_name in lev40:
            data = RCM2EC_functions.intp_NN_racmogrid(racmo, gpt, lat_EC_r, lon_EC_r, var_name)[0].T
            var = xr.DataArray(data, dims=["lev", "rlat"], attrs=values.attrs)
            var_dict[var_name] = var
        if var_name in tilevars:
            data_list = [RCM2EC_functions.intp_NN_racmogrid_1D(racmo, lat_EC_r, lon_EC_r, var_name, idx=i)[0]
             for i in range(len(racmo.sfc.values))]
            data = np.vstack(data_list)
            var = xr.DataArray(data, dims=["tile", "rlat"], attrs=values.attrs)
            var_dict[var_name] = var
        elif var_name in lev1:
            data = RCM2EC_functions.intp_NN_racmogrid_1D(racmo, lat_EC_r, lon_EC_r, var_name)[0]
            var = xr.DataArray(data, dims=["rlat"], attrs=values.attrs)
            var_dict[var_name] = var
    var_dict['height'] = xr.DataArray(height.T, dims=["lev", "rlat"], attrs={"long_name": "Height of full levels", "units": "m"})
    var_dict['height_half'] = xr.DataArray(height_half.T, dims=["lev_half", "rlat"], attrs={"long_name": "Height of half levels", "units": "m"})
    var_dict['rlon']   = xr.DataArray(lon_rac_r, dims=["rlon"], attrs={"long_name": "Rotated longitude", "units": "degrees"})
    var_dict['lat']    = xr.DataArray(lat_rac,   dims=["lat"],  attrs={"long_name": "Latitude", "units": "degrees_north"})
    var_dict['lon']    = xr.DataArray(lon_rac,   dims=["lon"],  attrs={"long_name": "Longitude", "units": "degrees_east"})
    lev = racmo.lev_2.values
    lev_half = racmo.lev_3.values
    tile = racmo.sfc.values
    ds = xr.Dataset(data_vars=var_dict,
                    coords=dict(lev=("lev", lev),
                                lev_half=("lev_half", lev_half),
                                tile=("tile", tile),
                                rlat=("rlat", lat_rac_r)))
    encoding = {}
    for var, da in var_dict.items():
        size = da.size  
        # Compress only if the variable is big enough
        if size > 10000:
            encoding[var] = {"zlib": True, "complevel": 3}
        else:
            encoding[var] = {}  
    ds.to_netcdf(savepath+'ec_traj_'+date+'.nc', encoding=encoding)
    return

def make_excel_file(excel_path, excel_in, date, data_path, domain, expname):
    f = pd.read_excel(excel_path+excel_in)
    f.iloc[2,4] = data_path
    f.iloc[4,4] = 'clm_in' + date + '.nc'
    f.to_excel(excel_path + domain + '/' + expname + '/Input_spread_sheet_RACMO_data_' + date + '.xlsx', index=False)
    return

def post_process_ec(domain, expname, month):
    savepath = f'/ec/res4/scratch/{user}/experiment/{domain}/{expname}/CLM_in_{month}/'
    savepath_grid = f'/ec/res4/scratch/{user}/experiment/{domain}/{expname}/EC_grid_{month}/'
    excel_path = '/ec/res4/scratch/{user}/CLM/settings/'
    dtop = xr.open_dataset(f'/perm/{user}/masks_grids/{domain}_masks.nc', engine='netcdf4')
    gridfile = RCMG.read_Setting_data(domain=domain,
                                      path='/perm/{user}/masks_grids/')
    path_aer = '/perm/{user}/RACMO24_data/ifsdata/aerosol_cams_3d_climatology_47r1.nc'
    path_RCM = f'/ec/res4/scratch/{user}/experiment/{domain}/{expname}/EC_TF_{month}/'
    savepath_traj = f'/ec/res4/scratch/{user}/experiment/{domain}/{expname}/EC_traj_{month}/'
    RCM_files = sorted((f for f in os.listdir(path_RCM) if not f.startswith(".") and month in f))
    path_EC = f'/ec/res4/scratch/{user}/{domain}-{expname}/EarthCARE_{month}/'
    EC_files = sorted((f for f in os.listdir(path_EC) if not f.startswith(".") and month in f[:36]))
    RCM_dates = []
    EC_dates = []
    
    for RCM in RCM_files:
        RCM_dates.append(datetime.strptime(RCM[2:14], '%Y%m%d%H%M'))
    for EC in EC_files:
        EC_dates.append(datetime.strptime(EC[20:35], '%Y%m%dT%H%M%S'))
        
    def process_file(s_EC_tuple):
        s, EC = s_EC_tuple
        RCM_match = RCM_files[np.argmin(np.abs(np.array(RCM_dates) - EC_dates[s]))]
        RCM_date = RCM_dates[np.argmin(np.abs(np.array(RCM_dates) - EC_dates[s]))]
        if np.abs(np.datetime64(EC_dates[s]) - np.datetime64(RCM_date)) < np.timedelta64(30, 'm'):
            path_rcm_in = path_RCM + RCM_match
            path_ec = path_EC + EC
            print(f"using {RCM_match} and {EC}", flush=True)
            
            try:
                if path_ec[-5] == 'M':
                    make_ec_traj(path_rcm_in, path_ec, dtop, gridfile, RCM_match[2:14]+path_ec[-5:-3],
                            savepath_traj, savepath_grid, domain)
                else: 
                    make_ec_traj(path_rcm_in, path_ec, dtop, gridfile, RCM_match[2:14]+path_ec[-4],
                            savepath_traj, savepath_grid, domain)
            except Exception as e:
                print(f"Failed making ec traj for {EC}: {e}", flush=True)
            try:
                make_clm_input(path_rcm_in, path_ec, savepath, path_aer, dtop, gridfile, domain)
            except Exception as e:
                print(f"Failed making clm input for {EC}: {e}", flush=True)
            try:
                if path_ec[-5] == 'M':
                    make_excel_file(excel_path, 'Input_spread_sheet_RACMO_data_example.xlsx',
                                RCM_match[2:14]+path_ec[-5:-3], savepath, domain, expname)
                else:
                    make_excel_file(excel_path, 'Input_spread_sheet_RACMO_data_example.xlsx',
                                RCM_match[2:14]+path_ec[-4], savepath, domain, expname)
            except Exception as e:
                print(f"Failed making excel for {EC}: {e}", flush=True)
    tasks = list(enumerate(EC_files))
    print(f"Found {len(tasks)} EC tasks")
    with ThreadPoolExecutor(max_workers=3) as executor:
        executor.map(process_file, tasks)
    return

def main():
    try:
        domain = sys.argv[1] 
        expname = sys.argv[2] 
        month = sys.argv[3]
    except IndexError:
        print('')
        print('Error: missing commandline inputs.')
        print('')
        print('Usage:')
        print(f'{sys.argv[0]} '+
              '<domain> <expname> <month>')
        print('')
        sys.exit(1)


    post_process_ec(domain, expname, str(month))

if __name__ == "__main__":
    main()

