#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
"""
Created on Tue Nov 19 13:33:58 2024

@author: Feens012
"""
import numpy as np
from tqdm import tqdm
import sys
import os
import re
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial import cKDTree
from datetime import datetime
import importlib
sys.path.append('../utils/')
import plot_functions 
import colormaps_ARS
import helpers
import RCM2EC_functions
import spatial_regridding
import scale_fluxes
importlib.reload(plot_functions)
importlib.reload(colormaps_ARS)
importlib.reload(helpers)
importlib.reload(RCM2EC_functions)
importlib.reload(spatial_regridding)
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({'font.size': 12, 
                     "legend.fontsize": 14, "axes.labelsize": 14,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, 
                     "hatch.color":'darkgrey', 
                    'mathtext.default': 'regular'})
# %% 
def extract_first_date(filename):
    '''extract first date from list of EarthCARE filenames
    (since EC filenames are observed time - process time)'''
    date_pattern = re.compile(r"(\d{8}T\d{6}Z)")
    match = date_pattern.search(filename)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ")
    return datetime.min
months = ['202409', '202410', '202411', '202412', '202501', '202502', '202503',
           '202504', '202505', '202506', '202507', '202508', '202509', '202510',
           '202511', '202512', '202601', '202602', '202603', '202604']
domain = 'FGRN055'
sim = 'bl'
d_topo = xr.open_dataset(f'{domain}_masks.nc', engine='netcdf4')
topo = d_topo.Topography.values
infiles_ec_list = []
infiles_quality_list = []
infiles_racmo = []
for month in months:
    path_ec = f'{domain}/L2b_ds_{month}/'
    path_q = f'{domain}/L2b_quality_{month}/'
    path_racmo = f'{domain}/RACMO2.4p1_v4_{sim}/EC_traj_{month}/'
    racmofiles = sorted((f for f in os.listdir(path_racmo) if month in f))
    infiles_ACM = sorted((f for f in os.listdir(path_ec) if month in f[:30]),
                            key=extract_first_date)
    infiles_q = sorted((f for f in os.listdir(path_q) if month in f[:30]),
                        key=extract_first_date)
    orb_ACM = []
    orb_q = []
    ACM_dates = []
    for ACM in infiles_ACM:
        orb_ACM.append(ACM[-11:-6])
    for Q in infiles_q:
        orb_q.append(Q[-11:-6])
    pop_idx_ec = []
    pop_idx_q = []
    for i in range(len(infiles_ACM)):
        if orb_ACM[i] not in orb_q:
            pop_idx_ec.append(i)
    for idx in pop_idx_ec[::-1]:
        infiles_ACM.pop(idx)
    for i in range(len(infiles_q)):
        if orb_q[i] not in orb_ACM:
            pop_idx_q.append(i)
    for idx in pop_idx_q[::-1]:
        infiles_q.pop(idx)
    for ACM in infiles_ACM:
        ACM_dates.append(datetime.strptime(ACM[24:39], '%Y%m%dT%H%M%S'))
    remove_idx = []
    ACM_idx = []
    for i,rac in enumerate(racmofiles):
        racmo_date = datetime.strptime(rac[8:-5], '%Y%m%d%H%M')
        if np.min(np.abs(np.array(ACM_dates) - racmo_date)) < np.timedelta64(30, 'm'):
            ACM_idx.append(np.argmin(np.abs(np.array(ACM_dates) - racmo_date)))
        else:
            remove_idx.append(i)
    for j in range(len(remove_idx)):
        racmofiles.pop(np.flip(remove_idx)[j])
    for k in range(len(racmofiles)):
        infiles_racmo.append(path_racmo+racmofiles[k])
    for idx in ACM_idx:
        infiles_ec_list.append(path_ec+infiles_ACM[idx])
        infiles_quality_list.append(path_q+infiles_q[idx])
#%%
bbr_lw = []
rac_lw = []
ec_cld = [] # 0 = cs, 1 = only ice, 2 = only liquid, 3 = mixed
rac_cld = []
ec_cld_h = [] # 0 = cs, 1 = only low, 2 = mid, 3 = high
rac_cld_h = []
rlat = d_topo.rlat.values
rlon = d_topo.rlon.values
rlat_grid, rlon_grid = np.meshgrid(rlat, rlon, indexing='ij')
latlon_pairs = np.column_stack((rlat_grid.ravel(), rlon_grid.ravel()))
latlon_tree = cKDTree(latlon_pairs)
is_only = False
if is_only:
    is_vals = d_topo.Icesheet_Only.values 
for ec, q, rac in tqdm(zip(infiles_ec_list, infiles_quality_list, infiles_racmo),
                    total=len(infiles_ec_list)):
    try:
        if rac[-11:-7] == '0100':
            continue
        d_ec = xr.open_dataset(ec)
        d_q = xr.open_dataset(q)
        d_r = xr.open_dataset(rac)
        lw_status = d_q.longwave_status.values
        com_status = d_q.quality_status_com.values
        clutter = d_q.clutter_mask.values
        clutter = clutter.astype(float)
        clutter[clutter==1] = np.nan
        clutter[clutter==0] = 1
        iwc_dem = d_q.iwc_dem_mask.values
        iwc_dem[iwc_dem==1] = np.nan
        iwc_dem[iwc_dem==0] = 1
        lwc_dem = d_q.lwc_dem_mask.values
        lwc_dem[lwc_dem==1] = np.nan
        lwc_dem[lwc_dem==0] = 1   
        if is_only:
            ilat = np.abs(rlat[:, None] - d_r.rlat.values[None, :]).argmin(axis=0)
            ilon = np.abs(rlon[:, None] - d_r.rlon.values[None, :]).argmin(axis=0)
            icesheetmask = is_vals[ilat,ilon]
            icesheetmask[icesheetmask>=0.5] = 1
            icesheetmask[icesheetmask<0.5] = np.nan
        bbr = d_ec.BBR_effective_flux_thermal.values
        racmo_lw = d_r.lwupa.values[0]
        lw_status[racmo_lw == 0] = 1
        ec_iwp = d_ec.ice_water_path_com.values
        ec_lwp = d_ec.liquid_water_path.values
        rac_iwc = helpers.kgkg_to_kgm3(d_r.cldi.values + d_r.clds.values,
                                    d_r.temp.values, 
                                    d_r.hum.values,
                                    d_r.p.values)*clutter*iwc_dem
        dh = np.abs(d_r.height.values[:-1] - d_r.height.values[1:])
        iwc_layer = 0.5 * (rac_iwc[:-1] + rac_iwc[1:])
        rac_iwp = np.nansum(iwc_layer*dh, axis=0)
        rac_lwc = helpers.kgkg_to_kgm3(d_r.cldw.values,
                                    d_r.temp.values, 
                                    d_r.hum.values,
                                    d_r.p.values)*clutter*lwc_dem
        lwc_layer = 0.5 * (rac_lwc[:-1] + rac_lwc[1:])
        rac_lwp = np.nansum(lwc_layer*dh, axis=0)
        ec_wp = ec_iwp.copy() + ec_lwp.copy()
        rac_wp = rac_iwp.copy() + rac_lwp.copy()
        wp_b = rac_wp - ec_wp
        ec_iwp[ec_iwp >= 1e-3] = 1
        ec_iwp[ec_iwp < 1e-3] = 0
        if is_only:
            ec_iwp = ec_iwp[(lw_status == 0) & (com_status == 0) & (icesheetmask == 1)]
        else:
            ec_iwp = ec_iwp[(lw_status == 0) & (com_status == 0)]
        ec_lwp[ec_lwp >= 1e-3] = 1
        ec_lwp[ec_lwp < 1e-3] = 0
        if is_only:
            ec_lwp = ec_lwp[(lw_status == 0) & (com_status == 0) & (icesheetmask == 1)]
        else:
            ec_lwp = ec_lwp[(lw_status == 0) & (com_status == 0)]
        rac_iwp[rac_iwp >= 1e-3] = 1
        rac_iwp[rac_iwp < 1e-3] = 0
        if is_only:
            rac_iwp = rac_iwp[(lw_status == 0) & (com_status == 0) & (icesheetmask == 1)]
        else:
            rac_iwp = rac_iwp[(lw_status == 0) & (com_status == 0)]
        rac_lwp[rac_lwp >= 1e-3] = 1
        rac_lwp[rac_lwp < 1e-3] = 0
        if is_only:
            rac_lwp = rac_lwp[(lw_status == 0) & (com_status == 0) & (icesheetmask == 1)]
        else:
            rac_lwp = rac_lwp[(lw_status == 0) & (com_status == 0)]
        ec_cld_st = np.zeros(len(ec_iwp))
        ec_cld_st[(ec_iwp == 1) & (ec_lwp == 0)] = 1
        ec_cld_st[(ec_lwp == 1) & (ec_iwp == 0)] = 2
        ec_cld_st[(ec_iwp == 1) & (ec_lwp == 1)] = 3
        rac_cld_st = np.zeros(len(rac_iwp))
        rac_cld_st[(rac_iwp == 1) & (rac_lwp == 0)] = 1
        rac_cld_st[(rac_lwp == 1) & (rac_iwp == 0)] = 2
        rac_cld_st[(rac_iwp == 1) & (rac_lwp == 1)] = 3
        if is_only:
            height = d_r.height.values[:,(lw_status == 0) & (com_status == 0) & (icesheetmask == 1)]
        else:
            height = d_r.height.values[:,(lw_status == 0) & (com_status == 0)]
        height[~np.isfinite(height)] = 0
        ec_iwc = d_ec.ice_water_content_com.values
        if is_only:
            ec_iwc = ec_iwc[:,(lw_status == 0) & (com_status == 0) & (icesheetmask == 1)]
        else:
            ec_iwc = ec_iwc[:,(lw_status == 0) & (com_status == 0)]
        ec_iwc[~np.isfinite(ec_iwc)] = 0
        rac_cloud_top = np.zeros(len(height[0]))
        ec_cloud_top = np.zeros(len(height[0]))
        points = np.column_stack((d_r.rlat.values, d_r.rlon.values))
        _, idx = latlon_tree.query(points)
        ilat, ilon = np.unravel_index(idx, (len(rlat), len(rlon)))
        if is_only:
            topo_height = topo[ilat,ilon][(lw_status == 0) & (com_status == 0) & (icesheetmask == 1)]
        else:
            topo_height = topo[ilat,ilon][(lw_status == 0) & (com_status == 0)]
        if len(rac_iwc[0]) > 0:
            for k in range(len(height[0])):
                heights = height[:,k]
                if len(heights.copy()[rac_iwc[:,k] >= 1e-6]) > 0:
                    heights_rac = heights.copy()[rac_iwc[:,k] >= 1e-6]
                    rac_cloud_top[k] = np.nanmax(heights_rac)
                if len(heights.copy()[ec_iwc[:,k] >= 1e-6]) > 0: 
                    heights_ec = heights.copy()[ec_iwc[:,k] >= 1e-6]
                    ec_cloud_top[k] = np.nanmax(heights_ec)
        rac_cloud_top = rac_cloud_top - topo_height
        ec_cloud_top = ec_cloud_top - topo_height
        rac_cloud_top[rac_cloud_top < 0] = 0
        ec_cloud_top[ec_cloud_top < 0] = 0
        ec_cld_height = np.zeros(len(ec_cloud_top))
        ec_cld_height[(ec_cloud_top > 0) & (ec_cloud_top <= 4000)] = 1
        ec_cld_height[(ec_cloud_top > 4000) & (ec_cloud_top <= 8000)] = 2
        ec_cld_height[(ec_cloud_top > 8000)] = 3
        rac_cld_height = np.zeros(len(rac_cloud_top))
        rac_cld_height[(rac_cloud_top > 0) & (rac_cloud_top <= 4000)] = 1
        rac_cld_height[(rac_cloud_top > 4000) & (rac_cloud_top <= 8000)] = 2
        rac_cld_height[(rac_cloud_top > 8000)] = 3
        if is_only:
            bbr_lw.append(bbr[(lw_status == 0) & (com_status == 0) & (icesheetmask == 1)])
            rac_lw.append(racmo_lw[(lw_status == 0) & (com_status == 0) & (icesheetmask == 1)])
        else:
            bbr_lw.append(bbr[(lw_status == 0) & (com_status == 0)])
            rac_lw.append(racmo_lw[(lw_status == 0) & (com_status == 0)])
        ec_cld_h.append(ec_cld_height)
        rac_cld_h.append(rac_cld_height)
        ec_cld.append(ec_cld_st)
        rac_cld.append(rac_cld_st)
    except:
        continue
bbr_lw_arr = np.array([xx for x in bbr_lw for xx in x])
rac_lw_arr = np.array([xx for x in rac_lw for xx in x])
ec_cld_arr = np.array([xx for x in ec_cld for xx in x])
rac_cld_arr = np.array([xx for x in rac_cld for xx in x])
ec_cld_h_arr = np.array([xx for x in ec_cld_h for xx in x])
rac_cld_h_arr = np.array([xx for x in rac_cld_h for xx in x])

ds = xr.Dataset(
    data_vars={
        "bbr_lw": (["points"], bbr_lw_arr),
        "rac_lw": (["points"], rac_lw_arr),
        "ec_cld": (["points"], ec_cld_arr),
        "rac_cld": (["points"], rac_cld_arr),
        "ec_cld_h": (["points"], ec_cld_h_arr),
        "rac_cld_h": (["points"], rac_cld_h_arr),
    },
    attrs={
        "description": "Longwave fluxes bbr and racmo, with cloud properties",
        "lw_units": "W m-2"}
)
ds["bbr_lw"] = ds["bbr_lw"].assign_attrs(long_name = "TOA upwelling longwave flux from BBR")
ds["rac_lw"] = ds["rac_lw"].assign_attrs(long_name = "TOA upwelling longwave flux from RACMO")
ds["ec_cld"] = ds["ec_cld"].assign_attrs(long_name = "0: clear sky, 1: only ice, 2: only liquid, 3: mixed")
ds["rac_cld"] = ds["rac_cld"].assign_attrs(long_name = "0: clear sky, 1: only ice, 2: only liquid, 3: mixed")
ds["ec_cld_h"] = ds["ec_cld_h"].assign_attrs(long_name = "0: clear sky, 1: low, 2: mid, 3: high")
ds["rac_cld_h"] = ds["rac_cld_h"].assign_attrs(long_name = "0: clear sky, 1: low, 2: mid, 3: high")
if is_only:
    ds.to_netcdf(f"LW_{domain}_icesheet.nc") 
else:
    ds.to_netcdf(f"LW_{domain}.nc")
# %%
bbr_sw = []
rac_sw = []
ec_cld = [] # 0 = cs, 1 = only ice, 2 = only liquid, 3 = mixed
rac_cld = []
ec_cld_h = [] # 0 = cs, 1 = only low, 2 = mid, 3 = high
rac_cld_h = []
racmo_swdn_hr = xr.open_mfdataset([f'{domain}/RACMO2.4p1_v4_{sim}-2024/NC_1H/rsdt.KNMI-2024.{domain}.RACMO2.4p1_v4_{sim}.1H.nc',
                                   f'{domain}/RACMO2.4p1_v4_{sim}-2025/NC_1H/rsdt.KNMI-2025.{domain}.RACMO2.4p1_v4_{sim}.1H.nc',
                                   f'{domain}/RACMO2.4p1_v4_{sim}-2026/NC_1H/rsdt.KNMI-2026.{domain}.RACMO2.4p1_v4_{sim}.1H.nc'],
                                   combine='nested', concat_dim='time').squeeze()
racmo_swup_hr = xr.open_mfdataset([f'{domain}/RACMO2.4p1_v4_{sim}-2024/NC_1H/rsut.KNMI-2024.{domain}.RACMO2.4p1_v4_{sim}.1H.nc',
                                   f'{domain}/RACMO2.4p1_v4_{sim}-2025/NC_1H/rsut.KNMI-2025.{domain}.RACMO2.4p1_v4_{sim}.1H.nc',
                                   f'{domain}/RACMO2.4p1_v4_{sim}-2026/NC_1H/rsut.KNMI-2026.{domain}.RACMO2.4p1_v4_{sim}.1H.nc'],
                                   combine='nested', concat_dim='time').squeeze()
rlat = racmo_swdn_hr.rlat.values
rlon = racmo_swdn_hr.rlon.values
rlat_grid, rlon_grid = np.meshgrid(rlat, rlon, indexing='ij')
latlon_pairs = np.column_stack((rlat_grid.ravel(), rlon_grid.ravel()))
latlon_tree = cKDTree(latlon_pairs)
is_only = False
if is_only:
    is_vals = d_topo.Icesheet_Only.values 
for ec, q, rac in tqdm(zip(infiles_ec_list, infiles_quality_list, infiles_racmo),
                    total=len(infiles_ec_list)):
    try:
        if rac[-11:-7] == '0100':
            continue
        d_ec = xr.open_dataset(ec)
        d_q = xr.open_dataset(q)
        d_r = xr.open_dataset(rac)
        sw_status = d_q.shortwave_status.values
        com_status = d_q.quality_status_com.values
        clutter = d_q.clutter_mask.values
        clutter = clutter.astype(float)
        clutter[clutter==1] = np.nan
        clutter[clutter==0] = 1
        iwc_dem = d_q.iwc_dem_mask.values
        iwc_dem[iwc_dem==1] = np.nan
        iwc_dem[iwc_dem==0] = 1
        lwc_dem = d_q.lwc_dem_mask.values
        lwc_dem[lwc_dem==1] = np.nan
        lwc_dem[lwc_dem==0] = 1
        if is_only:
            ilat = np.abs(rlat[:, None] - d_r.rlat.values[None, :]).argmin(axis=0)
            ilon = np.abs(rlon[:, None] - d_r.rlon.values[None, :]).argmin(axis=0)
            icesheetmask = is_vals[ilat,ilon]
            icesheetmask[icesheetmask>=0.5] = 1
            icesheetmask[icesheetmask<0.5] = np.nan
        bbr = d_ec.BBR_effective_flux_solar.values
        rac_swnet_toa = d_r.swnet.values[0]
        racmo_date = datetime.strptime(rac[-17:-5], '%Y%m%d%H%M')
        racmo_date = racmo_date.replace(minute=0, second=0, microsecond=0)
        racmo_swup = racmo_swup_hr.sel(time=racmo_date).squeeze()
        racmo_swdn = racmo_swdn_hr.sel(time=racmo_date).squeeze()
        points = np.column_stack((d_r.rlat.values, d_r.rlon.values))
        _, idx = latlon_tree.query(points)
        ilat, ilon = np.unravel_index(idx, (len(rlat), len(rlon)))
        fulloutput_swuptoa_traj = racmo_swup.rsut.values[ilat, ilon]
        fulloutput_swdntoa_traj = racmo_swdn.rsdt.values[ilat, ilon]
        racmo_sw = scale_fluxes.scale_shortwave_fluxes(fulloutput_swdntoa_traj,
                                                        fulloutput_swuptoa_traj,
                                                        rac_swnet_toa)[1]
        sw_status[racmo_sw == 0] = 1
        ec_iwp = d_ec.ice_water_path_com.values
        ec_lwp = d_ec.liquid_water_path.values
        rac_iwc = helpers.kgkg_to_kgm3(d_r.cldi.values + d_r.clds.values,
                                    d_r.temp.values, 
                                    d_r.hum.values,
                                    d_r.p.values)*clutter*iwc_dem
        dh = np.abs(d_r.height.values[:-1] - d_r.height.values[1:])
        iwc_layer = 0.5 * (rac_iwc[:-1] + rac_iwc[1:])
        rac_iwp = np.nansum(iwc_layer*dh, axis=0)
        rac_lwc = helpers.kgkg_to_kgm3(d_r.cldw.values,
                                    d_r.temp.values, 
                                    d_r.hum.values,
                                    d_r.p.values)*clutter*lwc_dem
        lwc_layer = 0.5 * (rac_lwc[:-1] + rac_lwc[1:])
        rac_lwp = np.nansum(lwc_layer*dh, axis=0)
        ec_wp = ec_iwp.copy() + ec_lwp.copy()
        rac_wp = rac_iwp.copy() + rac_lwp.copy()
        wp_b = rac_wp - ec_wp
        ec_iwp[ec_iwp >= 1e-3] = 1
        ec_iwp[ec_iwp < 1e-3] = 0
        if is_only:
            ec_iwp = ec_iwp[(sw_status == 0) & (com_status == 0) & (icesheetmask == 1)]
        else:
            ec_iwp = ec_iwp[(sw_status == 0) & (com_status == 0)]
        ec_lwp[ec_lwp >= 1e-3] = 1
        ec_lwp[ec_lwp < 1e-3] = 0
        if is_only:
            ec_lwp = ec_lwp[(sw_status == 0) & (com_status == 0) & (icesheetmask == 1)]
        else:
            ec_lwp = ec_lwp[(sw_status == 0) & (com_status == 0)]
        rac_iwp[rac_iwp >= 1e-3] = 1
        rac_iwp[rac_iwp < 1e-3] = 0
        if is_only:
            rac_iwp = rac_iwp[(sw_status == 0) & (com_status == 0) & (icesheetmask == 1)]
        else:
            rac_iwp = rac_iwp[(sw_status == 0) & (com_status == 0)]
        rac_lwp[rac_lwp >= 1e-3] = 1
        rac_lwp[rac_lwp < 1e-3] = 0
        if is_only:
            rac_lwp = rac_lwp[(sw_status == 0) & (com_status == 0) & (icesheetmask == 1)]
        else:
            rac_lwp = rac_lwp[(sw_status == 0) & (com_status == 0)]
        ec_cld_st = np.zeros(len(ec_iwp))
        ec_cld_st[(ec_iwp == 1) & (ec_lwp == 0)] = 1
        ec_cld_st[(ec_lwp == 1) & (ec_iwp == 0)] = 2
        ec_cld_st[(ec_iwp == 1) & (ec_lwp == 1)] = 3
        rac_cld_st = np.zeros(len(rac_iwp))
        rac_cld_st[(rac_iwp == 1) & (rac_lwp == 0)] = 1
        rac_cld_st[(rac_lwp == 1) & (rac_iwp == 0)] = 2
        rac_cld_st[(rac_iwp == 1) & (rac_lwp == 1)] = 3
        if is_only:
            height = d_r.height.values[:,(sw_status == 0) & (com_status == 0) & (icesheetmask == 1)]
        else:
            height = d_r.height.values[:,(sw_status == 0) & (com_status == 0)]
        height[~np.isfinite(height)] = 0
        ec_iwc = d_ec.ice_water_content_com.values
        if is_only:
            ec_iwc = ec_iwc[:,(sw_status == 0) & (com_status == 0) & (icesheetmask == 1)]
        else:
            ec_iwc = ec_iwc[:,(sw_status == 0) & (com_status == 0)]
        ec_iwc[~np.isfinite(ec_iwc)] = 0
        rac_cloud_top = np.zeros(len(height[0]))
        ec_cloud_top = np.zeros(len(height[0]))
        points = np.column_stack((d_r.rlat.values, d_r.rlon.values))
        _, idx = latlon_tree.query(points)
        ilat, ilon = np.unravel_index(idx, (len(rlat), len(rlon)))
        if is_only:
            topo_height = topo[ilat,ilon][(sw_status == 0) & (com_status == 0) & (icesheetmask == 1)]
        else:
            topo_height = topo[ilat,ilon][(sw_status == 0) & (com_status == 0)]
        if len(rac_iwc[0]) > 0:
            for k in range(len(height[0])):
                heights = height[:,k]
                if len(heights.copy()[rac_iwc[:,k] >= 1e-6]) > 0:
                    heights_rac = heights.copy()[rac_iwc[:,k] >= 1e-6]
                    rac_cloud_top[k] = np.nanmax(heights_rac)
                if len(heights.copy()[ec_iwc[:,k] >= 1e-6]) > 0: 
                    heights_ec = heights.copy()[ec_iwc[:,k] >= 1e-6]
                    ec_cloud_top[k] = np.nanmax(heights_ec)
        rac_cloud_top -= topo_height
        ec_cloud_top -= topo_height
        rac_cloud_top[rac_cloud_top < 0] = 0
        ec_cloud_top[ec_cloud_top < 0] = 0
        ec_cld_height = np.zeros(len(ec_cloud_top))
        ec_cld_height[(ec_cloud_top > 0) & (ec_cloud_top <= 4000)] = 1
        ec_cld_height[(ec_cloud_top > 4000) & (ec_cloud_top <= 8000)] = 2
        ec_cld_height[(ec_cloud_top > 8000)] = 3
        rac_cld_height = np.zeros(len(rac_cloud_top))
        rac_cld_height[(rac_cloud_top > 0) & (rac_cloud_top <= 4000)] = 1
        rac_cld_height[(rac_cloud_top > 4000) & (rac_cloud_top <= 8000)] = 2
        rac_cld_height[(rac_cloud_top > 8000)] = 3

        if is_only:
            bbr_sw.append(bbr[(sw_status == 0) & (com_status == 0) & (icesheetmask == 1)])
            rac_sw.append(racmo_sw[(sw_status == 0) & (com_status == 0) & (icesheetmask == 1)])
        else:
            bbr_sw.append(bbr[(sw_status == 0) & (com_status == 0)])
            rac_sw.append(racmo_sw[(sw_status == 0) & (com_status == 0)])
        ec_cld_h.append(ec_cld_height)
        rac_cld_h.append(rac_cld_height)
        ec_cld.append(ec_cld_st)
        rac_cld.append(rac_cld_st)
    except:
        continue
bbr_sw_arr = np.array([xx for x in bbr_sw for xx in x])
rac_sw_arr = np.array([xx for x in rac_sw for xx in x])
ec_cld_arr = np.array([xx for x in ec_cld for xx in x])
rac_cld_arr = np.array([xx for x in rac_cld for xx in x])
ec_cld_h_arr = np.array([xx for x in ec_cld_h for xx in x])
rac_cld_h_arr = np.array([xx for x in rac_cld_h for xx in x])
ds = xr.Dataset(
    data_vars={
        "bbr_sw": (["points"], bbr_sw_arr),
        "rac_sw": (["points"], rac_sw_arr),
        "ec_cld": (["points"], ec_cld_arr),
        "rac_cld": (["points"], rac_cld_arr),
        "ec_cld_h": (["points"], ec_cld_h_arr),
        "rac_cld_h": (["points"], rac_cld_h_arr),
    },
    attrs={
        "description": "Shortwave fluxes bbr and racmo, with cloud properties",
        "sw_units": "W m-2"}
)
ds["bbr_sw"] = ds["bbr_sw"].assign_attrs(long_name = "TOA upwelling shortwave flux from BBR")
ds["rac_sw"] = ds["rac_sw"].assign_attrs(long_name = "TOA upwelling shortwave flux from RACMO")
ds["ec_cld"] = ds["ec_cld"].assign_attrs(long_name = "0: clear sky, 1: only ice, 2: only liquid, 3: mixed")
ds["rac_cld"] = ds["rac_cld"].assign_attrs(long_name = "0: clear sky, 1: only ice, 2: only liquid, 3: mixed")
ds["ec_cld_h"] = ds["ec_cld_h"].assign_attrs(long_name = "0: clear sky, 1: low, 2: mid, 3: high")
ds["rac_cld_h"] = ds["rac_cld_h"].assign_attrs(long_name = "0: clear sky, 1: low, 2: mid, 3: high")
if is_only:
    ds.to_netcdf(f"SW_{domain}_icesheet.nc")
else:
    ds.to_netcdf(f"SW_{domain}.nc")
#%%
domains = ['FGRN055', 'PXANT11']
names = ['Greenland', 'Antarctica']
is_only = False
labels = ["Both clear-sky",
        "Both ice",
        "Both liquid",
        "Both mixed",
        "EarthCARE clear-sky,\nRACMO cloudy",
        "EarthCARE cloudy,\nRACMO clear-sky",
        "EarthCARE ice,\nRACMO liq./mixed",
        "EarthCARE liq./mixed,\nRACMO ice",
        "EarthCARE high,\nRACMO low",
        "EarthCARE high,\nRACMO mid"]
titles = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)',
          '(g)', '(h)', '(i)', '(j)', '(k)', '(l)',
          '(m)', '(n)', '(o)', '(p)', '(q)', '(r)',
          '(s)', '(t)']
fig, axes = plt.subplots(4, 5, figsize=(24, 16), sharex=True, sharey=True)
axes = axes.flatten()
xlim_low = 0
xlim_high = 350
ylim_low = 0
ylim_high = 350

for j,domain in enumerate(domains):
    if is_only:
        lw_ds = xr.open_dataset(f"LW_{domain}_icesheet.nc")
    else:
        lw_ds = xr.open_dataset(f"LW_{domain}.nc")
    bbr_lw_arr = lw_ds.bbr_lw.values
    rac_lw_arr = lw_ds.rac_lw.values
    ec_cld_arr = lw_ds.ec_cld.values
    rac_cld_h_arr = lw_ds.rac_cld.values
    ec_cld_h_arr = lw_ds.ec_cld_h.values
    rac_cld_arr = lw_ds.rac_cld_h.values
    wp_bias_arr = lw_ds.wp_bias.values
    phase = np.zeros(len(bbr_lw_arr))
    phase[(ec_cld_arr == 1) & (rac_cld_arr == 1)] = 1 # both ice
    phase[(ec_cld_arr == 2) & (rac_cld_arr == 2)] = 2 # both liq
    phase[(ec_cld_arr == 3) & (rac_cld_arr == 3)] = 3 # both mixed
    phase[(ec_cld_arr == 0) & (rac_cld_arr > 0)] = 4 # ec cs, rac cld
    phase[(ec_cld_arr > 0) & (rac_cld_arr == 0)] = 5 # ec cld, rac cs
    phase[(ec_cld_arr == 1) & (rac_cld_arr > 1)] = 6 # ec ice, rac liq/mix
    phase[(ec_cld_arr > 1) & (rac_cld_arr == 1)] = 7 # ec liq/mix, rac ice
    height = np.zeros(len(bbr_lw_arr))
    height[(ec_cld_h_arr == 3) & (rac_cld_h_arr == 1)] = 8 # ec high, rac low
    height[(ec_cld_h_arr == 3) & (rac_cld_h_arr == 2)] = 9 # ec high, rac mid
    if domain == 'FGRN055':
        norm = mcolors.LogNorm(1,1e3)
    elif domain == 'PXANT11':
        norm = mcolors.LogNorm(1,3e3)
    for i, ax in enumerate(axes[j*10:j*10+10]):
        if i < 8:
            mask = (phase == i)
        elif i >= 8:
            mask = (height == i)
        xedges, yedges = np.linspace(xlim_low, xlim_high, 70), np.linspace(ylim_low, ylim_high, 70)
        xplot_fit = np.linspace(xlim_low,xlim_high,11) 
        hist, xedges, yedges = np.histogram2d(bbr_lw_arr[mask], rac_lw_arr[mask], (xedges, yedges))
        xidx = np.clip(np.digitize(bbr_lw_arr[mask], xedges), 0, hist.shape[0]-1)
        yidx = np.clip(np.digitize(rac_lw_arr[mask], yedges), 0, hist.shape[1]-1)
        c = hist[xidx-1, yidx-1]
        plot = ax.scatter(bbr_lw_arr[mask], rac_lw_arr[mask], c=c, s=2, 
                    cmap=plt.cm.RdYlBu_r,norm=norm)
        ax.plot([xlim_low, xlim_high],
            [xlim_low, xlim_high],
            color='black',
            ls='--')
        ax.set_title(f"{labels[i]}")
        ax.set_title(titles[i+j*10], loc='left')
        ax.set_xlim(xlim_low, xlim_high)
        ax.set_ylim(ylim_low, ylim_high)
        ax.set_aspect('equal')
        if i+j*10 >= 15:
            ax.set_xlabel("EarthCARE BBR\nLWup TOA [W m$^{-2}$]")
        else:
            ax.tick_params(bottom=False)
        if (i == 0) | (i==5):
            ax.set_ylabel("RACMO LWup\nTOA [W m$^{-2}$]")
        else:
            ax.tick_params(left=False)
        ax.set_axisbelow(True)
        ax.grid('on')
        bias = np.mean(rac_lw_arr[mask] - bbr_lw_arr[mask])
        rmse = np.sqrt(np.mean((rac_lw_arr[mask] - bbr_lw_arr[mask])**2))
        r2 = np.corrcoef(rac_lw_arr[mask], bbr_lw_arr[mask])[1,0]**2
        ax.text(0.05, 0.98,
            f"N={mask.sum()}, Bias={bias:.1f}\nRMSE={rmse:.1f}, R$^2$={r2:.2f}",
            transform=ax.transAxes,
            va='top', fontsize=14)
    ax.text(-5.3, 1.15,
            names[j],
            rotation=90,
            va='center',
            ha='center',
            transform=ax.transAxes,
            fontsize=20)
cbar = fig.colorbar(plot, ax=axes, pad=0.02, shrink=1, aspect=30)
cbar.ax.tick_params(labelsize=16)
cbar.set_label('Number of points  [-]', fontsize=20)
if is_only:
    plt.savefig(f'GrIS_Ant_LWup_scatter_icesheet.png', 
                dpi=300, bbox_inches='tight', format='png')
else:
    plt.savefig(f'GrIS_Ant_LWup_scatter.png', 
                dpi=300, bbox_inches='tight', format='png')

#%%
domains = ['FGRN055', 'PXANT11']
names = ['Greenland', 'Antarctica']
is_only = False
labels = ["Both clear-sky",
        "Both ice",
        "Both liquid",
        "Both mixed",
        "EarthCARE clear-sky,\nRACMO cloudy",
        "EarthCARE cloudy,\nRACMO clear-sky",
        "EarthCARE ice,\nRACMO liq./mixed",
        "EarthCARE liq./mixed,\nRACMO ice"]
titles = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)',
          '(g)', '(h)', '(i)', '(j)', '(k)', '(l)',
          '(m)', '(n)', '(o)', '(p)']
fig, axes = plt.subplots(4, 4, figsize=(20, 17), sharex=True, sharey=True)
axes = axes.flatten()
xlim_low = 0
xlim_high = 1000
ylim_low = 0
ylim_high = 1000

for j,domain in enumerate(domains):
    if is_only:
        sw_ds = xr.open_dataset(f"SW_{domain}_icesheet.nc")
    else:
        sw_ds = xr.open_dataset(f"SW_{domain}.nc")
    bbr_sw_arr = sw_ds.bbr_sw.values
    rac_sw_arr = sw_ds.rac_sw.values
    ec_cld_arr = sw_ds.ec_cld.values
    rac_cld_h_arr = sw_ds.rac_cld.values
    ec_cld_h_arr = sw_ds.ec_cld_h.values
    rac_cld_arr = sw_ds.rac_cld_h.values
    wp_bias_arr = sw_ds.wp_bias.values
    phase = np.zeros(len(bbr_sw_arr))
    phase[(ec_cld_arr == 1) & (rac_cld_arr == 1)] = 1 # both ice
    phase[(ec_cld_arr == 2) & (rac_cld_arr == 2)] = 2 # both liq
    phase[(ec_cld_arr == 3) & (rac_cld_arr == 3)] = 3 # both mixed
    phase[(ec_cld_arr == 0) & (rac_cld_arr > 0)] = 4 # ec cs, rac cld
    phase[(ec_cld_arr > 0) & (rac_cld_arr == 0)] = 5 # ec cld, rac cs
    phase[(ec_cld_arr == 1) & (rac_cld_arr > 1)] = 6 # ec ice, rac liq/mix
    phase[(ec_cld_arr > 1) & (rac_cld_arr == 1)] = 7 # ec liq/mix, rac ice
    if domain == 'FGRN055':
        norm = mcolors.LogNorm(1,1e3)
    elif domain == 'PXANT11':
        norm = mcolors.LogNorm(1,3e3)
    for i, ax in enumerate(axes[j*8:j*8+8]):
        mask = (phase == i)
        xedges, yedges = np.linspace(xlim_low, xlim_high, 70), np.linspace(ylim_low, ylim_high, 70)
        xplot_fit = np.linspace(xlim_low,xlim_high,11) 
        hist, xedges, yedges = np.histogram2d(bbr_sw_arr[mask], rac_sw_arr[mask], (xedges, yedges))
        xidx = np.clip(np.digitize(bbr_sw_arr[mask], xedges), 0, hist.shape[0]-1)
        yidx = np.clip(np.digitize(rac_sw_arr[mask], yedges), 0, hist.shape[1]-1)
        c = hist[xidx-1, yidx-1]
        plot = ax.scatter(bbr_sw_arr[mask], rac_sw_arr[mask], c=c, s=2, 
                    cmap=plt.cm.RdYlBu_r,norm=norm)
        ax.plot([xlim_low, xlim_high],
            [xlim_low, xlim_high],
            color='black',
            ls='--')
        ax.set_title(f"{labels[i]}")
        ax.set_title(titles[i+j*8], loc='left')
        ax.set_xlim(xlim_low, xlim_high)
        ax.set_ylim(ylim_low, ylim_high)
        ax.set_aspect('equal')
        if i+j*8 >= 12:
            ax.set_xlabel("EarthCARE BBR\nSWup TOA [W m$^{-2}$]")
        else:
            ax.tick_params(bottom=False)
        if (i == 0) | (i==4):
            ax.set_ylabel("RACMO SWup\nTOA [W m$^{-2}$]")
        else:
            ax.tick_params(left=False)
        ax.set_axisbelow(True)
        ax.grid('on')
        bias = np.mean(rac_sw_arr[mask] - bbr_sw_arr[mask])
        rmse = np.sqrt(np.mean((rac_sw_arr[mask] - bbr_sw_arr[mask])**2))
        r2 = np.corrcoef(rac_sw_arr[mask], bbr_sw_arr[mask])[1,0]**2
        ax.text(0.05, 0.98,
            f"N={mask.sum()}, Bias={bias:.1f}\nRMSE={rmse:.1f}, R$^2$={r2:.2f}",
            transform=ax.transAxes,
            va='top', fontsize=14)
    ax.text(-4.1, 1.15,
            names[j],
            rotation=90,
            va='center',
            ha='center',
            transform=ax.transAxes,
            fontsize=20)
cbar = fig.colorbar(plot, ax=axes, pad=0.02, shrink=1, aspect=30)
cbar.ax.tick_params(labelsize=16)
cbar.set_label('Number of points  [-]', fontsize=20)
if is_only:
    plt.savefig(f'GrIS_Ant_SWup_scatter_icesheet.png', 
                dpi=300, bbox_inches='tight', format='png')
else:
    plt.savefig(f'GrIS_Ant_SWup_scatter.png', 
                dpi=300, bbox_inches='tight', format='png')
