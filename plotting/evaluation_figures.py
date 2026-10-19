#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
"""
Created on Tue Nov 19 13:33:58 2024

@author: Feens012
"""
import numpy as np
import sys
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from cmcrameri import cm as ccm
import cartopy.crs as crs
import cartopy.feature as ccrs
import cartopy.io.shapereader as shpreader
from cartopy.feature import ShapelyFeature
import importlib
sys.path.append('../utils/')
import plot_functions 
import colormaps_ARS
import helpers
import RCM2EC_functions
import spatial_regridding
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
                     "hatch.color":'dimgrey', 
                    'mathtext.default': 'regular'})
topo_g = xr.open_dataset('FGRN055_masks.nc', engine='netcdf4')
topo_a = xr.open_dataset('PXANT11_masks.nc', engine='netcdf4')


#%% 
ds_rac_g = xr.open_dataset(f'/FGRN055/merged/RACMO_FGRN055_bl_ice_top_combined.nc')
res_new_rac_g, lat_new_g, lon_new_g= spatial_regridding.downsample_grid(ds_rac_g, 'mean', 20, sample_type='mean')
res_new_rac_g[res_new_rac_g == 0] = np.nan
ds_ec_g = xr.open_dataset(f'/FGRN055/merged/EC_ice_top_com_combined.nc')
res_new_ec_g, lat_new_g, lon_new_g= spatial_regridding.downsample_grid(ds_ec_g, 'mean', 20, sample_type='mean')
res_new_ec_g[res_new_ec_g == 0] = np.nan
ds_rac_a = xr.open_dataset(f'/PXANT11/merged/RACMO_PXANT11_bl_qb_ice_top_combined.nc')
res_new_rac_a, lat_new_a, lon_new_a= spatial_regridding.downsample_grid(ds_rac_a, 'mean', 20, sample_type='mean')
res_new_rac_a[res_new_rac_a == 0] = np.nan
ds_ec_a = xr.open_dataset(f'/PXANT11/merged/EC_ice_top_com_combined.nc')
res_new_ec_a, lat_new_a, lon_new_a= spatial_regridding.downsample_grid(ds_ec_a, 'mean', 20, sample_type='mean')
res_new_ec_a[res_new_ec_a == 0] = np.nan
plot_functions.GrIS_Ant_maps_diff(res_new_ec_g, res_new_rac_g, 
                    res_new_ec_a, res_new_rac_a, 
                    lat_new_g, lon_new_g,
                    lat_new_a, lon_new_a, ccm.batlow_r, 
                    1000,8000, log=False,extend='both',
                    mdiff=mcolors.Normalize(vmin=-3000, vmax=3000),
                    title1=f'EarthCARE', 
                    title2=f'RACMO',
                    title3=f'RACMO - EarthCARE',
                    clabel='Ice cloud top height', 
                    unit = '[m]', masked=False)
#%% 
ds_rac_g = xr.open_dataset(f'/FGRN055/merged/RACMO_FGRN055_bl_liquid_top_combined.nc')
res_new_rac_g, lat_new_g, lon_new_g= spatial_regridding.downsample_grid(ds_rac_g, 'mean', 20, sample_type='mean')
res_new_rac_g[res_new_rac_g == 0] = np.nan
ds_ec_g = xr.open_dataset(f'/FGRN055/merged/EC_liquid_top_combined.nc')
res_new_ec_g, lat_new_g, lon_new_g= spatial_regridding.downsample_grid(ds_ec_g, 'mean', 20, sample_type='mean')
res_new_ec_g[res_new_ec_g == 0] = np.nan
ds_rac_a = xr.open_dataset(f'/PXANT11/merged/RACMO_PXANT11_bl_qb_liquid_top_combined.nc')
res_new_rac_a, lat_new_a, lon_new_a= spatial_regridding.downsample_grid(ds_rac_a, 'mean', 20, sample_type='mean')
res_new_rac_a[res_new_rac_a == 0] = np.nan
ds_ec_a = xr.open_dataset(f'/PXANT11/merged/EC_liquid_top_combined.nc')
res_new_ec_a, lat_new_a, lon_new_a= spatial_regridding.downsample_grid(ds_ec_a, 'mean', 20, sample_type='mean')
res_new_ec_a[res_new_ec_a == 0] = np.nan
plot_functions.GrIS_Ant_maps_diff(res_new_ec_g, res_new_rac_g, 
                    res_new_ec_a, res_new_rac_a, 
                    lat_new_g, lon_new_g,
                    lat_new_a, lon_new_a, ccm.batlow_r, 
                    1000,8000, log=False,extend='both',
                    mdiff=mcolors.Normalize(vmin=-1500, vmax=1500),
                    title1=f'EarthCARE', 
                    title2=f'RACMO',
                    title3=f'RACMO - EarthCARE',
                    clabel='Liquid cloud top height', 
                    unit = '[m]', masked=False)

#%% 
ds_rac_g = xr.open_dataset(f'/FGRN055/merged/RACMO_FGRN055_bl_ice_occ_2D_combined.nc')
res_new_rac_g, lat_new_g, lon_new_g= spatial_regridding.downsample_grid(ds_rac_g, 'fraction', 20)
res_new_rac_g[res_new_rac_g == 0] = np.nan
ds_ec_g = xr.open_dataset(f'/FGRN055/merged/EC_ice_occ_com_2D_combined.nc')
res_new_ec_g, lat_new_g, lon_new_g= spatial_regridding.downsample_grid(ds_ec_g, 'fraction', 20)
res_new_ec_g[res_new_ec_g == 0] = np.nan
ds_rac_a = xr.open_dataset(f'/PXANT11/merged/RACMO_PXANT11_bl_qb_ice_occ_2D_qb_combined.nc')
res_new_rac_a, lat_new_a, lon_new_a= spatial_regridding.downsample_grid(ds_rac_a, 'fraction', 20)
res_new_rac_a[res_new_rac_a == 0] = np.nan
ds_ec_a = xr.open_dataset(f'/PXANT11/merged/EC_ice_occ_com_2D_combined.nc')
res_new_ec_a, lat_new_a, lon_new_a= spatial_regridding.downsample_grid(ds_ec_a, 'fraction', 20)
res_new_ec_a[res_new_ec_a == 0] = np.nan
plot_functions.GrIS_Ant_maps_diff(res_new_ec_g, res_new_rac_g, 
                    res_new_ec_a, res_new_rac_a, 
                    lat_new_g, lon_new_g,
                    lat_new_a, lon_new_a, ccm.batlow_r, 
                    0,1, log=False,extend=None,
                    title1=f'EarthCARE', 
                    title2=f'RACMO',
                    title3=f'RACMO - EarthCARE',
                    clabel='Ice cloud occcurrence', 
                    unit = '[-]', masked=False)
#%% 
ds_rac_g = xr.open_dataset(f'/FGRN055/merged/RACMO_FGRN055_bl_liq_occ_2D_combined.nc')
res_new_rac_g, lat_new_g, lon_new_g= spatial_regridding.downsample_grid(ds_rac_g, 'fraction', 20)
res_new_rac_g[res_new_rac_g == 0] = np.nan
ds_ec_g = xr.open_dataset(f'/FGRN055/merged/EC_liq_occ_2D_combined.nc')
res_new_ec_g, lat_new_g, lon_new_g= spatial_regridding.downsample_grid(ds_ec_g, 'fraction', 20)
res_new_ec_g[res_new_ec_g == 0] = np.nan
ds_rac_a = xr.open_dataset(f'/PXANT11/merged/RACMO_PXANT11_bl_qb_liq_occ_2D_combined.nc')
res_new_rac_a, lat_new_a, lon_new_a= spatial_regridding.downsample_grid(ds_rac_a, 'fraction', 20)
res_new_rac_a[res_new_rac_a == 0] = np.nan
ds_ec_a = xr.open_dataset(f'/PXANT11/merged/EC_liq_occ_2D_combined.nc')
res_new_ec_a, lat_new_a, lon_new_a= spatial_regridding.downsample_grid(ds_ec_a, 'fraction', 20)
res_new_ec_a[res_new_ec_a == 0] = np.nan
plot_functions.GrIS_Ant_maps_diff(res_new_ec_g, res_new_rac_g, 
                    res_new_ec_a, res_new_rac_a, 
                    lat_new_g, lon_new_g,
                    lat_new_a, lon_new_a, ccm.batlow_r, 
                    0,1, log=False,extend=None,
                    title1=f'EarthCARE', 
                    title2=f'RACMO',
                    title3=f'RACMO - EarthCARE',
                    clabel='Liquid cloud occcurrence', 
                    unit = '[-]', masked=False)
#%% 
ds_rac_g = xr.open_dataset(f'/FGRN055/merged/RACMO_FGRN055_bl_IWP_conditional_combined.nc')
res_new_rac_g, lat_new_g, lon_new_g= spatial_regridding.downsample_grid(ds_rac_g, 'mean', 20, sample_type='mean')
res_new_rac_g[res_new_rac_g == 0] = np.nan
ds_ec_g = xr.open_dataset(f'/FGRN055/merged/EC_IWP_conditional_com_combined.nc')
res_new_ec_g, lat_new_g, lon_new_g= spatial_regridding.downsample_grid(ds_ec_g, 'mean', 20, sample_type='mean')
res_new_ec_g[res_new_ec_g == 0] = np.nan
ds_rac_a = xr.open_dataset(f'/PXANT11/merged/RACMO_PXANT11_bl_qb_IWP_conditional_qb_combined.nc')
res_new_rac_a, lat_new_a, lon_new_a= spatial_regridding.downsample_grid(ds_rac_a, 'mean', 20, sample_type='mean')
res_new_rac_a[res_new_rac_a == 0] = np.nan
ds_ec_a = xr.open_dataset(f'/PXANT11/merged/EC_IWP_conditional_com_combined.nc')
res_new_ec_a, lat_new_a, lon_new_a= spatial_regridding.downsample_grid(ds_ec_a, 'mean', 20, sample_type='mean')
res_new_ec_a[res_new_ec_a == 0] = np.nan
plot_functions.GrIS_Ant_maps_diff(res_new_ec_g, res_new_rac_g, 
                    res_new_ec_a, res_new_rac_a, 
                    lat_new_g, lon_new_g,
                    lat_new_a, lon_new_a, ccm.batlow_r, 
                    0,0.4, log=False,extend='max',
                    title1=f'EarthCARE', 
                    title2=f'RACMO',
                    title3=f'RACMO - EarthCARE',
                    clabel='Conditional IWP', 
                    unit = '[kg m$^{-2}$]', masked=False)
# %%
ds_rac_g = xr.open_dataset(f'/FGRN055/merged/RACMO_FGRN055_bl_snow_occ_topo_combined.nc')
res_new_rac_g, lat_new_g, lon_new_g, hnew = spatial_regridding.downsample_grid(ds_rac_g, 'fraction', 20, 1)
res_new_rac_g[res_new_rac_g== 0] = np.nan
ds_ec_g = xr.open_dataset(f'/FGRN055/merged/EC_snow_occ_com_topo_combined.nc')
res_new_ec_g, lat_new_g, lon_new_g, hnew = spatial_regridding.downsample_grid(ds_ec_g, 'fraction', 20, 1) 
res_new_ec_g[res_new_ec_g == 0] = np.nan
ds_rac_a = xr.open_dataset(f'/PXANT11/merged/RACMO_PXANT11_bl_qb_snow_occ_topo_combined.nc')
res_new_rac_a, lat_new_a, lon_new_a, hnew = spatial_regridding.downsample_grid(ds_rac_a, 'fraction', 20, 1)
res_new_rac_a[res_new_rac_a== 0] = np.nan
ds_ec_a = xr.open_dataset(f'/PXANT11/merged/EC_snow_occ_com_topo_combined.nc')
res_new_ec_a, lat_new_a, lon_new_a, hnew = spatial_regridding.downsample_grid(ds_ec_a, 'fraction', 20, 1) 
res_new_ec_a[res_new_ec_a == 0] = np.nan
plot_functions.GrIS_Ant_maps_diff(res_new_ec_g[:,:,-11], 
                   res_new_rac_g[:,:,-11], 
                   res_new_ec_a[:,:,-11], 
                   res_new_rac_a[:,:,-11],  
                    lat_new_g, lon_new_g,
                    lat_new_a, lon_new_a, ccm.batlow_r, 
                    0, 1,
                    log=False,extend=None,
                    title1=f'EarthCARE', 
                    title2=f'RACMO',
                    title3=f'RACMO - EarthCARE',
                    clabel='Snowfall occurrence at 1 km above surface', 
                    unit = '[-]', masked=False)
#%%
ds_rac_g = xr.open_dataset(f'/FGRN055/merged/RACMO_FGRN055_bl_snowrate_cond_topo_combined.nc')
res_new_rac_g, lat_new_g, lon_new_g, hnew = spatial_regridding.downsample_grid(ds_rac_g, 'mean', 20, 1, sample_type='mean')
res_new_rac_g[res_new_rac_g== 0] = np.nan
ds_ec_g = xr.open_dataset(f'/FGRN055/merged/EC_snowrate_cond_com_topo_combined.nc')
res_new_ec_g, lat_new_g, lon_new_g, hnew = spatial_regridding.downsample_grid(ds_ec_g, 'mean', 20, 1, sample_type='mean')
res_new_ec_g[res_new_ec_g == 0] = np.nan
ds_rac_a = xr.open_dataset(f'/PXANT11/merged/RACMO_PXANT11_bl_qb_snowrate_cond_topo_qb_combined.nc')
res_new_rac_a, lat_new_a, lon_new_a, hnew = spatial_regridding.downsample_grid(ds_rac_a, 'mean', 20, 1, sample_type='mean')
res_new_rac_a[res_new_rac_a== 0] = np.nan
ds_ec_a = xr.open_dataset(f'/PXANT11/merged/EC_snowrate_cond_com_topo_combined.nc')
res_new_ec_a, lat_new_a, lon_new_a, hnew = spatial_regridding.downsample_grid(ds_ec_a, 'mean', 20, 1, sample_type='mean')
res_new_ec_a[res_new_ec_a == 0] = np.nan
plot_functions.GrIS_Ant_maps_diff(res_new_ec_g[:,:,-11], 
                   res_new_rac_g[:,:,-11], 
                   res_new_ec_a[:,:,-11], 
                   res_new_rac_a[:,:,-11],  
                    lat_new_g, lon_new_g,
                    lat_new_a, lon_new_a, ccm.batlow_r, 
                    1e-6, 1e-3, mdiff='relative_perc',
                    log=True,extend=None,
                    title1=f'EarthCARE', 
                    title2=f'RACMO',
                    title3=f'RACMO - EarthCARE',
                    clabel='Conditional snowfall rate at 1 km above surface', 
                    unit = '[kg m$^{-2}$ s$^{-1}$]', masked=False)
#%%
ds_rac_g = xr.open_dataset(f'/FGRN055/merged/RACMO_FGRN055_bl_IWP_combined.nc')
res_new_rac_ec_g, lat_new_g, lon_new_g= spatial_regridding.downsample_grid(ds_rac_g, 'mean', 20, sample_type='mean')
res_new_rac_ec_g[res_new_rac_ec_g == 0] = np.nan
ds_rac_i = xr.open_mfdataset([f'/FGRN055/RACMO2.4p1_v4_bl-2024/NC_DEFAULT/clivi.KNMI-2024.FGRN055.RACMO2.4p1_v4_bl.DD.nc',
                              f'/FGRN055/RACMO2.4p1_v4_bl-2025/NC_DEFAULT/clivi.KNMI-2025.FGRN055.RACMO2.4p1_v4_bl.DD.nc',
                              f'/FGRN055/RACMO2.4p1_v4_bl-2026/NC_DEFAULT/clivi.KNMI-2026.FGRN055.RACMO2.4p1_v4_bl.DD.nc'],
                              concat_dim='time', combine='nested').squeeze().sel(time=slice('2024-09-01', '2026-04-30')).mean('time')
res_new_rac_i = spatial_regridding.downsample_grid(ds_rac_i, 'clivi', 20, sample_type='mean')[0]
ds_rac_s = xr.open_mfdataset([f'/FGRN055/RACMO2.4p1_v4_bl-2024/NC_DEFAULT/tcsw.KNMI-2024.FGRN055.RACMO2.4p1_v4_bl.DD.nc',
                              f'/FGRN055/RACMO2.4p1_v4_bl-2025/NC_DEFAULT/tcsw.KNMI-2025.FGRN055.RACMO2.4p1_v4_bl.DD.nc',
                              f'/FGRN055/RACMO2.4p1_v4_bl-2026/NC_DEFAULT/tcsw.KNMI-2026.FGRN055.RACMO2.4p1_v4_bl.DD.nc'],
                              concat_dim='time', combine='nested').squeeze().sel(time=slice('2024-09-01', '2026-04-30')).mean('time')
res_new_rac_s = spatial_regridding.downsample_grid(ds_rac_s, 'tcsw', 20, sample_type='mean')[0]
res_new_rac_g = res_new_rac_i + res_new_rac_s
res_new_rac_g[res_new_rac_g == 0] = np.nan
ds_rac_a = xr.open_dataset(f'/PXANT11/merged/RACMO_PXANT11_bl_qb_IWP_qb_combined.nc')
res_new_rac_ec_a, lat_new_a, lon_new_a= spatial_regridding.downsample_grid(ds_rac_a, 'mean', 20, sample_type='mean')
res_new_rac_ec_a[res_new_rac_ec_a == 0] = np.nan
ds_rac_i = xr.open_mfdataset([f'/PXANT11/RACMO2.4p1_v4_bl_qb-2024/NC_DEFAULT/clivi.KNMI-2024.PXANT11.RACMO2.4p1_v4_bl_qb.DD.nc',
                              f'/PXANT11/RACMO2.4p1_v4_bl_qb-2025/NC_DEFAULT/clivi.KNMI-2025.PXANT11.RACMO2.4p1_v4_bl_qb.DD.nc',
                              f'/PXANT11/RACMO2.4p1_v4_bl_qb-2026/NC_DEFAULT/clivi.KNMI-2026.PXANT11.RACMO2.4p1_v4_bl_qb.DD.nc'],
                              concat_dim='time', combine='nested').squeeze().sel(time=slice('2024-09-01', '2026-04-30')).mean('time')
res_new_rac_i = spatial_regridding.downsample_grid(ds_rac_i, 'clivi', 20, sample_type='mean')[0]
ds_rac_s = xr.open_mfdataset([f'/PXANT11/RACMO2.4p1_v4_bl_qb-2024/NC_DEFAULT/tcsw.KNMI-2024.PXANT11.RACMO2.4p1_v4_bl_qb.DD.nc',
                              f'/PXANT11/RACMO2.4p1_v4_bl_qb-2025/NC_DEFAULT/tcsw.KNMI-2025.PXANT11.RACMO2.4p1_v4_bl_qb.DD.nc',
                              f'/PXANT11/RACMO2.4p1_v4_bl_qb-2026/NC_DEFAULT/tcsw.KNMI-2026.PXANT11.RACMO2.4p1_v4_bl_qb.DD.nc'],
                              concat_dim='time', combine='nested').squeeze().sel(time=slice('2024-09-01', '2026-04-30')).mean('time')
res_new_rac_s = spatial_regridding.downsample_grid(ds_rac_s, 'tcsw', 20, sample_type='mean')[0]
res_new_rac_a = res_new_rac_i + res_new_rac_s
res_new_rac_a[res_new_rac_a == 0] = np.nan
plot_functions.GrIS_Ant_maps_diff(res_new_rac_g, res_new_rac_ec_g, 
                   res_new_rac_a, res_new_rac_ec_a,
                lat_new_g, lon_new_g, 
                lat_new_a, lon_new_a, ccm.batlow_r, 
                0, 0.3, log=False,
                extend='max',
                title1=f'RACMO all', 
                title2=f'RACMO during overpass',
                title3='During overpass - all',
                clabel='IWP', unit = '[kg m$^{-2}$]', masked=False)


# %% avg ice/liq occ
months = ['202409', '202410', '202411', '202412', '202501', '202502', '202503',
          '202504', '202505', '202506', '202507', '202508', '202509', '202510',
          '202511', '202512', '202601', '202602', '202603', '202604']
ec_io_g = np.zeros((len(months),5))
ec_lo_g = np.zeros((len(months),5))
rc_io_g = np.zeros((len(months),5))
rc_lo_g = np.zeros((len(months),5))
ec_io_a = np.zeros((len(months),5))
ec_lo_a = np.zeros((len(months),5))
rc_io_a = np.zeros((len(months),5))
rc_lo_a = np.zeros((len(months),5))
for i,month in enumerate(months):
    d_ec_io_g = xr.open_dataset(f'/FGRN055/EC_ice_occ_com_2D_{month}.nc')
    d_ec_lo_g = xr.open_dataset(f'/FGRN055/EC_liq_occ_2D_{month}.nc')
    d_rc_io_g = xr.open_dataset(f'/FGRN055/RACMO_FGRN055_bl_ice_occ_2D_{month}.nc')
    d_rc_lo_g = xr.open_dataset(f'/FGRN055/RACMO_FGRN055_bl_liq_occ_2D_{month}.nc')
    sice_g = xr.open_dataset(f'/FGRN055/RACMO2.4p1_v4_bl/Monthly/siconca_monthlyA_FGRN055_RACMO2.4p1_v4_bl_202406_202604.nc').sel(time=month[:4]+'-'+month[4:], method='nearest')
    ice_sheet_mask = topo_g.Icesheet_Only.values.copy()
    ice_sheet_mask[ice_sheet_mask == 0] = np.nan
    land_mask = topo_g.LSM.values.copy()
    land_mask[land_mask == 0] = np.nan
    ocean_mask = 1 - topo_g.LSM.values.copy()
    ocean_mask[ocean_mask == 0] = np.nan
    sice_mask = sice_g.siconca.values.copy()
    sice_mask[sice_mask >= 0.15] = 1
    sice_mask[sice_mask < 0.15] = np.nan
    sice_mask_inv = sice_g.siconca.values.copy()
    sice_mask_inv[sice_mask_inv >= 0.15] = np.nan
    sice_mask_inv[sice_mask_inv < 0.15] = 1
    ec_io_g[i,0] = np.nansum(d_ec_io_g.cond_result.values)/np.nansum(d_ec_io_g.overpasses.values)
    ec_lo_g[i,0] = np.nansum(d_ec_lo_g.cond_result.values)/np.nansum(d_ec_lo_g.overpasses.values)
    rc_io_g[i,0] = np.nansum(d_rc_io_g.cond_result.values)/np.nansum(d_rc_io_g.overpasses.values)
    rc_lo_g[i,0] = np.nansum(d_rc_lo_g.cond_result.values)/np.nansum(d_rc_lo_g.overpasses.values)
    ec_io_g[i,1] = np.nansum(d_ec_io_g.cond_result.values*ice_sheet_mask)/np.nansum(d_ec_io_g.overpasses.values*ice_sheet_mask)
    ec_lo_g[i,1] = np.nansum(d_ec_lo_g.cond_result.values*ice_sheet_mask)/np.nansum(d_ec_lo_g.overpasses.values*ice_sheet_mask)
    rc_io_g[i,1] = np.nansum(d_rc_io_g.cond_result.values*ice_sheet_mask)/np.nansum(d_rc_io_g.overpasses.values*ice_sheet_mask)
    rc_lo_g[i,1] = np.nansum(d_rc_lo_g.cond_result.values*ice_sheet_mask)/np.nansum(d_rc_lo_g.overpasses.values*ice_sheet_mask)
    ec_io_g[i,2] = np.nansum(d_ec_io_g.cond_result.values*land_mask)/np.nansum(d_ec_io_g.overpasses.values*land_mask)
    ec_lo_g[i,2] = np.nansum(d_ec_lo_g.cond_result.values*land_mask)/np.nansum(d_ec_lo_g.overpasses.values*land_mask)
    rc_io_g[i,2] = np.nansum(d_rc_io_g.cond_result.values*land_mask)/np.nansum(d_rc_io_g.overpasses.values*land_mask)
    rc_lo_g[i,2] = np.nansum(d_rc_lo_g.cond_result.values*land_mask)/np.nansum(d_rc_lo_g.overpasses.values*land_mask)
    ec_io_g[i,3] = np.nansum(d_ec_io_g.cond_result.values*ocean_mask*sice_mask_inv)/np.nansum(d_ec_io_g.overpasses.values*ocean_mask*sice_mask_inv)
    ec_lo_g[i,3] = np.nansum(d_ec_lo_g.cond_result.values*ocean_mask*sice_mask_inv)/np.nansum(d_ec_lo_g.overpasses.values*ocean_mask*sice_mask_inv)
    rc_io_g[i,3] = np.nansum(d_rc_io_g.cond_result.values*ocean_mask*sice_mask_inv)/np.nansum(d_rc_io_g.overpasses.values*ocean_mask*sice_mask_inv)
    rc_lo_g[i,3] = np.nansum(d_rc_lo_g.cond_result.values*ocean_mask*sice_mask_inv)/np.nansum(d_rc_lo_g.overpasses.values*ocean_mask*sice_mask_inv)
    ec_io_g[i,4] = np.nansum(d_ec_io_g.cond_result.values*sice_mask)/np.nansum(d_ec_io_g.overpasses.values*sice_mask)
    ec_lo_g[i,4] = np.nansum(d_ec_lo_g.cond_result.values*sice_mask)/np.nansum(d_ec_lo_g.overpasses.values*sice_mask)
    rc_io_g[i,4] = np.nansum(d_rc_io_g.cond_result.values*sice_mask)/np.nansum(d_rc_io_g.overpasses.values*sice_mask)
    rc_lo_g[i,4] = np.nansum(d_rc_lo_g.cond_result.values*sice_mask)/np.nansum(d_rc_lo_g.overpasses.values*sice_mask)
    d_ec_io_a = xr.open_dataset(f'/PXANT11/EC_ice_occ_com_2D_{month}.nc')
    d_ec_lo_a = xr.open_dataset(f'/PXANT11/EC_liq_occ_2D_{month}.nc')
    d_rc_io_a = xr.open_dataset(f'/PXANT11/RACMO_PXANT11_bl_qb_ice_occ_2D_qb_{month}.nc')
    d_rc_lo_a = xr.open_dataset(f'/PXANT11/RACMO_PXANT11_bl_qb_liq_occ_2D_{month}.nc')
    sice_a = xr.open_dataset(f'/PXANT11/RACMO2.4p1_v4_bl_qb/Monthly/siconca_monthlyA_PXANT11_RACMO2.4p1_v4_bl_qb_202408_202604.nc').sel(time=month[:4]+'-'+month[4:], method='nearest')
    ice_sheet_mask = topo_a.Icesheet_Only.values.copy()
    ice_sheet_mask[ice_sheet_mask == 0] = np.nan
    land_mask = topo_a.LSM.values.copy()
    land_mask[land_mask == 0] = np.nan
    ocean_mask = 1 - topo_a.LSM.values.copy()
    ocean_mask[ocean_mask == 0] = np.nan
    sice_mask = sice_a.siconca.values.copy()
    sice_mask[sice_mask >= 0.15] = 1
    sice_mask[sice_mask < 0.15] = np.nan
    sice_mask_inv = sice_a.siconca.values.copy()
    sice_mask_inv[sice_mask_inv >= 0.15] = np.nan
    sice_mask_inv[sice_mask_inv < 0.15] = 1
    ec_io_a[i,0] = np.nansum(d_ec_io_a.cond_result.values)/np.nansum(d_ec_io_a.overpasses.values)
    ec_lo_a[i,0] = np.nansum(d_ec_lo_a.cond_result.values)/np.nansum(d_ec_lo_a.overpasses.values)
    rc_io_a[i,0] = np.nansum(d_rc_io_a.cond_result.values)/np.nansum(d_rc_io_a.overpasses.values)
    rc_lo_a[i,0] = np.nansum(d_rc_lo_a.cond_result.values)/np.nansum(d_rc_lo_a.overpasses.values)
    ec_io_a[i,1] = np.nansum(d_ec_io_a.cond_result.values*ice_sheet_mask)/np.nansum(d_ec_io_a.overpasses.values*ice_sheet_mask)
    ec_lo_a[i,1] = np.nansum(d_ec_lo_a.cond_result.values*ice_sheet_mask)/np.nansum(d_ec_lo_a.overpasses.values*ice_sheet_mask)
    rc_io_a[i,1] = np.nansum(d_rc_io_a.cond_result.values*ice_sheet_mask)/np.nansum(d_rc_io_a.overpasses.values*ice_sheet_mask)
    rc_lo_a[i,1] = np.nansum(d_rc_lo_a.cond_result.values*ice_sheet_mask)/np.nansum(d_rc_lo_a.overpasses.values*ice_sheet_mask)
    ec_io_a[i,2] = np.nansum(d_ec_io_a.cond_result.values*land_mask)/np.nansum(d_ec_io_a.overpasses.values*land_mask)
    ec_lo_a[i,2] = np.nansum(d_ec_lo_a.cond_result.values*land_mask)/np.nansum(d_ec_lo_a.overpasses.values*land_mask)
    rc_io_a[i,2] = np.nansum(d_rc_io_a.cond_result.values*land_mask)/np.nansum(d_rc_io_a.overpasses.values*land_mask)
    rc_lo_a[i,2] = np.nansum(d_rc_lo_a.cond_result.values*land_mask)/np.nansum(d_rc_lo_a.overpasses.values*land_mask)
    ec_io_a[i,3] = np.nansum(d_ec_io_a.cond_result.values*ocean_mask*sice_mask_inv)/np.nansum(d_ec_io_a.overpasses.values*ocean_mask*sice_mask_inv)
    ec_lo_a[i,3] = np.nansum(d_ec_lo_a.cond_result.values*ocean_mask*sice_mask_inv)/np.nansum(d_ec_lo_a.overpasses.values*ocean_mask*sice_mask_inv)
    rc_io_a[i,3] = np.nansum(d_rc_io_a.cond_result.values*ocean_mask*sice_mask_inv)/np.nansum(d_rc_io_a.overpasses.values*ocean_mask*sice_mask_inv)
    rc_lo_a[i,3] = np.nansum(d_rc_lo_a.cond_result.values*ocean_mask*sice_mask_inv)/np.nansum(d_rc_lo_a.overpasses.values*ocean_mask*sice_mask_inv)
    ec_io_a[i,4] = np.nansum(d_ec_io_a.cond_result.values*sice_mask)/np.nansum(d_ec_io_a.overpasses.values*sice_mask)
    ec_lo_a[i,4] = np.nansum(d_ec_lo_a.cond_result.values*sice_mask)/np.nansum(d_ec_lo_a.overpasses.values*sice_mask)
    rc_io_a[i,4] = np.nansum(d_rc_io_a.cond_result.values*sice_mask)/np.nansum(d_rc_io_a.overpasses.values*sice_mask)
    rc_lo_a[i,4] = np.nansum(d_rc_lo_a.cond_result.values*sice_mask)/np.nansum(d_rc_lo_a.overpasses.values*sice_mask)
cmap = ccm.batlow_r
colors = cmap(np.linspace(0, 1, 10))
fig, axes = plt.subplots(2,3,figsize=(15,7), sharey=True, sharex=True)
plt.subplots_adjust(wspace=0.05, hspace=0.15)
(ax1, ax2, ax3), (ax4, ax5, ax6) = axes
ax1.set_title('Ice sheet only')
ax2.set_title('Open ocean only')
ax3.set_title('Sea ice only')
ax1.set_title('(a)', loc='left')
ax2.set_title('(b)', loc='left')
ax3.set_title('(c)', loc='left')
ax4.set_title('(d)', loc='left')
ax5.set_title('(e)', loc='left')
ax6.set_title('(f)', loc='left')
ax1.plot(months, ec_io_g[:,1], label='Ice, EarthCARE', color=colors[1], linewidth=2)
ax1.plot(months, rc_io_g[:,1], label='Ice, RACMO', color=colors[-1], linewidth=2)
ax1.plot(months, ec_lo_g[:,1], label='Liquid, EarthCARE', color=colors[1], ls ='--', linewidth=2)
ax1.plot(months, rc_lo_g[:,1], label='Liquid, RACMO', color=colors[-1], ls ='--', linewidth=2)
ax2.plot(months, ec_io_g[:,3], color=colors[1], linewidth=2)
ax2.plot(months, rc_io_g[:,3], color=colors[-1], linewidth=2)
ax2.plot(months, ec_lo_g[:,3], color=colors[1], ls ='--', linewidth=2)
ax2.plot(months, rc_lo_g[:,3], color=colors[-1], ls ='--', linewidth=2)
ax3.plot(months, ec_io_g[:,4], color=colors[1], linewidth=2)
ax3.plot(months, rc_io_g[:,4], color=colors[-1], linewidth=2)
ax3.plot(months, ec_lo_g[:,4], color=colors[1], ls ='--', linewidth=2)
ax3.plot(months, rc_lo_g[:,4], color=colors[-1], ls ='--', linewidth=2)
ax4.plot(months, ec_io_a[:,1], color=colors[1], linewidth=2)
ax4.plot(months, rc_io_a[:,1], color=colors[-1], linewidth=2)
ax4.plot(months, ec_lo_a[:,1], color=colors[1], ls ='--', linewidth=2)
ax4.plot(months, rc_lo_a[:,1], color=colors[-1], ls ='--', linewidth=2)
ax5.plot(months, ec_io_a[:,3], color=colors[1], linewidth=2)
ax5.plot(months, rc_io_a[:,3], color=colors[-1], linewidth=2)
ax5.plot(months, ec_lo_a[:,3], color=colors[1], ls ='--', linewidth=2)
ax5.plot(months, rc_lo_a[:,3], color=colors[-1], ls ='--', linewidth=2)
ax6.plot(months, ec_io_a[:,4], color=colors[1], linewidth=2)
ax6.plot(months, rc_io_a[:,4], color=colors[-1], linewidth=2)
ax6.plot(months, ec_lo_a[:,4], color=colors[1], ls ='--', linewidth=2)
ax6.plot(months, rc_lo_a[:,4], color=colors[-1], ls ='--', linewidth=2)
fig.legend(frameon=False, bbox_to_anchor=(0.75,-0.01), ncol=4)
ax1.grid(color='grey', linestyle='dashed')
ax2.grid(color='grey', linestyle='dashed')
ax3.grid(color='grey', linestyle='dashed')
ax4.grid(color='grey', linestyle='dashed')
ax5.grid(color='grey', linestyle='dashed')
ax6.grid(color='grey', linestyle='dashed')
ax4.set_xticks(months[::3], months[::3], rotation=90)
ax5.set_xticks(months[::3], months[::3], rotation=90)
ax6.set_xticks(months[::3], months[::3], rotation=90)
ax1.set_ylim(0, 1)
ax2.set_ylim(0, 1)
ax3.set_ylim(0, 1)
ax4.set_ylim(0, 1)
ax5.set_ylim(0, 1)
ax6.set_ylim(0, 1)
ax1.tick_params(bottom=False)
ax2.tick_params(left=False, bottom=False)
ax3.tick_params(left=False, bottom=False)
ax5.tick_params(left=False)
ax6.tick_params(left=False)
ax1.text(
    -0.25, 0.5,
    'Greenland',
    rotation=90,
    va='center',
    ha='center',
    transform=ax1.transAxes,
    fontsize=18)
ax4.text(
    -0.25, 0.5,
    'Antarctica',
    rotation=90,
    va='center',
    ha='center',
    transform=ax4.transAxes,
    fontsize=18)
ax1.set_ylabel('Occurrence [-]')
ax4.set_ylabel('Occurrence [-]')
plt.savefig(f'GrIS_Ant_cloud_occ_timeseries.png', 
            dpi=300, bbox_inches='tight', format='png')
#%% 
months = ['202409', '202410', '202411', '202412', '202501', '202502',
          '202503', '202504', '202505', '202506', '202507', '202508',
          '202509', '202510', '202511', '202512', '202601', '202602',
          '202603', '202604']
domains = ['FGRN055','PXANT11']
labels = ['Greenland','Antarctica']
results = {}
for domain in domains:
    if domain == 'FGRN055':
        topo = xr.open_dataset(
            'FGRN055_masks.nc',
            engine='netcdf4')
    else:
        topo = xr.open_dataset(
            'PXANT11_masks.nc',
            engine='netcdf4')
    ec_iwp = np.zeros((len(months),5))
    co_iwp = np.zeros((len(months),5))
    ec_lwp = np.zeros((len(months),5))
    rc_iwp = np.zeros((len(months),5))
    rc_lwp = np.zeros((len(months),5))
    for i,month in enumerate(months):
        d_ec_iwp = xr.open_dataset(
            f'/{domain}/EC_IWP_2b_{month}.nc')
        d_co_iwp = xr.open_dataset(
            f'/{domain}/EC_IWP_com_{month}.nc')
        d_ec_lwp = xr.open_dataset(
            f'/{domain}/EC_LWP_2b_{month}.nc')
        if domain == 'FGRN055':
            d_rc_iwp = xr.open_dataset(
                f'/{domain}/RACMO_{domain}_bl_IWP_{month}.nc')
            d_rc_lwp = xr.open_dataset(
                f'/{domain}/RACMO_{domain}_bl_LWP_{month}.nc')
            sice = xr.open_dataset(
                f'/{domain}/'
                f'RACMO2.4p1_v4_bl/Monthly/'
                f'siconca_monthlyA_{domain}_RACMO2.4p1_v4_bl_202406_202604.nc').sel(
                time=month[:4]+'-'+month[4:],
                method='nearest')
        elif domain == 'PXANT11':
            d_rc_iwp = xr.open_dataset(
                f'/{domain}/RACMO_{domain}_bl_qb_IWP_qb_{month}.nc')
            d_rc_lwp = xr.open_dataset(
                f'/{domain}/RACMO_{domain}_bl_qb_LWP_{month}.nc')
            sice = xr.open_dataset(
                f'/{domain}/'
                f'RACMO2.4p1_v4_bl_qb/Monthly/'
                f'siconca_monthlyA_{domain}_RACMO2.4p1_v4_bl_qb_202408_202604.nc').sel(
                time=month[:4]+'-'+month[4:],
                method='nearest')
        ice_sheet_mask = topo.Icesheet_Only.values.copy()
        ice_sheet_mask[ice_sheet_mask==0] = np.nan
        land_mask = topo.LSM.values.copy()
        land_mask[land_mask==0] = np.nan
        ocean_mask = 1-topo.LSM.values.copy()
        ocean_mask[ocean_mask==0] = np.nan
        sice_mask = sice.siconca.values.copy()
        sice_mask[sice_mask>=0.15] = 1
        sice_mask[sice_mask<0.15] = np.nan
        sice_mask_inv = sice.siconca.values.copy()
        sice_mask_inv[sice_mask_inv>=0.15] = np.nan
        sice_mask_inv[sice_mask_inv<0.15] = 1
        ec_iwp[i,0] = np.nansum(d_ec_iwp.cond_result.values)/np.nansum(d_ec_iwp.overpasses.values)
        co_iwp[i,0] = np.nansum(d_co_iwp.cond_result.values)/np.nansum(d_co_iwp.overpasses.values)
        ec_lwp[i,0] = np.nansum(d_ec_lwp.cond_result.values)/np.nansum(d_ec_lwp.overpasses.values)
        rc_iwp[i,0] = np.nansum(d_rc_iwp.cond_result.values)/np.nansum(d_rc_iwp.overpasses.values)
        rc_lwp[i,0] = np.nansum(d_rc_lwp.cond_result.values)/np.nansum(d_rc_lwp.overpasses.values)
        ec_iwp[i,1] = np.nansum(d_ec_iwp.cond_result.values*ice_sheet_mask)/np.nansum(d_ec_iwp.overpasses.values*ice_sheet_mask)
        co_iwp[i,1] = np.nansum(d_co_iwp.cond_result.values*ice_sheet_mask)/np.nansum(d_co_iwp.overpasses.values*ice_sheet_mask)
        ec_lwp[i,1] = np.nansum(d_ec_lwp.cond_result.values*ice_sheet_mask)/np.nansum(d_ec_lwp.overpasses.values*ice_sheet_mask)
        rc_iwp[i,1] = np.nansum(d_rc_iwp.cond_result.values*ice_sheet_mask)/np.nansum(d_rc_iwp.overpasses.values*ice_sheet_mask)
        rc_lwp[i,1] = np.nansum(d_rc_lwp.cond_result.values*ice_sheet_mask)/np.nansum(d_rc_lwp.overpasses.values*ice_sheet_mask)
        ec_iwp[i,3] = np.nansum(d_ec_iwp.cond_result.values*ocean_mask*sice_mask_inv)/np.nansum(d_ec_iwp.overpasses.values*ocean_mask*sice_mask_inv)
        co_iwp[i,3] = np.nansum(d_co_iwp.cond_result.values*ocean_mask*sice_mask_inv)/np.nansum(d_co_iwp.overpasses.values*ocean_mask*sice_mask_inv)
        ec_lwp[i,3] = np.nansum(d_ec_lwp.cond_result.values*ocean_mask*sice_mask_inv)/np.nansum(d_ec_lwp.overpasses.values*ocean_mask*sice_mask_inv)
        rc_iwp[i,3] = np.nansum(d_rc_iwp.cond_result.values*ocean_mask*sice_mask_inv)/np.nansum(d_rc_iwp.overpasses.values*ocean_mask*sice_mask_inv)
        rc_lwp[i,3] = np.nansum(d_rc_lwp.cond_result.values*ocean_mask*sice_mask_inv)/np.nansum(d_rc_lwp.overpasses.values*ocean_mask*sice_mask_inv)
        ec_iwp[i,4] = np.nansum(d_ec_iwp.cond_result.values*sice_mask)/np.nansum(d_ec_iwp.overpasses.values*sice_mask)
        co_iwp[i,4] = np.nansum(d_co_iwp.cond_result.values*sice_mask)/np.nansum(d_co_iwp.overpasses.values*sice_mask)
        ec_lwp[i,4] = np.nansum(d_ec_lwp.cond_result.values*sice_mask)/np.nansum(d_ec_lwp.overpasses.values*sice_mask)
        rc_iwp[i,4] = np.nansum(d_rc_iwp.cond_result.values*sice_mask)/np.nansum(d_rc_iwp.overpasses.values*sice_mask)
        rc_lwp[i,4] = np.nansum(d_rc_lwp.cond_result.values*sice_mask)/np.nansum(d_rc_lwp.overpasses.values*sice_mask)
    results[domain] = [ec_iwp, co_iwp, ec_lwp, rc_iwp, rc_lwp]
labels = ['Greenland','Antarctica']
cmap = ccm.batlow_r
colors = cmap(np.linspace(0,1,10))
fig, axes = plt.subplots(
    2,3,
    figsize=(14,7),
    sharex=True,
    sharey=True)
plt.subplots_adjust(wspace=0.05,hspace=0.15)
titles = ['Ice sheet only',
          'Open ocean only',
          'Sea ice only']
letters = [['(a)', '(b)', '(c)'],
           ['(d)', '(e)', '(f)']]
plotcols=[1,3,4]
for row,domain in enumerate(domains):
    ec_iwp,co_iwp,ec_lwp,rc_iwp,rc_lwp = results[domain]
    for col,idx in enumerate(plotcols):
        ax=axes[row,col]
        ax.plot(months,ec_iwp[:,idx],
                color=colors[1],linewidth=2)
        ax.plot(months,co_iwp[:,idx],
                color=colors[5],linewidth=2)
        ax.plot(months,rc_iwp[:,idx],
                color=colors[-1],linewidth=2)
        if row==0:
            ax.set_title(titles[col])
            ax.tick_params(bottom=False)
        if col==0:
            ax.set_ylabel('IWP [kg m$^{-2}$]')
            ax.text(-0.25, 0.5,
                    labels[row],
                    rotation=90,
                    va='center',
                    ha='center',
                    transform=ax.transAxes,
                    fontsize=18)
        else:
            ax.tick_params(left=False)
        ax.grid(color='grey',
                linestyle='dashed')
        ax.set_ylim(0,0.5)
        ax.tick_params(axis='x',rotation=90)
        ax.set_title(letters[row][col], loc='left')
fig.legend(['ACM-CAP','ACM-COM','RACMO'],
            frameon=False,
            bbox_to_anchor=(0.5,-0.03),
            ncol=3)
axes[1,0].set_xticks(months[::3], months[::3], rotation=90)
axes[1,1].set_xticks(months[::3], months[::3], rotation=90)
axes[1,2].set_xticks(months[::3], months[::3], rotation=90)
plt.savefig(f'GrIS_Ant_ACM_CAP_COM_IWP.png', 
            dpi=300, bbox_inches='tight', format='png')
#%% make map with temperature and sea ice cover
tas_g = xr.open_dataset('/FGRN055/RACMO2.4p1_v4_bl/Monthly/tas_monthlyA_FGRN055_RACMO2.4p1_v4_bl_202406_202604.nc').sel(time=slice('2024-09-01','2026-05-01'))
tas_a = xr.open_dataset('/PXANT11/RACMO2.4p1_v4_bl_qb/Monthly/tas_monthlyA_PXANT11_RACMO2.4p1_v4_bl_qb_202408_202604.nc').sel(time=slice('2024-09-01','2026-05-01'))
pr_g = xr.open_dataset('/FGRN055/RACMO2.4p1_v4_bl/Monthly/pr_monthlyS_FGRN055_RACMO2.4p1_v4_bl_202406_202604.nc').sel(time=slice('2024-09-01','2026-05-01'))
pr_a = xr.open_dataset('/PXANT11/RACMO2.4p1_v4_bl_qb/Monthly/pr_monthlyS_PXANT11_RACMO2.4p1_v4_bl_qb_202408_202604.nc').sel(time=slice('2024-09-01','2026-05-01'))
siconca_g = xr.open_dataset(f'/FGRN055/RACMO2.4p1_v4_bl/Monthly/siconca_monthlyA_FGRN055_RACMO2.4p1_v4_bl_202406_202604.nc').sel(time=slice('2024-09-01','2026-05-01'))
siconca_a = xr.open_dataset(f'/PXANT11/RACMO2.4p1_v4_bl_qb/Monthly/siconca_monthlyA_PXANT11_RACMO2.4p1_v4_bl_qb_202408_202604.nc').sel(time=slice('2024-09-01','2026-05-01'))
tas_g_DJF = tas_g.copy().sel(time=tas_g.time.dt.season == "DJF").tas.mean(dim=['height','time']).values
tas_g_JJA = tas_g.copy().sel(time=tas_g.time.dt.season == "JJA").tas.mean(dim=['height','time']).values
tas_a_DJF = tas_a.copy().sel(time=tas_a.time.dt.season == "DJF").tas.mean(dim=['height','time']).values
tas_a_JJA = tas_a.copy().sel(time=tas_a.time.dt.season == "JJA").tas.mean(dim=['height','time']).values
pr_g_ANN = pr_g.copy().pr.mean(dim=['height','time'])*12
pr_a_ANN = pr_a.copy().pr.mean(dim=['height','time'])*12
siconca_g_DJF = siconca_g.copy().sel(time=siconca_g.time.dt.season == "DJF").siconca.mean(dim=['height','time']).values
siconca_g_JJA = siconca_g.copy().sel(time=siconca_g.time.dt.season == "JJA").siconca.mean(dim=['height','time']).values
siconca_a_DJF = siconca_a.copy().sel(time=siconca_a.time.dt.season == "DJF").siconca.mean(dim=['height','time']).values
siconca_a_JJA = siconca_a.copy().sel(time=siconca_a.time.dt.season == "JJA").siconca.mean(dim=['height','time']).values
lat_g = tas_g.lat.values
lon_g = tas_g.lon.values
lat_a = tas_a.lat.values
lon_a = tas_a.lon.values

topo_g = xr.open_dataset(f'FGRN055_masks.nc', engine='netcdf4')
topo_a = xr.open_dataset(f'PXANT11_masks.nc', engine='netcdf4')
siconca_a_DJF[topo_a.LSM.values > 0] = 0
siconca_a_JJA[topo_a.LSM.values > 0] = 0
siconca_g_DJF[topo_g.LSM.values > 0] = 0
siconca_g_JJA[topo_g.LSM.values > 0] = 0
cmap = 'RdYlBu_r'
cmap2 = ccm.batlow_r
projection_a=crs.RotatedPole(pole_latitude=185, pole_longitude=20.0,
                        central_rotated_longitude=0)
MapLL_TR_lat_a = np.array([  -44.884, -44.034])
MapLL_TR_lon_a = np.array([ -109.982, 62.260])

projection_g=crs.RotatedPole(pole_latitude=-18, pole_longitude=-37.5,
                        central_rotated_longitude=0)
MapLL_TR_lat_g = np.array([  56.119, 77.922])  
MapLL_TR_lon_g = np.array([ -55.15, 34.106])
xy_g = projection_g.transform_points(
    crs.PlateCarree(),
    lon_g,
    lat_g)
x_g = xy_g[..., 0]
y_g = xy_g[..., 1]
xy_a = projection_a.transform_points(
    crs.PlateCarree(),
    lon_a,
    lat_a)
x_a = xy_a[..., 0]
y_a = xy_a[..., 1]
fig = plt.figure(figsize=(15,10))
gs = gridspec.GridSpec(2, 3, hspace=0.0)
ax1 = fig.add_subplot(gs[0, 0], projection=projection_g,
                    frameon=False)
ax2 = fig.add_subplot(gs[0, 1], projection=projection_g,
                    frameon=False)
ax3 = fig.add_subplot(gs[0, 2], projection=projection_g,
                    frameon=False)
ax4 = fig.add_subplot(gs[1, 0], projection=projection_a,
                    frameon=False)
ax5 = fig.add_subplot(gs[1, 1], projection=projection_a,
                    frameon=False)
ax6 = fig.add_subplot(gs[1, 2], projection=projection_a,
                    frameon=False)
MapXextent_g, MapYextent_g, _ = projection_g.transform_points(crs.PlateCarree(), 
                                                            MapLL_TR_lon_g, MapLL_TR_lat_g).T
MapXextent_a, MapYextent_a, _ = projection_a.transform_points(crs.PlateCarree(), 
                                                            MapLL_TR_lon_a, MapLL_TR_lat_a).T
ax1.set_xlim([MapXextent_g[0], MapXextent_g[1]])
ax1.set_ylim([MapYextent_g[0], MapYextent_g[1]])
ax2.set_xlim([MapXextent_g[0], MapXextent_g[1]])
ax2.set_ylim([MapYextent_g[0], MapYextent_g[1]])
ax3.set_xlim([MapXextent_g[0], MapXextent_g[1]])
ax3.set_ylim([MapYextent_g[0], MapYextent_g[1]])
ax4.set_xlim([MapXextent_a[0], MapXextent_a[1]])
ax4.set_ylim([MapYextent_a[0], MapYextent_a[1]])
ax5.set_xlim([MapXextent_a[0], MapXextent_a[1]])
ax5.set_ylim([MapYextent_a[0], MapYextent_a[1]])
ax6.set_xlim([MapXextent_a[0], MapXextent_a[1]])
ax6.set_ylim([MapYextent_a[0], MapYextent_a[1]])
plot = ax1.pcolormesh(x_g, y_g, tas_g_DJF, shading='auto',
                norm=mcolors.Normalize(200,290), cmap=cmap, rasterized=True)
ax1.add_feature(ccrs.COASTLINE, zorder=10, 
                    edgecolor='black')
siplot = ax1.contour(lon_g, lat_g, siconca_g_DJF, colors='black',
                        transform=crs.PlateCarree(), levels=[0.15], 
                        linewidths=1)#, linestyles='dashed')
sihatch = ax1.contourf(lon_g, lat_g, siconca_g_DJF, transform=crs.PlateCarree(), 
                        levels=[0,0.15,1], colors='none', 
                        hatches=['', 'xx'], alpha=0)
plot = ax2.pcolormesh(x_g,y_g, tas_g_JJA, shading='auto',
                norm=mcolors.Normalize(200,290), cmap=cmap, rasterized=True)
ax2.add_feature(ccrs.COASTLINE, zorder=10, 
                    edgecolor='black')
siplot = ax2.contour(lon_g, lat_g, siconca_g_JJA, colors='black',
                        transform=crs.PlateCarree(), levels=[0.15], 
                        linewidths=1)#, linestyles='dashed')
sihatch = ax2.contourf(lon_g, lat_g, siconca_g_JJA, transform=crs.PlateCarree(), 
                        levels=[0,0.15,1], colors='none', 
                        hatches=['', 'xx'], alpha=0)
plot = ax3.pcolormesh(x_g,y_g, pr_g_ANN, shading='auto',
                norm=mcolors.Normalize(0,2000), cmap=cmap2, rasterized=True)
ax3.add_feature(ccrs.COASTLINE, zorder=10, 
                    edgecolor='black')
gl = ax1.gridlines(draw_labels=True, dms=True, 
                    linewidth=2, color='gray', alpha=0.5, linestyle='--',
                    zorder=11, x_inline=False,y_inline=False, 
                    rotate_labels=0, xpadding=10)
gl.xlabel_style = {'size': 10}
gl.ylabel_style = {'size': 10}
gl.top_labels = False
gl.left_labels = False
ax1.set_title('DJF')
gl = ax2.gridlines(draw_labels=True, dms=True, 
                    linewidth=2, color='gray', alpha=0.5, linestyle='--',
                    zorder=11, x_inline=False,y_inline=False, 
                    rotate_labels=0, xpadding=10)
gl.xlabel_style = {'size': 10}
gl.ylabel_style = {'size': 10}
gl.top_labels = False
gl.left_labels = False
ax2.set_title('JJA')
gl = ax3.gridlines(draw_labels=True, dms=True, 
                    linewidth=2, color='gray', alpha=0.5, linestyle='--',
                    zorder=11, x_inline=False,y_inline=False, 
                    rotate_labels=0, xpadding=10)
gl.xlabel_style = {'size': 10}
gl.ylabel_style = {'size': 10}
gl.top_labels = False
gl.left_labels = False
ax3.set_title('Annual')
ax1.set_title('(a)', loc='left')
ax2.set_title('(b)', loc='left')
ax3.set_title('(c)', loc='left')
reader = shpreader.Reader("ne_50m_antarctic_ice_shelves_lines.shp")
geometries = []
for record in reader.records():
    geometries.append(record.geometry)
shape_feature = ShapelyFeature(geometries, crs.PlateCarree(), facecolor='none', edgecolor='black', lw=1)
plot = ax4.pcolormesh(x_a, y_a, tas_a_DJF, shading='auto',
                norm=mcolors.Normalize(200,290), cmap=cmap, rasterized=True)
ax4.add_feature(ccrs.COASTLINE, zorder=10, 
                    edgecolor='black')
ax4.add_feature(shape_feature)
lat_sic = siconca_a.lat.values
lon_sic = siconca_a.lon.values
xy_sic = projection_a.transform_points(crs.PlateCarree(), lon_sic, lat_sic)
x_sic, y_sic = xy_sic[..., 0], xy_sic[..., 1]
ax4.contour(x_sic, y_sic, siconca_a_DJF,
            colors='black', levels=[0.15])
ax4.contourf(x_sic, y_sic, siconca_a_DJF,
            levels=[0, 0.15, 1],
            colors='none', hatches=['', 'xx'], alpha=0)
plot = ax5.pcolormesh(x_a,y_a, tas_a_JJA, shading='auto',
                norm=mcolors.Normalize(200,290), cmap=cmap, rasterized=True)
ax5.add_feature(ccrs.COASTLINE, zorder=10, 
                    edgecolor='black')
ax5.add_feature(shape_feature)
ax5.contour(x_sic, y_sic, siconca_a_JJA,
            colors='black', levels=[0.15])
ax5.contourf(x_sic, y_sic, siconca_a_JJA,
            levels=[0, 0.15, 1],
            colors='none', hatches=['', 'xx'], alpha=0)
plot2 = ax6.pcolormesh(x_a,y_a, pr_a_ANN, shading='auto',
                norm=mcolors.Normalize(0,2000), cmap=cmap2, rasterized=True)
ax6.add_feature(ccrs.COASTLINE, zorder=10, 
                    edgecolor='black')
ax6.add_feature(shape_feature)
gl = ax4.gridlines(draw_labels=True, dms=True, 
                    linewidth=2, color='gray', alpha=0.5, linestyle='--',
                    zorder=11, x_inline=False,y_inline=False, 
                    rotate_labels=0, xpadding=10)
gl.xlabel_style = {'size': 10}
gl.ylabel_style = {'size': 10}
gl.top_labels = False
gl.left_labels = False
gl = ax5.gridlines(draw_labels=True, dms=True, 
                    linewidth=2, color='gray', alpha=0.5, linestyle='--',
                    zorder=11, x_inline=False,y_inline=False, 
                    rotate_labels=0, xpadding=10)
gl.xlabel_style = {'size': 10}
gl.ylabel_style = {'size': 10}
gl.top_labels = False
gl.left_labels = False
gl = ax6.gridlines(draw_labels=True, dms=True, 
                    linewidth=2, color='gray', alpha=0.5, linestyle='--',
                    zorder=11, x_inline=False,y_inline=False, 
                    rotate_labels=0, xpadding=10)
gl.xlabel_style = {'size': 10}
gl.ylabel_style = {'size': 10}
gl.top_labels = False
gl.left_labels = False
ax4.set_title('DJF')
ax5.set_title('JJA')
ax6.set_title('Annual')
ax4.set_title('(d)', loc='left')
ax5.set_title('(e)', loc='left')
ax6.set_title('(f)', loc='left')

cbar_ax = fig.add_axes([0.125, 0.0, 0.5, 0.015])
cb = fig.colorbar(plot, cax=cbar_ax, 
                    label=f'2m air temperature [K]', extend='both',
                    orientation="horizontal")
cb.ax.minorticks_off()
cbar_ax = fig.add_axes([0.67, 0.0, 0.23, 0.015])
cb = fig.colorbar(plot2, cax=cbar_ax, 
                    label='Precipitation [mm yr$^{-1}$]', extend='both',
                    orientation="horizontal")
cb.ax.minorticks_off()
fig.subplots_adjust(bottom=0.01)
plt.savefig(f'GrIS_Ant_temp_2m_pr_sea_ice.png', dpi=300, bbox_inches='tight', format='png')
#%%
ds_rac_g = xr.open_dataset(f'/FGRN055/merged/RACMO_FGRN055_bl_ice_occ_2D_combined.nc')
samples_g, lat_new_g, lon_new_g = spatial_regridding.downsample_grid(ds_rac_g, 'overpasses', 20, sample_type='sum')
samples_g[samples_g == 0] = np.nan
overpasses_g, lat_new_g, lon_new_g = spatial_regridding.downsample_grid(ds_rac_g, 'overpasses', 20, sample_type='mean')
overpasses_g *= 20 # number of months
overpasses_g[overpasses_g == 0] = np.nan
ds_rac_a = xr.open_dataset(f'/PXANT11/merged/RACMO_PXANT11_bl_qb_ice_occ_2D_combined.nc')
samples_a, lat_new_a, lon_new_a = spatial_regridding.downsample_grid(ds_rac_a, 'overpasses', 20, sample_type='sum')
samples_a[samples_a == 0] = np.nan
overpasses_a, lat_new_a, lon_new_a = spatial_regridding.downsample_grid(ds_rac_a, 'overpasses', 20, sample_type='mean')
overpasses_a *= 20 # number of months
overpasses_a[overpasses_a == 0] = np.nan

projection_a=crs.RotatedPole(pole_latitude=185, pole_longitude=20.0,
                        central_rotated_longitude=0)
MapLL_TR_lat_a = np.array([  -44.884, -44.034])
MapLL_TR_lon_a = np.array([ -109.982, 62.260])

projection_g=crs.RotatedPole(pole_latitude=-18, pole_longitude=-37.5,
                        central_rotated_longitude=0)
MapLL_TR_lat_g = np.array([  56.119, 77.922])  
MapLL_TR_lon_g = np.array([ -55.15, 34.106])
xy_g = projection_g.transform_points(
    crs.PlateCarree(),
    lon_new_g,
    lat_new_g)
x_g = xy_g[..., 0]
y_g = xy_g[..., 1]
xy_a = projection_a.transform_points(
    crs.PlateCarree(),
    lon_new_a,
    lat_new_a)
x_a = xy_a[..., 0]
y_a = xy_a[..., 1]
cmap = ccm.batlow_r
fig = plt.figure(figsize=(10,10))
gs = gridspec.GridSpec(2, 2, hspace=0.0)
ax1 = fig.add_subplot(gs[0, 0], projection=projection_g,
                    frameon=False)
ax2 = fig.add_subplot(gs[0, 1], projection=projection_g,
                    frameon=False)
ax3 = fig.add_subplot(gs[1, 0], projection=projection_a,
                    frameon=False)
ax4 = fig.add_subplot(gs[1, 1], projection=projection_a,
                    frameon=False)
MapXextent_g, MapYextent_g, _ = projection_g.transform_points(crs.PlateCarree(), 
                                                            MapLL_TR_lon_g, MapLL_TR_lat_g).T
MapXextent_a, MapYextent_a, _ = projection_a.transform_points(crs.PlateCarree(), 
                                                            MapLL_TR_lon_a, MapLL_TR_lat_a).T
ax1.set_xlim([MapXextent_g[0], MapXextent_g[1]])
ax1.set_ylim([MapYextent_g[0], MapYextent_g[1]])
ax2.set_xlim([MapXextent_g[0], MapXextent_g[1]])
ax2.set_ylim([MapYextent_g[0], MapYextent_g[1]])
ax3.set_xlim([MapXextent_a[0], MapXextent_a[1]])
ax3.set_ylim([MapYextent_a[0], MapYextent_a[1]])
ax4.set_xlim([MapXextent_a[0], MapXextent_a[1]])
ax4.set_ylim([MapYextent_a[0], MapYextent_a[1]])

plot = ax1.pcolormesh(x_g, y_g, samples_g, shading='auto',
                norm=mcolors.Normalize(0,25000), cmap=cmap, rasterized=True)
ax1.add_feature(ccrs.COASTLINE, zorder=10, 
                    edgecolor='black')
plot = ax2.pcolormesh(x_g,y_g, overpasses_g, shading='auto',
                norm=mcolors.Normalize(0,2500), cmap=cmap, rasterized=True)
ax2.add_feature(ccrs.COASTLINE, zorder=10, 
                    edgecolor='black')
gl = ax1.gridlines(draw_labels=True, dms=True, 
                    linewidth=2, color='gray', alpha=0.5, linestyle='--',
                    zorder=11, x_inline=False,y_inline=False, 
                    rotate_labels=0, xpadding=10)
gl.xlabel_style = {'size': 10}
gl.ylabel_style = {'size': 10}
gl.top_labels = False
gl.left_labels = False
gl = ax2.gridlines(draw_labels=True, dms=True, 
                    linewidth=2, color='gray', alpha=0.5, linestyle='--',
                    zorder=11, x_inline=False,y_inline=False, 
                    rotate_labels=0, xpadding=10)
gl.xlabel_style = {'size': 10}
gl.ylabel_style = {'size': 10}
gl.top_labels = False
gl.left_labels = False
ax1.set_title('(a)', loc='left')
ax2.set_title('(b)', loc='left')

reader = shpreader.Reader("ne_50m_antarctic_ice_shelves_lines.shp")
geometries = []
for record in reader.records():
    geometries.append(record.geometry)
shape_feature = ShapelyFeature(geometries, crs.PlateCarree(), facecolor='none', edgecolor='black', lw=1)
plot = ax3.pcolormesh(x_a, y_a, samples_a, shading='auto',
                norm=mcolors.Normalize(0,25000), cmap=cmap, rasterized=True)
ax3.add_feature(ccrs.COASTLINE, zorder=10, 
                    edgecolor='black')
ax3.add_feature(shape_feature)
plot2 = ax4.pcolormesh(x_a,y_a, overpasses_a, shading='auto',
                norm=mcolors.Normalize(0,2500), cmap=cmap, rasterized=True)
ax4.add_feature(ccrs.COASTLINE, zorder=10, 
                    edgecolor='black')
ax4.add_feature(shape_feature)
gl = ax3.gridlines(draw_labels=True, dms=True, 
                    linewidth=2, color='gray', alpha=0.5, linestyle='--',
                    zorder=11, x_inline=False,y_inline=False, 
                    rotate_labels=0, xpadding=10)
gl.xlabel_style = {'size': 10}
gl.ylabel_style = {'size': 10}
gl.top_labels = False
gl.left_labels = False
gl = ax4.gridlines(draw_labels=True, dms=True, 
                    linewidth=2, color='gray', alpha=0.5, linestyle='--',
                    zorder=11, x_inline=False,y_inline=False, 
                    rotate_labels=0, xpadding=10)
gl.xlabel_style = {'size': 10}
gl.ylabel_style = {'size': 10}
gl.top_labels = False
gl.left_labels = False
ax3.set_title('(c)', loc='left')
ax4.set_title('(d)', loc='left')
cbar_ax = fig.add_axes([0.125, 0.0, 0.35, 0.015])
cb = fig.colorbar(plot, cax=cbar_ax, 
                    label=f'Number of data points [-]', extend='max',
                    orientation="horizontal")
cb.ax.minorticks_off()
cbar_ax = fig.add_axes([0.55, 0.0, 0.35, 0.015])
cb = fig.colorbar(plot2, cax=cbar_ax, 
                    label='Number of overpasses [-]', extend='max',
                    orientation="horizontal")
cb.ax.minorticks_off()
fig.subplots_adjust(bottom=0.01)
plt.savefig(f'overpasses.png', dpi=300, bbox_inches='tight', format='png')
