#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
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

#%%
ds_rac_i = xr.open_mfdataset([f'FGRN055/RACMO2.4p1_v4_bl-2024/NC_1H/clivi.KNMI-2024.FGRN055.RACMO2.4p1_v4_bl.1H.nc',
                              f'FGRN055/RACMO2.4p1_v4_bl-2025/NC_1H/clivi.KNMI-2025.FGRN055.RACMO2.4p1_v4_bl.1H.nc',
                              f'FGRN055/RACMO2.4p1_v4_bl-2026/NC_1H/clivi.KNMI-2026.FGRN055.RACMO2.4p1_v4_bl.1H.nc'],
                              concat_dim='time', combine='nested').squeeze().sel(time=slice('2024-09-01', '2026-04-30'))
ds_rac_s = xr.open_mfdataset([f'FGRN055/RACMO2.4p1_v4_bl-2024/NC_1H/tcsw.KNMI-2024.FGRN055.RACMO2.4p1_v4_bl.1H.nc',
                              f'FGRN055/RACMO2.4p1_v4_bl-2025/NC_1H/tcsw.KNMI-2025.FGRN055.RACMO2.4p1_v4_bl.1H.nc',
                              f'FGRN055/RACMO2.4p1_v4_bl-2026/NC_1H/tcsw.KNMI-2026.FGRN055.RACMO2.4p1_v4_bl.1H.nc'],
                              concat_dim='time', combine='nested').squeeze().sel(time=slice('2024-09-01', '2026-04-30'))
res_new_rac_i = spatial_regridding.downsample_grid(ds_rac_i.copy().mean('time'), 'clivi', 20, sample_type='mean')[0]
res_new_rac_s = spatial_regridding.downsample_grid(ds_rac_s.copy().mean('time'), 'tcsw', 20, sample_type='mean')[0]
res_new_rac_g = res_new_rac_i + res_new_rac_s
res_new_rac_g[res_new_rac_g == 0] = np.nan
lon = ds_rac_i.lon.values
UTCm = np.round(lon / 15 - 2) % 24
UTCa = np.round(lon / 15 - 14) % 24
hour = ds_rac_i.time.dt.hour
hour = hour.values[:, None, None]
UTCm = UTCm[None, :, :]
UTCa = UTCa[None, :, :]
mask = (hour == UTCm) | (hour == UTCa)
clivi = ds_rac_i.clivi.where(mask)
ds_rac_i["clivi"] = clivi
tcsw = ds_rac_s.tcsw.where(mask)
ds_rac_s["tcsw"] = tcsw
ds_rac_i.to_netcdf('FGRN055_clivi_202409_202604.nc')
ds_rac_s.to_netcdf('FGRN055_tcsw_202409_202604.nc')
res_time_rac_i = spatial_regridding.downsample_grid(ds_rac_i.copy().mean('time'), 'clivi', 20, sample_type='mean')[0]
res_time_rac_s = spatial_regridding.downsample_grid(ds_rac_s.copy().mean('time'), 'tcsw', 20, sample_type='mean')[0]
res_time_rac_g = res_time_rac_i + res_time_rac_s
res_time_rac_g[res_time_rac_g == 0] = np.nan
ds_rac_g = xr.open_dataset(f'RACMO_FGRN055_bl_IWP_combined.nc')
res_new_rac_ec_g, lat_new_g, lon_new_g= spatial_regridding.downsample_grid(ds_rac_g, 'mean', 20, sample_type='mean')
res_new_rac_ec_g[res_new_rac_ec_g == 0] = np.nan

ds_rac_i = xr.open_mfdataset([f'PXANT11/RACMO2.4p1_v4_bl_qb-2024/NC_1H/clivi.KNMI-2024.PXANT11.RACMO2.4p1_v4_bl_qb.1H.nc',
                              f'PXANT11/RACMO2.4p1_v4_bl_qb-2025/NC_1H/clivi.KNMI-2025.PXANT11.RACMO2.4p1_v4_bl_qb.1H.nc',
                              f'PXANT11/RACMO2.4p1_v4_bl_qb-2026/NC_1H/clivi.KNMI-2026.PXANT11.RACMO2.4p1_v4_bl_qb.1H.nc'],
                              concat_dim='time', combine='nested').squeeze().sel(time=slice('2024-09-01', '2026-04-30'))
ds_rac_s = xr.open_mfdataset([f'PXANT11/RACMO2.4p1_v4_bl_qb-2024/NC_1H/tcsw.KNMI-2024.PXANT11.RACMO2.4p1_v4_bl_qb.1H.nc',
                              f'PXANT11/RACMO2.4p1_v4_bl_qb-2025/NC_1H/tcsw.KNMI-2025.PXANT11.RACMO2.4p1_v4_bl_qb.1H.nc',
                              f'PXANT11/RACMO2.4p1_v4_bl_qb-2026/NC_1H/tcsw.KNMI-2026.PXANT11.RACMO2.4p1_v4_bl_qb.1H.nc'],
                              concat_dim='time', combine='nested').squeeze().sel(time=slice('2024-09-01', '2026-04-30'))
lon = ds_rac_i.lon.values
UTCm = np.round(lon / 15 - 2) % 24
UTCa = np.round(lon / 15 - 14) % 24
hour = ds_rac_i.time.dt.hour
hour = hour.values[:, None, None]
UTCm = UTCm[None, :, :]
UTCa = UTCa[None, :, :]
mask = (hour == UTCm) | (hour == UTCa)##
tcsw = ds_rac_s.tcsw.where(mask).load()
tcsw_new = xr.DataArray(
    np.asarray(tcsw.values),
    dims=tcsw.dims,
    coords=tcsw.coords,
    name="tcsw",
    attrs=tcsw.attrs)
tcsw_new.to_netcdf("PXANT11_tcsw_202409_202604.nc")
ds_rac_i.to_netcdf('PXANT11_clivi_202409_202604.nc')
ds_rac_s.to_netcdf('PXANT11_tcsw_202409_202604.nc')
res_time_rac_i = spatial_regridding.downsample_grid(ds_rac_i.copy().mean('time'), 'clivi', 20, sample_type='mean')[0]
res_time_rac_s = spatial_regridding.downsample_grid(ds_rac_s.copy().mean('time'), 'tcsw', 20, sample_type='mean')[0]
res_time_rac_a = res_time_rac_i + res_time_rac_s
res_time_rac_a[res_time_rac_a == 0] = np.nan
ds_rac_a = xr.open_dataset(f'RACMO_FGRN055_bl_qb_IWP_combined.nc')
res_new_rac_ec_a, lat_new_a, lon_new_a= spatial_regridding.downsample_grid(ds_rac_a, 'mean', 20, sample_type='mean')
ds_rac_i = xr.open_mfdataset([f'FGRN055/RACMO2.4p1_v4_bl-2024/NC_DEFAULT/clivi.KNMI-2024.FGRN055.RACMO2.4p1_v4_bl.DD.nc',
                              f'FGRN055/RACMO2.4p1_v4_bl-2025/NC_DEFAULT/clivi.KNMI-2025.FGRN055.RACMO2.4p1_v4_bl.DD.nc',
                              f'FGRN055/RACMO2.4p1_v4_bl-2026/NC_DEFAULT/clivi.KNMI-2026.FGRN055.RACMO2.4p1_v4_bl.DD.nc'],
                              concat_dim='time', combine='nested').squeeze().sel(time=slice('2024-09-01', '2026-04-30'))
ds_rac_s = xr.open_mfdataset([f'FGRN055/RACMO2.4p1_v4_bl-2024/NC_DEFAULT/tcsw.KNMI-2024.FGRN055.RACMO2.4p1_v4_bl.DD.nc',
                              f'FGRN055/RACMO2.4p1_v4_bl-2025/NC_DEFAULT/tcsw.KNMI-2025.FGRN055.RACMO2.4p1_v4_bl.DD.nc',
                              f'FGRN055/RACMO2.4p1_v4_bl-2026/NC_DEFAULT/tcsw.KNMI-2026.FGRN055.RACMO2.4p1_v4_bl.DD.nc'],
                              concat_dim='time', combine='nested').squeeze().sel(time=slice('2024-09-01', '2026-04-30'))
res_new_rac_i = spatial_regridding.downsample_grid(ds_rac_i.copy().mean('time'), 'clivi', 20, sample_type='mean')[0]
res_new_rac_s, lat_new_g, lon_new_g = spatial_regridding.downsample_grid(ds_rac_s.copy().mean('time'), 'tcsw', 20, sample_type='mean')
res_new_rac_g = res_new_rac_i + res_new_rac_s
res_new_rac_g[res_new_rac_g == 0] = np.nan
print("loaded RACMO IWP", flush=True)
data = xr.Dataset(
    data_vars=dict(
        result=(["lat_ix", "lon_ix"], res_new_rac_g),
        latitude=(["lat_ix", "lon_ix"], lat_new_g),
        longitude=(["lat_ix", "lon_ix"], lon_new_g)
    ),
    coords=dict(
        lon=("lon_ix", np.arange(0,len(res_new_rac_g[0]),1)),
        lat=("lat_ix", np.arange(0,len(res_new_rac_g),1))))
data.to_netcdf("FGRN055_racmo_iwp_202409_202604.nc")
ds_rac_i = xr.open_dataset('FGRN055_clivi_202409_202604.nc')
ds_rac_s = xr.open_dataset('FGRN055_tcsw_202409_202604.nc')
res_time_rac_i = spatial_regridding.downsample_grid(ds_rac_i.copy().mean('time'), 'clivi', 20, sample_type='mean')[0]
res_time_rac_s = spatial_regridding.downsample_grid(ds_rac_s.copy().mean('time'), 'tcsw', 20, sample_type='mean')[0]
res_time_rac_g = res_time_rac_i + res_time_rac_s
res_time_rac_g[res_time_rac_g == 0] = np.nan
print("loaded time variability", flush=True)
data = xr.Dataset(
    data_vars=dict(
        result=(["lat_ix", "lon_ix"], res_time_rac_g),
        latitude=(["lat_ix", "lon_ix"], lat_new_g),
        longitude=(["lat_ix", "lon_ix"], lon_new_g)
    ),
    coords=dict(
        lon=("lon_ix", np.arange(0,len(res_time_rac_g[0]),1)),
        lat=("lat_ix", np.arange(0,len(res_time_rac_g),1))))
data.to_netcdf("FGRN055_racmo_iwp_time_202409_202604.nc")
ds_rac_g = xr.open_dataset(f'RACMO_FGRN055_bl_IWP_combined.nc')
res_new_rac_ec_g, lat_new_g, lon_new_g = spatial_regridding.downsample_grid(ds_rac_g, 'mean', 20, sample_type='mean')
res_new_rac_ec_g[res_new_rac_ec_g == 0] = np.nan
print("loaded EC tracked RACMO", flush=True)
ds_rac_i = xr.open_mfdataset([f'PXANT11/RACMO2.4p1_v4_bl_qb-2024/NC_DEFAULT/clivi.KNMI-2024.PXANT11.RACMO2.4p1_v4_bl_qb.DD.nc',
                             f'RACMO2.4p1_v4_bl_qb-2025/NC_DEFAULT/clivi.KNMI-2025.PXANT11.RACMO2.4p1_v4_bl_qb.DD.nc',
                              f'PXANT11/RACMO2.4p1_v4_bl_qb-2026/NC_DEFAULT/clivi.KNMI-2026.PXANT11.RACMO2.4p1_v4_bl_qb.DD.nc'],
                              concat_dim='time', combine='nested').squeeze().sel(time=slice('2024-09-01', '2026-04-30'))
ds_rac_s = xr.open_mfdataset([f'PXANT11/RACMO2.4p1_v4_bl_qb-2024/NC_DEFAULT/tcsw.KNMI-2024.PXANT11.RACMO2.4p1_v4_bl_qb.DD.nc',
                              f'PXANT11/RACMO2.4p1_v4_bl_qb-2025/NC_DEFAULT/tcsw.KNMI-2025.PXANT11.RACMO2.4p1_v4_bl_qb.DD.nc',
                              f'PXANT11/RACMO2.4p1_v4_bl_qb-2026/NC_DEFAULT/tcsw.KNMI-2026.PXANT11.RACMO2.4p1_v4_bl_qb.DD.nc'],
                              concat_dim='time', combine='nested').squeeze().sel(time=slice('2024-09-01', '2026-04-30'))
res_new_rac_i = spatial_regridding.downsample_grid(ds_rac_i.copy().mean('time'), 'clivi', 20, sample_type='mean')[0]
res_new_rac_s, lat_new_a, lon_new_a = spatial_regridding.downsample_grid(ds_rac_s.copy().mean('time'), 'tcsw', 20, sample_type='mean')
res_new_rac_a = res_new_rac_i + res_new_rac_s
res_new_rac_a[res_new_rac_a == 0] = np.nan
print("loaded RACMO IWP", flush=True)
data = xr.Dataset(
    data_vars=dict(
        result=(["lat_ix", "lon_ix"], res_new_rac_a),
        latitude=(["lat_ix", "lon_ix"], lat_new_a),
        longitude=(["lat_ix", "lon_ix"], lon_new_a)
    ),
    coords=dict(
        lon=("lon_ix", np.arange(0,len(res_new_rac_a[0]),1)),
        lat=("lat_ix", np.arange(0,len(res_new_rac_a),1))))
data.to_netcdf("PXANT11_racmo_iwp_202409_202604.nc")
ds_rac_i = xr.open_dataset('PXANT11_clivi_202409_202604.nc')
ds_rac_s = xr.open_dataset('PXANT11_tcsw_202409_202604.nc')
res_time_rac_i = spatial_regridding.downsample_grid(ds_rac_i.copy().mean('time'), 'clivi', 20, sample_type='mean')[0]
res_time_rac_s = spatial_regridding.downsample_grid(ds_rac_s.copy().mean('time'), 'tcsw', 20, sample_type='mean')[0]
res_time_rac_a = res_time_rac_i + res_time_rac_s
res_time_rac_a[res_time_rac_a == 0] = np.nan
print("loaded time variability", flush=True)
data = xr.Dataset(
    data_vars=dict(
        result=(["lat_ix", "lon_ix"], res_time_rac_a),
        latitude=(["lat_ix", "lon_ix"], lat_new_a),
        longitude=(["lat_ix", "lon_ix"], lon_new_a)
    ),
    coords=dict(
        lon=("lon_ix", np.arange(0,len(res_time_rac_a[0]),1)),
        lat=("lat_ix", np.arange(0,len(res_time_rac_a),1))))
data.to_netcdf("PXANT11_racmo_iwp_time_202409_202604.nc")
 
#%%
res_new_rac_g = xr.open_dataset("FGRN055_racmo_iwp_202409_202604.nc").result.values
res_time_rac_g = xr.open_dataset("FGRN055_racmo_iwp_time_202409_202604.nc").result.values
ds_rac_g = xr.open_dataset(f'RACMO_FGRN055_bl_IWP_combined.nc')
res_new_rac_ec_g, lat_new_g, lon_new_g = spatial_regridding.downsample_grid(ds_rac_g, 'mean', 20, sample_type='mean')
res_new_rac_ec_g[res_new_rac_ec_g == 0] = np.nan
print("loaded EC tracked RACMO", flush=True)
#%%
res_new_rac_a = xr.open_dataset("PXANT11_racmo_iwp_202409_202604.nc").result.values
res_time_rac_a = xr.open_dataset("PXANT11_racmo_iwp_time_202409_202604.nc").result.values
ds_rac_a = xr.open_dataset(f'RACMO_PXANT11_bl_qb_IWP_combined.nc')
res_new_rac_ec_a, lat_new_a, lon_new_a= spatial_regridding.downsample_grid(ds_rac_a, 'mean', 20, sample_type='mean')
res_new_rac_ec_a[res_new_rac_ec_a == 0] = np.nan
print("loaded EC tracked RACMO", flush=True)

#%%
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
fig = plt.figure(figsize=(18.5,9))
gs = gridspec.GridSpec(2, 4, hspace=0.0)
ax1 = fig.add_subplot(gs[0, 0], projection=projection_g,
                    frameon=False)
ax2 = fig.add_subplot(gs[0, 1], projection=projection_g,
                    frameon=False)
ax3 = fig.add_subplot(gs[0, 2], projection=projection_g,
                    frameon=False)
ax4 = fig.add_subplot(gs[0, 3], projection=projection_g,
                    frameon=False)
ax5 = fig.add_subplot(gs[1, 0], projection=projection_a,
                    frameon=False)
ax6 = fig.add_subplot(gs[1, 1], projection=projection_a,
                    frameon=False)
ax7 = fig.add_subplot(gs[1, 2], projection=projection_a,
                    frameon=False)
ax8 = fig.add_subplot(gs[1, 3], projection=projection_a,
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
ax4.set_xlim([MapXextent_g[0], MapXextent_g[1]])
ax4.set_ylim([MapYextent_g[0], MapYextent_g[1]])
ax5.set_xlim([MapXextent_a[0], MapXextent_a[1]])
ax5.set_ylim([MapYextent_a[0], MapYextent_a[1]])
ax6.set_xlim([MapXextent_a[0], MapXextent_a[1]])
ax6.set_ylim([MapYextent_a[0], MapYextent_a[1]])
ax7.set_xlim([MapXextent_a[0], MapXextent_a[1]])
ax7.set_ylim([MapYextent_a[0], MapYextent_a[1]])
ax8.set_xlim([MapXextent_a[0], MapXextent_a[1]])
ax8.set_ylim([MapYextent_a[0], MapYextent_a[1]])

plot = ax1.pcolormesh(x_g, y_g, res_new_rac_g, shading='auto',
                norm=mcolors.Normalize(0,0.3), cmap=ccm.batlow_r, rasterized=True)
ax1.add_feature(ccrs.COASTLINE, zorder=10, 
                    edgecolor='black')
plot = ax2.pcolormesh(x_g, y_g, res_new_rac_ec_g, shading='auto',
                norm=mcolors.Normalize(0,0.3), cmap=ccm.batlow_r, rasterized=True)
ax2.add_feature(ccrs.COASTLINE, zorder=10, 
                    edgecolor='black')
diff_g = (res_new_rac_ec_g - res_new_rac_g)/res_new_rac_g*100
diff_g[diff_g == 0] = np.nan
plot3 = ax3.pcolormesh(x_g, y_g, diff_g, shading='auto',
                norm=mcolors.Normalize(-100,100), cmap=ccm.vik, rasterized=True)
ax3.add_feature(ccrs.COASTLINE, zorder=10, 
                    edgecolor='black')
diff_time_g = (res_time_rac_g - res_new_rac_g)/res_new_rac_g*100
diff_time_g[diff_time_g == 0] = np.nan
plot2 = ax4.pcolormesh(x_g,y_g, diff_time_g, shading='auto',
                norm=mcolors.Normalize(-100,100), cmap=ccm.vik, rasterized=True)
ax4.add_feature(ccrs.COASTLINE, zorder=10, 
                    edgecolor='black')
gl = ax1.gridlines(draw_labels=True, dms=True, 
                    linewidth=2, color='gray', alpha=0.5, linestyle='--',
                    zorder=11, x_inline=False,y_inline=False, 
                    rotate_labels=0, xpadding=10)
gl.xlabel_style = {'size': 10}
gl.ylabel_style = {'size': 10}
gl.top_labels = False
gl.left_labels = False
ax1.set_title('RACMO all')
gl = ax2.gridlines(draw_labels=True, dms=True, 
                    linewidth=2, color='gray', alpha=0.5, linestyle='--',
                    zorder=11, x_inline=False,y_inline=False, 
                    rotate_labels=0, xpadding=10)
gl.xlabel_style = {'size': 10}
gl.ylabel_style = {'size': 10}
gl.top_labels = False
gl.left_labels = False
ax2.set_title('RACMO during overpass')
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
ax3.set_title('Total difference')
ax4.set_title('Difference due to \n overpass time')
ax1.set_title('(a)', loc='left')
ax2.set_title('(b)', loc='left')
ax3.set_title('(c)', loc='left')
ax4.set_title('(d)', loc='left')
reader = shpreader.Reader("ne_50m_antarctic_ice_shelves_lines.shp")
geometries = []
for record in reader.records():
    geometries.append(record.geometry)
shape_feature = ShapelyFeature(geometries, crs.PlateCarree(), facecolor='none', edgecolor='black', lw=1)
plot = ax5.pcolormesh(x_a, y_a, res_new_rac_a, shading='auto',
                norm=mcolors.Normalize(0,0.3), cmap=ccm.batlow_r, rasterized=True)
ax5.add_feature(ccrs.COASTLINE, zorder=10, 
                    edgecolor='black')
ax5.add_feature(shape_feature)
plot = ax6.pcolormesh(x_a,y_a, res_new_rac_ec_a, shading='auto',
                norm=mcolors.Normalize(0,0.3), cmap=ccm.batlow_r, rasterized=True)
ax6.add_feature(ccrs.COASTLINE, zorder=10, 
                    edgecolor='black')
ax6.add_feature(shape_feature)
diff_a = (res_new_rac_ec_a - res_new_rac_a)/res_new_rac_a*100
diff_a[diff_a == 0] = np.nan
plot3 = ax7.pcolormesh(x_a, y_a, diff_a, shading='auto',
                norm=mcolors.Normalize(-100,100), cmap=ccm.vik, rasterized=True)
ax7.add_feature(ccrs.COASTLINE, zorder=10, 
                    edgecolor='black')
ax7.add_feature(shape_feature)
diff_time_a = (res_time_rac_a - res_new_rac_a)/res_new_rac_a*100
diff_time_a[diff_time_a == 0] = np.nan
plot3 = ax8.pcolormesh(x_a, y_a, diff_time_a, shading='auto',
                norm=mcolors.Normalize(-100,100), cmap=ccm.vik, rasterized=True)
ax8.add_feature(ccrs.COASTLINE, zorder=10, 
                    edgecolor='black')
ax8.add_feature(shape_feature)
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
gl = ax7.gridlines(draw_labels=True, dms=True, 
                    linewidth=2, color='gray', alpha=0.5, linestyle='--',
                    zorder=11, x_inline=False,y_inline=False, 
                    rotate_labels=0, xpadding=10)
gl.xlabel_style = {'size': 10}
gl.ylabel_style = {'size': 10}
gl.top_labels = False
gl.left_labels = False
gl = ax8.gridlines(draw_labels=True, dms=True, 
                    linewidth=2, color='gray', alpha=0.5, linestyle='--',
                    zorder=11, x_inline=False,y_inline=False, 
                    rotate_labels=0, xpadding=10)
gl.xlabel_style = {'size': 10}
gl.ylabel_style = {'size': 10}
gl.top_labels = False
gl.left_labels = False
ax5.set_title('(e)', loc='left')
ax6.set_title('(f)', loc='left')
ax7.set_title('(g)', loc='left')
ax8.set_title('(h)', loc='left')

cbar_ax = fig.add_axes([0.125, 0.0, 0.37, 0.015])
cb = fig.colorbar(plot, cax=cbar_ax, 
                    label=r'IWP [kg m$^{-2}$]', extend='max',
                    orientation="horizontal")
cb.ax.minorticks_off()
cbar_ax2 = fig.add_axes([0.53, 0.0, 0.37, 0.015])
cb2 = fig.colorbar(plot2, cax=cbar_ax2, 
                        label=f'Relative difference [%]', extend='max',
                        orientation="horizontal")

cb2.ax.locator_params(nbins=5)
cb2.ax.minorticks_off()
fig.subplots_adjust(bottom=0.01)
plt.savefig('GrIS_Ant_RACMOvsOverpass_IWP_split.png', dpi=300, 
            bbox_inches='tight', format='png')
