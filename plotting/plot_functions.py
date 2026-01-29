#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot functions for comparison between RACMO and 
EarthCARE. 
@author: Thirza Feenstra
"""
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator, FixedLocator
from cmcrameri import cm as ccm
import matplotlib.colors as mcolors
import cartopy.crs as crs
import cartopy.feature as ccrs
from datetime import datetime
sys.path.append('../utils/')
import RCM2EC_functions 
import helpers
import Convert_From_To_RacmoGrid as CFTR

def format_lat_lon_height(ax1, ax2, lat, lon, height=None, x1label=True, 
                          x2label=True, ylabel=True, grid='below'):
    '''Format the lat lon height labels and axis in scene plots
    input:
    ax1, ax2 = axes of which ax2 is the twiny axis of ax1
    lat, lon = real lat and lon with same length as plotted matrix
    height = height matrix
    x1label, x2label, ylabel = boolean for where to include labels
    grid = 'below' if plotted below data, or 'above' if plotted above
    date (e.g. when data fills all pixels)'''
    ax1.set_xticks(np.arange(0,len(lat),1), np.round(lat,1))
    ax1.xaxis.set_major_locator(MaxNLocator(8))
    ax2.set_xticks(np.arange(0,len(lon),1), np.round(lon,1))
    ax2.xaxis.set_major_locator(MaxNLocator(8))
    if height is not None:
        ax1.set_yticks(np.arange(0,len(height),1), np.round(height))
        ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    if x1label:
        ax1.set_xlabel(r'Latitude [$^\circ$N]')
    if x2label:
        ax2.set_xlabel(r'Longitude [$^\circ$E]')
    if ylabel:
        ax1.set_ylabel('Height [km]')
    if grid == 'below':
        ax1.set_axisbelow(True)
        ax1.grid(color='black', linestyle='dashed')
    elif grid == 'above':
        ax1.set_axisbelow(False)
        ax1.grid(color='black', linestyle='dashed')
    ax1.set_ylim(0, 15)
    ax1.set_yticks(np.arange(0, 15.01, 2.5))
        
def plot_topo(ax, x, heights_srf, heights):
    '''
    plot topography over 2D EarthCARE/RACMO scene plot
    Input:
        - ax: axes to plot on
        - x: x-coordinates of grid
        - heights_surf: topographic heights of surface
        - heigts: heights of grid, should be a regular grid'''
    heights_srf_2D = np.tile(heights_srf, (len(heights[0]),1)).T
    mask = heights > heights_srf_2D
    cmap = mcolors.ListedColormap([(0, 0, 0, 1),
                                (0, 0, 0, 0)])
    ax.pcolormesh(x, heights, mask, cmap=cmap, vmin=0, vmax=1)


def RACMOvsEC_panel_plot(racmo_data, ec_data, height, lat, lon, cmap_name, 
                         norm, cbar_label, savedir=None, varname=None, date=None, 
                         order='horizontal', grid='below', clutter=None):
    """
    Plots a comparison between RACMO and EarthCARE data for one timestep.
    Input:
    - racmo_data: data of RACMO scene (2D)
    - ec_data: data of EarthCARE scene (2D, same grid as racmo_data)
    - height: height coordinates, not necessarily regular grid (2D)
    - lat, lon: latitude and longitude of scene (1D)
    - cmap_name: colormap name
    - norm: norm of colormap 
    - cbar_label: label of colorbar (with units, string)
    - savedir: place to save plot on computer
    - varname: necessary for name of saved plot (if savedir != None)
    - date: necessary for name of saved plot (if savedir != None)
    - order: horizontal or vertical
    - grid: below or above (for gridlines)
    - clutter: mask to mask out the clutter
    """
    if order=='horizontal':
        fig, axes = plt.subplots(1, 2, figsize=(13,5), sharex=True)
        cbar_ax = fig.add_axes([0.9, 0.1, 0.02, 0.65])
        orientation = 'vertical'
    elif order=='vertical':
        fig, axes = plt.subplots(2, 1, figsize=(6,9))
        cbar_ax = fig.add_axes([0.12, 0.07, 0.78, 0.02])
        orientation = 'horizontal'
    ax1, ax3 = axes
    ax2, ax4 = ax1.twiny(), ax3.twiny()
    x = np.tile(np.arange(0, len(lat), 1), (len(height[0]),1)).T
    if cmap_name == 'tc':
        cmap = ccm.batlow_r
        colors = cmap(np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0]))
        discrete_cmap = mcolors.ListedColormap(colors)
        ec_data, snow_ec, rain_ec = ec_data
        racmo_data, snow_rac, rain_rac = racmo_data
        if not isinstance(clutter, type(None)):
            ax1.pcolormesh(x, height, clutter, shading='auto', 
                            cmap='Greys', vmin=0, vmax=2)
        plot = ax1.pcolormesh(x, height, ec_data, shading='auto', 
                            cmap=discrete_cmap, vmin=0, vmax=1.41)
        plot = ax3.pcolormesh(x, height, racmo_data, shading='auto', 
                                cmap=discrete_cmap, vmin=0, vmax=1.41)
        ax1.pcolor(x, height, np.ma.array(snow_ec, mask=snow_ec==0), cmap=mcolors.ListedColormap(['none']),
                    hatch='///', edgecolors='black', linewidth=0)
        ax1.pcolor(x, height, np.ma.array(rain_ec, mask=rain_ec==0), cmap=mcolors.ListedColormap(['none']),
                    hatch='xx', edgecolors='black', linewidth=0)
        ax3.pcolor(x, height, np.ma.array(snow_rac, mask=snow_rac==0), cmap=mcolors.ListedColormap(['none']),
                    hatch='///', edgecolors='black', linewidth=0)
        ax3.pcolor(x, height, np.ma.array(rain_rac, mask=snow_rac==0), cmap=mcolors.ListedColormap(['none']),
                    hatch='xx', edgecolors='black', linewidth=0)
        cbar = fig.colorbar(plot, cax=cbar_ax, orientation=orientation,
                            ticks=[0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3])
        cbar.set_ticklabels(['All \nice','Mostly \nice', 'Mixed \nphase', 'Mostly \nliquid', 
                             'All \nliquid', 'Snow-\nfall', 'Rain-\nfall'])
        if order=='vertical':
            hatch_rect = patches.Rectangle((0.86, 0), 0.14, 1, transform=cbar_ax.transAxes,
                            facecolor='white', edgecolor='black', hatch='xx', clip_on=False)
            cbar_ax.add_patch(hatch_rect)
            hatch_rect = patches.Rectangle((0.72, 0), 0.14, 1, transform=cbar_ax.transAxes,
                            facecolor='white', edgecolor='black', hatch='///', clip_on=False)
            cbar_ax.add_patch(hatch_rect)
        elif order=='horizontal':
            hatch_rect = patches.Rectangle((0, 0.86), 1, 0.14, transform=cbar_ax.transAxes,
                            facecolor='white', edgecolor='black', hatch='xx', clip_on=False)
            cbar_ax.add_patch(hatch_rect)
            hatch_rect = patches.Rectangle((0, 0.72), 1, 0.14, transform=cbar_ax.transAxes,
                            facecolor='white', edgecolor='black', hatch='///', clip_on=False)
            cbar_ax.add_patch(hatch_rect)
    else:
        if not isinstance(clutter, type(None)):
            ax1.pcolormesh(x, height, clutter, shading='auto', 
                            cmap='Greys', vmin=0, vmax=2)
        plot = ax1.pcolormesh(x, height, ec_data, shading='auto', 
                            cmap=cmap_name, norm=norm)
        plot = ax3.pcolormesh(x, height, racmo_data, shading='auto', 
                                cmap=cmap_name, norm=norm)
        cbar = fig.colorbar(plot, cax=cbar_ax, orientation=orientation,
                            label=cbar_label)
    if order=='horizontal':
        format_lat_lon_height(ax1, ax2, lat, lon, height=None, grid=grid)
        format_lat_lon_height(ax3, ax4, lat, lon, height=None, ylabel=False, grid=grid)
        ax3.set_title('RACMO')
        ax1.set_title('EarthCARE')
        ax3.tick_params(labelleft=False, left=False)
        fig.subplots_adjust(top=0.75, right=0.88)
    if order=='vertical':
        format_lat_lon_height(ax1, ax2, lat, lon, height=None, x1label=False, x2label=True, grid=grid)
        format_lat_lon_height(ax3, ax4, lat, lon, height=None, x1label=True, x2label=False, grid=grid)
        ax1.tick_params(labelbottom=False, bottom=False)
        ax4.tick_params(labeltop=False, top=False)
        fig.subplots_adjust(bottom=0.16, left=0.12, hspace=0.05)
    ax1.set_title('(a)', loc='left')
    ax3.set_title('(b)', loc='left')
    height_reg = np.tile(np.arange(0,15.01,0.1), (len(lat),1))
    x_reg = np.tile(np.arange(0, len(lat), 1), (len(height_reg[0]),1)).T
    plot_topo(ax1, x_reg, height[:,-1], height_reg)
    plot_topo(ax3, x_reg, height[:,-1], height_reg)
    if date is not None:
        plt.savefig(f"{savedir}{varname}{date.strftime('%Y%m%d%H%M')}.png", dpi=300,  bbox_inches='tight', format='png')
    return

def single_colormesh_plot(data, height, lat, lon, cmap_name, 
                         norm, cbar_label, savedir=None, varname=None, 
                         title=None, date=None, order='horizontal',
                         grid='below'):
    """
    Plots a single scene plot
    Input:
    - data: data of scene (2D)
    - height: height coordinates, not necessarily regular grid (2D)
    - lat, lon: latitude and longitude of scene (1D)
    - cmap_name: colormap name
    - norm: norm of colormap 
    - cbar_label: label of colorbar (with units, string)
    - savedir: place to save plot on computer
    - varname: necessary for name of saved plot (if savedir != None)
    - title: title of plot (optional)
    - date: necessary for name of saved plot (if savedir != None)
    - order: horizontal or vertical
    - grid: below or above (for gridlines)
    """
    if order=='horizontal':
        fig, ax1 = plt.subplots(1, 1, figsize=(6.5, 4))
        cbar_ax = fig.add_axes([0.8, 0.11, 0.03, 0.74])
        orientation = 'vertical'
    elif order=='vertical':
        fig, ax1 = plt.subplots(1, 1, figsize=(6.3,5))
        cbar_ax = fig.add_axes([0.15, 0.10, 0.75, 0.035])
        orientation = 'horizontal'
    ax2 = ax1.twiny()
    x = np.tile(np.arange(0, len(lat), 1), (len(height[0]),1)).T
    
    if cmap_name == 'tc':
        cmap = ccm.batlow_r
        colors = cmap(np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0]))
        discrete_cmap = mcolors.ListedColormap(colors)
        data, snow = data
        plot = ax1.pcolormesh(x, height, data, shading='auto', 
                            cmap=discrete_cmap, vmin=0, vmax=1.21)
        ax1.pcolor(x, height, np.ma.array(snow, mask=snow==0), cmap=mcolors.ListedColormap(['none']),
                    hatch='///', edgecolors='black', linewidth=0)
        cbar = fig.colorbar(plot, cax=cbar_ax, orientation=orientation,
                            ticks=[0.1, 0.3, 0.5, 0.7, 0.9, 1.1])
        cbar.set_ticklabels(['All \nice','Mostly \nice', 'Mixed \nphase', 'Mostly \nliquid',
                             'All \nliquid', 'Snowfall'])
        hatch_rect = patches.Rectangle((0.83, 0), 0.17, 1, transform=cbar_ax.transAxes,
                           facecolor='white', edgecolor='black', hatch='///', clip_on=False)
        cbar_ax.add_patch(hatch_rect)
    else:
        plot = ax1.pcolormesh(x, height, data, shading='auto', 
                            cmap=cmap_name, norm=norm)
        cbar = fig.colorbar(plot, cax=cbar_ax, orientation=orientation,
                            label=cbar_label)
    format_lat_lon_height(ax1, ax2, lat, lon, height=None, grid=grid)
    ax1.set_ylim(0, 15)
    if title:
        fig.suptitle(title, y=0.95, fontsize=14)
    if order=='horizontal':
        fig.subplots_adjust(top=0.85, bottom=0.15, left=0.15, right=0.75)
    if savedir is not None:
        plt.savefig(f"{savedir}{varname}{date.strftime('%Y%m%d%H%M')}.png", bbox_inches='tight', dpi=300, format='png')
    height_reg = np.tile(np.arange(0,15.01,0.1), (len(lat),1))
    x_reg = np.tile(np.arange(0, len(lat), 1), (len(height_reg[0]),1)).T
    plot_topo(ax1, x_reg, height[:,-1], height_reg)
    plt.close(fig)    
    return
    
def plot_l1_comparison(pathclm, pathtraj, pathatl, pathcpr, gridfile, savedir=None, date=None):
    """
    Plots a comparison between RACMO and EarthCARE data.

    Parameters:
    pathclm (str): Path to the RACMO CLM output file.
    pathtraj (str): path to the RACMO output file.
    pathatl (str): Path to the EarthCARE ATLID data file.
    pathcpr (str): path to the EarthCARE CPR data file.
    gridfile (str): Path to the grid settings file.
    savedir (str): Directory to save the output plots.
    date (datetime, optional): Date of the data for the plot title.

    Returns:
    None
    """
    clmfile = xr.open_dataset(pathclm)
    atl = xr.open_dataset(pathatl, engine='h5netcdf')
    cpr = xr.open_dataset(pathcpr, engine='h5netcdf')
    racfile = xr.open_dataset(pathtraj).squeeze()
    h_clm = clmfile.altitude_mid.values
    clm_mie = clmfile.ATB_Mie_co.values + clmfile.ATB_Mie_cr.values
    clm_ray = clmfile.ATB_Ray.values
    h_traj = racfile.height.values
    rlat, rlon = CFTR.RealWorld2RotatedGrid(clmfile.Latitude.values, clmfile.Longitude.values, gridfile)
    (lat_EC_r, lon_EC_r), h_EC_r, lat_EC, lon_EC = RCM2EC_functions.get_EC_traj(atl, [55, 90, -105, 35], gridfile)
    ec_mie, h_atl = RCM2EC_functions.get_EC_r(atl, 'mie_total_attenuated_backscatter_355nm', [55, 90, -105, 35], 
                                            h_clm[h_clm > 0], return_H=True)
    ec_mie, (lat_new, lon_new) = RCM2EC_functions.downsample(ec_mie, lat_EC, lon_EC, h_atl,
                                            rlat, rlon, h_clm,
                                            gridfile)
    ec_ray, h_atl = RCM2EC_functions.get_EC_r(atl, 'rayleigh_attenuated_backscatter_355nm', [55, 90, -105, 35], 
                                            h_clm[h_clm > 0], return_H=True)
    ec_ray, (lat_new, lon_new) = RCM2EC_functions.downsample(ec_ray, lat_EC, lon_EC, h_atl,
                                            rlat, rlon, h_clm,
                                            gridfile)    

    ec_mie[np.abs(ec_mie) > 1e20] = np.nan
    ec_ray[np.abs(ec_ray) > 1e20] = np.nan
    (lat_EC_r, lon_EC_r), h_EC_r, lat_EC, lon_EC = RCM2EC_functions.get_EC_traj(cpr, [55, 90, -105, 35], gridfile)
    rac_refl = helpers.radar_sim(racfile.cldi.values, racfile.cldw.values, racfile.clds.values, 
                                 racfile.cldr.values, racfile.temp.values, racfile.hum.values, 
                                 racfile.p.values, racfile.height.values)

    ecval, hec = RCM2EC_functions.get_EC_r(cpr, 'reflectivity_corrected', [55, 90, -105, 35], h_traj[h_traj > 0], 
                                           return_H=True)    
    ecval[~np.isfinite(ecval)] = -100
    ecval, (lat_new2, lon_new2) = RCM2EC_functions.downsample(ecval, lat_EC, lon_EC, hec,
                                            racfile.rlat.values, racfile.rlon.values, h_traj,
                                            gridfile)
    ecval[(ecval < -35)] = np.nan
    clut, hec = RCM2EC_functions.get_EC_r(cpr, 'significant_detection_classification', [55, 90, -105, 35], h_traj[h_traj > 0], 
                                           return_H=True)
    clut = clut.astype('float')
    clut_plot = np.full(np.shape(clut), np.nan)
    clut_plot[(clut >= 3) & (clut <=4)] = 1
    clut_plot, (lat_new, lon_new) = RCM2EC_functions.downsample(clut_plot, lat_EC, lon_EC, hec,
                                                                racfile.rlat.values, racfile.rlon.values, h_traj,
                                                            gridfile, fillna=False)
    for i in range(len(clut_plot[0])):
        valid = np.where(~np.isnan(clut_plot[:,i]))[0]
        if valid.size > 0:
            terrain_idx = valid[0]
            clut_plot[terrain_idx:,i] = 1
    valid_lat = np.isin(np.round(lat_new2,2), np.round(lat_new,2)) & np.isin(np.round(lon_new2,2), np.round(lon_new,2))
    lat_new2, lon_new2 = lat_new2[valid_lat], lon_new2[valid_lat]
    valid_lat2 = np.isin(np.round(lat_new,2), np.round(lat_new2,2)) & np.isin(np.round(lon_new,2), np.round(lon_new2,2))
    try:
        ecval = ecval[:, valid_lat]
        lat_new, lon_new = lat_new[valid_lat2], lon_new[valid_lat2]
        lat_new2, lon_new2 = lat_new2[valid_lat], lon_new2[valid_lat]
    except:
        lat_new, lon_new = lat_new[valid_lat2], lon_new[valid_lat2]
        valid_lat = np.isin(lat_new2, lat_new) & np.isin(lon_new2, lon_new)
        ecval = ecval[:, valid_lat]
        lat_new, lon_new = lat_new[valid_lat2], lon_new[valid_lat2]
    valid_indices = np.isin(np.round(clmfile.Latitude.values,2), np.round(lat_new,2)) & np.isin(np.round(clmfile.Longitude.values,2), np.round(lon_new,2))
    valid_indices2 = np.isin(np.round(racfile.lat.values,2), np.round(lat_new,2)) & np.isin(np.round(racfile.lon.values,2), np.round(lon_new,2))
    rac_refl = rac_refl[:, valid_indices2]
    clm_mie = clm_mie[:, valid_indices]
    clm_ray = clm_ray[:, valid_indices]
    ec_mie = ec_mie[:,valid_lat2]
    ec_ray = ec_ray[:,valid_lat2]
    h_traj = h_traj[:, valid_indices2]
    if len(pathclm) == 2:
        un_idx = np.unique(lat_new, return_index=True)[1]
        lat_new, lon_new = lat_new[un_idx], lon_new[un_idx]
        lat_new2, lon_new2 = lat_new2[un_idx], lon_new2[un_idx]
        rac_refl = rac_refl[:, un_idx]
        ecval = ecval[:, un_idx]
        clm_mie = clm_mie[:, un_idx]
        clm_ray = clm_ray[:, un_idx]
        ec_mie = ec_mie[:,un_idx]
        ec_ray = ec_ray[:,un_idx]
        h_traj = h_traj[:, un_idx]
    for i in range(clm_mie.shape[1]):
        col = clm_mie[:, i] 
        nan_idx = np.where(np.isnan(col))[0]  # Find NaN indices in column
        for j in nan_idx:
            if j < len(clm_mie[0]) - 1 and not np.isnan(clm_mie[j, i+1]):  # Use value right if available
                clm_mie[j, i] = clm_mie[j, i+1]
            elif i > 0 and not np.isnan(clm_mie[j, i-1]):  # Use value left if available
                clm_mie[j, i] = clm_mie[j, i-1]
    for i in range(clm_ray.shape[1]):
        col = clm_ray[:, i] 
        nan_idx = np.where(np.isnan(col))[0]  # Find NaN indices in column
        for j in nan_idx:
            if j < len(clm_ray[0]) - 1 and not np.isnan(clm_ray[j, i+1]):  # Use value right if available
                clm_ray[j, i] = clm_ray[j, i+1]
            elif i > 0 and not np.isnan(clm_ray[j, i-1]):  # Use value left if available
                clm_ray[j, i] = clm_ray[j, i-1]
    rac_refl[rac_refl < -1000] = np.nan
    rac_refl[rac_refl > 1e36] = np.nan
    ecval[np.abs(ecval) > 1e20] = np.nan
    fig, axes = plt.subplots(3, 2, figsize=(12,12))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.subplots_adjust(right=0.2)
    norm1 = mcolors.SymLogNorm(linthresh=1e-10, vmin=1e-8, vmax=1e-5)
    norm2 = mcolors.Normalize(vmin=-35, vmax=30)
    cbar_ax1 = fig.add_axes([1, 0.665, 0.02, 0.255])
    cbar_ax2 = fig.add_axes([1, 0.365, 0.02, 0.255])
    cbar_ax3 = fig.add_axes([1, 0.065, 0.02, 0.255])
    (ax1, ax3), (ax5, ax7), (ax9, ax11) = axes
    ax2, ax4, ax6, ax8, ax10, ax12 = ax1.twiny(), ax3.twiny(), ax5.twiny(), ax7.twiny(), ax9.twiny(), ax11.twiny()
    height = np.tile(h_clm/1000, (len(lat_new),1))
    x1 = np.tile(np.arange(0, len(lat_new), 1), (len(height[0]),1)).T
    plot1 = ax1.pcolormesh(x1, height, ec_mie.T, shading='auto', 
                        cmap='calipso', norm=norm1)
    plot2 = ax3.pcolormesh(x1, height, clm_mie.T, shading='auto', 
                        cmap='calipso', norm=norm1)
    plot3 = ax5.pcolormesh(x1, height, ec_ray.T, shading='auto', 
                            cmap='plasma', norm=norm1)
    plot4 = ax7.pcolormesh(x1, height, clm_ray.T, shading='auto', 
                            cmap='plasma', norm=norm1)
    cbar1 = fig.colorbar(plot1, cax=cbar_ax1, orientation='vertical',
                        label=r'Mie att. backscatter [sr$^{-1}$ m$^{-1}$]',
                        extend='both')
    cbar2 = fig.colorbar(plot3, cax=cbar_ax2, orientation='vertical',
                        label=r'Rayleigh att. backscatter [sr$^{-1}$ m$^{-1}$]',
                        extend='both')
    cbar1.ax.tick_params(labelsize=14)
    cbar2.ax.tick_params(labelsize=14)
    x2 = np.tile(np.arange(0, len(lat_new), 1), (len(h_traj[:,0]),1)).T
    plot5 = ax9.pcolormesh(x2, h_traj.T/1000, ecval.T, shading='auto', 
                        cmap=ccm.batlow_r, norm=norm2)
    ax9.pcolormesh(x2, h_traj.T/1000, clut_plot.T, shading='auto', 
                       cmap='Greys', norm=mcolors.Normalize(0,2))
    plot6 = ax11.pcolormesh(x2, h_traj.T/1000, rac_refl.T, shading='auto', 
                        cmap=ccm.batlow_r, norm=norm2)
    cbar3 = fig.colorbar(plot5, cax=cbar_ax3, orientation='vertical',
                            label='Radar reflectivity [dBZ]',
                            extend='max')
    cbar3.ax.tick_params(labelsize=14)
    plot_topo(ax2, x1, h_traj[-1]/1000,height)
    plot_topo(ax4, x1, h_traj[-1]/1000,height)
    plot_topo(ax6, x1, h_traj[-1]/1000,height)
    plot_topo(ax8, x1, h_traj[-1]/1000,height)
    plot_topo(ax10, x1, h_traj[-1]/1000,height)
    plot_topo(ax12, x1, h_traj[-1]/1000,height)
    format_lat_lon_height(ax1, ax2, lat_new, lon_new, height=None, x1label=False, grid='above')
    format_lat_lon_height(ax3, ax4, lat_new, lon_new, height=None, ylabel=False, x1label=False,
                          grid='above')
    format_lat_lon_height(ax5, ax6, lat_new, lon_new, height=None, x1label=False, x2label=False,
                          grid='above')
    format_lat_lon_height(ax7, ax8, lat_new, lon_new, height=None, ylabel=False, 
                          x1label=False, x2label=False, grid='above')
    format_lat_lon_height(ax9, ax10, lat_new, lon_new2, height=None, 
                          x2label=False)
    format_lat_lon_height(ax11, ax12, lat_new, lon_new2, height=None, 
                          ylabel=False, x2label=False)
    ax1.set_title('EarthCARE')
    ax3.set_title('RACMO')
    ax1.tick_params(labelbottom=False, bottom=False)
    ax3.tick_params(labelleft=False, left=False,
                    labelbottom=False, bottom=False)
    ax5.tick_params(labelbottom=False, bottom=False)
    ax6.tick_params(labeltop=False, top=False)
    ax7.tick_params(labelleft=False, labelbottom=False, labeltop=False,
                    left=False, bottom=False, top=False)
    ax8.tick_params(labeltop=False, top=False)
    ax10.tick_params(labeltop=False, top=False)
    ax11.tick_params(labelleft=False, left=False)
    ax12.tick_params(labeltop=False, top=False)
    ax1.set_title('(a)', loc='left')
    ax3.set_title('(b)', loc='left')
    ax5.set_title('(c)', loc='left')
    ax7.set_title('(d)', loc='left')
    ax9.set_title('(e)', loc='left')
    ax11.set_title('(f)', loc='left')
    plt.tight_layout()
    if savedir != None:
        plt.savefig(f"{savedir}l1_plot{date.strftime('%Y%m%d%H%M')}.png", bbox_inches='tight', dpi=300, format='png')
    return lat_new, lon_new


def plot_l2a_IWC_precip(pathtraj, pathice, pathcld, pathtc, gridfile, savedir=None, 
                      date=None, lat_lon = None, rain=False, only_composite=False):
    """
    Plots a comparison between RACMO and EarthCARE data.
    - Ice water content
    - Precipitation (snow or snow and rain if rain = True)

    Parameters:
    pathtraj (str): Path to the RACMO trajectory file.
    pathice (str): Path to the EarthCARE data file (ATL_ICE).
    pathcld (str): Path to the EarthCARE data file (CPR_CLD).
    pathtc (str): Path to the EarthCARE data file (AC__TC).
    gridfile (str): Path to the grid settings file.
    savedir (str): Directory to save the output plots.
    date (datetime, optional): Date of the data for the plot title.

    Returns:
    None
    """
    racmo = xr.open_dataset(pathtraj).squeeze()
    ecice = xr.open_dataset(pathice, engine='h5netcdf')
    ectc = xr.open_dataset(pathtc, engine='h5netcdf')
    eccld = xr.open_dataset(pathcld, engine='h5netcdf')
    (lat_EC_r, lon_EC_r), h_EC_r, lat_EC, lon_EC = RCM2EC_functions.get_EC_traj(ecice, [55, 90, -105, 35], 
                                                                                gridfile)
    racmo_ice = racmo['cldi'].values + racmo['clds'].values
    racmo_ice = helpers.kgkg_to_kgm3(racmo_ice, racmo.temp.values, 
                                     racmo.hum.values, racmo.p.values)
    racmo_snow = racmo['clds'].values 
    racmo_snow = helpers.kgkg_to_kgm3(racmo_snow, racmo.temp.values, 
                                     racmo.hum.values, racmo.p.values)
    racmo_rain_kgkg = racmo['cldr'].values 
    racmo_rain = helpers.kgkg_to_kgm3(racmo_rain_kgkg, racmo.temp.values, 
                                     racmo.hum.values, racmo.p.values)
    h_traj = racmo.height.values
    lat_rac = racmo.lat.values
    lon_rac = racmo.lon.values
    temp = racmo.temp.values
    hum = racmo.hum.values
    pres = racmo.p.values
    aice, hec = RCM2EC_functions.get_EC_r(ecice, 'ice_water_content', [55, 90, -105, 35], h_traj[h_traj > 0], return_H=True)
    aice[~np.isfinite(aice)] = 0
    if lat_lon is None:
        lat_in, lon_in = racmo.rlat.values, racmo.rlon.values
    else:
        lat_in, lon_in = CFTR.RealWorld2RotatedGrid(lat_lon[0], lat_lon[1], gridfile)
    aice, (lat_new, lon_new) = RCM2EC_functions.downsample(aice, lat_EC, lon_EC, hec,
                                                            lat_in, lon_in, h_traj, gridfile)
    aice[np.abs(aice) > 1e20] = np.nan
    aice = aice/1e6
    aice[~np.isfinite(aice)] = 0
    racmo_ice[racmo_ice <= 1e-7] = np.nan
    (lat_EC_r, lon_EC_r), h_EC_r, lat_EC, lon_EC = RCM2EC_functions.get_EC_traj(eccld, [55, 90, -105, 35], gridfile)
    hclas, hec = RCM2EC_functions.get_EC_r(eccld, 'hydrometeor_classification', [55, 90, -105, 35], h_traj[h_traj > 0], 
                                           return_H=True)
    hclas = hclas.astype('float')
    hclas_plot = np.full(np.shape(hclas), np.nan)
    hclas_plot[(hclas >= 16) & (hclas <=19)] = 1
    hclas[(hclas >= 16) & (hclas <=19)] = np.nan
    hclas[np.isfinite(hclas)] = 1
    hclas, (lat_new, lon_new) = RCM2EC_functions.downsample(hclas, lat_EC, lon_EC, hec,
                                                            lat_in, lon_in, h_traj,
                                                            gridfile, fillna=False)
    hclas_plot, (lat_new, lon_new) = RCM2EC_functions.downsample(hclas_plot, lat_EC, lon_EC, hec,
                                                            lat_in, lon_in, h_traj,
                                                            gridfile, fillna=False)
    for i in range(len(hclas_plot[0])):
        valid = np.where(~np.isnan(hclas_plot[:,i]))[0]
        if valid.size > 0:
            terrain_idx = valid[0]
            hclas_plot[terrain_idx:,i] = 1
    (lat_EC_r, lon_EC_r), h_EC_r, lat_EC, lon_EC = RCM2EC_functions.get_EC_traj(eccld, [55, 90, -105, 35], gridfile)
    cice, hec = RCM2EC_functions.get_EC_r(eccld, 'water_content', [55, 90, -105, 35], h_traj[h_traj > 0], return_H=True)
    cice[~np.isfinite(cice)] = 0
    cice, (lat_new2, lon_new2) = RCM2EC_functions.downsample(cice, lat_EC, lon_EC, hec,
                                            lat_in, lon_in, h_traj, gridfile)
    cwat = RCM2EC_functions.get_EC_r(eccld, 'liquid_water_content', [55, 90, -105, 35], h_traj[h_traj > 0])
    cwat[~np.isfinite(cwat)] = 0
    cwat, (lat_new2, lon_new2) = RCM2EC_functions.downsample(cwat, lat_EC, lon_EC, hec,
                                            lat_in, lon_in, h_traj, gridfile)
    cwat[~np.isfinite(cwat)] = 0
    cice *= hclas
    cwat *= hclas
    cice -= cwat
    cice[cice <= 1e-7] = np.nan
    cice[~np.isfinite(cice)] = 0
    cwat[cwat <= 1e-7] = np.nan
    cprec_flx, hec = RCM2EC_functions.get_EC_r(eccld, 'mass_flux', [55, 90, -105, 35], h_traj[h_traj > 0], return_H=True)
    cprec_flx[~np.isfinite(cprec_flx)] = 0
    cprec_flx, (lat_new2, lon_new2) = RCM2EC_functions.downsample(cprec_flx, lat_EC, lon_EC, hec,
                                                                lat_in, lon_in, h_traj, gridfile)
    cprec_flx *= hclas
    racmo_snow_flx = racmo_snow * 2 # to go to the flux, based on fall speed of 2 m/s
    racmo_snow_flx[racmo_snow_flx <= 1e-7] = np.nan
    rho = helpers.density(temp, hum, pres)
    racmo_rain_flx = racmo_rain * helpers.rain_fallspeed(racmo_rain_kgkg, rho) # to go to the flux, based on fall speed 
    racmo_rain_flx[racmo_rain_flx <= 1e-7] = np.nan
    cvel = RCM2EC_functions.get_EC_r(eccld, 'sedimentation_velocity', [55, 90, -105, 35], h_traj[h_traj > 0])
    cvel, (lat_new2, lon_new2) = RCM2EC_functions.downsample(cvel, lat_EC, lon_EC, hec,
                                            lat_in, lon_in, h_traj, gridfile)
    cvel *= hclas
    racmo_snow[racmo_snow <= 1e-7] = np.nan
    racmo_rain[racmo_rain <= 1e-7] = np.nan
    cvel[cvel <= 1e-3] = np.nan
    tc, hectc = RCM2EC_functions.get_EC_r(ectc, 'synergetic_target_classification', [55, 90, -105, 35], 
                                        h_traj[h_traj >= 0], return_H=True)
    tc, ec_snow, ec_rain, (lat_new, lon_new) = RCM2EC_functions.downsample_l2b_classification(tc, lat_EC, lon_EC, hectc,
                                                                lat_in, lon_in, h_traj, gridfile)
    cprec_flx[cprec_flx <= 1e-7] = np.nan
    ec_rain[~np.isfinite(ec_rain)] = 0
    snow_mask = np.ones(np.shape(ec_rain))
    snow_mask[ec_rain > 0] = np.nan
    csnow = cprec_flx*snow_mask / cvel # go to snow water content
    rain_mask = np.ones(np.shape(ec_rain))
    rain_mask[ec_rain == 0] = np.nan
    crain = cprec_flx*rain_mask / cvel # go to rain water content
    crain2 = crain.copy()
    crain2[~np.isfinite(crain2)] = 0
    cice = cice - crain2
    csnow_flx = cprec_flx*snow_mask 
    crain_flx = cprec_flx*rain_mask 
    cvel[(~np.isfinite(csnow_flx)) & (~np.isfinite(crain_flx))] = np.nan
    if lat_lon is None:
        valid_lat = np.isin(np.round(lat_new2, 2), np.round(lat_new, 2)) & np.isin(np.round(lon_new2, 2), np.round(lon_new, 2))
        lat_new2, lon_new2 = lat_new2[valid_lat], lon_new2[valid_lat]
        valid_lat2 = np.isin(np.round(lat_new, 2), np.round(lat_new2, 2)) & np.isin(np.round(lon_new, 2), np.round(lon_new2, 2))
        lat_new, lon_new = lat_new[valid_lat2], lon_new[valid_lat2]
        valid_indices = np.isin(np.round(lat_in, 2), np.round(lat_new, 2)) & np.isin(np.round(lon_in, 2), np.round(lon_new, 2))
        aice = aice[:, valid_lat2]
        cice = cice[:, valid_lat]
        csnow = csnow[:, valid_lat]
        crain = crain[:, valid_lat]
        cvel = cvel[:, valid_lat]
        racmo_ice = racmo_ice[:, valid_indices]
        racmo_snow = racmo_snow[:, valid_indices]
        racmo_rain = racmo_rain[:, valid_indices]
        racmo_snow_flx = racmo_snow_flx[:, valid_indices]
        racmo_rain_flx = racmo_rain_flx[:, valid_indices]
        h_traj = h_traj[:, valid_indices]
    else:
        valid_indices = np.isin(np.round(lat_rac, 2), np.round(lat_new, 2)) & np.isin(np.round(lon_rac, 2), np.round(lon_new, 2))
        h_traj = h_traj[:, valid_indices]
        racmo_ice = racmo_ice[:, valid_indices]
        racmo_snow = racmo_snow[:, valid_indices]
        racmo_rain = racmo_rain[:, valid_indices]
        racmo_snow_flx = racmo_snow_flx[:, valid_indices]
        racmo_rain_flx = racmo_rain_flx[:, valid_indices]
    # input for the composite function
    aice_sig, hec = RCM2EC_functions.get_EC_r(ecice, 'ice_water_content_error', [55, 90, -105, 35], 
                                              h_traj[h_traj > 0], return_H=True)
    aice_sig = aice_sig / 1e6
    aice_sig = RCM2EC_functions.downsample(aice_sig, lat_EC, lon_EC, hec,
                                        lat_in, lon_in, h_traj, gridfile, sample_type='error')[0]
    if lat_lon is None:
        aice_sig = aice_sig[:, valid_lat2]
    a_re, hec = RCM2EC_functions.get_EC_r(ecice, 'ice_effective_radius', [55, 90, -105, 35], 
                                          h_traj[h_traj > 0], return_H=True)
    a_re[~np.isfinite(a_re)] = 0
    a_re = RCM2EC_functions.downsample(a_re, lat_EC, lon_EC, hec,
                                        lat_in, lon_in, h_traj, gridfile)[0]
    if lat_lon is None:
        a_re = a_re[:, valid_lat2]
    a_re_sig, hec = RCM2EC_functions.get_EC_r(ecice, 'ice_effective_radius_error', [55, 90, -105, 35], 
                                              h_traj[h_traj > 0], return_H=True)
    a_re_sig = RCM2EC_functions.downsample(a_re_sig, lat_EC, lon_EC, hec,
                                        lat_in, lon_in, h_traj, gridfile, sample_type='error')[0]
    if lat_lon is None:
        a_re_sig = a_re_sig[:, valid_lat2]
    cice_sig, hec = RCM2EC_functions.get_EC_r(eccld, 'water_content_log_error', [55, 90, -105, 35], 
                                              h_traj[h_traj > 0], return_H=True)
    cice_sig = RCM2EC_functions.downsample(cice_sig, lat_EC, lon_EC, hec,
                                        lat_in, lon_in, h_traj, gridfile, sample_type='error')[0]
    if lat_lon is None:
        cice_sig = cice_sig[:, valid_lat]
    c_re_sig, hec = RCM2EC_functions.get_EC_r(eccld, 'characteristic_diameter_log_error', [55, 90, -105, 35], 
                                              h_traj[h_traj > 0], return_H=True)
    c_re_sig = RCM2EC_functions.downsample(c_re_sig, lat_EC, lon_EC, hec,
                                        lat_in, lon_in, h_traj, gridfile, sample_type='error')[0]
    if lat_lon is None:
        c_re_sig = c_re_sig[:, valid_lat]
    comp = helpers.compute_composite(aice, aice_sig, a_re, a_re_sig, 
                    cice, cice_sig, c_re_sig)
    comp *= hclas
    comp[comp < 1e-7] = np.nan
    aice[aice <= 1e-7] = np.nan
    cice[cice <= 1e-7] = np.nan
    height = np.flip(np.flipud(h_traj.T/1000), axis=0)
    x1 = np.tile(np.arange(0, len(lat_new), 1), (len(height[0]),1)).T
    xlim_low = 7e-8
    xlim_high = 2e-3
    ylim_low = 7e-8
    ylim_high = 2e-3
    totalbins=15
    a = (xlim_high/xlim_low)**(1/totalbins)
    b = (ylim_high/ylim_low)**(1/totalbins)
    xedges, yedges = xlim_low*a**np.arange(1,totalbins+1,1), ylim_low*b**np.arange(1,totalbins+1,1)
    cellheight = height.copy()
    cellheight[:,1:] = height[:,:-1] - height[:,1:]
    fractionsec = np.zeros(totalbins)
    fractionsrac = np.zeros(totalbins)
    eci = comp.flatten()
    ecw = cellheight.flatten()
    ecw = ecw[np.isfinite(eci)] 
    eci = eci[np.isfinite(eci)]
    ri = racmo_ice.flatten()*hclas.flatten()
    rw = cellheight.flatten()
    rw = rw[np.isfinite(ri)] 
    ri = ri[np.isfinite(ri)]
    counts, bins, patches = plt.hist(eci, bins=xedges,
                                    weights=ecw)
    fractionsec[1:] = counts / np.nansum(ecw) 
    counts, bins, patches = plt.hist(ri, bins=xedges, 
                                    weights=rw)
    fractionsrac[1:] = counts / np.nansum(rw)
    plt.clf()
    cmap = ccm.batlow_r
    colors = cmap(np.linspace(0, 1, 10))
    if only_composite:
        fig, axes = plt.subplots(1, 3, figsize=(12,4),
                                 width_ratios=[0.35,0.35,0.3])
        plt.subplots_adjust(wspace=0.0, hspace=0.0)
        norm1 = mcolors.SymLogNorm(linthresh=1e-10, vmin=1e-7, vmax=1e-3)
        cbar_ax1 = fig.add_axes([0.08, -0.025, 0.59, 0.045])
        ax1, ax3, ax5 = axes
        ax2, ax4 = ax1.twiny(), ax3.twiny()
        ax1.pcolormesh(x1, height, hclas_plot.T, shading='auto', 
                       cmap='Greys', norm=mcolors.Normalize(0,2))
        plot1 = ax1.pcolormesh(x1, height, comp.T, shading='auto', 
                                cmap=ccm.batlow_r, norm=norm1)
        plot2 = ax3.pcolormesh(x1, height, racmo_ice.T, shading='auto', 
                                cmap=ccm.batlow_r, norm=norm1)
        ax5.step(xedges, fractionsec, where="mid", color=colors[1],
                 label='EarthCARE',linewidth=2)
        ax5.step(xedges, fractionsrac, where="mid", color=colors[-1],
                 label='RACMO',linewidth=2)
        cbar1 = fig.colorbar(plot1, cax=cbar_ax1, orientation='horizontal',
                            label=r'Ice water content [kg m$^{-3}$]',
                            extend='both')
        cbar1.ax.tick_params(labelsize=14)
        height_reg = np.tile(np.arange(0,15.01,0.1), (len(lat_new),1))
        x_reg = np.tile(np.arange(0, len(lat_new), 1), (len(height_reg[0]),1)).T
        plot_topo(ax2, x_reg, h_traj[-1]/1000,height_reg)
        plot_topo(ax4, x_reg, h_traj[-1]/1000,height_reg)
        format_lat_lon_height(ax1, ax2, lat_new, lon_new, height=None)
        format_lat_lon_height(ax3, ax4, lat_new, lon_new, height=None, ylabel=False)
        ax1.set_title('ATLID - CPR composite')
        ax3.set_title('RACMO')
        ax3.tick_params(labelleft=False, left=False)
        ax1.set_title('(a)', loc='left')
        ax3.set_title('(b)', loc='left')
        ax5.set_title('(c)', loc='left', y=1.22)
        ax5.set_xscale('log')
        ax5.set_xlabel(r'Ice water content [kg m$^{-3}$]')
        ax5.set_ylim(0, 0.2)
        ax5.set_ylabel('Fraction of gridcell area [-]')
        plt.tight_layout()
        ax5.legend(frameon=False, bbox_to_anchor=(-0.25,-0.55), ncol=2, loc='lower left', columnspacing=1)
        if savedir != None:
            plt.savefig(f"{savedir}IWCplot_{date.strftime('%Y%m%d%H%M')}.png", 
                        bbox_inches='tight', dpi=300, format='png') 
    else:
        fig, axes = plt.subplots(2, 3, figsize=(12,7))
        norm1 = mcolors.SymLogNorm(linthresh=1e-10, vmin=1e-7, vmax=1e-3)
        cbar_ax1 = fig.add_axes([0.075, -0.015, 0.6, 0.03])
        (ax1, ax3, ax9), (ax5, ax7, ax11) = axes
        fig.delaxes(ax11)
        ax2, ax4, ax6, ax8 = ax1.twiny(), ax3.twiny(), ax5.twiny(), ax7.twiny()
        plot1 = ax1.pcolormesh(x1, height, aice.T, shading='auto', 
                            cmap=ccm.batlow_r, norm=norm1)
        ax3.pcolormesh(x1, height, hclas_plot.T, shading='auto', 
                       cmap='Greys', norm=mcolors.Normalize(0,2))
        plot2 = ax3.pcolormesh(x1, height, cice.T, shading='auto', 
                            cmap=ccm.batlow_r, norm=norm1)
        ax5.pcolormesh(x1, height, hclas_plot.T, shading='auto', 
                      cmap='Greys', norm=mcolors.Normalize(0,2))
        plot3 = ax5.pcolormesh(x1, height, comp.T, shading='auto', 
                                cmap=ccm.batlow_r, norm=norm1)
        plot4 = ax7.pcolormesh(x1, height, racmo_ice.T, shading='auto', 
                                cmap=ccm.batlow_r, norm=norm1)
        ax9.step(xedges, fractionsec, where="mid", color=colors[1],
                 label='EarthCARE',linewidth=2)
        ax9.step(xedges, fractionsrac, where="mid", color=colors[-1],
                 label='RACMO',linewidth=2)
        cbar1 = fig.colorbar(plot1, cax=cbar_ax1, orientation='horizontal',
                            label=r'Ice water content [kg m$^{-3}$]',
                            extend='both')
        cbar1.ax.tick_params(labelsize=14)
        height_reg = np.tile(np.arange(0,15.01,0.1), (len(lat_new),1))
        x_reg = np.tile(np.arange(0, len(lat_new), 1), (len(height_reg[0]),1)).T
        plot_topo(ax2, x_reg, h_traj[-1]/1000,height_reg)
        plot_topo(ax4, x_reg, h_traj[-1]/1000,height_reg)
        plot_topo(ax6, x_reg, h_traj[-1]/1000,height_reg)
        plot_topo(ax8, x_reg, h_traj[-1]/1000,height_reg)
        format_lat_lon_height(ax1, ax2, lat_new, lon_new, height=None, x1label=False)
        format_lat_lon_height(ax3, ax4, lat_new, lon_new, height=None, ylabel=False, x1label=False)
        format_lat_lon_height(ax5, ax6, lat_new, lon_new, height=None, 
                            x2label=False)
        format_lat_lon_height(ax7, ax8, lat_new, lon_new, height=None, 
                            ylabel=False, x2label=False)
        ax1.set_title('ATLID')
        ax3.set_title('CPR')
        ax5.set_title('ATLID - CPR composite')
        ax7.set_title('RACMO')
        ax1.tick_params(labelbottom=False, bottom=False)
        ax3.tick_params(labelleft=False, left=False,
                        labelbottom=False, bottom=False)
        ax6.tick_params(labeltop=False, top=False)
        ax6.tick_params(labeltop=False, top=False)
        ax7.tick_params(labelleft=False, left=False)
        ax8.tick_params(labeltop=False, top=False)
        ax1.set_title('(a)', loc='left')
        ax3.set_title('(b)', loc='left')
        ax5.set_title('(d)', loc='left')
        ax7.set_title('(e)', loc='left')
        ax9.set_title('(c)', loc='left', y=1.22)
        ax9.set_xscale('log')
        ax9.set_xlabel(r'Ice water content [kg m$^{-3}$]')
        ax9.set_ylim(0, 0.2)
        ax9.set_ylabel('Area-weighted fraction [-]')
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.tight_layout()
        ax9.legend(frameon=False, bbox_to_anchor=(0,-0.55), ncol=2, loc='lower left', columnspacing=1)
        if savedir != None:
            plt.savefig(f"{savedir}IWCplot_{date.strftime('%Y%m%d%H%M')}.png", 
                        bbox_inches='tight', dpi=300, format='png')
    
    if not rain:
        fig, axes = plt.subplots(1, 3, figsize=(12,3.8))
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.subplots_adjust(right=0.2)
        norm1 = mcolors.SymLogNorm(linthresh=1e-10, vmin=1e-7, vmax=1e-3)
        norm2 = mcolors.Normalize(vmin=0, vmax=2)
        cbar_ax1 = fig.add_axes([0.06, -0.025, 0.59, 0.05])
        cbar_ax2  = fig.add_axes([0.7, -0.025, 0.27, 0.05])
        (ax1, ax3, ax5) = axes
        ax2, ax4, ax6 = ax1.twiny(), ax3.twiny(), ax5.twiny()
        height = np.flip(np.flipud(h_traj.T/1000), axis=0)
        x1 = np.tile(np.arange(0, len(lat_new), 1), (len(height[0]),1)).T
        ax1.pcolormesh(x1, height, hclas_plot.T, shading='auto', 
                       cmap='Greys', norm=mcolors.Normalize(0,2))
        plot1 = ax1.pcolormesh(x1, height, csnow_flx.T, shading='auto', 
                            cmap=ccm.batlow_r, norm=norm1)
        plot2 = ax3.pcolormesh(x1, height, racmo_snow_flx.T, shading='auto', 
                            cmap=ccm.batlow_r, norm=norm1)
        ax5.pcolormesh(x1, height, hclas_plot.T, shading='auto', 
                       cmap='Greys', norm=mcolors.Normalize(0,2))
        plot3 = ax5.pcolormesh(x1, height, cvel.T, shading='auto', 
                                cmap=ccm.batlow_r, norm=norm2)
        cbar1 = fig.colorbar(plot1, cax=cbar_ax1, orientation='horizontal',
                            label=r'Snowfall rate [kg m$^{-2}$ s$^{-1}$]',
                            extend='both')
        cbar1.ax.tick_params(labelsize=14)
        cbar2 = fig.colorbar(plot3, cax=cbar_ax2, orientation='horizontal',
                            label=r'Sedimentation velocity [m s$^{-1}$]',
                            extend='max')
        cbar2.ax.tick_params(labelsize=14)
        height_reg = np.tile(np.arange(0,15.01,0.1), (len(lat_new),1))
        x_reg = np.tile(np.arange(0, len(lat_new), 1), (len(height_reg[0]),1)).T
        plot_topo(ax2, x_reg, h_traj[-1]/1000,height_reg)
        plot_topo(ax4, x_reg, h_traj[-1]/1000,height_reg)
        plot_topo(ax6, x_reg, h_traj[-1]/1000,height_reg)
        format_lat_lon_height(ax1, ax2, lat_new, lon_new, height=None)
        format_lat_lon_height(ax3, ax4, lat_new, lon_new, height=None, ylabel=False)
        format_lat_lon_height(ax5, ax6, lat_new, lon_new, height=None, 
                            ylabel=False)
        ax1.set_title('CPR')
        ax3.set_title('RACMO')
        ax5.set_title('CPR')
        ax3.tick_params(labelleft=False, left=False)
        ax5.tick_params(labelleft=False, left=False)
        ax1.set_title('(a)', loc='left')
        ax3.set_title('(b)', loc='left')
        ax5.set_title('(c)', loc='left')
        plt.tight_layout()
        if savedir != None:
            plt.savefig(f"{savedir}prec_plot_{date.strftime('%Y%m%d%H%M')}.png", 
                        bbox_inches='tight', dpi=300, format='png')
    else:
        fig, axes = plt.subplots(2, 3, figsize=(12,6.5))
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.subplots_adjust(right=0.2)
        norm1 = mcolors.SymLogNorm(linthresh=1e-10, vmin=1e-7, vmax=1e-3)
        norm2 = mcolors.Normalize(vmin=0, vmax=2)
        cbar_ax1 = fig.add_axes([0.075, -0.025, 0.575, 0.03])
        cbar_ax2  = fig.add_axes([0.705, 0.47, 0.265, 0.03])
        (ax1, ax3, ax5), (ax7, ax9, ax11) = axes
        fig.delaxes(ax11)
        ax2, ax4, ax6, ax8, ax10 = ax1.twiny(), ax3.twiny(), ax5.twiny(), ax7.twiny(), ax9.twiny()
        height = np.flip(np.flipud(h_traj.T/1000), axis=0)
        x1 = np.tile(np.arange(0, len(lat_new), 1), (len(height[0]),1)).T
        ax1.pcolormesh(x1, height, hclas_plot.T, shading='auto', 
                       cmap='Greys', norm=mcolors.Normalize(0,2))
        plot1 = ax1.pcolormesh(x1, height, csnow_flx.T, shading='auto', 
                            cmap=ccm.batlow_r, norm=norm1)
        plot2 = ax3.pcolormesh(x1, height, racmo_snow_flx.T, shading='auto', 
                            cmap=ccm.batlow_r, norm=norm1)
        ax5.pcolormesh(x1, height, hclas_plot.T, shading='auto', 
                       cmap='Greys', norm=mcolors.Normalize(0,2))
        plot3 = ax5.pcolormesh(x1, height, cvel.T, shading='auto', 
                                cmap=ccm.batlow_r, norm=norm2)
        ax7.pcolormesh(x1, height, hclas_plot.T, shading='auto', 
                       cmap='Greys', norm=mcolors.Normalize(0,2))
        plot4 = ax7.pcolormesh(x1, height, crain_flx.T, shading='auto', 
                            cmap=ccm.batlow_r, norm=norm1)
        plot5 = ax9.pcolormesh(x1, height, racmo_rain_flx.T, shading='auto', 
                            cmap=ccm.batlow_r, norm=norm1)
        cbar1 = fig.colorbar(plot1, cax=cbar_ax1, orientation='horizontal',
                            label=r'Precipitation rate [kg m$^{-2}$ s$^{-1}$]',
                            extend='both')
        cbar1.ax.tick_params(labelsize=14)
        cbar2 = fig.colorbar(plot3, cax=cbar_ax2, orientation='horizontal',
                            label=r'Sedimentation velocity [m s$^{-1}$]',
                            extend='max')
        cbar2.ax.tick_params(labelsize=14)
        height_reg = np.tile(np.arange(0,15.01,0.1), (len(lat_new),1))
        x_reg = np.tile(np.arange(0, len(lat_new), 1), (len(height_reg[0]),1)).T
        plot_topo(ax2, x_reg, h_traj[-1]/1000,height_reg)
        plot_topo(ax4, x_reg, h_traj[-1]/1000,height_reg)
        plot_topo(ax6, x_reg, h_traj[-1]/1000,height_reg)
        plot_topo(ax8, x_reg, h_traj[-1]/1000,height_reg)
        plot_topo(ax10, x_reg, h_traj[-1]/1000,height_reg)
        format_lat_lon_height(ax1, ax2, lat_new, lon_new, height=None, x1label=False)
        format_lat_lon_height(ax3, ax4, lat_new, lon_new, height=None, ylabel=False, 
                              x1label=False)
        format_lat_lon_height(ax5, ax6, lat_new, lon_new, height=None, 
                              ylabel=False, x1label=False)
        format_lat_lon_height(ax7, ax8, lat_new, lon_new, height=None, x2label=False)
        format_lat_lon_height(ax9, ax10, lat_new, lon_new, height=None, 
                              ylabel=False, x2label=False)
        ax1.set_title('CPR - snowfall')
        ax3.set_title('RACMO - snowfall')
        ax5.set_title('CPR')
        ax7.set_title('CPR - rainfall')
        ax9.set_title('RACMO - rainfall')
        ax1.tick_params(labelbottom=False, bottom=False)
        ax3.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)
        ax5.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)
        ax8.tick_params(labeltop=False, top=False)
        ax9.tick_params(labelleft=False, left=False)
        ax10.tick_params(labeltop=False, top=False)
        ax1.set_title('(a)', loc='left')
        ax3.set_title('(b)', loc='left')
        ax5.set_title('(c)', loc='left')
        ax7.set_title('(d)', loc='left')
        ax9.set_title('(e)', loc='left')
        plt.tight_layout()
        if savedir != None:
            plt.savefig(f"{savedir}prec_plot_{date.strftime('%Y%m%d%H%M')}.png", 
                        bbox_inches='tight', dpi=300, format='png')
    return


def plot_TC2b_comparison(pathtraj, pathtc, gridfile, savedir=None, date=None, 
                          order='horizontal',
                         lat_lon=None, clutter=None):
    """
    Plots a comparison between RACMO and EarthCARE data.
    Parameters:
    pathtraj (str): Path to the RACMO trajectory file.
    pathtc (str): Path to the EarthCARE data file (A_TC).
    gridfile (str): Path to the grid settings file.
    savedir (str): Directory to save the output plots.
    date (datetime, optional): Date of the data for the plot title.
    lat_lon (array): lat, lon coordinates to align with other plots
    clutter (array): array with clutter mask from other plots
    Returns:
    None
    """
    racmo = xr.open_dataset(pathtraj, engine='netcdf4').squeeze()
    ectc = xr.open_dataset(pathtc, engine='h5netcdf')
    racmo_ice = racmo['cldi'].values 
    racmo_liq = racmo['cldw'].values
    racmo_snow = racmo['clds'].values
    racmo_rain = racmo['cldr'].values
    racmo_tc, rac_snow, rac_rain = helpers.make_RACMO_classification(racmo_ice, racmo_liq, racmo_snow, racmo_rain)
    h_traj = racmo.height.values
    (lat_EC_r, lon_EC_r), h_EC_r, lat_EC, lon_EC = RCM2EC_functions.get_EC_traj(ectc, [55, 90, -105, 35], gridfile)
    tc, hec = RCM2EC_functions.get_EC_r(ectc, 'synergetic_target_classification', [55, 90, -105, 35], 
                                        h_traj[h_traj >= 0], return_H=True)
    if lat_lon is None:
        lat_in, lon_in = racmo.rlat.values, racmo.rlon.values
    else:
        lat_in, lon_in = lat_lon
    tc, ec_snow, ec_rain, (lat_new, lon_new) = RCM2EC_functions.downsample_l2b_classification(tc, lat_EC, lon_EC, hec,
                                                                lat_in, lon_in, h_traj, gridfile)
    tc[tc > 1] = np.nan
    valid_indices = np.isin(np.round(racmo.lat.values,2), np.round(lat_new,2)) & np.isin(np.round(racmo.lon.values,2), np.round(lon_new,2))
    racmo_tc = racmo_tc[:, valid_indices]
    rac_snow = rac_snow[:, valid_indices]
    rac_rain = rac_rain[:, valid_indices]
    h_traj = h_traj[:, valid_indices]
    RACMOvsEC_panel_plot([np.flip(np.flipud(racmo_tc.T), axis=0), 
                          np.flip(np.flipud(rac_snow.T), axis=0),
                          np.flip(np.flipud(rac_rain.T), axis=0)], 
                        [tc.T, ec_snow.T, ec_rain.T],
                        np.flip(np.flipud(h_traj.T/1000), axis=0), 
                        lat_new, lon_new, 
                        'tc', None, r'', savedir, 
                        'tc_2b', date=date, order=order,
                        clutter=clutter)
    return
    
    
def racmo_overview_plot(path_rac_3D, path_traj, path_siconca, 
                        dtop, savedir=None, date=None, 
                        lat_lon=None, add_loc=None):
    ''' 
    Makes RACMO overview plot of the scene, containing
    total cloud water content spatial plot, and transects
    along EarthCARE flight line of cloud content in all four
    phases (ice, liquid, rain, snow).
    Input:
    - path_rac_3D: path to 3D RACMO data of whole domain
    - path_traj: path to RACMO data along EarthCARE flight line
    - path_siconca: path to 2D grid of sea ice extent
    - dtop: dataframe with topographic information of RACMO
    - savedir: path to save plot
    - date: for extracting correct sea ice extent
    - lat_lon: specific targeted lat/lon combinations
    - add_loc: string with location names to add to panel b
    '''
    racmo3D = xr.open_dataset(path_rac_3D).squeeze()
    date = datetime.strptime(path_rac_3D[-17:-5], '%Y%m%d%H%M')
    siconca = xr.open_dataset(path_siconca).squeeze().sel(time=date, method='nearest')
    if len(path_traj) != 2:
        racmo = xr.open_dataset(path_traj).squeeze()
    else:
        rac1 = helpers.preprocess_racmo_traj(xr.open_dataset(path_traj[0]))
        rac2 = helpers.preprocess_racmo_traj(xr.open_dataset(path_traj[1]))
        racmo = xr.concat([rac1, rac2], dim='rlat')
    height = RCM2EC_functions.plev2gph(racmo3D.temp.values, racmo3D.hum.values, 
                                       racmo3D.ps.values, dtop.Geopotential.values)
    dh = height[:-1] - height[1:]
    cldt = racmo3D.cldi.values + racmo3D.cldw.values + racmo3D.clds.values + racmo3D.cldr.values
    cldt = helpers.kgkg_to_kgm3(cldt, racmo3D.temp.values, racmo3D.hum.values, 
                                racmo3D.p.values)
    twp = np.sum(cldt[:-1]*dh, axis=0) 
    twp[twp == 0] = np.nan
    cmap = mpl.colors.ListedColormap(ccm.batlow_r(np.linspace(0, 0.8, 256)))
    norm=mcolors.SymLogNorm(linthresh=1e-2,
                                vmin=1, vmax=1000)
    fig = plt.figure(figsize=(15,6.3))
    gs1 = gridspec.GridSpec(1, 1, right=0.36)
    gs2 = gridspec.GridSpec(2, 2, left=0.46,
                           wspace=0.1, hspace=0.15)
    projection=crs.RotatedPole(pole_latitude=-18, pole_longitude=-37.5,
                               central_rotated_longitude=0)
    ax1 = fig.add_subplot(gs1[0, 0],projection=projection)  
    ax2 = fig.add_subplot(gs2[0, 0])
    ax4 = fig.add_subplot(gs2[0, 1])
    ax6 = fig.add_subplot(gs2[1, 0]) 
    ax8 = fig.add_subplot(gs2[1, 1])  
    MapLL_TR_lat = np.array([  56.119, 77.922])  
    MapLL_TR_lon = np.array([ -55.15, 34.106])
    MapXextent, MapYextent, _ = projection.transform_points(crs.PlateCarree(), MapLL_TR_lon, MapLL_TR_lat).T
    ax1.set_xlim([MapXextent[0], MapXextent[1]])
    ax1.set_ylim([MapYextent[0], MapYextent[1]])
    plot = ax1.pcolormesh(racmo3D.lon.values, racmo3D.lat.values, twp*1000, 
                  norm=norm, cmap=cmap, transform=crs.PlateCarree(), rasterized=True)
    
    ax1.add_feature(ccrs.COASTLINE, zorder=10, 
                        edgecolor='black')
    cbar_ax = fig.add_axes([0.125, -0.03, 0.235, 0.03])
    cb = fig.colorbar(plot, cax=cbar_ax, 
                      label='Total cloud water path, \n vertically integrated [g $m^{-2}$]', 
                      extend='both',
                      orientation="horizontal")
    cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=14)
    seaice = siconca.siconca.values
    seaice[dtop.LSM.values > 0] = 0
    siplot = ax1.contour(siconca.lon.values, siconca.lat.values,
                          seaice, colors='black',
                          transform=crs.PlateCarree(), levels=[0.15], 
                          linewidths=1)#, linestyles='dashed')
    sihatch = ax1.contourf(siconca.lon.values, siconca.lat.values,
                        seaice, transform=crs.PlateCarree(), 
                        levels=[0,0.15,1], colors='none', 
                        hatches=['', 'x'], alpha=0)
    fi500 = helpers.geopotential_500hpa(racmo3D.p.values, height)
    levels = np.arange(4600, 6001, 100)
    gp = ax1.contour(racmo3D.lon.values, racmo3D.lat.values,
                     fi500, colors='black', transform=crs.PlateCarree(),
                     levels = levels, linewidths=1, linestyles='dashed') 
    ax1.clabel(gp, levels[::],  
        fmt='%1.0f', fontsize=8)
    gl = ax1.gridlines(draw_labels=True, dms=True, 
                        linewidth=1, color='black', alpha=0.5, linestyle='--',
                        zorder=11, x_inline=False,y_inline=False, 
                        rotate_labels=0, xpadding=10)
    gl.top_labels = False
    gl.left_labels = False
    ax1.scatter(racmo.lon.values, racmo.lat.values,  s=20, marker='.', c='black', 
               zorder=100,
               transform=crs.PlateCarree())
    ax1.set_title('(a)', loc='left', y=1.12)
    height = racmo.height.values/1000
    lat = racmo.lat.values
    lon = racmo.lon.values
    cldi = racmo.cldi.values
    cldw = racmo.cldw.values
    clds = racmo.clds.values
    cldr = racmo.cldr.values
    temp = racmo.temp.values
    cldi = helpers.kgkg_to_kgm3(cldi, racmo.temp.values, racmo.hum.values, 
                                racmo.p.values)
    cldw = helpers.kgkg_to_kgm3(cldw, racmo.temp.values, racmo.hum.values, 
                                racmo.p.values)
    clds = helpers.kgkg_to_kgm3(clds, racmo.temp.values, racmo.hum.values, 
                                racmo.p.values)
    cldr = helpers.kgkg_to_kgm3(cldr, racmo.temp.values, racmo.hum.values, 
                                racmo.p.values)
    cldi[cldi < 1e-7] = np.nan
    cldw[cldw < 1e-7] = np.nan
    clds[clds < 1e-7] = np.nan
    cldr[cldr < 1e-7] = np.nan
    if lat_lon is not None:
        lat_val, lon_val = lat_lon
        valid_indices = np.isin(lat, lat_val) & np.isin(lon, lon_val)
        height = height[:, valid_indices]
        cldi = cldi[:, valid_indices]
        cldw = cldw[:, valid_indices]
        clds = clds[:, valid_indices]
        cldr = cldr[:, valid_indices]
        temp = temp[:, valid_indices]
        lat = lat[valid_indices]
        lon = lon[valid_indices]
    if len(path_traj) == 2:
        un_idx = np.unique(lat, return_index=True)[1]
        cldi = cldi[:, un_idx]
        cldw = cldw[:, un_idx]
        clds = clds[:, un_idx]
        cldr = cldr[:, un_idx]
        height = height[:, un_idx]
        temp = temp[:, un_idx]
        lat = lat[un_idx]
        lon = lon[un_idx]
    norm = mcolors.SymLogNorm(linthresh=1e-10, vmin=1e-7, vmax=1e-3) 
    cbar_ax1 = fig.add_axes([0.46, -0.03, 0.42, 0.03])
    orientation = 'horizontal'
    ax3, ax5, ax7, ax9 = ax2.twiny(), ax4.twiny(), ax6.twiny(), ax8.twiny()
    x = np.tile(np.arange(0, len(lat), 1), (len(height),1))
    plot1 = ax2.pcolormesh(x, height, cldi, shading='auto', 
                        cmap=cmap, norm=norm, zorder=5)
    plot2 = ax4.pcolormesh(x, height, cldw, shading='auto', 
                        cmap=cmap, norm=norm, zorder=5)
    plot3 = ax6.pcolormesh(x, height, clds, shading='auto', 
                            cmap=cmap, norm=norm, zorder=5)
    plot4 = ax8.pcolormesh(x, height, cldr, shading='auto', 
                            cmap=cmap, norm=norm, zorder=5)
    tlevels = np.arange(-50, 20.1, 10)
    tempC = temp - 273.16
    t = ax8.contour(x, height, tempC, 
                     colors='black', #xkcd:bluegrey
                     levels = tlevels,
                    linewidths=0.5, linestyles='dashed')
    ax8.clabel(t, tlevels[::],  
                fmt='%1.0f', fontsize=6)
    ax2.contour(x, height, tempC, 
                     colors='black',
                     levels = tlevels,
                    linewidths=0.5, linestyles='dashed')
    ax4.contour(x, height, tempC, 
                     colors='black',
                     levels = tlevels,
                    linewidths=0.5, linestyles='dashed')
    ax6.contour(x, height, tempC, 
                     colors='black',
                     levels = tlevels,
                    linewidths=0.5, linestyles='dashed')
    cbar1 = fig.colorbar(plot1, cax=cbar_ax1, orientation=orientation,
                        label='Cloud water content [kg m$^{-3}$]',
                        extend='both')
    cbar1.ax.tick_params(labelsize=14)
    height_reg = np.tile(np.arange(0,15.01,0.1), (len(lat),1))
    x_reg = np.tile(np.arange(0, len(lat), 1), (len(height_reg[0]),1)).T
    plot_topo(ax2, x_reg, height[-1], height_reg)
    plot_topo(ax4, x_reg, height[-1], height_reg)
    plot_topo(ax6, x_reg, height[-1], height_reg)
    plot_topo(ax8, x_reg, height[-1], height_reg)
    format_lat_lon_height(ax2, ax3, lat, lon, height=None, x1label=False)
    format_lat_lon_height(ax4, ax5, lat, lon, height=None, ylabel=False, x1label=False)
    format_lat_lon_height(ax6, ax7, lat, lon, height=None, x2label=False)
    format_lat_lon_height(ax8, ax9, lat, lon, height=None, ylabel=False, x2label=False)
    ax2.set_title('(b)', loc='left')
    ax4.set_title('(c)', loc='left')
    ax6.set_title('(d)', loc='left')
    ax8.set_title('(e)', loc='left')
    ax2.set_title('Ice water content')
    ax4.set_title('Liquid water content')
    ax6.set_title('Snow water content')
    ax8.set_title('Rain water content')
    ax2.tick_params(labelbottom=False, bottom=False)
    ax4.tick_params(labelleft=False, left=False,
                    labelbottom=False, bottom=False)
    ax7.tick_params(labeltop=False, top=False)
    ax9.tick_params(labeltop=False, top=False)
    ax8.tick_params(labelleft=False, left=False)
    fig.subplots_adjust(top=0.85, right=0.88)
    if add_loc != None:
        ax2.text(10, 14, add_loc, fontsize=10)
    fig.tight_layout()
    if savedir != None:
        plt.savefig(f"{savedir}racmo_overview_{date.strftime('%Y%m%d%H%M')}.png", bbox_inches='tight', dpi=300, format='png')

def racmo_overview_plot_ERA5(path_rac_3D, path_traj, path_siconca, 
                             path_era5_cld, path_era5_si, path_era5_z500,
                            dtop, savedir=None, date=None, 
                            lat_lon=None, add_loc=None):
    racmo3D = xr.open_dataset(path_rac_3D).squeeze()
    date = datetime.strptime(path_rac_3D[-17:-5], '%Y%m%d%H%M')
    siconca = xr.open_dataset(path_siconca).squeeze().sel(time=date, method='nearest')
    era5_cld = xr.open_dataset(path_era5_cld)
    era5_si = xr.open_dataset(path_era5_si)
    era5_z500 = xr.open_dataset(path_era5_z500)
    lat_era = era5_cld.latitude.values
    lon_era = era5_cld.longitude.values
    if len(path_traj) != 2:
        racmo = xr.open_dataset(path_traj).squeeze()
    else:
        rac1 = helpers.preprocess_racmo_traj(xr.open_dataset(path_traj[0]))
        rac2 = helpers.preprocess_racmo_traj(xr.open_dataset(path_traj[1]))
        racmo = xr.concat([rac1, rac2], dim='rlat')
    height = RCM2EC_functions.plev2gph(racmo3D.temp.values, racmo3D.hum.values, 
                                       racmo3D.ps.values, dtop.Geopotential.values)
    dh = height[:-1] - height[1:]
    cldt = racmo3D.cldi.values + racmo3D.cldw.values + racmo3D.clds.values + racmo3D.cldr.values
    cldt = helpers.kgkg_to_kgm3(cldt, racmo3D.temp.values, racmo3D.hum.values, 
                                racmo3D.p.values)
    twp = np.sum(cldt[:-1]*dh, axis=0) 
    twp[twp == 0] = np.nan
    twp_era5 = era5_cld.tciw.values + era5_cld.tclw.values + era5_cld.tcrw.values + era5_cld.tcsw.values
    twp_era5[twp_era5 == 0] = np.nan
    box = np.ones(np.shape(twp))
    cmap = mpl.colors.ListedColormap(ccm.batlow_r(np.linspace(0, 0.8, 256)))
    norm=mcolors.SymLogNorm(linthresh=1e-4,
                                vmin=1e-3, vmax=1)
    fig = plt.figure(figsize=(16,6.3))
    gs1 = gridspec.GridSpec(1, 1, right=0.39)
    gs2 = gridspec.GridSpec(2, 2, left=0.46,
                           wspace=0.1, hspace=0.15)
    projection=crs.RotatedPole(pole_latitude=-18, pole_longitude=-37.5,
                               central_rotated_longitude=0)
    ax1 = fig.add_subplot(gs1[0, 0],projection=projection)  
    ax2 = fig.add_subplot(gs2[0, 0])
    ax4 = fig.add_subplot(gs2[0, 1])
    ax6 = fig.add_subplot(gs2[1, 0]) 
    ax8 = fig.add_subplot(gs2[1, 1])  
    MapLL_TR_lat = np.array([  42, 71])  
    MapLL_TR_lon = np.array([-60, 60])
    MapXextent, MapYextent, _ = projection.transform_points(crs.PlateCarree(), MapLL_TR_lon, MapLL_TR_lat).T
    ax1.set_xlim([MapXextent[0], MapXextent[1]])
    ax1.set_ylim([MapYextent[0], MapYextent[1]])
    plot = ax1.pcolormesh(lon_era, lat_era, twp_era5.squeeze(), 
                  norm=norm, cmap=cmap, transform=crs.PlateCarree(), rasterized=False,
                  zorder=1)
    siplot = ax1.contour(lon_era, lat_era,
                          era5_si.siconc.values.squeeze(), colors='black',
                          transform=crs.PlateCarree(), levels=[0.15], 
                          linewidths=1, zorder=2)
    sihatch = ax1.contourf(lon_era, lat_era,
                          era5_si.siconc.values.squeeze(), transform=crs.PlateCarree(), 
                        levels=[0,0.15,1], colors='none', 
                        hatches=['', 'x'], alpha=0, zorder=3)
    levels = np.arange(4600, 6001, 100)
    gp = ax1.contour(lon_era, lat_era,
                     era5_z500.z.values.squeeze()/9.81, colors='purple', transform=crs.PlateCarree(),
                     levels = levels, linewidths=1, linestyles='dashed',zorder=4) 
    ax1.clabel(gp, levels[::],  
        fmt='%1.0f', fontsize=8, zorder=5)
    gl = ax1.gridlines(crs=crs.PlateCarree(),
                    draw_labels=True,
                    linewidth=0)
    gl.xlocator = FixedLocator(np.arange(-180, 181, 20))
    gl.ylocator = FixedLocator(np.arange(-90, 91, 10))
    gl.top_labels = False
    gl.left_labels = False
    gl.x_inline = False
    gl.y_inline = False
    b = ax1.pcolormesh(racmo3D.lon.values, racmo3D.lat.values, box, vmin=0, vmax=1, cmap='Greys_r',
                       transform=crs.PlateCarree(), zorder=6, rasterized=False)
    plot = ax1.pcolormesh(racmo3D.lon.values, racmo3D.lat.values, twp,
                        norm=norm, cmap=cmap, transform=crs.PlateCarree(), 
                        zorder=7, rasterized=False)
    box[0] = 0
    box[-1] = 0
    box[:,0] = 0
    box[:,-1] = 0
    b = ax1.contour(racmo3D.lon.values, racmo3D.lat.values, box, levels=[0.5], colors='black',
                    transform=crs.PlateCarree(), rasterized=False, linewidths=2, zorder=8)
    ax1.add_feature(ccrs.COASTLINE, zorder=9, 
                        edgecolor='black')
    cbar_ax = fig.add_axes([0.135, -0.03, 0.245, 0.03])
    cb = fig.colorbar(plot, cax=cbar_ax, 
                      label='Total cloud and precipitation water path, \n vertically integrated [kg $m^{-2}$]', 
                      extend='max',
                      orientation="horizontal")
    cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=14)
    seaice = siconca.siconca.values
    seaice[dtop.LSM.values > 0] = 0
    siplot = ax1.contour(siconca.lon.values, siconca.lat.values,
                          seaice, colors='black',
                          transform=crs.PlateCarree(), levels=[0.15], 
                          linewidths=1, zorder=10)#, linestyles='dashed')
    sihatch = ax1.contourf(siconca.lon.values, siconca.lat.values,
                        seaice, transform=crs.PlateCarree(), 
                        levels=[0,0.15,1], colors='none', 
                        hatches=['', 'x'], alpha=0, zorder=11)
    fi500 = helpers.geopotential_500hpa(racmo3D.p.values, height)
    levels = np.arange(4600, 6001, 100)
    gp = ax1.contour(racmo3D.lon.values, racmo3D.lat.values,
                     fi500, colors='purple', transform=crs.PlateCarree(),
                     levels = levels, linewidths=1, linestyles='dashed', zorder=12) 
    ax1.clabel(gp, levels[::],  
        fmt='%1.0f', fontsize=8,zorder=13)
    ax1.scatter(racmo.lon.values, racmo.lat.values,  s=20, marker='.', c='black', 
               zorder=14,
               transform=crs.PlateCarree())
    ax1.set_title('(a)', loc='left', y=1.1)
    lons = np.arange(-180, 181, 20)
    lats = np.arange(-90, 91, 10)
    for lat in lats:
        ax1.plot(
            np.linspace(-180, 180, 361),
            np.full(361, lat),
            transform=crs.PlateCarree(),
            color='black',
            linestyle='--',
            linewidth=1,
            alpha=0.5,
            zorder=100
        )
    for lon in lons:
        ax1.plot(
            np.full(181, lon),
            np.linspace(-90, 90, 181),
            transform=crs.PlateCarree(),
            color='black',
            linestyle='--',
            linewidth=1,
            alpha=0.5,
            zorder=100
        )
    box[box == 1] = np.nan
    cmapblack = mcolors.ListedColormap(['black'])
    cmapblack.set_bad(color=(0, 0, 0, 0))
    b = ax1.pcolormesh(racmo3D.lon.values, racmo3D.lat.values, box, vmin=0, vmax=1, cmap=cmapblack,
                       transform=crs.PlateCarree(), zorder=15, rasterized=False, facecolor=None)
    height = racmo.height.values/1000
    lat = racmo.lat.values
    lon = racmo.lon.values
    cldi = racmo.cldi.values
    cldw = racmo.cldw.values
    clds = racmo.clds.values
    cldr = racmo.cldr.values
    temp = racmo.temp.values
    cldi = helpers.kgkg_to_kgm3(cldi, racmo.temp.values, racmo.hum.values, 
                                racmo.p.values)
    cldw = helpers.kgkg_to_kgm3(cldw, racmo.temp.values, racmo.hum.values, 
                                racmo.p.values)
    clds = helpers.kgkg_to_kgm3(clds, racmo.temp.values, racmo.hum.values, 
                                racmo.p.values)
    cldr = helpers.kgkg_to_kgm3(cldr, racmo.temp.values, racmo.hum.values, 
                                racmo.p.values)
    cldi[cldi < 1e-7] = np.nan
    cldw[cldw < 1e-7] = np.nan
    clds[clds < 1e-7] = np.nan
    cldr[cldr < 1e-7] = np.nan
    if lat_lon is not None:
        lat_val, lon_val = lat_lon
        valid_indices = np.isin(np.round(lat,2), np.round(lat_val,2)) & np.isin(np.round(lon,2), np.round(lon_val,2))
        height = height[:, valid_indices]
        cldi = cldi[:, valid_indices]
        cldw = cldw[:, valid_indices]
        clds = clds[:, valid_indices]
        cldr = cldr[:, valid_indices]
        temp = temp[:, valid_indices]
        lat = lat[valid_indices]
        lon = lon[valid_indices]
    if len(path_traj) == 2:
        un_idx = np.unique(lat, return_index=True)[1]
        cldi = cldi[:, un_idx]
        cldw = cldw[:, un_idx]
        clds = clds[:, un_idx]
        cldr = cldr[:, un_idx]
        height = height[:, un_idx]
        temp = temp[:, un_idx]
        lat = lat[un_idx]
        lon = lon[un_idx]
    norm = mcolors.SymLogNorm(linthresh=1e-10, vmin=1e-7, vmax=1e-3) 
    cbar_ax1 = fig.add_axes([0.46, -0.03, 0.42, 0.03])
    orientation = 'horizontal'
    ax3, ax5, ax7, ax9 = ax2.twiny(), ax4.twiny(), ax6.twiny(), ax8.twiny()
    x = np.tile(np.arange(0, len(lat), 1), (len(height),1))
    plot1 = ax2.pcolormesh(x, height, cldi, shading='auto', 
                        cmap=cmap, norm=norm, zorder=5)
    plot2 = ax4.pcolormesh(x, height, cldw, shading='auto', 
                        cmap=cmap, norm=norm, zorder=5)
    plot3 = ax6.pcolormesh(x, height, clds, shading='auto', 
                            cmap=cmap, norm=norm, zorder=5)
    plot4 = ax8.pcolormesh(x, height, cldr, shading='auto', 
                            cmap=cmap, norm=norm, zorder=5)
    tlevels = np.arange(-50, 20.1, 10)
    tempC = temp - 273.16
    t = ax8.contour(x, height, tempC, 
                     colors='black', #xkcd:bluegrey
                     levels = tlevels,
                    linewidths=0.5, linestyles='dashed')
    ax8.clabel(t, tlevels[::],  
                fmt='%1.0f', fontsize=6)
    ax2.contour(x, height, tempC, 
                     colors='black',
                     levels = tlevels,
                    linewidths=0.5, linestyles='dashed')
    ax4.contour(x, height, tempC, 
                     colors='black',
                     levels = tlevels,
                    linewidths=0.5, linestyles='dashed')
    ax6.contour(x, height, tempC, 
                     colors='black',
                     levels = tlevels,
                    linewidths=0.5, linestyles='dashed')
    cbar1 = fig.colorbar(plot1, cax=cbar_ax1, orientation=orientation,
                        label='Water content [kg m$^{-3}$]',
                        extend='max')
    cbar1.ax.tick_params(labelsize=14)
    height_reg = np.tile(np.arange(0,15.01,0.1), (len(lat),1))
    x_reg = np.tile(np.arange(0, len(lat), 1), (len(height_reg[0]),1)).T
    plot_topo(ax2, x_reg, height[-1], height_reg)
    plot_topo(ax4, x_reg, height[-1], height_reg)
    plot_topo(ax6, x_reg, height[-1], height_reg)
    plot_topo(ax8, x_reg, height[-1], height_reg)
    format_lat_lon_height(ax2, ax3, lat, lon, height=None, x1label=False)
    format_lat_lon_height(ax4, ax5, lat, lon, height=None, ylabel=False, x1label=False)
    format_lat_lon_height(ax6, ax7, lat, lon, height=None, x2label=False)
    format_lat_lon_height(ax8, ax9, lat, lon, height=None, ylabel=False, x2label=False)
    ax2.set_title('(b)', loc='left')
    ax4.set_title('(c)', loc='left')
    ax6.set_title('(d)', loc='left')
    ax8.set_title('(e)', loc='left')
    ax2.set_title('Ice water content')
    ax4.set_title('Liquid water content')
    ax6.set_title('Snow water content')
    ax8.set_title('Rain water content')
    ax2.tick_params(labelbottom=False, bottom=False)
    ax4.tick_params(labelleft=False, left=False,
                    labelbottom=False, bottom=False)
    ax7.tick_params(labeltop=False, top=False)
    ax9.tick_params(labeltop=False, top=False)
    ax8.tick_params(labelleft=False, left=False)
    fig.subplots_adjust(top=0.85, right=0.88)
    if add_loc != None:
        ax2.text(10, 14, add_loc, fontsize=10)
    fig.tight_layout()
    if savedir != None:
        plt.savefig(f"{savedir}racmo_overview_{date.strftime('%Y%m%d%H%M')}.png", 
                    bbox_inches='tight', dpi=300, format='png')