#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%

import numpy as np
import xarray as xr
import re
import os
from tqdm import tqdm
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from cmcrameri import cm as ccm
import matplotlib.colors as mcolors
from datetime import datetime
import sys
import importlib
sys.path.append('../utils/')
import RCM2EC_functions 
import plot_functions
import helpers
importlib.reload(plot_functions)
importlib.reload(helpers)
importlib.reload(RCM2EC_functions)
import warnings
warnings.filterwarnings("ignore")
#%% 
def extract_first_date(filename):
    '''extract first date from list of EarthCARE filenames
    (since EC filenames are observed time - process time)'''
    date_pattern = re.compile(r"(\d{8}T\d{6}Z)")
    match = date_pattern.search(filename)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ")
    return datetime.min
months = ['202409', '202410', '202411', '202412',
          '202501', '202502', '202503',
          '202504', '202505', '202506', '202507', '202508', '202509', '202510',
          '202511', '202512', '202601', '202602', '202603', '202604']

domain = 'PXANT11'
sim = 'bl_qb'
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
rac_iwc_list = []
rac_h_list = []
com_iwc_list = []
com_h_list = []
topo = True
if topo:
    dtopo = xr.open_dataset(f'{domain}_masks.nc', engine='netcdf4')
    rlat = dtopo.rlat.values
    rlon = dtopo.rlon.values
    rlat_grid, rlon_grid = np.meshgrid(rlat, rlon, indexing='ij')
    latlon_pairs = np.column_stack((rlat_grid.ravel(), rlon_grid.ravel()))
    latlon_tree = cKDTree(latlon_pairs)
    t_arr = dtopo.Topography.values
for e, q, r in tqdm(zip(infiles_ec_list, infiles_quality_list, infiles_racmo),
                    total=len(infiles_ec_list)):
    try:
        rac = xr.open_dataset(r)
        ec = xr.open_dataset(e)
        qu = xr.open_dataset(q)
    except:
        continue
    try:
        com_status = qu.quality_status_com.values
        clutter = qu.clutter_mask.values
        clutter = clutter.astype(float)
        clutter[clutter==1] = np.nan
        clutter[clutter==0] = 1
        iwc_dem = qu.iwc_dem_mask.values
        iwc_dem[iwc_dem==1] = np.nan
        iwc_dem[iwc_dem==0] = 1
        rac_iwc = helpers.kgkg_to_kgm3(rac.cldi.values + rac.clds.values,
                                rac.temp.values, 
                                rac.hum.values,
                                rac.p.values)*clutter*iwc_dem
        if topo:
            points = np.column_stack((rac.rlat.values, rac.rlon.values))
            _, idx = latlon_tree.query(points)
            ilat, ilon = np.unravel_index(idx, (len(rlat), len(rlon)))
            topo_height = t_arr[ilat,ilon]
            rac_h = (rac.height.values-topo_height)*clutter*iwc_dem
            com_h = (ec.height.values-topo_height)*clutter*iwc_dem
        else:
            rac_h = rac.height.values*clutter*iwc_dem
            com_h = ec.height.values*clutter*iwc_dem
        rac_iwc_list.append(rac_iwc[:,com_status == 0].flatten())
        rac_h_list.append(rac_h[:,com_status == 0].flatten())
        com_iwc = ec.ice_water_content_com.values*clutter*iwc_dem
        com_iwc_list.append(com_iwc[:,com_status == 0].flatten())
        com_h_list.append(com_h[:,com_status == 0].flatten())
    except:
        continue
rac_iwc_arr = np.array([xx for x in rac_iwc_list for xx in x])
rac_h_iwc_arr = np.array([xx for x in rac_h_list for xx in x])
com_iwc_arr = np.array([xx for x in com_iwc_list for xx in x])
com_h_arr = np.array([xx for x in com_h_list for xx in x])
ds = xr.Dataset(
    data_vars={
        "rac_iwc": (["points"], rac_iwc_arr),
        "rac_iwc_height": (["points"], rac_h_iwc_arr),
        "com_iwc": (["points"], com_iwc_arr),
        "com_iwc_height": (["points"], com_h_arr)
    },
    attrs={
        "description": "IWC and height arrays used for CFAD generation",
        "iwc_units": "kg m-3",
        "height_units": "m above surface",
    }
)
ds.to_netcdf(f"cfad_{domain}_topo.nc")
#%%
def make_cfad(iwc, h, height_bins, iwc_bins):
    total_counts, h_edges = np.histogram(h,bins=height_bins)
    valid = ((iwc > 0) &
            np.isfinite(iwc) &
            np.isfinite(h))
    log_iwc = np.log10(iwc[valid])
    H, _, iwc_edges = np.histogram2d(h[valid],log_iwc,
                                    bins=[height_bins, iwc_bins])
    H_norm = np.divide(H, total_counts[:, np.newaxis],
                        out=np.full_like(H, np.nan, dtype=float),
                        where=total_counts[:, np.newaxis] != 0)
    return H_norm, h_edges, iwc_edges, total_counts
#%%
domains = ['FGRN055','PXANT11']
results = {}
height_bins = np.arange(0, 16.001, 0.500)
iwc_bins = np.arange(-8.125, -2.3, 0.25)

for domain in domains:
    ds = xr.open_dataset(
        f"cfad_{domain}_topo.nc")
    rac_iwc_arr = ds.rac_iwc.values.astype(np.float32)
    rac_h_arr = ds.rac_iwc_height.values.astype(np.float32)/1000
    com_iwc_arr = ds.com_iwc.values.astype(np.float32)
    com_h_arr = ds.com_iwc_height.values.astype(np.float32)/1000
    H_com_iwc, h_edges, iwc_edges, _ = make_cfad(com_iwc_arr, com_h_arr, height_bins, iwc_bins)
    H_rac_iwc, _, _, _ = make_cfad(rac_iwc_arr, rac_h_arr, height_bins, iwc_bins)
    H_rac_iwc_nonan = H_rac_iwc.copy()*1/(iwc_bins[-1]-iwc_bins[-2])
    H_rac_iwc_nonan[np.isnan(H_rac_iwc_nonan)] = 0
    H_rac_iwc_nonan[H_rac_iwc_nonan < 0.00005] = 0
    H_com_iwc_nonan = H_com_iwc.copy()*1/(iwc_bins[-1]-iwc_bins[-2])
    H_com_iwc_nonan[np.isnan(H_com_iwc_nonan)] = 0
    H_com_iwc_nonan[H_com_iwc_nonan < 0.00005] = 0
    diff = H_rac_iwc_nonan - H_com_iwc_nonan
    diff[diff == 0] = np.nan
    results[domain] = {'com': H_com_iwc_nonan,
                        'rac': H_rac_iwc_nonan,
                        'diff': diff}
cmap = plt.get_cmap(ccm.vik)
colors = cmap(np.linspace(0, 1, 10))
fig, axes = plt.subplots(2, 4,figsize=(14,8.2),
                         width_ratios=[1,1,1,0.6],
                         sharey=True)
plt.subplots_adjust(hspace=0.1, wspace=0.1)
ax1, ax2, ax3, ax_l1 = axes[0]
ax4, ax5, ax6, ax_l2 = axes[1]
levels = np.arange(0,0.2501,0.01)
levels_diff = np.arange(-0.1,0.101,0.01)
iwc_widths = np.diff(iwc_edges)
row_info = [('FGRN055', axes[0]),
             ('PXANT11', axes[1])]
for domain, row_axes in row_info:
    datasets = [(results[domain]['com'], 'EarthCARE'),
                (results[domain]['rac'], 'RACMO'),
                (results[domain]['diff'], 'RACMO - EarthCARE')]
    for ax, (H, title) in zip(row_axes, datasets):
        H_nonzero = H.copy()
        H_nonzero[H_nonzero < 0.00005] = np.nan
        if ax != row_axes[2]:
            pcm = ax.contourf(
                (iwc_edges[:-1] + iwc_edges[1:]) / 2,
                (h_edges[:-1] + h_edges[1:]) / 2,
                H_nonzero,
                levels=levels,
                cmap='RdYlBu_r',
                extend='max',
                norm=mcolors.Normalize(
                    vmin=0,
                    vmax=0.25))
        else:
            diff_plot = ax.contourf(
                (iwc_edges[:-1] + iwc_edges[1:]) / 2,
                (h_edges[:-1] + h_edges[1:]) / 2,
                H,
                levels=levels_diff,
                cmap=ccm.vik,
                extend='both',
                norm=mcolors.Normalize(
                    vmin=-0.1,
                    vmax=0.1))
        if (ax==ax1) or (ax==ax2) or (ax==ax3):
            ax.set_title(title)
            ax.tick_params(bottom=False, labelbottom=False)
        if (ax==ax2) or (ax==ax3) or (ax==ax5) or (ax==ax6):
            ax.tick_params(left=False)
        if (ax==ax4) or (ax==ax5) or (ax==ax6):
            ax.set_xlabel(r'$\log_{10}(\mathrm{IWC}\,[kg\,m^{-3}])$')
            ax.set_xticks([-8,-7,-6,-5,-4,-3])
        if (ax==ax1) or (ax==ax4):
            ax.set_yticks([0,2,4,6,8,10,12,14,16])
            ax.set_ylabel('Height above surface [km]')
        ax.axhline(1, color='black', ls='--')
        ax.hlines([2, 4, 6, 8, 10, 12, 14], xmin=-8, xmax=-2.5, color='grey', linestyle='--',
                linewidth=0.5)
        ax.vlines([-7, -6, -5, -4, -3], ymin=0, ymax=16, color='grey', linestyle='--',
                linewidth=0.5)
        ax.set_ylim(0,16)
    if domain == 'FGRN055':
        ax_l1.plot(np.round(np.sum(datasets[0][0]* iwc_widths[None, :], axis=1),2),(h_edges[:-1] + h_edges[1:]) / 2,
                   label='EarthCARE', color=colors[2])
        ax_l1.plot(np.round(np.sum(datasets[1][0]* iwc_widths[None, :], axis=1),2),(h_edges[:-1] + h_edges[1:]) / 2,
                   label='RACMO', color=colors[-3])
        ax_l1.legend(frameon=False, handlelength=1)
        ax_l1.tick_params(left=False, bottom=False, labelbottom=False)
        ax_l1.set_xlim(0,0.6)
        ax_l1.axhline(1, color='black', ls='--')
        ax_l1.set_ylim(0,16)
        ax_l1.hlines([2, 4, 6, 8, 10, 12, 14], xmin=0, xmax=0.6, color='grey', linestyle='--',
                linewidth=0.5)
        ax_l1.vlines([0.2,0.4], ymin=0, ymax=16, color='grey', linestyle='--',
                linewidth=0.5)
    elif domain == 'PXANT11':
        ax_l2.plot(np.round(np.sum(datasets[0][0]* iwc_widths[None, :], axis=1),2),(h_edges[:-1] + h_edges[1:]) / 2,
                   color=colors[2])
        ax_l2.plot(np.round(np.sum(datasets[1][0]* iwc_widths[None, :], axis=1),2),(h_edges[:-1] + h_edges[1:]) / 2,
                   color=colors[-3])
        ax_l2.tick_params(left=False)
        ax_l2.set_xlabel('Ice cloud\noccurrence [-]')
        ax_l2.set_xlim(0,0.6)
        ax_l2.axhline(1, color='black', ls='--')
        ax_l2.set_ylim(0,16)
        ax_l2.hlines([2, 4, 6, 8, 10, 12, 14], xmin=0, xmax=0.6, color='grey', linestyle='--',
                linewidth=0.5)
        ax_l2.vlines([0.2,0.4], ymin=0, ymax=16, color='grey', linestyle='--',
                linewidth=0.5)
ax1.set_title('(a)',loc='left')
ax2.set_title('(b)',loc='left')
ax3.set_title('(c)',loc='left')
ax_l1.set_title('(d)',loc='left')
ax4.set_title('(e)',loc='left')
ax5.set_title('(f)',loc='left')
ax6.set_title('(g)',loc='left')
ax_l2.set_title('(h)',loc='left')
ax1.text(
    -0.3, 0.5,
    'Greenland',
    rotation=90,
    va='center',
    ha='center',
    transform=ax1.transAxes,
    fontsize=18)
ax4.text(
    -0.3, 0.5,
    'Antarctica',
    rotation=90,
    va='center',
    ha='center',
    transform=ax4.transAxes,
    fontsize=18)
cbar_ax = fig.add_axes([0.95, 0.10, 0.02, 0.78])
cb = fig.colorbar(
    pcm,
    cax=cbar_ax,
    label='Occurrence density [per unit log$_{10}$(IWC)]',
    orientation='vertical')
cbar_ax2 = fig.add_axes([1.06, 0.10, 0.02, 0.78])
cb2 = fig.colorbar(
    diff_plot,
    cax=cbar_ax2,
    label='Difference in occurrence density [per unit log$_{10}$(IWC)]',
    orientation='vertical')
plt.savefig(f'GrIS_Ant_CFAD_IWC.png', 
            dpi=300, bbox_inches='tight', format='png')

