#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import numpy as np
import xarray as xr
from scipy import stats
import sys
sys.path.append('../utils/')
import warnings
from scipy.spatial import cKDTree
import helpers
import importlib
import os
import sys
from datetime import datetime
import stat_functions
import scale_fluxes
importlib.reload(stat_functions)
importlib.reload(scale_fluxes)
warnings.filterwarnings("ignore")

def compute_stats_overpass(ecpath, qualitypath, racpath, rac_swdnpath, rac_swuppath, 
                           topo, siconca, vs=1, vi=0.13,
                           thr_wc=1e-6, thr_wc2=1e-4, thr_wp=1e-3, thr_wp2=1e-1):
    ec = xr.open_dataset(ecpath)
    quality = xr.open_dataset(qualitypath)
    racmo = xr.open_dataset(racpath)
    rlat = racmo.rlat.values
    rlon = racmo.rlon.values
    topo_vals = topo.Topography.values     
    lsm_vals  = topo.LSM.values   
    is_vals = topo.Icesheet_Only.values         
    ilat = np.abs(topo.rlat.values[:, None] - rlat[None, :]).argmin(axis=0)
    ilon = np.abs(topo.rlon.values[:, None] - rlon[None, :]).argmin(axis=0)
    landsea = lsm_vals[ilat, ilon]  
    topo_vals = topo_vals[ilat, ilon]
    height = racmo.height.values          
    landmask = landsea[None, :]
    oceanmask = (1 - landsea)[None, :]
    landmask[landmask>=0.5] = 1
    landmask[landmask<0.5] = np.nan
    oceanmask[oceanmask>=0.5] = 1
    oceanmask[oceanmask<0.5] = np.nan
    icesheet = is_vals[ilat,ilon]
    icesheetmask = icesheet[None,:]
    icesheetmask[icesheetmask>=0.5] = 1
    icesheetmask[icesheetmask<0.5] = np.nan
    siconca = siconca.sel(time=racpath[-17:-13]+'-'+racpath[-13:-11]+'-'+racpath[-11:-9], method='nearest').squeeze()
    sice = siconca.siconca.values.copy()[ilat, ilon]
    sice_mask = sice[None, :]
    sice_mask[sice_mask >= 0.15] = 1
    sice_mask[sice_mask < 0.15] = np.nan
    sice_inv = siconca.siconca.values.copy()[ilat, ilon]
    sice_mask_inv = sice_inv[None, :]
    sice_mask_inv[sice_mask_inv >= 0.15] = np.nan
    sice_mask_inv[sice_mask_inv < 0.15] = 1
    openoceanmask = oceanmask*sice_mask_inv
    quality_mask = np.where(quality.quality_status_com.values == 0, 1.0, np.nan)
    clutter_mask = np.where(quality.clutter_mask.values == 0, 1.0, np.nan)
    sw_mask = np.where(quality.shortwave_status.values == 0, 1.0, np.nan)
    lw_mask = np.where(quality.longwave_status.values == 0, 1.0, np.nan)
    iwc_dem_mask = np.where(quality.iwc_dem_mask.values == 0, 1.0, np.nan)
    lwc_dem_mask = np.where(quality.lwc_dem_mask.values == 0, 1.0, np.nan)
    iwc_ec = ec.ice_water_content_com.values * quality_mask*clutter_mask*iwc_dem_mask
    iwc_rac = (helpers.kgkg_to_kgm3(racmo.cldi.values, racmo.temp.values,
                                racmo.hum.values, racmo.p.values)
            + helpers.kgkg_to_kgm3(racmo.clds.values, racmo.temp.values,
                                racmo.hum.values, racmo.p.values)) \
            * quality_mask * clutter_mask*iwc_dem_mask
    lwc_ec = ec.liquid_water_content.values * quality_mask *clutter_mask*lwc_dem_mask
    lwc_rac = helpers.kgkg_to_kgm3(racmo.cldw.values, racmo.temp.values,
                                racmo.hum.values, racmo.p.values) \
            * quality_mask *  clutter_mask*lwc_dem_mask
    lwc_ec = ec.liquid_water_content.values * quality_mask * clutter_mask*lwc_dem_mask
    lwc_rac = helpers.kgkg_to_kgm3(racmo.cldw.values, racmo.temp.values,
                                racmo.hum.values, racmo.p.values) \
            * quality_mask * clutter_mask*lwc_dem_mask
    dh = np.abs(height[:-1] - height[1:])
    riwp = np.nansum(0.5*(iwc_rac[1:] + iwc_rac[:-1]) *iwc_dem_mask[1:] * dh, axis=0)
    eiwp = ec.ice_water_path_com.values
    rlwp = np.nansum(0.5*(lwc_rac[1:] + lwc_rac[:-1])*lwc_dem_mask[1:] * dh, axis=0)
    elwp = ec.liquid_water_path.values
    ritop = np.full(len(riwp), np.nan)
    eitop = np.full(len(riwp), np.nan)
    rltop = np.full(len(riwp), np.nan)
    eltop = np.full(len(riwp), np.nan)
    for l in range(len(ritop)):
        height_top = height[:,l]
        riwc = iwc_rac.copy()[:,l]
        riwc[~np.isfinite(riwc)] = 0
        eiwc = iwc_ec.copy()[:,l]
        eiwc[~np.isfinite(eiwc)] = 0
        rlwc = lwc_rac.copy()[:,l]
        rlwc[~np.isfinite(rlwc)] = 0
        elwc = lwc_ec.copy()[:,l]
        elwc[~np.isfinite(elwc)] = 0
        if len(height_top.copy()[riwc >= 1e-6]):
            ritop[l] = np.nanmax(height_top.copy()[riwc >= 1e-6])
        if len(height_top.copy()[eiwc >= 1e-6]):
            eitop[l] = np.nanmax(height_top.copy()[eiwc >= 1e-6])
        if len(height_top.copy()[rlwc >= 1e-6]):
            rltop[l] = np.nanmax(height_top.copy()[rlwc >= 1e-6])
        if len(height_top.copy()[elwc >= 1e-6]):
            eltop[l] = np.nanmax(height_top.copy()[elwc >= 1e-6])
    rr_ec = ec.rain_rate.values * quality_mask * clutter_mask
    rho = helpers.density(racmo.temp.values, racmo.hum.values, racmo.p.values)
    vr = helpers.rain_fallspeed(racmo.cldr.values, rho)
    rr_rac = helpers.kgkg_to_kgm3(racmo.cldr.values, racmo.temp.values,
                                racmo.hum.values, racmo.p.values) \
            * vr * quality_mask * clutter_mask
    sr_ec = ec.snow_water_content_com.values * ec.sedimentation_velocity.values * quality_mask * clutter_mask *iwc_dem_mask
    icesnow = np.zeros(np.shape(racmo.clds.values))
    icesnow[racmo.clds.values > thr_wc] = racmo.cldi.values[racmo.clds.values > thr_wc]
    if vs == 'BP':
        vs = helpers.compute_vs_BP(racmo.temp.values, racmo.p.values)
    elif vs == 'T_joppe':
        vs = helpers.compute_vs_T_joppe(racmo.temp.values)
    if vi == 'heymsfield':
        rho = helpers.density(racmo.temp.values, racmo.hum.values, racmo.p.values)
        vi = helpers.ice_fallspeed(racmo.cldi.values, rho)
    sr_rac = (helpers.kgkg_to_kgm3(racmo.clds.values, racmo.temp.values,
                                racmo.hum.values, racmo.p.values) \
            * vs + helpers.kgkg_to_kgm3(icesnow, racmo.temp.values,
                                racmo.hum.values, racmo.p.values) * vi) \
                                    * quality_mask * clutter_mask *iwc_dem_mask
    # need rain and snow at 1 km above the surface
    height_idx = np.zeros(len(sr_rac[0]))
    for k in range(len(height_idx)):
        height_idx[k] = np.where(height[:,k] > topo_vals[k]+1000)[0][-1]
    height_idx = np.array(height_idx, dtype=int)
    rr_ec = rr_ec[height_idx, np.arange(rr_ec.shape[1])]
    rr_rac = rr_rac[height_idx, np.arange(rr_rac.shape[1])]
    sr_ec = sr_ec[height_idx, np.arange(sr_ec.shape[1])]
    sr_rac = sr_rac[height_idx, np.arange(sr_rac.shape[1])]
    sw_ec = ec.BBR_effective_flux_solar.values * quality_mask * sw_mask
    lw_ec = ec.BBR_effective_flux_thermal.values * quality_mask * lw_mask
    rac_lwup_toa = racmo.lwupa.values[0]
    rac_lwdn_toa = racmo.lwdna.values[0]
    rac_swnet_toa = racmo.swnet.values[0]
    rac_lwnet_toa = racmo.lwnet.values[0]
    racmo_swdn_hr = xr.open_dataset(rac_swdnpath)
    racmo_swup_hr = xr.open_dataset(rac_swuppath)
    racmo_date = datetime.strptime(racpath[-17:-5], '%Y%m%d%H%M')
    racmo_date = racmo_date.replace(minute=0, second=0, microsecond=0)
    racmo_swup = racmo_swup_hr.sel(time=racmo_date).squeeze()
    racmo_swdn = racmo_swdn_hr.sel(time=racmo_date).squeeze()
    points = np.column_stack((racmo.rlat.values, racmo.rlon.values))
    rlat_grid_2D, rlon_grid_2D = np.meshgrid(racmo_swdn_hr.rlat.values, racmo_swdn_hr.rlon.values)
    latlon_pairs = np.column_stack((rlat_grid_2D.ravel(), rlon_grid_2D.ravel()))
    latlon_tree = cKDTree(latlon_pairs)
    nn_coords = latlon_pairs[latlon_tree.query(points)[1]]
    lat_indices = {lat: np.where(racmo_swdn_hr.rlat.values == lat)[0] for lat in np.unique(nn_coords[:, 0])}
    lon_indices = {lon: np.where(racmo_swdn_hr.rlon.values == lon)[0] for lon in np.unique(nn_coords[:, 1])}
    fulloutput_swuptoa_traj = np.zeros(len(racmo.swupa.values[0]))
    fulloutput_swdntoa_traj = np.zeros(len(racmo.swdna.values[0]))
    for j, (lat, lon) in enumerate(nn_coords[np.sort(np.unique(nn_coords, axis=0, return_index=True)[1])]):
        i_lat = lat_indices[lat][0]
        j_lon = lon_indices[lon][0]
        fulloutput_swuptoa_traj[j] = racmo_swup.rsut.values[i_lat,j_lon]
        fulloutput_swdntoa_traj[j] = racmo_swdn.rsdt.values[i_lat,j_lon]
    sw_rac = scale_fluxes.scale_shortwave_fluxes(fulloutput_swdntoa_traj, fulloutput_swuptoa_traj, rac_swnet_toa)[1]
    lw_rac = rac_lwup_toa
    sw_rac *= quality_mask 
    sw_rac *= sw_mask
    lw_rac *= quality_mask 
    lw_rac *= lw_mask
    bias = np.full((10, 6), np.nan)
    se = np.full((10, 6), np.nan)
    log_bias = np.full((10, 6), np.nan)
    log_se = np.full((10, 6), np.nan)
    sum_ec = np.full((10, 6), np.nan)
    sum_rac = np.full((10, 6), np.nan)
    sum_ec_rac = np.full((10, 6), np.nan)
    sum_ec2 = np.full((10, 6), np.nan)
    sum_rac2 = np.full((10, 6), np.nan)
    sum_ec_log = np.full((10, 6), np.nan)
    sum_rac_log = np.full((10, 6), np.nan)
    sum_ec_rac_log = np.full((10, 6), np.nan)
    sum_ec2_log = np.full((10, 6), np.nan)
    sum_rac2_log = np.full((10, 6), np.nan)
    TPwc1 = np.full((10, 6), np.nan)
    FPwc1 = np.full((10, 6), np.nan)
    FNwc1 = np.full((10, 6), np.nan)
    TNwc1 = np.full((10, 6), np.nan)
    TPwc2 = np.full((10, 6), np.nan)
    FPwc2 = np.full((10, 6), np.nan)
    FNwc2 = np.full((10, 6), np.nan)
    TNwc2 = np.full((10, 6), np.nan)
    data = [(iwc_rac, iwc_ec, thr_wc, thr_wc2, True, height),
            (lwc_rac, lwc_ec, thr_wc, thr_wc2, True, height),
            (riwp, eiwp, thr_wp, thr_wp2, True, None),
            (rlwp, elwp, thr_wp, thr_wp2, True, None),
            (ritop, eitop, 0, 0, False, None),
            (rltop, eltop, 0, 0, False, None),
            (sr_rac, sr_ec, thr_wc, thr_wc2, True, None),
            (rr_rac, rr_ec, thr_wc, thr_wc2, True, None),
            (sw_rac, sw_ec, 0, 0, False, None),
            (lw_rac, lw_ec, 0, 0, False, None)]
    masks = [None, landmask, oceanmask, icesheetmask, openoceanmask, sice_mask]
    for j, mask in enumerate(masks):
        for i, (mod, obs, thr, thr2, skill, w) in enumerate(data):
            m = mod if mask is None else mod * mask
            o = obs if mask is None else obs * mask
            if skill:
                b, s, lb, ls, se_sum, sm_sum, smx, se2, sm2, lse_sum, lsm_sum, lsmx, lse2, lsm2, tp1, fp1, fn1, tn1, tp2, fp2, fn2, tn2 = \
                    stat_functions.compute_bias_se_r2_skill(
                        m, o, thr, thr2, skill=True, weights=w)
            else:
                b, s, lb, ls, se_sum, sm_sum, smx, se2, sm2, lse_sum, lsm_sum, lsmx, lse2, lsm2 = \
                    stat_functions.compute_bias_se_r2_skill(
                        m, o, thr, skill=False)
            bias[i, j] = b
            se[i, j] = s
            log_bias[i, j] = lb
            log_se[i, j] = ls
            sum_ec[i, j] = se_sum
            sum_rac[i, j] = sm_sum
            sum_ec_rac[i, j] = smx
            sum_ec2[i, j] = se2
            sum_rac2[i, j] = sm2
            sum_ec_log[i, j] = lse_sum
            sum_rac_log[i, j] = lsm_sum
            sum_ec_rac_log[i, j] = lsmx
            sum_ec2_log[i, j] = lse2
            sum_rac2_log[i, j] = lsm2
            if skill:
                TPwc1[i, j] = tp1
                FPwc1[i, j] = fp1
                FNwc1[i, j] = fn1
                TNwc1[i, j] = tn1
                TPwc2[i, j] = tp2
                FPwc2[i, j] = fp2
                FNwc2[i, j] = fn2
                TNwc2[i, j] = tn2
    n = np.array([len(rlat),
            np.nansum(landmask),
            np.nansum(oceanmask),
            np.nansum(icesheetmask),
            np.nansum(openoceanmask),
            np.nansum(sice_mask)])
    n_cond = np.zeros((10, 6))
    for j, mask in enumerate(masks):
        n_cond[0, j] = np.sum((iwc_rac* (1 if mask is None else mask) > thr_wc) &
                            (iwc_ec * (1 if mask is None else mask) > thr_wc))
        n_cond[1, j] = np.sum((lwc_rac* (1 if mask is None else mask) > thr_wc) &
                            (lwc_ec * (1 if mask is None else mask) > thr_wc))
        n_cond[2, j] = np.sum((riwp* (1 if mask is None else mask) > thr_wp) &
                            (eiwp * (1 if mask is None else mask) > thr_wp))
        n_cond[3, j] = np.sum((rlwp* (1 if mask is None else mask) > thr_wp) &
                            (elwp * (1 if mask is None else mask) > thr_wp))
        n_cond[4, j] = np.sum((ritop* (1 if mask is None else mask) > 0) &
                            (eitop * (1 if mask is None else mask) > 0))
        n_cond[5, j] = np.sum((rltop* (1 if mask is None else mask) > 0) &
                            (eltop * (1 if mask is None else mask) > 0))
        n_cond[6, j] = np.sum((sr_rac* (1 if mask is None else mask) > thr_wc) &
                            (sr_ec * (1 if mask is None else mask) > thr_wc))
        n_cond[7, j] = np.sum((rr_rac* (1 if mask is None else mask) > thr_wc) &
                            (rr_ec * (1 if mask is None else mask) > thr_wc))
        n_cond[8, j] = np.sum((sw_rac* (1 if mask is None else mask) > 0) &
                            (sw_ec * (1 if mask is None else mask) > 0))
        n_cond[9, j] = np.sum((lw_rac* (1 if mask is None else mask) > 0) &
                            (lw_ec * (1 if mask is None else mask) > 0))
    return (bias, se, log_bias, log_se,
            sum_ec, sum_rac, sum_ec_rac, sum_ec2, sum_rac2,
            sum_ec_log, sum_rac_log, sum_ec_rac_log, sum_ec2_log, sum_rac2_log,
            TPwc1, FPwc1, FNwc1, TNwc1, TPwc2, FPwc2, FNwc2, TNwc2, n, n_cond, thr_wc, thr_wc2,
            thr_wp, thr_wp2)
    
def stat_file(ecfolder, qualityfolder, racfolder, rac_swdnpath, 
              rac_swuppath, savepath, siconca, topo,
              vs=1, vi=0.13):
    racmofiles = sorted((f for f in os.listdir(racfolder) if not f.startswith('.')))
    ecfiles = sorted((f for f in os.listdir(ecfolder) if not f.startswith('.')))
    qfiles = sorted((f for f in os.listdir(qualityfolder) if not f.startswith('.')))
    ec_dates = []
    q_dates = []
    for EC in ecfiles:
        ec_dates.append(datetime.strptime(EC[24:39], '%Y%m%dT%H%M%S'))
    for Q in qfiles:
        q_dates.append(datetime.strptime(Q[24:39], '%Y%m%dT%H%M%S'))
    pop_idx_ec = []
    pop_idx_q = []
    for i in range(len(ecfiles)):
        if ec_dates[i] not in q_dates:
            pop_idx_ec.append(i)
    for idx in pop_idx_ec[::-1]:
        ecfiles.pop(idx)
        ec_dates.pop(idx)
    for i in range(len(qfiles)):
        if q_dates[i] not in ec_dates:
            pop_idx_q.append(i)
    for idx in pop_idx_q[::-1]:
        qfiles.pop(idx)
        q_dates.pop(idx)
    remove_idx = []
    EC_idx = []
    for i,rac in enumerate(racmofiles):
        racmo_date = datetime.strptime(rac[8:-5], '%Y%m%d%H%M')
        if np.min(np.abs(np.array(ec_dates) - racmo_date)) < np.timedelta64(30, 'm'):
            EC_idx.append(np.argmin(np.abs(np.array(ec_dates) - racmo_date)))
        else:
            remove_idx.append(i)
    for j in range(len(remove_idx)):
        racmofiles.pop(np.flip(remove_idx)[j])
    ec_files = []
    q_files = []
    for idx in EC_idx:
        ec_files.append(ecfiles[idx])
        q_files.append(qfiles[idx])
    n_orb = len(ec_files)
    bias = np.full((10, 6, n_orb), np.nan)
    se = np.full((10, 6, n_orb), np.nan)
    log_bias = np.full((10, 6, n_orb), np.nan)
    log_se = np.full((10, 6, n_orb), np.nan)
    sum_ec = np.full((10, 6, n_orb), np.nan)
    sum_rac = np.full((10, 6, n_orb), np.nan)
    sum_ec_rac = np.full((10, 6, n_orb), np.nan)
    sum_ec2 = np.full((10, 6, n_orb), np.nan)
    sum_rac2 = np.full((10, 6, n_orb), np.nan)
    sum_ec_log = np.full((10, 6, n_orb), np.nan)
    sum_rac_log = np.full((10, 6, n_orb), np.nan)
    sum_ec_rac_log = np.full((10, 6, n_orb), np.nan)
    sum_ec2_log = np.full((10, 6, n_orb), np.nan)
    sum_rac2_log = np.full((10, 6, n_orb), np.nan)
    TPwc1 = np.full((10, 6, n_orb), np.nan)
    FPwc1 = np.full((10, 6, n_orb), np.nan)
    FNwc1 = np.full((10, 6, n_orb), np.nan)
    TNwc1 = np.full((10, 6, n_orb), np.nan)
    TPwc2 = np.full((10, 6, n_orb), np.nan)
    FPwc2 = np.full((10, 6, n_orb), np.nan)
    FNwc2 = np.full((10, 6, n_orb), np.nan)
    TNwc2 = np.full((10, 6, n_orb), np.nan)
    n = np.full((6, n_orb), np.nan)
    n_cond = np.full((10, 6, n_orb), np.nan)
    orb = np.full(n_orb, np.nan)
    date = np.full(n_orb, np.nan)
    count = 0
    for i, (r, e, q) in enumerate(zip(racmofiles, ec_files, q_files)):
        try:
            stats = compute_stats_overpass(
                ecfolder+e, qualityfolder+q, racfolder+r, rac_swdnpath, rac_swuppath, 
                topo=topo, siconca=siconca, vs=vs, vi=vi)
            (bias[:,:,i], se[:,:,i], log_bias[:,:,i], log_se[:,:,i],
            sum_ec[:,:,i], sum_rac[:,:,i], sum_ec_rac[:,:,i],
            sum_ec2[:,:,i], sum_rac2[:,:,i],
            sum_ec_log[:,:,i], sum_rac_log[:,:,i], sum_ec_rac_log[:,:,i],
            sum_ec2_log[:,:,i], sum_rac2_log[:,:,i],
            TPwc1[:,:,i], FPwc1[:,:,i], FNwc1[:,:,i], TNwc1[:,:,i], 
            TPwc2[:,:,i], FPwc2[:,:,i], FNwc2[:,:,i], TNwc2[:,:,i], 
            n[:,i], n_cond[:,:,i], thr_wc1, thr_wc2, thr_wp1, thr_wp2) = stats
        except:
            count += 1
            continue
        orb[i] = e[-11:-6]
        date[i] = str(r[-17:-5])
    ls = np.array(['All', 'Land', 'Ocean', 'Ice Sheet', 'Open ocean', 'Sea ice'])
    vars = np.array(['IWC', 'LWC', 'IWP', 'LWP', 'Ice cloud top', 'Liquid cloud top',
                     'Snow rate', 'Rain rate', 'SWup TOA', 'LWup TOA'])
    varunit = np.array([
                '[kg m$^{-3}$]', '[kg m$^{-3}$]',
                '[kg m$^{-2}$]', '[kg m$^{-2}$]',
                '[m]', '[m]',
                '[kg m$^{-2}$ s$^{-1}$]', '[kg m$^{-2}$ s$^{-1}$]',
                '[W m$^{-2}$]', '[W m$^{-2}$]'])
    ds = xr.Dataset(
        data_vars={
            "date": (["orbit"], date),
            "bias": (["variable", "LS", "orbit"], bias),
            "se": (["variable", "LS", "orbit"], se),
            "log_bias": (["variable", "LS", "orbit"], log_bias),
            "log_se": (["variable", "LS", "orbit"], log_se),
            "sum_ec": (["variable", "LS", "orbit"], sum_ec),
            "sum_rac": (["variable", "LS", "orbit"], sum_rac),
            "sum_ec_rac": (["variable", "LS", "orbit"], sum_ec_rac),
            "sum_ec2": (["variable", "LS", "orbit"], sum_ec2),
            "sum_rac2": (["variable", "LS", "orbit"], sum_rac2),
            "sum_ec_log": (["variable", "LS", "orbit"], sum_ec_log),
            "sum_rac_log": (["variable", "LS", "orbit"], sum_rac_log),
            "sum_ec_rac_log": (["variable", "LS", "orbit"], sum_ec_rac_log),
            "sum_ec2_log": (["variable", "LS", "orbit"], sum_ec2_log),
            "sum_rac2_log": (["variable", "LS", "orbit"], sum_rac2_log),
            "TPwc1": (["variable", "LS", "orbit"], TPwc1),
            "FPwc1": (["variable", "LS", "orbit"], FPwc1),
            "FNwc1": (["variable", "LS", "orbit"], FNwc1),
            "TNwc1": (["variable", "LS", "orbit"], TNwc1),
            "TPwc2": (["variable", "LS", "orbit"], TPwc2),
            "FPwc2": (["variable", "LS", "orbit"], FPwc2),
            "FNwc2": (["variable", "LS", "orbit"], FNwc2),
            "TNwc2": (["variable", "LS", "orbit"], TNwc2),
            "n": (["LS", "orbit"], n),
            "n_cond": (["variable", "LS", "orbit"], n_cond),
            "varunit": (["variable"], varunit)
        },
        coords={"orbit": orb, "LS": ls, "variable": vars}
    )
    ds["TPwc1"] = ds["TPwc1"].assign_attrs(long_name = f"thr WC = {thr_wc1} kg/m3, thr WP = {thr_wp1} kg/m2")
    ds["FPwc1"] = ds["FPwc1"].assign_attrs(long_name = f"thr WC = {thr_wc1} kg/m3, thr WP = {thr_wp1} kg/m2")
    ds["FNwc1"] = ds["FNwc1"].assign_attrs(long_name = f"thr WC = {thr_wc1} kg/m3, thr WP = {thr_wp1} kg/m2")
    ds["TNwc1"] = ds["TNwc1"].assign_attrs(long_name = f"thr WC = {thr_wc1} kg/m3, thr WP = {thr_wp1} kg/m2")
    ds["TPwc2"] = ds["TPwc2"].assign_attrs(long_name = f"thr WC = {thr_wc2} kg/m3, thr WP = {thr_wp2} kg/m2")
    ds["FPwc2"] = ds["FPwc2"].assign_attrs(long_name = f"thr WC = {thr_wc2} kg/m3, thr WP = {thr_wp2} kg/m2")
    ds["FNwc2"] = ds["FNwc2"].assign_attrs(long_name = f"thr WC = {thr_wc2} kg/m3, thr WP = {thr_wp2} kg/m2")
    ds["TNwc2"] = ds["TNwc2"].assign_attrs(long_name = f"thr WC = {thr_wc1} kg/m3, thr WP = {thr_wp1} kg/m2")
    ds.to_netcdf(savepath)
    print(count)
    return ds
#%%
months = ['202409', '202410', '202411', '202412', '202501', '202502', '202503',
          '202504', '202505', '202506', '202507', '202508', '202509', '202510',
          '202511', '202512', '202601', '202602', '202603', '202604']
domain = 'FGRN055' # 'PXANT11'
sim = 'bl' # 'bl_qb'
topo = xr.open_dataset(f'/perm/nld4845/masks_grids/{domain}_masks.nc', engine='netcdf4')
for month in months:
    siconcafile = xr.open_dataset(f'/ec/res4/scratch/nld4845/experiment/{domain}/RACMO2.4p1_v4_{sim}-{month[:4]}/NC_DEFAULT/siconca.KNMI-{month[:4]}.{domain}.RACMO2.4p1_v4_{sim}.DD.nc')
    ecfolder = f'/ec/res4/scratch/nld4845/EarthCARE/{domain}/L2b_ds_{month}/'
    qualityfolder = f'/ec/res4/scratch/nld4845/EarthCARE/{domain}/L2b_quality_{month}/'
    racfolder = f'/ec/res4/scratch/nld4845/experiment/{domain}/RACMO2.4p1_v4_{sim}/EC_traj_{month}/'
    rac_swdnpath = f'/ec/res4/scratch/nld4845/experiment/{domain}/RACMO2.4p1_v4_{sim}-{month[:4]}/NC_1H/rsdt.KNMI-{month[:4]}.{domain}.RACMO2.4p1_v4_{sim}.1H.nc'
    rac_swuppath = f'/ec/res4/scratch/nld4845/experiment/{domain}/RACMO2.4p1_v4_{sim}-{month[:4]}/NC_1H/rsut.KNMI-{month[:4]}.{domain}.RACMO2.4p1_v4_{sim}.1H.nc'
    savepath = f'/ec/res4/scratch/nld4845/EarthCARE/{domain}/stats/stats_{domain}_{sim}_{month}.nc'
    stat_file(ecfolder, qualityfolder, racfolder, rac_swdnpath, rac_swuppath, savepath, siconcafile, topo=topo, vs=1)

