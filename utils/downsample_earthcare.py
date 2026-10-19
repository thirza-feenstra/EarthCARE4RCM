import numpy as np
import xarray as xr
from scipy.ndimage import label, binary_dilation
import Convert_From_To_RacmoGrid as CFTR
import RCM_gridSettings as RCMG
import warnings
import helpers
import importlib
import os
import sys
import RCM2EC_functions
from datetime import datetime
importlib.reload(RCM2EC_functions)
from tqdm import tqdm
from cmcrameri import cm as ccm
import matplotlib.colors as mcolors
warnings.filterwarnings("ignore")


def downsample_earthcare(acmpath, compath, cprpath, dfpath, racpath, domain, 
                         user, save_path=None, save_name=None, save_path_quality=None, 
                         save_name_quality=None):
    if domain == "PXANT11":
        extent = [-90,-40,-180,180]
    elif domain[:7] == "FGRN055":   
        extent = [50,90,-110,35]
    varlist_acm = ['ice_water_content', 'ice_mass_flux', 'rain_rate', 'liquid_water_content',
            'ice_effective_radius', 'liquid_effective_radius', 'liquid_number_concentration',
            'ice_normalized_number_concentration']    
    varunitlist_acm = ['kg m-3', 'kg m-2 s-1', 'kg m-2 s-1', 'kg m-3',
                'm', 'm', 'm-3', 'm-4']    
    varlongnamelist_acm = ['Ice water content', 'Ice mass flux (snowfall rate)',
                    'Rainfall rate', 'Liquid water content',
                    'Ice effective radius', 'Liquid effective radius',
                    'Cloud droplet number concentration (liquid)',
                    'Ice crystal number concentration (normalized)']
    varlist_acm_1D = ['tropopause_height']
    varunitlist_acm_1D = ['m']
    varlongnamelist_acm_1D = ['Topopause height']
    varlist_com = ['ice_water_content', 'snow_water_content', 'rain_rate', 'liquid_water_content',
            'ice_effective_radius', 'liquid_effective_radius']    
    varunitlist_com = ['kg m-3', 'kg m-3', 'kg m-2 s-1', 'kg m-3',
                'm', 'm']
    varlongnamelist_com = ['Ice water content', 'Snow water content',
                    'Rainfall rate', 'Liquid water content',
                    'Ice effective radius', 'Liquid effective radius']
    varlist_cpr = ['sedimentation_velocity']
    varunitlist_cpr = ['m s-1']
    varlongnamelist_cpr = ['Sedimentation velocity']
    varlist_df = ['flux_thermal_1d', 'flux_solar_1d', 'effective_flux_thermal_3d',
                  'effective_flux_solar_3d', 'BBR_effective_flux_thermal',
                  'BBR_effective_flux_solar']    
    varunitlist_df = ['W m$^{-2}$', 'W m$^{-2}$', 'W m$^{-2}$', 'W m$^{-2}$',
                      'W m$^{-2}$', 'W m$^{-2}$']
    varlongnamelist_df = ['Thermal flux 1D modeled', 'Solar flux 1D modeled',
                          'Thermal flux 3D modeled', 'Solar flux 3D modeled',
                          'Thermal flux BBR', 'Solar flux BBR']
    if domain == "PXANT11":
        gridfile = RCMG.read_Setting_data(domain=domain, 
                                        path=f'{user}/settings')
    elif domain[:7] == "FGRN055":   
        gridfile = RCMG.read_Setting_data(domain=domain, 
                                        path=f'{user}/settings')
    acm = xr.open_dataset(acmpath)
    com = xr.open_dataset(compath).sel(atmosphere=0)
    cpr = xr.open_dataset(cprpath)
    df = xr.open_dataset(dfpath).squeeze()
    rac = xr.open_dataset(racpath)
    (latec_r, lonec_r), hec, latec, lonec = RCM2EC_functions.get_EC_traj(acm, extent, gridfile)
    latrac_r = rac.rlat.values
    lonrac_r = rac.rlon.values
    hrac = rac.height.values
    latrac, lonrac = CFTR.RotatedGrid2RealWorld(latrac_r, lonrac_r, gridfile)
    ds = xr.Dataset(data_vars={
                "lat": (["along_track"], latrac),
                "lon": (["along_track"], lonrac),
                "rlat": (["along_track"], latrac_r),
                "rlon": (["along_track"], lonrac_r),
                "height": (["h_ix","along_track"], hrac)
            },
            coords={"along_track": np.arange(len(latrac)), "h_ix": np.arange(len(hrac))}
        )
    for var, varunit, varlongname in zip(varlist_acm, varunitlist_acm, varlongnamelist_acm):
        v, (filtlat, filtlon) = RCM2EC_functions.downsample(RCM2EC_functions.get_EC_r(acm, var, extent, hrac), 
                                        latec, lonec, 
                                        RCM2EC_functions.get_EC_r(acm, var, extent,
                                                                    hrac, return_H=True)[1],
                                        latrac_r, lonrac_r, hrac, gridfile=gridfile, fillna=False)
        if len(ds.lat.values) != len(v[0]):
            coords = np.column_stack((np.round(latrac,2), np.round(lonrac,2)))  
            allowed = np.column_stack((np.round(filtlat,2), np.round(filtlon,2))) 
            coords_view   = coords.view([('lat', coords.dtype), ('lon', coords.dtype)])
            allowed_view  = allowed.view([('lat', allowed.dtype), ('lon', allowed.dtype)])
            mask = np.array(np.isin(coords_view, allowed_view)).squeeze()
            v_new = np.full(np.shape(hrac), np.nan)
            v_new[:,mask == True] = v
            v = v_new
        ds[var] = (("h_ix","along_track"), v)
        ds[var] = ds[var].assign_attrs(units = varunit, long_name = varlongname)
    for var, varunit, varlongname in zip(varlist_acm_1D, varunitlist_acm_1D, varlongnamelist_acm_1D):
        v, (filtlat, filtlon) = RCM2EC_functions.downsample_1D(RCM2EC_functions.get_EC_r(acm, var, extent), 
                                        latec, lonec, latrac_r, lonrac_r, gridfile=gridfile, fillna=False)
        if len(ds.lat.values) != len(v):
            coords = np.column_stack((np.round(latrac,2), np.round(lonrac,2)))  
            allowed = np.column_stack((np.round(filtlat,2), np.round(filtlon,2))) 
            coords_view   = coords.view([('lat', coords.dtype), ('lon', coords.dtype)])
            allowed_view  = allowed.view([('lat', allowed.dtype), ('lon', allowed.dtype)])
            mask = np.array(np.isin(coords_view, allowed_view)).squeeze()
            v_new = np.full(np.shape(latrac), np.nan)
            v_new[mask == True] = v
            v = v_new
        ds[var] = (("along_track"), v)
        ds[var] = ds[var].assign_attrs(units = varunit, long_name = varlongname)
    for var, varunit, varlongname in zip(varlist_com, varunitlist_com, varlongnamelist_com):
        v, (filtlat, filtlon) = RCM2EC_functions.downsample(RCM2EC_functions.get_EC_r(com, var, extent, hrac), 
                                        latec, lonec, 
                                        RCM2EC_functions.get_EC_r(com, var, extent,
                                                                    hrac, return_H=True)[1],
                                        latrac_r, lonrac_r, hrac, gridfile=gridfile, fillna=False)
        if len(ds.lat.values) != len(v[0]):
            coords = np.column_stack((np.round(latrac,2), np.round(lonrac,2)))  
            allowed = np.column_stack((np.round(filtlat,2), np.round(filtlon,2))) 
            coords_view   = coords.view([('lat', coords.dtype), ('lon', coords.dtype)])
            allowed_view  = allowed.view([('lat', allowed.dtype), ('lon', allowed.dtype)])
            mask = np.array(np.isin(coords_view, allowed_view)).squeeze()
            v_new = np.full(np.shape(hrac), np.nan)
            v_new[:,mask == True] = v
            v = v_new
        ds[var+'_com'] = (("h_ix","along_track"), v)
        ds[var+'_com'] = ds[var+'_com'].assign_attrs(units = varunit, long_name = varlongname+' (ACM-COM)')
    for var, varunit, varlongname in zip(varlist_cpr, varunitlist_cpr, varlongnamelist_cpr):
        v, (filtlat, filtlon) = RCM2EC_functions.downsample(RCM2EC_functions.get_EC_r(cpr, var, extent, hrac), 
                                        latec, lonec, 
                                        RCM2EC_functions.get_EC_r(cpr, var, extent,
                                                                    hrac, return_H=True)[1],
                                        latrac_r, lonrac_r, hrac, gridfile=gridfile, fillna=False)
        if len(ds.lat.values) != len(v[0]):
            coords = np.column_stack((np.round(latrac,2), np.round(lonrac,2)))  
            allowed = np.column_stack((np.round(filtlat,2), np.round(filtlon,2))) 
            coords_view   = coords.view([('lat', coords.dtype), ('lon', coords.dtype)])
            allowed_view  = allowed.view([('lat', allowed.dtype), ('lon', allowed.dtype)])
            mask = np.array(np.isin(coords_view, allowed_view)).squeeze()
            v_new = np.full(np.shape(hrac), np.nan)
            v_new[:,mask == True] = v
            v = v_new
        ds[var] = (("h_ix","along_track"), v)
        ds[var] = ds[var].assign_attrs(units = varunit, long_name = varlongname)
    for var, varunit, varlongname in zip(varlist_df, varunitlist_df, varlongnamelist_df):
        v, (filtlat, filtlon) = RCM2EC_functions.downsample_1D(RCM2EC_functions.get_EC_r(df, var, extent), 
                                                                latec, lonec, latrac_r, lonrac_r, gridfile=gridfile,
                                                                fillna=False)
        if len(ds.lat.values) != len(v):
            coords = np.column_stack((np.round(latrac,2), np.round(lonrac,2)))  
            allowed = np.column_stack((np.round(filtlat,2), np.round(filtlon,2))) 
            coords_view   = coords.view([('lat', coords.dtype), ('lon', coords.dtype)])
            allowed_view  = allowed.view([('lat', allowed.dtype), ('lon', allowed.dtype)])
            mask = np.array(np.isin(coords_view, allowed_view)).squeeze()
            v_new = np.full(np.shape(latrac), np.nan)
            v_new[mask == True] = v
            v = v_new
        ds[var] = (("along_track"), v)
        ds[var] = ds[var].assign_attrs(units = varunit, long_name = varlongname)
    
    msi = RCM2EC_functions.get_EC_r(acm, 'MSI_shortwave_albedo_assimilation_status', extent)
    msi[msi > 0] = 1
    msi = RCM2EC_functions.downsample_1D(msi, latec, lonec, latrac_r, lonrac_r, gridfile)[0]
    msi[msi > 0.1] = 1
    msi[msi <= 0.1] = 0
    quality = RCM2EC_functions.get_EC_r(acm, 'quality_status', extent)
    quality[quality < 8] = 0
    quality[quality >= 8] = 1
    quality, (filtlat, filtlon) = RCM2EC_functions.downsample_1D(quality, latec, lonec, 
                                                                    latrac_r, lonrac_r, gridfile)
    quality[quality > 0.1] = 1
    quality[quality <= 0.1] = 0    
    quality_com = RCM2EC_functions.get_EC_r(com, 'quality_status', extent)
    quality_com[quality_com == 4] = 0
    quality_com[quality_com == 1] = 0
    quality_com[quality_com == 2] = 1
    quality_com[quality_com == 3] = 1
    quality_com, (filtlat, filtlon) = RCM2EC_functions.downsample_1D(quality_com, latec, lonec, 
                                                                    latrac_r, lonrac_r, gridfile)
    quality_com[quality_com > 0.1] = 1
    quality_com[quality_com <= 0.1] = 0    
    tph = ds['tropopause_height'].values
    tph_mask = np.zeros(np.shape(hrac))
    for i in range(len(tph)):
        abovetph = np.where(hrac[:,i] > tph[i])[0]
        if abovetph.size > 0:
            idx = abovetph[0]
            tph_mask[:idx,i] = 1
            
    structure = np.array([[1, 1, 1],
                        [1, 1, 1], 
                        [0, 0, 0]])
    n_it = 6
    max_size = 5000
    high_mask_n = (ds.ice_water_content.values > 3e-5) & (np.isfinite(ds.ice_water_content.values))
    low_mask_n  = (ds.ice_water_content.values > 5e-6) & (np.isfinite(ds.ice_water_content.values))
    iwc_quality = np.zeros(np.shape(low_mask_n))
    labeled, num = label(high_mask_n, structure=np.array([[1,1,1],[1,1,1],[1,1,1]]))
    for k in range(1, num + 1):
        region = (labeled == k)
        dilated = binary_dilation(region, structure=structure, iterations=n_it) 
        if np.sum(low_mask_n[dilated]) >= 0.8*(np.sum(structure)*n_it): 
            continue
        else:
            ys, xs = np.where(region)
            if len(np.unique(ys)) <= 4:
                iwc_quality[region] = 1
            if np.sum(region*hrac) <= max_size:
                iwc_quality[region] = 1
    mask1 = np.full(np.shape(iwc_quality), 1, dtype=float)
    mask1[iwc_quality == 1] = np.nan
    high_mask_n = (ds.ice_water_content.values*mask1 > 1e-4) & (np.isfinite(ds.ice_water_content.values*mask1))
    low_mask_n  = (ds.ice_water_content.values*mask1 > 1e-5) & (np.isfinite(ds.ice_water_content.values*mask1))
    labeled, num = label(high_mask_n, structure=np.array([[1,1,1],[1,1,1],[1,1,1]]))
    for k in range(1, num + 1):
        region = (labeled == k)
        dilated = binary_dilation(region, structure=structure)
        if np.sum(low_mask_n[dilated]) >= (np.sum(structure)-1):
            continue
        else:
            ys, xs = np.where(region)
            if len(np.unique(ys)) <= 4:
                iwc_quality[region] = 1
            if np.sum(region*hrac) <= max_size:
                iwc_quality[region] = 1
    iwc_quality[ds.ice_water_content.values > 8e-3] = 1
    iwc_quality[ds.ice_water_content.values > 8e-3] = 1
    
    # masking for DEM problems
    height_low = hrac.copy()
    dh_low = np.abs(height_low[:-1] - height_low[1:])
    iwc_dem_mask = np.full(np.shape(ds.ice_water_content.values), 0, dtype=float)
    lwc_dem_mask = np.full(np.shape(ds.liquid_water_content.values), 0, dtype=float)
    iwc_low = ds.ice_water_content.values.copy()
    iwc_low[hrac<2000] = 0
    iwpn_low = np.nansum(0.5*(iwc_low[:-1] + iwc_low[1:])*dh_low, axis=0)
    iwpn_low = np.tile(iwpn_low, (40, 1))
    # print(iwpn_low)
    if len(iwc_dem_mask[(hrac[-1] >= 100) & (hrac[-1] <= 2000) & (hrac < hrac[-1] + 1000) & (iwpn_low < 0.01) & (ds.ice_water_content.values > 1e-7)]) > 0:
        iwc_dem_mask[(hrac[-1] >= 100) & (hrac[-1] <= 2000) & (hrac < hrac[-1] + 1000) & (iwpn_low < 0.01) & (ds.ice_water_content.values > 1e-7)] = 1
    lwc_low = ds.liquid_water_content.values.copy()
    lwc_low[hrac<2000] = 0
    lwpn_low = np.nansum(0.5*(lwc_low[:-1] + lwc_low[1:])*dh_low, axis=0)
    lwpn_low = np.tile(lwpn_low, (40, 1))
    if len(lwc_dem_mask[(hrac[-1] >= 100) & (hrac[-1] <= 2000) & (hrac < hrac[-1] + 1000) & (lwpn_low < 0.01) & (ds.liquid_water_content.values > 1e-7)]) > 0:
        lwc_dem_mask[(hrac[-1] >= 100) & (hrac[-1] <= 2000) & (hrac < hrac[-1] + 1000) & (lwpn_low < 0.01) & (ds.liquid_water_content.values > 1e-7)] = 1
    dh = np.abs(ds.height.values[:-1] - ds.height.values[1:])
    lwc = ds.liquid_water_content.values
    lwc[lwc_dem_mask == 1] = np.nan
    lwp_comp = np.nansum(0.5*(lwc[:-1] + lwc[1:])*dh , axis=0)
    ds['liquid_water_path'] = (("along_track"), lwp_comp)
    ds['liquid_water_path'] = ds['liquid_water_path'].assign_attrs(units = 'kg m-2', long_name = 'Liquid water path')
    iwc = ds.ice_water_content.values
    iwc[iwc_dem_mask == 1] = np.nan
    iwc[iwc_quality == 1] = np.nan
    iwp_comp = np.nansum(0.5*(iwc[:-1] + iwc[1:])*dh , axis=0)
    ds['ice_water_path'] = (("along_track"), iwp_comp)
    ds['ice_water_path'] = ds['ice_water_path'].assign_attrs(units = 'kg m-2', long_name = 'Ice water path')

    lwc_com = ds.liquid_water_content_com.values
    lwc_com[lwc_dem_mask == 1] = np.nan
    lwp_comp_com = np.nansum(0.5*(lwc_com[:-1] + lwc_com[1:])*dh , axis=0)
    ds['liquid_water_path_com'] = (("along_track"), lwp_comp_com)
    ds['liquid_water_path_com'] = ds['liquid_water_path_com'].assign_attrs(units = 'kg m-2', long_name = 'Liquid water path (ACM-COM)')
    iwc_com = ds.ice_water_content.values
    iwc_com[iwc_dem_mask == 1] = np.nan
    iwp_comp_com = np.nansum(0.5*(iwc_com[:-1] + iwc_com[1:])*dh , axis=0)
    ds['ice_water_path_com'] = (("along_track"), iwp_comp_com)
    ds['ice_water_path_com'] = ds['ice_water_path_com'].assign_attrs(units = 'kg m-2', long_name = 'Ice water path (ACM-COM)')
    hclas = RCM2EC_functions.get_EC_r(cpr, 'hydrometeor_classification', extent, hrac)
    clutter = hclas.astype('float')
    clutter[~np.isfinite(clutter)] = 0
    clutter[(clutter < 16) | (clutter > 19)] = 0
    clutter[(clutter >= 16) & (clutter <=19)] = 1
    clutter, (filtlat, filtlon) = RCM2EC_functions.downsample(clutter, latec, lonec, 
                                        RCM2EC_functions.get_EC_r(cpr, 'hydrometeor_classification', extent,
                                                                    hrac, return_H=True)[1],
                                        latrac_r, lonrac_r, hrac, gridfile=gridfile, fillna=False)
    clutter[clutter > 0] = 1
    for i in range(len(clutter[0])):
        invalid = np.where(clutter[:,i] == 1)[0]
        if invalid.size > 0:
            terrain_idx = invalid[0]
            clutter[terrain_idx:,i] = 1
    clutter[clutter != 1] = 0
    sw_mod = ds.flux_solar_1d.values
    sw_obs = ds.BBR_effective_flux_solar.values
    dsw = np.abs(sw_mod - sw_obs)
    dsw[dsw <= 10] = 0
    dsw[dsw > 10] = 1
    dsw[np.isnan(sw_mod - sw_obs)] = np.nan
    lw_mod = ds.flux_thermal_1d.values
    lw_obs = ds.BBR_effective_flux_thermal.values
    dlw = np.abs(lw_mod - lw_obs)
    dlw[dlw <= 10] = 0
    dlw[dlw > 10] = 1
    dlw[np.isnan(lw_mod - lw_obs)] = np.nan
    drad = dlw + dsw
    drad[drad > 1] = 1
    drad[np.isnan(dlw)] = np.nan
    drad[(np.isnan(dsw) & np.isfinite(dlw))] = dlw[(np.isnan(dsw) & np.isfinite(dlw))]
    sw_stat = sw_obs.copy()
    sw_stat[(sw_obs == 0)] = 1
    sw_stat[(~np.isfinite(sw_obs))] = 1
    sw_stat[(sw_obs > 0)] = 0
    sw_stat[(np.isfinite(sw_obs))] = 0
    lw_stat = lw_obs.copy()
    lw_stat[(lw_obs == 0)] = 1
    lw_stat[(~np.isfinite(lw_obs))] = 1
    lw_stat[(lw_obs > 0)] = 0
    lw_stat[(np.isfinite(lw_obs))] = 0
    if len(latrac) != len(msi):
        coords = np.column_stack((np.round(latrac,2), np.round(lonrac,2)))  
        allowed = np.column_stack((np.round(filtlat,2), np.round(filtlon,2))) 
        coords_view = coords.view([('lat', coords.dtype), ('lon', coords.dtype)])
        allowed_view = allowed.view([('lat', allowed.dtype), ('lon', allowed.dtype)])
        mask = np.array(np.isin(coords_view, allowed_view)).squeeze()
        msi_new = np.full(np.shape(latrac), 1)
        msi_new[mask] = msi
        msi = msi_new
        quality_new = np.full(np.shape(latrac), 1)
        quality_new[mask] = quality
        quality = quality_new
        quality_com_new = np.full(np.shape(latrac), 1)
        quality_com_new[mask] = quality_com
        quality_com = quality_com_new
        clutter_new = np.full(np.shape(hrac), 1)
        clutter_new[:,mask] = clutter
        clutter = clutter_new
        clutter[clutter < 0] = 1
    dq = xr.Dataset(data_vars={
            "lat": (["along_track"], latrac),
            "lon": (["along_track"], lonrac),
            "rlat": (["along_track"], latrac_r),
            "rlon": (["along_track"], lonrac_r),
            "height": (["h_ix","along_track"], hrac),
            "msi_status": (["along_track"], msi),
            "rad_status": (["along_track"], drad),
            "quality_status_cap": (["along_track"], quality),
            "quality_status_com": (["along_track"], quality_com),
            "clutter_mask": (["h_ix", "along_track"], clutter),
            "shortwave_status": (["along_track"], sw_stat),
            "longwave_status": (["along_track"], lw_stat),
            "quality_iwc": (["h_ix","along_track"], iwc_quality),
            "iwc_dem_mask": (["h_ix","along_track"], iwc_dem_mask),
            "lwc_dem_mask": (["h_ix","along_track"], lwc_dem_mask),
            "tph_mask": (["h_ix","along_track"], tph_mask)
        },
        coords={"along_track": np.arange(len(latrac)), "h_ix": np.arange(len(hrac))}
    )
    dq["msi_status"] = dq["msi_status"].assign_attrs(long_name = "0: MSI assimilated, 1: MSI not assimilated")
    dq["quality_status_cap"] = dq["quality_status_cap"].assign_attrs(long_name = "0: usable, 1: unusable (ACM-CAP)")
    dq["quality_status_com"] = dq["quality_status_com"].assign_attrs(long_name = "0: usable, 1: unusable (ACM-COM)")
    dq["clutter_mask"] = dq["clutter_mask"].assign_attrs(long_name = "0: no clutter, 1: clutter")
    dq["rad_status"] = dq["rad_status"].assign_attrs(long_name = "0: radiation difference <= 10 W/m2, 1: radiation difference > 10 W/m2")
    dq["shortwave_status"] = dq["shortwave_status"].assign_attrs(long_name = "0: SW observed, 1: no SW observed")
    dq["longwave_status"] = dq["longwave_status"].assign_attrs(long_name = "0: LW observed, 1: no LW observed")
    dq["quality_iwc"] = dq["quality_iwc"].assign_attrs(long_name = "0: usable, 1: unusable (iwc spike detected in ACM-CAP)")
    dq["iwc_dem_mask"] = dq["iwc_dem_mask"].assign_attrs(long_name = "0: usable, 1: unusable (dem artifact detected)")
    dq["lwc_dem_mask"] = dq["lwc_dem_mask"].assign_attrs(long_name = "0: usable, 1: unusable (dem artifact detected)")
    dq["tph_mask"] = dq["tph_mask"].assign_attrs(long_name = "0: below tropopause height, 1: above tropopause height")
    if save_path_quality != None:
        ds.to_netcdf(save_path + save_name + '.nc')
        dq.to_netcdf(save_path_quality + save_name_quality + '.nc')
    return ds, dq

def keep_first_per_orb(file_list):
    seen = set()
    result = []
    for f in file_list:
        orb = f[-10:-5]
        if orb not in seen:
            seen.add(orb)
            result.append(f)
    return result

def downsample_all_ec(acmfolder, comfolder, cprfolder, dffolder, racfolder, domain, user):
    racmofiles = sorted((f for f in os.listdir(racfolder) if not f.startswith('.')))
    acmfiles = sorted((f for f in os.listdir(acmfolder) if not f.startswith('.')))
    print(len(acmfiles))
    comfiles = sorted((f for f in os.listdir(comfolder) if not f.startswith('.')))
    cprfiles = sorted((f for f in os.listdir(cprfolder) if not f.startswith('.')))
    dffiles = sorted((f for f in os.listdir(dffolder) if not f.startswith('.')))
    if domain == 'FGRN055' or domain == 'PXANT11':
        save_path = f'{user}/EarthCARE/{domain}/L2b_ds_{acmfolder[-7:-1]}/'
        save_path_quality = f'{user}/EarthCARE/{domain}/L2b_quality_{acmfolder[-7:-1]}/'
    elif domain == 'FGRN055_65' or domain == 'PXANT11_65':
        save_path = f'{user}/EarthCARE/{domain}/L2b_ds_{acmfolder[-7:-1]}_65/'
        save_path_quality = f'{user}/EarthCARE/{domain}/L2b_quality_{acmfolder[-7:-1]}_65/'
    acmfiles = keep_first_per_orb(acmfiles)
    comfiles = keep_first_per_orb(comfiles)
    cprfiles = keep_first_per_orb(cprfiles)
    dffiles  = keep_first_per_orb(dffiles)

    orb_ACM = [ACM[-10:-5] for ACM in acmfiles]
    orb_COM = [COM[-10:-5] for COM in comfiles]
    orb_CPR = [CPR[-10:-5] for CPR in cprfiles]
    orb_DF  = [DF[-10:-5]  for DF in dffiles]
    common_orbs = set(orb_ACM) & set(orb_CPR) & set(orb_DF) & set(orb_COM)

    acmfiles = [f for f in acmfiles if f[-10:-5] in common_orbs]
    comfiles = [f for f in comfiles if f[-10:-5] in common_orbs]
    cprfiles = [f for f in cprfiles if f[-10:-5] in common_orbs]
    dffiles  = [f for f in dffiles  if f[-10:-5] in common_orbs]
    ec_dates = []
    for EC in acmfiles:
        ec_dates.append(datetime.strptime(EC[20:35], '%Y%m%dT%H%M%S'))
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
    infiles_ACM = []
    infiles_COM = []
    infiles_CPR = []
    infiles_DF = []
    for idx in EC_idx:
        infiles_ACM.append(acmfiles[idx])
        infiles_COM.append(comfiles[idx])
        infiles_CPR.append(cprfiles[idx])
        infiles_DF.append(dffiles[idx])
    for acm, com, cpr, df, rac in zip(infiles_ACM, infiles_COM, infiles_CPR, infiles_DF, racmofiles):
        save_name = acm[:9] + 'CAP_CLD_DF__' + acm[17:-3] + 'd'
        save_name_quality = acm[:9] + 'CAP_CLD_DF__' + acm[17:-3] + 'q'
        # print(acm, com, cpr, df, rac)
        try:
            downsample_earthcare(acmfolder+acm, comfolder+com, cprfolder+cpr, 
                                 dffolder+df, racfolder+rac, domain, user, 
                                save_path, save_name, save_path_quality, 
                                save_name_quality)
        except Exception as e:
            print(f"exception {e} occurred", flush=True)
            continue
    return

def main():
    try:
        acmfolder = sys.argv[1]
        comfolder = sys.argv[2]
        cprfolder = sys.argv[3]
        dffolder = sys.argv[4] 
        racfolder = sys.argv[5] 
        domain = sys.argv[6]
        user = sys.argv[7]
    except IndexError:
        print('')
        print('Error: missing commandline inputs.')
        print('')
        print('Usage:')
        print(f'{sys.argv[0]} '+
              '<acmfolder> <comfolder> <cprfolder> <dffolder> <racfolder> <product><domain><user>')
        print('')
        sys.exit(1)
    downsample_all_ec(acmfolder, comfolder, cprfolder, dffolder, racfolder, domain, user)

if __name__ == "__main__":
    main()
    

