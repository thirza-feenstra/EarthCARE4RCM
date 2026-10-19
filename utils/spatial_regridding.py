
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree
import warnings
from datetime import timedelta
import helpers
import RCM2EC_functions
from tqdm import tqdm
warnings.filterwarnings("ignore")


def conditional_grid_file_3D(func, var_list, lat_grid, lon_grid, 
                             rlat_grid, rlon_grid, h_grid, path, 
                             infiles, 
                             maskpath=None, maskfiles=None,
                             qmask='cap',
                             msimask=False, radmask=False,
                             iwcmask=False, iwcdemmask=False,
                             lwcdemmask=False,
                             no_msi=False,
                             topo=None, type='cond', save=True,
                             savepath=None, savename=None,
                             vs=None, wrt_topo=False):
    '''
    Maps all overpasses onto the RACMO 3D grid.
    Input:
    - func (one of specified below)
    - varlist: names of variables that are inputed to func
    - lat_grid: latitude of the RCM grid
    - lon_grid: longitude of the RCM grid
    - rlat_grid: rotated latitude of the RCM grid
    - rlon_grid: rotated longitude of the RCM grid
    - h_grid: heights of the 3D you want to map on
    - path: path to input files
    - infiles: list with names of input files
    - optional masks as described in maskfiles
    - topo: topography of the grid
    - wrt_topo: if True output defined as height above surface
    - vs: fall speed of snow in case this is used from RCM
    - type: cond = condition 0/1, sum = sum all up (e.g. for IWC)
    '''
    rlat_grid_2D, rlon_grid_2D = np.meshgrid(rlat_grid, rlon_grid)
    overpasses = np.zeros((len(lat_grid), len(lat_grid[0]), len(h_grid)))
    cond_result = np.zeros((len(lat_grid), len(lat_grid[0]), len(h_grid)))
    latlon_pairs = np.column_stack((rlat_grid_2D.ravel(), rlon_grid_2D.ravel()))
    latlon_tree = cKDTree(latlon_pairs)
    lat_min, lat_max = np.min(rlat_grid), np.max(rlat_grid)
    lon_min, lon_max = np.min(rlon_grid), np.max(rlon_grid)
    count = 0
    for i,f in enumerate(tqdm(infiles, desc="Processing files")):
        d = xr.open_dataset(path + f, engine="netcdf4")
        if maskpath != None:
            fmask = xr.open_dataset(maskpath+maskfiles[i])
            clutter_mask = np.where(fmask.clutter_mask.values == 0, 1.0, np.nan)
            if msimask:
                msi_mask = np.where(fmask.msi_status.values == 0, 1.0, np.nan)
            if qmask == 'cap':
                quality_mask = np.where(fmask.quality_status_cap.values == 0, 1.0, np.nan)
            if qmask == 'com':
                quality_mask = np.where(fmask.quality_status_com.values == 0, 1.0, np.nan)
            if radmask:
                rad_mask = np.where(fmask.rad_status.values == 0, 1.0, np.nan)
            if iwcmask:
                iwc_mask = np.where(fmask.quality_iwc.values == 0, 1.0, np.nan)
            if iwcdemmask:
                iwc_dem_mask = np.where(fmask.iwc_dem_mask.values == 0, 1.0, np.nan)
            if lwcdemmask:
                lwc_dem_mask = np.where(fmask.lwc_dem_mask.values == 0, 1.0, np.nan)
            if no_msi:
                no_msi_mask = np.where(fmask.msi_status.values == 1, 1.0, np.nan)
        latin = d.rlat.values
        lonin = d.rlon.values
        hin = d.height.values
        if len(latin) != len(quality_mask):
            unequal = True
            print("length not equal")
            print(len(latin), len(quality_mask))
            coords = np.column_stack((np.round(latin,2), np.round(lonin,2)))  
            allowed = np.column_stack((np.round(fmask.rlat.values,2), np.round(fmask.rlon.values,2))) 
            coords_view   = coords.view([('lat', coords.dtype), ('lon', coords.dtype)])
            allowed_view  = allowed.view([('lat', allowed.dtype), ('lon', allowed.dtype)])
            mask = np.array(np.isin(coords_view, allowed_view)).squeeze()
            latin = latin[mask == True] 
            lonin = lonin[mask == True] 
            hin = hin[:,mask == True] 
            print("new shape: ", np.shape(latin))
        else:
            unequal = False
        
        valid_mask = (latin >= lat_min) & (latin <= lat_max) & (lonin >= lon_min) & (lonin <= lon_max)
        if np.any(valid_mask):  
            valid_indices = np.where(valid_mask)[0]
            latin_valid = latin[valid_indices]
            lonin_valid = lonin[valid_indices]
            latlon_input = np.column_stack((latin_valid, lonin_valid))
            dist, tree_idx = latlon_tree.query(latlon_input)
            nn_coords = latlon_pairs[tree_idx]
            lat_indices = {lat: np.where(rlat_grid == lat)[0][0] for lat in np.unique(nn_coords[:, 0])}
            lon_indices = {lon: np.where(rlon_grid == lon)[0][0] for lon in np.unique(nn_coords[:, 1])}
            if not wrt_topo:
                if unequal:
                    var_data = [RCM2EC_functions.intp_heights(d[var].values[:,mask].T*clutter_mask[:, orig_idx], hin.T, h_grid).T for var in var_list]
                else:
                    var_data = [RCM2EC_functions.intp_heights(d[var].values.T*clutter_mask[:, orig_idx], hin.T, h_grid).T for var in var_list]
            for k, orig_idx in enumerate(valid_indices):
                overpasses_temp = np.zeros(len(h_grid))
                cond_result_temp = np.zeros(len(h_grid))
                lat, lon = nn_coords[k]
                i = lat_indices[lat]
                j_lon = lon_indices[lon]
                if maskpath is not None:
                    if msimask and not np.isfinite(msi_mask[orig_idx]):
                        continue
                    if qmask is not False and not np.isfinite(quality_mask[orig_idx]):
                        continue
                    if radmask and not np.isfinite(rad_mask[orig_idx]):
                        continue
                    if no_msi and not np.isfinite(no_msi_mask[orig_idx]):
                        continue
                topo_i = topo[i,j_lon]
                if not wrt_topo:
                    inp = np.array([var_data[v][:, orig_idx] for v in range(len(var_list))])
                    if iwcmask:
                        inp *= iwc_mask[:,orig_idx]
                    if iwcdemmask:
                        inp *= iwc_dem_mask[:, orig_idx]
                    if lwcdemmask:
                        inp *= lwc_dem_mask[:, orig_idx]
                else:
                    inp_profiles = []
                    for v, var in enumerate(var_list):
                        profile = d[var].values[:, orig_idx] * clutter_mask[:, orig_idx]
                        if iwcmask:
                            profile *= iwc_mask[:, orig_idx]
                        if iwcdemmask:
                            profile *= iwc_dem_mask[:, orig_idx]
                        if lwcdemmask:
                            profile *= lwc_dem_mask[:, orig_idx]    
                        h_rel = hin[:, orig_idx] - topo_i 
                        inp_profiles.append(RCM2EC_functions.intp_heights(
                            profile,
                            h_rel,
                            h_grid))
                    inp = np.array(inp_profiles)
                try:
                    if vs != None:
                        cond_result_temp = np.squeeze(func(inp, vs))
                    else:
                        cond_result_temp = np.squeeze(func(inp))
                except:
                    continue
                if not np.any(np.isfinite(cond_result_temp)):
                    continue
                if not wrt_topo:
                    cond_result_temp[h_grid < topo_i] = 0
                if type in ['cond', 'mean']:
                    overpasses_temp[:] = 1
                elif type == 'sum':
                    overpasses_temp[cond_result_temp > 0] = 1
                if not wrt_topo:
                    overpasses_temp[h_grid < topo_i] = 0
                cond_result[i, j_lon] += cond_result_temp
                overpasses[i, j_lon] += overpasses_temp
        count += 1
    print(count)    
    if wrt_topo:
        h_out = h_grid[None, None, :] + topo[:, :, None]
    else:
        grid = np.ones(np.shape(overpasses))
        h_out = h_grid[None, None, :] * grid
    res = cond_result/overpasses
    res[~np.isfinite(res)] = np.nan
    if type == 'cond':       
        ds = xr.Dataset(
            data_vars={
                "lat": (["lat_ix", "lon_ix"], lat_grid),
                "lon": (["lat_ix", "lon_ix"], lon_grid),
                "height": (["lat_ix", "lon_ix", "h_ix"], h_out),
                "overpasses": (["lat_ix", "lon_ix", "h_ix"], overpasses),
                "cond_result": (["lat_ix", "lon_ix", "h_ix"], cond_result),
                "fraction": (["lat_ix", "lon_ix", "h_ix"], res)
            },
            coords={"lat_ix": np.arange(len(lat_grid)), "lon_ix": np.arange(len(lat_grid[0])), "h_ix": np.arange(len(h_out[0][0]))}
        )
    else:
        ds = xr.Dataset(
            data_vars={
                "lat": (["lat_ix", "lon_ix"], lat_grid),
                "lon": (["lat_ix", "lon_ix"], lon_grid),
                "height": (["lat_ix", "lon_ix", "h_ix"], h_out),
                "overpasses": (["lat_ix", "lon_ix", "h_ix"], overpasses),
                "cond_result": (["lat_ix", "lon_ix", "h_ix"], cond_result),
                "mean": (["lat_ix", "lon_ix", "h_ix"], res)
            },
            coords={"lat_ix": np.arange(len(lat_grid)), "lon_ix": np.arange(len(lat_grid[0])), "h_ix": np.arange(len(h_out[0][0]))}
        )
    if save:
        ds.to_netcdf(savepath + savename + ".nc")
    return ds


def conditional_grid_file_2D(func, var_list, lat_grid, lon_grid, 
                             rlat_grid, rlon_grid, path, 
                             infiles, qmask='cap',
                             msimask=False, radmask=False,
                             iwcmask=False, iwcdemmask=False,
                             lwcdemmask=False,
                             swmask=False, lwmask=False,
                             no_msi=False,
                             maskpath=None, maskfiles=None,
                             type='cond', save=True,
                             savepath=None, savename=None,
                             vert_int=False):
    '''
    Maps all overpasses onto the RACMO 2D grid.
    Input:
    - func (one of specified below)
    - varlist: names of variables that are inputed to func
    - lat_grid: latitude of the RCM grid
    - lon_grid: longitude of the RCM grid
    - rlat_grid: rotated latitude of the RCM grid
    - rlon_grid: rotated longitude of the RCM grid
    - path: path to input files
    - infiles: list with names of input files
    - optional masks as described in maskfiles
    - vs: fall speed of snow in case this is used from RCM
    - type: cond = condition 0/1, sum = sum all up (e.g. for IWC),
    mean = unconditional mean (for radiation, avg. water path)
    '''
    rlat_grid_2D, rlon_grid_2D = np.meshgrid(rlat_grid, rlon_grid)
    overpasses = np.zeros((len(lat_grid), len(lat_grid[0])))
    cond_result = np.zeros((len(lat_grid), len(lat_grid[0])))
    latlon_pairs = np.column_stack((rlat_grid_2D.ravel(), rlon_grid_2D.ravel()))
    latlon_tree = cKDTree(latlon_pairs)
    lat_min, lat_max = np.min(rlat_grid), np.max(rlat_grid)
    lon_min, lon_max = np.min(rlon_grid), np.max(rlon_grid)
    count = 0
    for i,f in enumerate(tqdm(infiles, desc="Processing files")):
        d = xr.open_dataset(path + f, engine="netcdf4")
        if maskpath != None:
            fmask = xr.open_dataset(maskpath+maskfiles[i])
            clutter_mask = np.where(fmask.clutter_mask.values == 0, 1.0, np.nan)
            if msimask:
                msi_mask = np.where(fmask.msi_status.values == 0, 1.0, np.nan)
            if qmask == 'cap':
                quality_mask = np.where(fmask.quality_status_cap.values == 0, 1.0, np.nan)
            if qmask == 'com':
                quality_mask = np.where(fmask.quality_status_com.values == 0, 1.0, np.nan)
            if radmask:
                rad_mask = np.where(fmask.rad_status.values == 0, 1.0, np.nan)
            if swmask:
                sw_mask = np.where(fmask.shortwave_status.values == 0, 1.0, np.nan)
            if lwmask:
                lw_mask = np.where(fmask.longwave_status.values == 0, 1.0, np.nan)
            if iwcmask:
                iwc_mask = np.where(fmask.quality_iwc.values == 0, 1.0, np.nan)
            if iwcdemmask:
                iwc_dem_mask = np.where(fmask.iwc_dem_mask.values == 0, 1.0, np.nan)
            if lwcdemmask:
                lwc_dem_mask = np.where(fmask.lwc_dem_mask.values == 0, 1.0, np.nan)
            if no_msi:
                no_msi_mask = np.where(fmask.msi_status.values == 1, 1.0, np.nan)
        latin = d.rlat.values
        lonin = d.rlon.values
        if len(latin) != len(quality_mask):
            unequal = True
            print("length not equal")
            print(len(latin), len(quality_mask))
            coords = np.column_stack((np.round(latin,2), np.round(lonin,2)))  
            allowed = np.column_stack((np.round(fmask.rlat.values,2), np.round(fmask.rlon.values,2))) 
            coords_view   = coords.view([('lat', coords.dtype), ('lon', coords.dtype)])
            allowed_view  = allowed.view([('lat', allowed.dtype), ('lon', allowed.dtype)])
            mask = np.array(np.isin(coords_view, allowed_view)).squeeze()
            latin = latin[mask == True] 
            lonin = lonin[mask == True] 
            print("new shape: ", np.shape(latin))
        else:
            unequal = False
        valid_mask = (latin >= lat_min) & (latin <= lat_max) & (lonin >= lon_min) & (lonin <= lon_max)
        if np.any(valid_mask):  
            valid_indices = np.where(valid_mask)[0]
            latin_valid = latin[valid_indices]
            lonin_valid = lonin[valid_indices]
            latlon_input = np.column_stack((latin_valid, lonin_valid))
            dist, tree_idx = latlon_tree.query(latlon_input)
            nn_coords = latlon_pairs[tree_idx]
            lat_indices = {lat: np.where(rlat_grid == lat)[0][0] for lat in np.unique(nn_coords[:, 0])}
            lon_indices = {lon: np.where(rlon_grid == lon)[0][0] for lon in np.unique(nn_coords[:, 1])}
            if isinstance(var_list[0], str):
                if unequal:
                    var_data = [d[var].values[:,mask] for var in var_list]
                else:
                    var_data = [d[var].values for var in var_list]
            for k, orig_idx in enumerate(valid_indices):
                cond_result_temp = 0
                overpasses_temp = 0
                lat, lon = nn_coords[k]
                i_lat = lat_indices[lat]
                j_lon = lon_indices[lon]
                if maskpath is not None:
                    if msimask and not np.isfinite(msi_mask[orig_idx]):
                        continue
                    if qmask is not False and not np.isfinite(quality_mask[orig_idx]):
                        continue
                    if radmask and not np.isfinite(rad_mask[orig_idx]):
                        continue
                    if swmask and not np.isfinite(sw_mask[orig_idx]):
                        continue
                    if lwmask and not np.isfinite(lw_mask[orig_idx]):
                        continue
                    if no_msi and not np.isfinite(no_msi_mask[orig_idx]):
                        continue
                if isinstance(var_list[0], str):
                    if vert_int:
                        inp = np.array([var_data[v][:, orig_idx] * (clutter_mask[:, orig_idx]) for v in range(len(var_list))])
                        if iwcmask:
                            inp *= iwc_mask[:, orig_idx]
                        if iwcdemmask:
                            inp *= iwc_dem_mask[:, orig_idx]
                        if lwcdemmask:
                            inp *= lwc_dem_mask[:, orig_idx]
                    elif len(np.shape(var_data)) == 3:
                        inp = np.array([var_data[v][0, orig_idx] for v in range(len(var_list))])
                    else:
                        inp = np.array([var_data[v][orig_idx] for v in range(len(var_list))])
                else:
                    inp = var_list
                try:
                    cond_result_temp = np.squeeze(func(inp))
                except:
                    try:
                        cond_result_temp = np.squeeze(func(inp[i],d.lon.values[orig_idx]))
                    except:
                        continue
                if not np.isfinite(cond_result_temp):
                    continue
                if type in ['cond', 'mean']:
                    overpasses_temp = 1
                elif type == 'sum':
                    overpasses_temp = 1 if cond_result_temp > 0 else 0
                cond_result[i_lat, j_lon] += cond_result_temp
                overpasses[i_lat, j_lon] += overpasses_temp
            count += 1
    print(count) 
    res = cond_result/overpasses
    res[~np.isfinite(res)] = np.nan
    if type == 'cond':       
        ds = xr.Dataset(
            data_vars={
                "lat": (["lat_ix", "lon_ix"], lat_grid),
                "lon": (["lat_ix", "lon_ix"], lon_grid),
                "overpasses": (["lat_ix", "lon_ix"], overpasses),
                "cond_result": (["lat_ix", "lon_ix"], cond_result),
                "fraction": (["lat_ix", "lon_ix"], res)
            },
            coords={"lat_ix": np.arange(len(lat_grid)), "lon_ix": np.arange(len(lat_grid[0]))}
        )
    else:
        ds = xr.Dataset(
            data_vars={
                "lat": (["lat_ix", "lon_ix"], lat_grid),
                "lon": (["lat_ix", "lon_ix"], lon_grid),
                "overpasses": (["lat_ix", "lon_ix"], overpasses),
                "cond_result": (["lat_ix", "lon_ix"], cond_result),
                "mean": (["lat_ix", "lon_ix"], res)
            },
            coords={"lat_ix": np.arange(len(lat_grid)), "lon_ix": np.arange(len(lat_grid[0]))}
        )
    if save:
        ds.to_netcdf(savepath + savename + ".nc")
    return ds

def snow_cloud_pixel(inp, vs):
    swc, temp, hum, p = inp
    swr = helpers.kgkg_to_kgm3(swc,temp, hum, p) * vs # x vs for fall speed
    res = np.zeros(np.shape(swr))
    res[swr > 1e-6] = 1
    return res

def snowfall_rate(inp, vs):
    swc, iwc, temp, hum,p = inp
    icesnow = np.zeros((np.shape(swc)))
    icesnow[swc >= 1e-6] = iwc[swc >= 1e-6]
    swr = helpers.kgkg_to_kgm3(swc,temp, hum, p) * vs + \
        helpers.kgkg_to_kgm3(icesnow,temp, hum, p) * 0.13 # x vs for fall speed
    swr[swr < 1e-6] = 0
    return swr

def snowfall_rate_qb(inp, vs):
    swc, qb, iwc, temp, hum,p, height = inp
    icesnow = np.zeros((np.shape(swc)))
    icesnow[swc >= 1e-7] = iwc[swc >= 1e-7]
    vb = helpers.blowing_snow_fallspeed(p,
                                        temp,
                                        qb,
                                        height)
    swr = helpers.kgkg_to_kgm3(swc,temp, hum, p) * vs + \
        helpers.kgkg_to_kgm3(icesnow,temp, hum, p) * 0.13 + \
        helpers.kgkg_to_kgm3(qb,temp,hum,p) * vb # x vs for fall speed
    swr[swr < 1e-7] = 0
    return swr

def snowfall_rate_EC_com(inp):
    swc, vsed = inp
    swr = swc * vsed
    swr[swr < 1e-6] = 0
    swr[~np.isfinite(swr)] = 0
    return swr

def snow_cloud_pixel_EC_com(inp):
    swc, vsed = inp
    snow = swc * vsed
    res = np.zeros(np.shape(snow))
    res[snow >= 1e-6] = 1
    res[~np.isfinite(snow)] = 0
    return res

def IWP_L2b_EC(inp):
    iwp = inp
    if not np.isfinite(iwp):
        return np.nan
    else:
        return iwp

def IWP(inp):
    iwc, swc, temp, hum, p, height = inp
    iwc = iwc + swc
    iwc = helpers.kgkg_to_kgm3(iwc,temp, hum, p)
    dh = np.abs(height[:-1] - height[1:])
    iwc_layer = 0.5 * (iwc[:-1] + iwc[1:])
    iwp = np.nansum(iwc_layer*dh) 
    return iwp

def IWP_qb(inp):
    iwc, swc, qb, temp, hum, p, height = inp
    iwc = iwc + swc + qb
    iwc = helpers.kgkg_to_kgm3(iwc,temp, hum, p)
    dh = np.abs(height[:-1] - height[1:])
    iwc_layer = 0.5 * (iwc[:-1] + iwc[1:])
    iwp = np.nansum(iwc_layer*dh) 
    return iwp

def IWP_conv_snow(inp):
    iwc, swc, cons, temp, hum, p, height = inp
    iwc = iwc + swc + cons
    iwc = helpers.kgkg_to_kgm3(iwc,temp, hum, p)
    dh = np.abs(height[:-1] - height[1:])
    iwc_layer = 0.5 * (iwc[:-1] + iwc[1:])
    iwp = np.nansum(iwc_layer*dh) 
    return iwp

def LWP_L2b_EC(inp):
    lwp = inp
    if not np.isfinite(lwp):
        return np.nan
    else:
        return lwp

def LWP(inp):
    lwc, temp, hum, p, height = inp
    lwc = helpers.kgkg_to_kgm3(lwc,temp, hum, p)
    dh = np.abs(height[:-1] - height[1:])
    lwc_layer = 0.5 * (lwc[:-1] + lwc[1:])
    lwp = np.nansum(lwc_layer*dh) 
    return lwp

def cloud_top(inp):
    wc, temp, hum, p, height = inp
    wc = helpers.kgkg_to_kgm3(wc,temp, hum, p)
    wc[~np.isfinite(wc)] = 0
    height[~np.isfinite(height)] = 0
    wc_heights = height.copy()
    if len(wc_heights[wc >= 1e-6]) > 0:
        wc_heights = wc_heights[wc >= 1e-6]
        cloud_top = np.nanmax(wc_heights)
    else:
        cloud_top = np.nan
    return cloud_top

def cloud_top_EC(inp):
    wc, height = inp
    wc[~np.isfinite(wc)] = 0
    height[~np.isfinite(height)] = 0
    wc_heights = height.copy()
    if len(wc_heights[wc >= 1e-6]) > 0:
        wc_heights = wc_heights[wc >= 1e-6]
        cloud_top = np.nanmax(wc_heights)
    else:
        cloud_top = np.nan
    return cloud_top

def ice_occ_2D_EC(inp):
    iwp = inp
    if np.isnan(iwp):
        return np.nan
    elif iwp >= 1e-3:
        return 1
    elif iwp < 1e-3:
        return 0

def ice_occ_2D(inp):
    iwc, swc, temp, hum, p, height = inp
    iwc = iwc + swc
    iwc = helpers.kgkg_to_kgm3(iwc,temp, hum, p)
    dh = np.abs(height[:-1] - height[1:])
    iwc_layer = 0.5 * (iwc[:-1] + iwc[1:])
    iwp = np.nansum(iwc_layer*dh) 
    if np.all(np.isnan(iwc_layer)):
        return np.nan
    elif np.isnan(iwp):
        return np.nan
    elif iwp >= 1e-3:
        return 1
    elif iwp < 1e-3:
        return 0
    
def ice_occ_2D_qb(inp):
    iwc, swc, qb, temp, hum, p, height = inp
    iwc = iwc + swc + qb
    iwc = helpers.kgkg_to_kgm3(iwc,temp, hum, p)
    dh = np.abs(height[:-1] - height[1:])
    iwc_layer = 0.5 * (iwc[:-1] + iwc[1:])
    iwp = np.nansum(iwc_layer*dh) 
    if np.all(np.isnan(iwc_layer)):
        return np.nan
    elif np.isnan(iwp):
        return np.nan
    elif iwp >= 1e-3:
        return 1
    elif iwp < 1e-3:
        return 0

def liq_occ_2D_EC(inp):
    lwp = inp
    if np.isnan(lwp):
        return np.nan
    elif lwp >= 1e-3:
        return 1
    elif lwp < 1e-3:
        return 0

def liq_occ_2D(inp):
    lwc, temp, hum, p, height = inp
    lwc = helpers.kgkg_to_kgm3(lwc,temp, hum, p)
    dh = np.abs(height[:-1] - height[1:])
    lwc_layer = 0.5 * (lwc[:-1] + lwc[1:])
    lwp = np.nansum(lwc_layer*dh) 
    if np.all(np.isnan(lwc_layer)):
        return np.nan
    elif np.isnan(lwp):
        return np.nan
    elif lwp >= 1e-3:
        return 1
    elif lwp < 1e-3:
        return 0
    
def overpass_time(inp, lon):
    time = inp
    timej = time + timedelta(hours=lon*24/360)
    LST = timej.hour + timej.minute/60 + timej.second/3600
    return LST

def downsample_grid(d, var, nh, nv=None, sample_type='mean', thr=None):
    ''' d = dataset with (lat, lon, height) or (lat, lon) dimensions
        nh = number of gridpoints in horizontal plane (nh x nh)
        nv = number of gridpoints in vertical (not None if 3D)
        sample_type = 'mean' or 'sum' or 'cond_mean' (conditional)
    '''
    lat = d.lat.values
    lon = d.lon.values
    var = d[var].values
    nx = np.ceil(len(var) / nh)
    ny = np.ceil(len(var[0]) / nh)
    lat_new = np.zeros((int(nx), int(ny)))
    lon_new = np.zeros((int(nx), int(ny)))
    if sample_type == 'cond_mean':
        var[var == 0] = np.nan
    if thr != None:
        var[var < thr] = np.nan
    if nv is not None:
        h = d.height.values
        nz = np.ceil(len(var[0,0]) / nv)
        var_new = np.full((int(nx), int(ny), int(nz)), np.nan)
        h_new = np.zeros((int(nz)))
        for i in range(int(nx)):
            for j in range(int(ny)):
                try:
                    lat_block = lat[i*nh:(i*nh+nh), j*nh:(j*nh+nh)]
                    lat_new[i,j] = np.nanmean(lat_block)
                except:
                    lat_new[i,j] = np.nanmean(lat[i*nh:-1, j*nh:-1])
                try:
                    lon_block = lon[i*nh:(i*nh+nh), j*nh:(j*nh+nh)]
                except:
                    lon_block = lon[i*nh:-1, j*nh:-1]
                lon_rad = np.deg2rad(lon_block)
                x = np.nanmean(np.cos(lon_rad))
                y = np.nanmean(np.sin(lon_rad))
                lon_new[i,j] = np.rad2deg(np.arctan2(y, x))
                for k in range(int(nz)):
                    slice_block = var[i*nh:(i*nh+nh), j*nh:(j*nh+nh), k*nv:(k*nv+nv)]
                    if sample_type in ['mean', 'cond_mean']:
                        var_new[i,j,k] = np.nanmean(slice_block)
                    elif sample_type == 'sum':
                        var_new[i,j,k] = np.nansum(slice_block)
                    if i == 0 and j == 0:
                        h_new[k] = np.nanmean(slice_block) if isinstance(slice_block, np.ndarray) else np.nanmean(h[k*nv:(k*nv+nv)])
        var_new[var_new == 0] = np.nan
        lat_new[lat_new <= -90] = -89.99
        lat_new[lat_new >= 90] = 89.99
        return var_new, lat_new, lon_new, h_new
    else:
        var_new = np.zeros((int(nx), int(ny)))
        for i in range(int(nx)):
            for j in range(int(ny)):
                try:
                    lat_block = lat[i*nh:(i*nh+nh), j*nh:(j*nh+nh)]
                    lat_new[i,j] = np.nanmean(lat_block)
                except:
                    lat_new[i,j] = np.nanmean(lat[i*nh:-1, j*nh:-1])
                try:
                    lon_block = lon[i*nh:(i*nh+nh), j*nh:(j*nh+nh)]
                except:
                    lon_block = lon[i*nh:-1, j*nh:-1]
                lon_rad = np.deg2rad(lon_block)
                x = np.nanmean(np.cos(lon_rad))
                y = np.nanmean(np.sin(lon_rad))
                lon_new[i,j] = np.rad2deg(np.arctan2(y, x))
                slice_block = var[i*nh:(i*nh+nh), j*nh:(j*nh+nh)]
                if sample_type in ['mean', 'cond_mean']:
                    var_new[i,j] = np.nanmean(slice_block)
                elif sample_type == 'sum':
                    var_new[i,j] = np.nansum(slice_block)
                lat_new[lat_new <= -90] = -89.99
                lat_new[lat_new >= 90] = 89.99
        var_new[var_new == 0] = np.nan
        return var_new, lat_new, lon_new

