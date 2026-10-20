"""
Merged monthly files of mapped output on RACMO grid
"""
import numpy as np
import sys
import xarray as xr
import matplotlib.pyplot as plt
import importlib
from pathlib import Path
sys.path.append('../utils/')
import helpers
import RCM2EC_functions
import spatial_regridding
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

def merge_regrid(filetype_prefix,
                 domain='FGRN055',
                base_path="/gridded_output/",
                type="fraction", dim='3D', only_BC=False):
    base_path = base_path + domain + '/'
    save_path = base_path+'merged/'
    path = Path(base_path)
    files = sorted(path.glob(f"{filetype_prefix}_2*.nc"))
    if not files:
        raise FileNotFoundError(f"No files found for pattern {filetype_prefix}_2*.nc")
    if only_BC:
        pop_idx = []
        for i,f in enumerate(files):
            fname = f.name
            if "202512" not in fname and "2026" not in fname:
                pop_idx.append(i)
        for idx in pop_idx[::-1]:
            files.pop(idx)
    if dim == '3D':
        ds = xr.open_mfdataset(
            files,
            combine="nested",
            concat_dim="stack",
            chunks={"lat_ix": 100, "lon_ix": 100, "h_ix": 10},
        )
    elif dim == '2D':
        ds = xr.open_mfdataset(
            files,
            combine="nested",
            concat_dim="stack",
            chunks={"lat_ix": 100, "lon_ix": 100},
        )
    ref = xr.open_dataset(files[0]).fillna(0)
    overpasses = ds.overpasses.fillna(0).sum("stack")
    cond_result = ds.cond_result.fillna(0).sum("stack")
    overpasses = overpasses.where(overpasses != 0)
    result = cond_result / overpasses
    if dim == '3D':
        if type == 'fraction':
            combined = xr.Dataset(
                data_vars={
                    "lat": (["lat_ix", "lon_ix"], ref.lat.data),
                    "lon": (["lat_ix", "lon_ix"], ref.lon.data),
                    "height": ("h_ix", ref.height.values[0,0]),
                    "overpasses": (["lat_ix", "lon_ix", "h_ix"], overpasses.data),
                    "cond_result": (["lat_ix", "lon_ix", "h_ix"], cond_result.data),
                    "fraction": (["lat_ix", "lon_ix", "h_ix"], result.data),
                },
                coords={
                    "lat_ix": np.arange(ref.lat.shape[0]),
                    "lon_ix": np.arange(ref.lat.shape[1]),
                    "h_ix": np.arange(np.shape(ref.height.values)[2]),
                },
            )
        elif type == 'mean':
            combined = xr.Dataset(
                data_vars={
                    "lat": (["lat_ix", "lon_ix"], ref.lat.data),
                    "lon": (["lat_ix", "lon_ix"], ref.lon.data),
                    "height": ("h_ix", ref.height.values[0,0]),
                    "overpasses": (["lat_ix", "lon_ix", "h_ix"], overpasses.data),
                    "cond_result": (["lat_ix", "lon_ix", "h_ix"], cond_result.data),
                    "mean": (["lat_ix", "lon_ix", "h_ix"], result.data),
                },
                coords={
                    "lat_ix": np.arange(ref.lat.shape[0]),
                    "lon_ix": np.arange(ref.lat.shape[1]),
                    "h_ix": np.arange(np.shape(ref.height.values)[2]),
                },
            )        
        else:
            raise TypeError(f"The type {type} is unknown")
    elif dim =='2D':
        if type == 'fraction':
            combined = xr.Dataset(
                data_vars={
                    "lat": (["lat_ix", "lon_ix"], ref.lat.data),
                    "lon": (["lat_ix", "lon_ix"], ref.lon.data),
                    "overpasses": (["lat_ix", "lon_ix"], overpasses.data),
                    "cond_result": (["lat_ix", "lon_ix"], cond_result.data),
                    "fraction": (["lat_ix", "lon_ix"], result.data),
                },
                coords={
                    "lat_ix": np.arange(ref.lat.shape[0]),
                    "lon_ix": np.arange(ref.lat.shape[1]),
                },
            )
        elif type == 'mean':
            combined = xr.Dataset(
                data_vars={
                    "lat": (["lat_ix", "lon_ix"], ref.lat.data),
                    "lon": (["lat_ix", "lon_ix"], ref.lon.data),
                    "overpasses": (["lat_ix", "lon_ix"], overpasses.data),
                    "cond_result": (["lat_ix", "lon_ix"], cond_result.data),
                    "mean": (["lat_ix", "lon_ix"], result.data),
                },
                coords={
                    "lat_ix": np.arange(ref.lat.shape[0]),
                    "lon_ix": np.arange(ref.lat.shape[1]),
                },
            )        
        else:
            raise TypeError(f"The type {type} is unknown")
    if only_BC:
        combined.to_netcdf(save_path + filetype_prefix + "_combined_BC.nc",
                       engine="netcdf4", compute=True)
    else:
        combined.to_netcdf(save_path + filetype_prefix + "_combined.nc",
                       engine="netcdf4", compute=True)
    return combined

domain='PXANT11'
sim = 'bl_qb'
merge_regrid("EC_IWP_2b", domain=domain, type='mean', dim='2D')
merge_regrid(f"RACMO_{domain}_{sim}_IWP", domain=domain, type='mean', dim='2D')