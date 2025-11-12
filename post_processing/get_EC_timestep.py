#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code to compute the overpass time of EarthCARE
over a specific RACMO domain and write a file
that can be used as input for the simulation.
Works for only one month at the time.
Input:
- domain: RACMO domain name
- exp: RACMO experiment name (for location)
- dtgin: start date to get the date of month
@author: Thirza Feenstra
"""

import numpy as np
sys.path.append('../utils/')
import Convert_From_To_RacmoGrid as CFTR
import RCM_gridSettings as RCMG
import xarray as xr
from scipy.spatial import KDTree
import os
import re
import sys
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

user = 'username'

def get_EC_coords(f, topofile, gridfile):
    ''' f is the loaded h5 file
    topofile is the maskfile for the chosen domain
    gridfile contains RCM grid
    returns regridded trajectory, EC coordinate trajectory and time, only for
    points within the domain
    '''
    try:
        lat_EC = f['latitude'].values
        lon_EC = f['longitude'].values
        try:
            time_EC = f['time'].values
        except:
            time_EC = f['profileTime'].values
    except:
        lat_EC = f['ellipsoid_latitude'].values
        lon_EC = f['ellipsoid_longitude'].values
        time_EC = f['time'].values
        
    lat_R = topofile['rlat'].values
    lon_R = topofile['rlon'].values
    lat_min = np.min(lat_R)
    lat_max = np.max(lat_R)
    lon_min = np.min(lon_R)
    lon_max = np.max(lon_R)
    lat_EC_R, lon_EC_R = CFTR.RealWorld2RotatedGrid(lat_EC, lon_EC, gridfile)
    idx = np.where((lat_EC_R > lat_min) & (lat_EC_R < lat_max) & \
                   (lon_EC_R > lon_min) & (lon_EC_R < lon_max))
    lat_EC = lat_EC[idx]
    lon_EC = lon_EC[idx]
    lat_EC_R = lat_EC_R[idx]
    lon_EC_R = lon_EC_R[idx]
    time_EC = time_EC[idx]    
    return lat_EC_R, lon_EC_R, lat_EC, lon_EC, time_EC

def get_overpass_time(f, topofile, gridfile):
    ''' f is the loaded h5 file
    topofile is the maskfile for the chosen domain
    gridfile contains RCM grid
    returns overpass time of EarthCARE in middle of 
    the path within the domain
    '''
    lat_EC_R, lon_EC_R, lat_EC, lon_EC, time_EC = get_EC_coords(f, topofile, 
                                                              gridfile)
    try:
        op_time = time_EC[int(len(time_EC)/2)]
    except:
        op_time = np.nan
    return op_time
    
def time_to_numbers(timestep):
    ''' time as EC timestep converted to year, month, day, hour, minute
    '''
    timestring = str(timestep)
    year = np.array([int(timestring[:4])])
    month = np.array([int(timestring[5:7])])
    day = np.array([int(timestring[8:10])])
    hour = np.array([int(timestring[11:13])])
    minute = np.array([int(timestring[14:16])])
    second = np.array([int(timestring[17:19])])
    if second[0] >= 30:
        minute[0] += 1
    return year, month, day, hour, minute

def save_timesteps(years, months, days, hours, minutes, path, domain, exp, yearmonth):
    '''saves the timesteps in a file with list of dates, which will be input to RACMO'''
    f = open(path+'EC_timesteps_'+domain+'_'+exp+'_'+yearmonth+'.rcp', 'w')
    f.write('# define time steps for separate output files \n')
    if len(np.array([years])[0]) > 1:
        dstrs = [f'set ecyear = {tuple(years)} \n',
                  f'set ecmonth = {tuple(months)} \n',
                  f'set ecday = {tuple(days)} \n',
                  f'set ecchour = {tuple(hours)} \n',
                  f'set ecminute = {tuple(minutes)} \n']
    else:
        dstrs = [f'set ecyear = ({years[0]}) \n',
                  f'set ecmonth = ({months[0]}) \n',
                  f'set ecday = ({days[0]}) \n',
                  f'set ecchour = ({hours[0]}) \n',
                  f'set ecminute = ({minutes[0]}) \n']
    strs = ['set eyear = ( $ecyear ) \n',
            'set emonth = ( $ecmonth ) \n',
            'set eday = ( $ecday ) \n',
            'set ehour = ( $ecchour ) \n',
            'set eminute = ( $ecminute ) \n']
    f.writelines(dstrs)
    f.writelines(strs)
    f.close()
    return f

def extract_first_date(filename):
    '''extract first date from list of EarthCARE filenames
    (since EC filenames are observed time - process time)'''
    date_pattern = re.compile(r"(\d{8}T\d{6}Z)")
    match = date_pattern.search(filename)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ")
    return datetime.min

def run_ec_timesteps(domain, exp, dtgin):
    '''function to make the list of timesteps for certain domain,
    experiment and start time (dtgin)
    For specific setup on ECMWF supercomputer for RACMO'''
    dtopo = xr.open_dataset(f'/perm/{user}/masks_grids/{domain}_masks.nc', engine='netcdf4')
    gridfile = RCMG.read_Setting_data(domain=domain, 
                                  path='/perm/{user}/masks_grids/', log=False)
    yearmonth = str(dtgin)[0:6]
    path = f'/ec/res4/scratch/{user}/{domain}-{exp}/EarthCARE_{yearmonth}/'
    ec_files = sorted((f for f in os.listdir(path) if not f.startswith(".")),
                        key=extract_first_date)
    years = np.zeros(len(ec_files))
    months = np.zeros(len(ec_files))
    days = np.zeros(len(ec_files))
    hours = np.zeros(len(ec_files)) 
    minutes = np.zeros(len(ec_files))
    for i, file in enumerate(ec_files):
        dec = xr.open_dataset(path+file, engine='h5netcdf')
        op_time = get_overpass_time(dec, dtopo, gridfile)
        if np.isfinite(op_time):
            years[i], months[i], days[i], hours[i], minutes[i] = time_to_numbers(op_time)
        else:
            os.remove(path+file)
            years[i], months[i], days[i], hours[i], minutes[i] = np.nan, np.nan, np.nan, np.nan, np.nan
    years = years[np.isfinite(years)]
    months = months[np.isfinite(months)]
    days = days[np.isfinite(days)]
    hours = hours[np.isfinite(hours)]
    minutes = minutes[np.isfinite(minutes)]
    savepath = '/home/{user}/run_2v4/'
    rcp = save_timesteps(years, months, days, hours, minutes, savepath, domain, exp, yearmonth)
    return

def main():
    try:
        domain = sys.argv[1]
        expname = sys.argv[2]
        dtgin = sys.argv[3]
    except IndexError:
        print('')
        print('Error: missing commandline inputs.')
        print('')
        print('Usage:')
        print(f'{sys.argv[0]} '+
              '<domain> <expname>')
        print('')
        sys.exit(1)


    run_ec_timesteps(domain, expname, dtgin)

if __name__ == "__main__":
    main()

