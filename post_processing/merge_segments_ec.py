#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code to merge EarthCARE segments into one nc file,
discarding the overlap between segments.
Input:
- folder containing input
- folder to save new merged files to
- location: RACMO domain (FGRN055 for Greenland,
    PXANT11 for Antarctica), this determines which
    segments are used
- product name: name of the EarthCARE product that 
    is merged (e.g. ATL_NOM_1B)

This code is written by MSc thesis student Joppe Visch
"""

import xarray as xr 
import numpy as np 
import sys
from functools import partial
import os

# Functions to put all segments together in one folder

def HeightSlice(ds, product_name):
    """
    Slice lidar height dimension to ensure consistent shape 
    
    Removes the top row if the height has 242 bins instead of 241 bins
    """
    if product_name == 'ATL_ICE_2A':
        if ds["JSG_height"].shape[0] == 242:  
            ds = ds.isel(JSG_height=slice(1, 242))
        if ds["JSG_height"].shape[0] != 241:
            print("trouble, the across track is ", ds["JSG_height"].shape[0])
    return ds

def SideSlice(ds, segment_letter, product_name, orbit_nr_source, segments_in_order, EndTimeDict):
    """
    Slice the the ends of two adjacent segments across-track to ensure no overlapping 

    Slice a defined number of indices (slice_length) of first segment.
    Slices the rest of the overlapping segments based on latitude overlap.
    """
    if product_name == 'ATL_ICE_2A':
        slice_length = -50
    elif product_name == 'CPR_CLD_2A':
        slice_length = -5

    if segment_letter == segments_in_order[0]:
        ds = ds.isel(along_track=slice(None, slice_length))
        EndTimeDict[f"{orbit_nr_source}{segments_in_order[0]}"] = ds['time'].values[-1]
        
    elif segment_letter == segments_in_order[1]:
        slice_step1 = EndTimeDict[f"{orbit_nr_source}{segments_in_order[0]}"]
        slice_value_t = np.where(ds['time'].values[0:1000] <= slice_step1)
        ds = ds.isel(along_track=slice(len(slice_value_t[0]), slice_length))
        EndTimeDict[f"{orbit_nr_source}{segments_in_order[1]}"] = ds['time'].values[-1]
       
    elif segment_letter == segments_in_order[2]:
        slice_step2 = EndTimeDict[f"{orbit_nr_source}{segments_in_order[1]}"]
        slice_value_t = np.where(ds['time'].values <= slice_step2)  
        ds = ds.isel(along_track=slice(len(slice_value_t[0]), None))
    return ds

def preprocess(ds, product_name, segments_in_order, EndTimeDict):
    """
    Combine the two individual preprocessing steps. 
    """
    segment_letter = ds.encoding["source"][-4]
    orbit_nr_source = ds.encoding["source"][-9:-4]
    ds = SideSlice(ds, 
                   segment_letter=segment_letter, 
                   product_name=product_name, 
                   orbit_nr_source=orbit_nr_source,
                   segments_in_order=segments_in_order,
                   EndTimeDict=EndTimeDict
                   )
    

    ds = HeightSlice(ds, product_name)
    return ds

def merge_segments(input_folder, 
                   product_name,  
                   save_folder,
                   location,
                   EndTimeDict
                   ):
    ''' Merge the different segments into one segment of EarthCare. 
    Input:
        input_folder: The loaded .h5 EarthCare files of all parts of the EarthCare segment. 
        product_name: The name of the product_name input 
        start_orbit, end_orbit: The first orbit number input and the last orbit number input
        segments_in_order: The segment numbers (e.g "F") you want to export (in order of merging)
        save_folder: The folder to save the output in. 
        drop_var = The desired variables to be dropped out.    
    '''
    # Choosing location #
    if location == "MA": #Antartica
        segments_in_order = ["F","G", "H"]
    elif location == "MG": #Greenland 
        segments_in_order = ["B", "C", "D"] 


    # Finding start and end orbit 
    StartEndOrbit = []
    for file_name in os.listdir(input_folder):
        StartEndOrbit.append(int(file_name[-8:-4]))
    
    start_orbit = min(StartEndOrbit)
    end_orbit = max(StartEndOrbit)
    
    #making a list of all sequences of segments to be merged, and adding the corresponding input names.
    orbit_file_names = []
    orbit_nrs = [f"Z_{orbit_nr:05d}" for orbit_nr in range(start_orbit, end_orbit)]

    for orbit_nr in orbit_nrs:
        f_per_orbit_nr = [f for f in os.listdir(input_folder) if orbit_nr in f]
        f_per_orbit_nr = sorted(f_per_orbit_nr, key=lambda x: segments_in_order.index(x.split("_")[-1].split(".")[0][-1]))
        f_per_orbit_nr_full = [os.path.join(input_folder, f) for f in f_per_orbit_nr]

        # 2: Sorting how many input files are actually in the input_folder, looking whether these
        # segments are adjacent and otherwise individual output. 
        if len(f_per_orbit_nr) == 3: 
            orbit_file_names.append(f_per_orbit_nr_full)
        elif  len(f_per_orbit_nr) == 2:
            if (f_per_orbit_nr[0][-4] == 'F' and f_per_orbit_nr[1][-4] == 'G') or (
                f_per_orbit_nr[0][-4] == 'G' and f_per_orbit_nr[1][-4] == 'H' ):
                orbit_file_names.append(f_per_orbit_nr_full)
            else:
                orbit_file_names.append(f_per_orbit_nr_full[0])
                orbit_file_names.append(f_per_orbit_nr_full[1])
                
        elif len(f_per_orbit_nr) == 1:
            orbit_file_names.append(f_per_orbit_nr_full)

    if product_name == "CPR_CLD_2A":
        drop_var = ['geoid_offset', 'surface_bin_number', 'surface_estimation_flag', 'land_flag', 
                  'water_content_log_error', 
                  'characteristic_diameter_log_error', 'water_content_prior', 
                  'water_content_prior_log_error', 'characteristic_diameter_prior', 
                  'characteristic_diameter_prior_log_error', 'reflectivity_factor_forward', 
                  'doppler_velocity_forward', 'attenuation', 'path_integrated_attenuation_forward',
                  'liquid_water_path_integrated_attenuation', 'melting_layer_path_integrated_attenuation',
                  'number_concentration_parameter_N0_star', 'number_concentration_parameter_N0_star_log_error', 
                  'mass_flux_absolute_error', 'sedimentation_velocity', 
                  'sedimentation_velocity_absolute_error', 'ice_parameter_alpha', 'maximum_dimension_L', 
                  'maximum_dimension_L_lower_error_bound', 'maximum_dimension_L_upper_error_bound', 
                  'ice_water_path', 'ice_water_path_error', 'rain_water_path', 'rain_water_path_error', 
                  'chi_square_normalized', 'retrieval_status', 'retrieval_classification', 'doppler_velocity_classification', 
                  'doppler_velocity_quality_status', 
                  'liquid_water_content_relative_error', 'liquid_effective_radius', 
                  'liquid_effective_radius_relative_error', 'liquid_water_path', 'liquid_water_path_error', 
                  'drizzle_mass_flux', 'drizzle_mass_flux_log_error']
    elif product_name == "ATL_ICE_2A":
        drop_var = []
    
    # 3: Opening the list in mf_dataset and saving them in corresponding name. The opening of 
    # the dataset happens with the preprocess step which is defined above. 
    for i, segments in enumerate(orbit_file_names, start=0):
        try: 
            merged = xr.open_mfdataset(
                segments,
                drop_variables = drop_var,
                group='ScienceData',
                engine='h5netcdf',
                concat_dim='along_track',
                combine='nested',
                preprocess=partial(preprocess, product_name=product_name, segments_in_order=segments_in_order,
                        EndTimeDict=EndTimeDict)
            )

            str = segments[0].split("ECA_")[1][:55]
            path_to_save = f"{save_folder}/ECA_{str}{location}.h5"

            merged.to_netcdf(path_to_save, format="NETCDF4")
            print(f"Orbit number {str[-4:]} saved!")
        except:
            print(f"File does not exist")

# Main CLI

def main():
    try:
        input_folder = sys.argv[1]
        save_folder = sys.argv[2]
        location = sys.argv[3]
        product_name = sys.argv[4]
    except IndexError:
        print('')
        print('Error: missing commandline inputs.')
        print('')
        print('Usage:')
        print(f"{sys.argv[0]} <input_folder> <start_orbit> <end_orbit> <segments_in_order> <save_folder> <location_tag> <product_name>\n")
        print('')
        sys.exit(1)

    EndTimeDict = {}
    merge_segments(
        input_folder=input_folder,
        product_name=product_name,
        save_folder=save_folder,
        location=location,
        EndTimeDict=EndTimeDict
    )

if __name__ == "__main__":
    main()
