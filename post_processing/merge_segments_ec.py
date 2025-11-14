import xarray as xr 
import numpy as np 
import sys
from functools import partial
import os
import warnings
warnings.filterwarnings("ignore")

# Function to put all segments together in one folder

def HeightSlice(ds, product_name):
    """
    Slice lidar height dimension to ensure consistent shape 
    
    Removes the top row if the height has 242 bins instead of 241 bins
    """
    if product_name == 'ATL_ICE_2A' or product_name == 'AC__TC__2B':
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
    if product_name[:3] == 'ATL':
        slice_length = -50
    elif product_name[:3] == 'CPR' or product_name[:3] == 'ACM' or product_name[:3] == 'AC_':
        slice_length = -5

    if segment_letter == segments_in_order[0]:
        if product_name != 'CPR_NOM_1B':
            ds = ds.isel(along_track=slice(None, slice_length))
            EndTimeDict[f"{orbit_nr_source}{segments_in_order[0]}"] = ds['time'].values[-1]
        else:
            try:
                ds = ds.isel(phony_dim_14=slice(None, slice_length))
                EndTimeDict[f"{orbit_nr_source}{segments_in_order[0]}"] = ds['profileTime'].values[-1]
            except:
                ds = ds.isel(phony_dim_10=slice(None, slice_length))
        
    elif segment_letter == segments_in_order[1]:
        try:
            slice_step1 = EndTimeDict[f"{orbit_nr_source}{segments_in_order[0]}"]
            slice_value_t = np.where(ds['time'].values[0:1000] <= slice_step1)
            if product_name != 'CPR_NOM_1B':
                ds = ds.isel(along_track=slice(len(slice_value_t[0]), slice_length))
                EndTimeDict[f"{orbit_nr_source}{segments_in_order[1]}"] = ds['time'].values[-1]
            else:
                try:
                    ds = ds.isel(phony_dim_14=slice(len(slice_value_t[0]), slice_length))
                    EndTimeDict[f"{orbit_nr_source}{segments_in_order[1]}"] = ds['profileTime'].values[-1]
                except:
                    ds = ds.isel(phony_dim_10=slice(len(slice_value_t[0]), slice_length))
        except:
            if product_name != 'CPR_NOM_1B':
                ds = ds.isel(along_track=slice(None, slice_length))
                EndTimeDict[f"{orbit_nr_source}{segments_in_order[1]}"] = ds['time'].values[-1]
            else:
                try:
                    ds = ds.isel(phony_dim_14=slice(None, slice_length))
                    EndTimeDict[f"{orbit_nr_source}{segments_in_order[1]}"] = ds['profileTime'].values[-1]
                except:
                    ds = ds.isel(phony_dim_10=slice(None, slice_length))
    elif segment_letter == segments_in_order[2]:
        try:
            slice_step2 = EndTimeDict[f"{orbit_nr_source}{segments_in_order[1]}"]
            slice_value_t = np.where(ds['time'].values <= slice_step2)  
            if product_name != 'CPR_NOM_1B':
                ds = ds.isel(along_track=slice(len(slice_value_t[0]), None))
                EndTimeDict[f"{orbit_nr_source}{segments_in_order[2]}"] = ds['time'].values[-1]
            else:
                try:
                    ds = ds.isel(phony_dim_14=slice(len(slice_value_t[0]), None))
                    EndTimeDict[f"{orbit_nr_source}{segments_in_order[2]}"] = ds['profileTime'].values[-1]
                except:
                    ds = ds.isel(phony_dim_10=slice(len(slice_value_t[0]), None))
        except:
            ds = ds 
            if product_name != 'CPR_NOM_1B':
                EndTimeDict[f"{orbit_nr_source}{segments_in_order[2]}"] = ds['time'].values[-1]
            else:
                try:
                    EndTimeDict[f"{orbit_nr_source}{segments_in_order[2]}"] = ds['profileTime'].values[-1]
                except:
                    pass
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
    if location == "PXANT11": #Antartica
        segments_in_order = ["F","G", "H"]
        location_tag = "MA"
    elif location == "FGRN055": #Greenland 
        segments_in_order = ["B","C","D"]
        location_tag = "MG"

    # Finding start and end orbit 
    StartEndOrbit = []
    for file_name in os.listdir(input_folder):
        StartEndOrbit.append(int(file_name[-8:-4]))
    
    start_orbit = min(StartEndOrbit)
    end_orbit = max(StartEndOrbit)
    
    #making a list of all sequences of segments to be merged, and adding the corresponding input names.
    orbit_file_names = []
    orbit_nrs = [f"Z_{orbit_nr:05d}" for orbit_nr in range(start_orbit, end_orbit+1)]

    for orbit_nr in orbit_nrs:
        f_per_orbit_nr = [f for f in os.listdir(input_folder) if orbit_nr in f]
        f_per_orbit_nr = sorted(f_per_orbit_nr, key=lambda x: segments_in_order.index(x.split("_")[-1].split(".")[0][-1]))
        f_per_orbit_nr_full = [os.path.join(input_folder, f) for f in f_per_orbit_nr]

        # 2: Sorting how many input files are actually in the input_folder, looking whether these
        # segments are adjacent and otherwise individual output. 
        if len(f_per_orbit_nr) == 3: 
            orbit_file_names.append(f_per_orbit_nr_full)
        elif  len(f_per_orbit_nr) == 2:
            if (f_per_orbit_nr[0][-4] == segments_in_order[0] and f_per_orbit_nr[1][-4] == segments_in_order[1]) or (
                f_per_orbit_nr[0][-4] == segments_in_order[1] and f_per_orbit_nr[1][-4] == segments_in_order[2] ):
                orbit_file_names.append(f_per_orbit_nr_full)
            else:
                orbit_file_names.append(f_per_orbit_nr_full[0])
                orbit_file_names.append(f_per_orbit_nr_full[1])
                
        elif len(f_per_orbit_nr) == 1:
            orbit_file_names.append(f_per_orbit_nr_full)

    if product_name == "CPR_CLD_2A":
        drop_var = ['water_content_prior', 'water_content_prior_log_error', 'characteristic_diameter_prior', 'characteristic_diameter_prior_log_error', 
                    'reflectivity_factor_forward', 'doppler_velocity_forward', 'maximum_dimension_L', 'maximum_dimension_L_lower_error_bound', 
                    'maximum_dimension_L_upper_error_bound', 'chi_square_normalized', 'retrieval_classification', 'doppler_velocity_classification', 
                    'doppler_velocity_quality_status']

    elif product_name == "ATL_ICE_2A":
        drop_var = []
    elif product_name == 'ATL_NOM_1B':
        drop_var = ['mie_raw_signal','rayleigh_raw_signal', 'crosspolar_raw_signal','mie_offset',
                    'rayleigh_offset','crosspolar_offset','mie_offset_standard_deviation',
                    'rayleigh_offset_standard_deviation','crosspolar_offset_standard_deviation',
                    'mie_offset_variation','rayleigh_offset_variation','crosspolar_offset_variation',
                    'mie_background_signal','rayleigh_background_signal','crosspolar_background_signal',
                    'intersection_error_flag','atmospheric_interpolation_error_flag',
                    'rayleigh_raw_spectral_crosstalk_invalid_flag','floor_index',
                    'rayleigh_raw_spectral_crosstalk','mie_spectral_crosstalk_correction_factor',
                    'mie_spectral_crosstalk_reference_temperature','averaged_laser_energy',
                    'energy_error_flag','state_vector_quality_status','time_synchronisation_status',
                    'ccdb_redundancy_flag','mie_spectral_crosstalk_segment','mie_spectral_crosstalk_segment_error',
                    'segments_first_index','spike_flag_rayleigh','spike_flag_mie',
                    'spike_flag_crosspolar','valid_surface_rayleigh_spectral_crosstalk_segment_flag',
                    'rayleigh_spectral_crosstalk_surface_evaluations','rayleigh_spectral_crosstalk_surface_evaluations_error',
                    'valid_STRAP_rayleigh_spectral_crosstalk_segment_flag','rayleigh_spectral_crosstalk_STRAP_evaluations',
                    'rayleigh_spectral_crosstalk_STRAP_evaluations_error','rayleigh_averaged_spectral_crosstalk',
                    'rayleigh_averaged_spectral_crosstalk_error','mie_averaged_spectral_crosstalk',
                    'mie_averaged_spectral_crosstalk_error','crosspolar_polarisation_crosstalk',
                    'crosspolar_polarisation_crosstalk_error','copolar_polarisation_crosstalk',
                    'copolar_polarisation_crosstalk_error','crosspolar_polarsation_crosstalk_segment',
                    'crosspolar_polarsation_crosstalk_segment_error','copolar_polarsation_crosstalk_segment',
                    'copolar_polarsation_crosstalk_segment_error','ray_20km_spike_factor',
                    'mie_20km_spike_factor','background_correction_factor_rayleigh',
                    'background_correction_factor_mie','background_correction_factor_crosspolar',
                    'background_correction_factor_error_rayleigh','background_correction_factor_error_mie',
                    'background_correction_factor_error_crosspolar','hot_pixel_flag_rayleigh',
                    'hot_pixel_flag_mie','hot_pixel_flag_crosspolar','hot_pixel_level_rayleigh',
                    'hot_pixel_level_mie','hot_pixel_level_crosspolar']
    elif product_name == 'CPR_NOM_1B':
        drop_var = []
    elif product_name == 'AC__TC__2B':
        drop_var = []
    elif product_name == 'ACM_CAP_2B':
        drop_var = []
    elif product_name == 'ACM_COM_2B':
        drop_var = []
    # 3: Opening the list in mf_dataset and saving them in corresponding name. The opening of 
    # the dataset happens with the preprocess step which is defined above.
    for i, segments in enumerate(orbit_file_names, start=0):
        # print(segments)
        try:
            print(f'started orbit number {segments[0].split("ECA_")[1][50:55]}', flush=True)
        except:
            print(f'started orbit number {segments.split("ECA_")[1][50:55]}', flush=True)
        try: 
            if product_name != 'CPR_NOM_1B':
                merged = xr.open_mfdataset(
                    segments,
                    drop_variables = drop_var,
                    group='ScienceData',
                    engine='h5netcdf',
                    concat_dim='along_track',
                    combine='nested',
                    preprocess=partial(preprocess, product_name=product_name, 
                                    segments_in_order=segments_in_order,
                                        EndTimeDict=EndTimeDict)
                )
            else:
                data = xr.open_mfdataset(
                    segments,
                    group='ScienceData/Data',
                    engine='h5netcdf',
                    concat_dim='phony_dim_10',
                    combine='nested',
                    phony_dims='sort',
                    drop_variables=['binStatusFlag', 'covarianceCoeff', 'dopplerStatusFlag',
                                            'dopplerVelocity', 'dopplerVelocityAtSurfaceBin', 
                                            'integrationNumberDoppler', 'integrationNumberEcho', 
                                            'noiseFloorPower', 'operationalMode', 'pulseShapeWarnFlag',
                                            'pulseWidth', 'radarCoefficient', 'rangeBinValidNumber',
                                            'rayHeaderCalVers', 'rayHeaderLambda', 'rayQualityFlag',
                                            'rayStatusFlag', 'rayStatusPrf', 'receivedEchoPower',
                                            'satelliteVelocityContaminationInLOS', 'sigmaZero',
                                            'spectrumWidth', 'subOperationalMode', 'surfaceBinFraction',
                                            'surfaceBinNumber', 'surfaceEstimationFlag', 'transmitPower',
                                            'transmitPowerAvg', 'txRxStatusFlag'],
                    preprocess=partial(preprocess, product_name=product_name, 
                                    segments_in_order=segments_in_order,
                                        EndTimeDict=EndTimeDict)
                )
                geo = xr.open_mfdataset(
                    segments,
                    group='ScienceData/Geo',
                    engine='h5netcdf',
                    concat_dim='phony_dim_14',
                    combine='nested',
                    phony_dims='sort',
                    drop_variables=['navigationLandWaterFlg', 'pitchAngle', 'processingFrameNo',
                                    'rangeBinMaxNumber', 'rangeToFirstBin',
                                    'rangeToIntercept', 'rayHeaderRangeBinSize', 'rayHeaderSpatAvg',
                                    'rayNumber', 'rollAngle', 'satelliteVelocityX', 
                                    'satelliteVelocityY', 'satelliteVelocityZ', 'solarAzimuthAngle',
                                    'solarElevationAngle', 'surfaceElevation', 'timeFlag', 
                                    'xPosition', 'yPosition', 'yawAngle', 'zPosition'],
                    preprocess=partial(preprocess, product_name=product_name, 
                                    segments_in_order=segments_in_order,
                                        EndTimeDict=EndTimeDict)
                )
                geo.rename_dims({'phony_dim_14': 'phony_dim_10'})
                merged = xr.merge([data, geo])
            str = segments[0].split("ECA_")[1][:55]
            path_to_save = f"{save_folder}/ECA_{str}{location_tag}.h5"

            merged.to_netcdf(path_to_save, format="NETCDF4")
            print(f"Orbit number {str[-4:]} saved!", flush=True)
        except Exception as e:
            print(f"File does not exist, exception {e} occurred " , flush=True)

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
        print(f"{sys.argv[0]}")
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
