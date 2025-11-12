# EarthCARE4RCM
Software to compare EarthCARE satellite observations of clouds, aerosols and radiation with regional climate model output. This has been applied to the RCM RACMO2.4p1, but it should, with modifcations, be possible to use for other RCMs.

**Included in this software:**
- Code for extracting the RCM data along the EarthCARE flight line (under post_processing). Note that for RACMO, we wrote additional output at the closest timestep to the overpass time. Code for computing overpass times can be found here as well. The implementation of this additional output stream in the RACMO model is, however, not included, as it is very model-specific and the RACMO model is not open-source. As the Greenland domain covers multiple EarthCARE orbit segments, code to merge these segments is also included here. The RACMO post-processing includes making input for the ATLID simulator (https://gitlab.com/KNMI-OSS/satellite-data-research-tools/cardinal-campaign-tools). 
- Simple radar simulator (under utils), based on relationships between water content and reflectivity and correction for attenuation.
- Plot functions to plot RACMO co-located output next to EarthCARE scene, for Level 1 ATLID backscatter and CPR reflectivity, Level 2a ice water content, snowfall rate and rainfall rate, and Level 2b cloud and precipitation target classification. Note that the calipso colormap that is used for ATLID Mie backscatter can be obtained from the ECtools code by Shannon Mason (https://bitbucket.org/smason/ectools/src/main/).
- The grid setting of RACMO Greenland domain can be found under RACMO_grid, which is used to convert to the rotated pole grid.

RACMO output from and for use of these tools can be found on Zenodo: 

**Authors:** Thirza Feenstra (t.n.feenstra@uu.nl), with contributions from Willem Jan van de Berg and Joppe Visch.

**Questions or interested in working with these tools to evaluate other RCMs?** Please contact Thirza Feenstra (t.n.feenstra@uu.nl)


