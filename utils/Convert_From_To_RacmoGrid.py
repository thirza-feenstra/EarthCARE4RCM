#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 20:07:42 2019

@author: Willem Jan van de Berg, w.j.vandeberg@uu.nl
Code to convert back and forth the rotated pole grids of RACMO, which are
type 10 grids in the good old GRIB conventions.

Thanks to Michael Bevis who initially sorted out these calculations and wrote programs in Matlab.

I believe this code is error free but I'm open for suggestions, improvements and corrections.
"""
import numpy as np
#import RCM_gridSettings as RgS See this routine for definition of GridInfoType

def printline(name, latin, lonin, latout, lonout, linp=False, loutp=False):
    if linp and lonin<0.:
        lonin = lonin+360.
    if loutp and lonout<0.:
        lonout = lonout+360.
    
    print('{0:s} lon/lat: {1:8.3f} {2:8.3f} -> {3:8.3f} {4:8.3f}'.format(name, 
                lonin, latin, lonout, latout))
    return
        

def RealWorld2RotatedGrid(rwLat, rwLon, GridInfo, log=False):
    """
    This routine converts the array-types of rwLat and rwLon into rgLat and rgLon
    sp_lat and sp_lon (shifted pole) are the input variables, they are elements 
    32 (sp_lat) and 35 (sp_lon), counting from 0, in the "block2" entry of properly 
    converted RACMO2 NetCDF files.
    All input is expected to be in degrees (thus not in radians or in thousands of a degree)
    
    The method is:
        1) convert the entries into unit vectors on a sphere
        2) rotate these vectors obeying the shifted pole
        3) convert the rotated vectors back into lat-lon coordinates
    """
    
    # shortcuts
    p2g = 180./np.pi
    g2p = np.pi/180.
    
    ca = np.cos((-GridInfo.polat - 90.)*g2p)  # - 90 is easier for the calculations
    sa = np.sin((-GridInfo.polat - 90.)*g2p)  # - 90 is easier for the calculations
    co = np.cos(  GridInfo.polon       *g2p)
    so = np.sin(  GridInfo.polon       *g2p)
    
    # 1a convert to radians
    rwLatr = rwLat*g2p
    rwLonr = rwLon*g2p
    # 1b convert to a unit vector e
    ex     = np.cos(rwLatr)*np.cos(rwLonr)
    ey     = np.cos(rwLatr)*np.sin(rwLonr)
    ez     = np.sin(rwLatr)
    
    # 2 rotate this vector assuming a round Earth
    gx     = ( co*ex + so*ey)*ca - sa*ez
    gy     =  -so*ex + co*ey
    gz     = ( co*ex + so*ey)*sa + ca*ez
    
    # 3 translate vector back into lat-lon, including some security checks
    # for numerical rounding errors
    agz    = np.where(np.abs(gz)> 1.,                       1., np.abs(gz))
    hx     = np.where(     agz <= 1., gx/np.cos(np.arcsin(gz)),         0.)
    hx     = np.where(     hx  < -1.,    -1.,    np.where(hx > 1., 1., hx))
    
    rgLat  = np.arcsin(gz)*p2g
    rgLon  = np.where(hx <= -1., 180., \
                      np.where(gy > 0., np.arccos(hx)*p2g, -np.arccos(hx)*p2g))

#    print('Real world ll = {0:10.5f} {1:10.5f}'.format(rwLat[-1,-1], rwLon[-1,-1]))
#    print('Initial e     = {0:10.6f} {1:10.6f}  {2:10.6f}'.format(ex[-1,-1], ey[-1,-1] , ez[-1,-1]))
#    print('Rotated g     = {0:10.6f} {1:10.6f}  {2:10.6f}'.format(gx[-1,-1], gy[-1,-1] , gz[-1,-1]))
#    print('Dummy hx      = {0:10.6f}'.format(hx[-1,-1]))
#    print('Racmo lat lon = {0:10.5f} {1:10.5f}'.format(rgLat[-1,-1], rgLon[-1,-1]))
    
    return rgLat, rgLon


def RotatedGrid2RealWorld(rgLat, rgLon, GridInfo, log=False, ListEdges=False):
    """
    This routine converts the array-types of rgLat and rgLon into rwLat and rwLon
    sp_lat and sp_lon (shifted pole) are the input variables, they are elements 
    32 (sp_lat) and 35 (sp_lon), counting from 0, in the "block2" entry of properly 
    converted RACMO2 NetCDF files.
    All input is expected to be in degrees (thus not in radians or in thousands of a degree)
    
    The method is:
        1) convert the entries into unit vectors on a sphere
        2) rotate these vectors correcting for the shifted pole
        3) convert the rotated vectors back into real world lat-lon coordinates
    """
    
    # shortcuts
    p2g = 180./np.pi
    g2p = np.pi/180.
    
    ca = np.cos((GridInfo.polat + 90.)*g2p)  # + 90 is easier for the calculations
    sa = np.sin((GridInfo.polat + 90.)*g2p)  # + 90 is easier for the calculations
    co = np.cos( GridInfo.polon       *g2p)
    so = np.sin( GridInfo.polon       *g2p)
    
    # 1a convert to radians
    rgLatr = rgLat*g2p
    rgLonr = rgLon*g2p
    # 1b convert to unit vector e
    ex     = np.cos(rgLatr)*np.cos(rgLonr)
    ey     = np.cos(rgLatr)*np.sin(rgLonr)
    ez     = np.sin(rgLatr)
    
    # 2 rotate this vector assuming a round Earth
    gx     = (ca*ex - sa*ez)*co - so*ey
    gy     = (ca*ex - sa*ez)*so + co*ey
    gz     =  sa*ex + ca*ez
    
    # 3 translate vector back into lat-lon, including some security checks
    # for numerical rounding errors
    agz    = np.where(np.abs(gz)> 1.,                       1., np.abs(gz))
    hx     = np.where(     agz <= 1., gx/np.cos(np.arcsin(gz)),         0.)
    hx     = np.where(     hx  < -1.,    -1.,    np.where(hx > 1., 1., hx))
    
    rwLat  = np.arcsin(gz)*p2g
    rwLon  = np.where(hx <= -1., 180., \
                      np.where(gy > 0., np.arccos(hx)*p2g, -np.arccos(hx)*p2g))
    
#    print('Racmo lat lon = {0:10.5f} {1:10.5f}'.format(rgLat[-1,-1], rgLon[-1,-1]))
#    print('Initial e     = {0:10.6f} {1:10.6f}  {2:10.6f}'.format(ex[-1,-1], ey[-1,-1] , ez[-1,-1]))
#    print('Rotated g     = {0:10.6f} {1:10.6f}  {2:10.6f}'.format(gx[-1,-1], gy[-1,-1] , gz[-1,-1]))
#    print('Dummy hx      = {0:10.6f}'.format(hx[-1,-1]))
#    print('Real world ll = {0:10.5f} {1:10.5f}'.format(rwLat[-1,-1], rwLon[-1,-1]))
    if ListEdges:
        [nx, ny]   = rgLon.shape
        [xh, yh] = [(nx-1)//2, (ny-1)//2]        # get integer of half of the domain

        printline('TLC', rgLat[-1, 0], rgLon[-1, 0], rwLat[-1, 0], rwLon[-1, 0], linp=True, loutp=True)
        printline('CNB', rgLat[-1,yh], rgLon[-1,yh], rwLat[-1,yh], rwLon[-1,yh], loutp=True)
        printline('TRC', rgLat[-1,-1], rgLon[-1,-1], rwLat[-1,-1], rwLon[-1,-1], loutp=True)
        printline('CWB', rgLat[xh, 0], rgLon[xh, 0], rwLat[xh, 0], rwLon[xh, 0], loutp=True)
        printline('CPD', rgLat[xh,yh], rgLon[xh,yh], rwLat[xh,yh], rwLon[xh,yh], loutp=True)
        printline('CEB', rgLat[xh,-1], rgLon[xh,-1], rwLat[xh,-1], rwLon[xh,-1], loutp=True)
        printline('BLC', rgLat[ 0, 0], rgLon[ 0, 0], rwLat[ 0, 0], rwLon[ 0, 0], loutp=True)
        printline('CSB', rgLat[ 0,yh], rgLon[ 0,yh], rwLat[ 0,yh], rwLon[ 0,yh], loutp=True)
        printline('BRC', rgLat[ 0,-1], rgLon[ 0,-1], rwLat[ 0,-1], rwLon[ 0,-1], loutp=True)
        
    elif log:
        printline('BLC', rgLat[ 0, 0], rgLon[ 0, 0], rwLat[ 0, 0], rwLon[ 0, 0], loutp=True)
        printline('TLC', rgLat[-1, 0], rgLon[-1, 0], rwLat[-1, 0], rwLon[-1, 0], loutp=True)
        printline('TRC', rgLat[-1,-1], rgLon[-1,-1], rwLat[-1,-1], rwLon[-1,-1], loutp=True)
        printline('BRC', rgLat[ 0,-1], rgLon[ 0,-1], rwLat[ 0,-1], rwLon[ 0,-1], loutp=True)
    
    return rwLat, rwLon

def get_area(GridInfo):
    GridArea = np.zeros([GridInfo.nlat,GridInfo.nlon])
    REarth = 6.37E6
    DisPLS  = (np.pi*REarth/180.)**2
    for i in range (GridInfo.nlat):
        GridArea[i,:] = GridInfo.dlat * GridInfo.dlon * np.cos(GridInfo.rglat[i]*np.pi/180.) * DisPLS
    return GridArea
    
    

def rgrid(GridInfo):
# this method uses the GridInfoType type    

# get a rotated grid for the given grid information
    rglat = np.zeros([GridInfo.nlat, GridInfo.nlon])
    rglon = np.zeros([GridInfo.nlat, GridInfo.nlon])
# the loops over i are needed as process a vector of the matrix at once (at least I don't know how to do 
#   that more efficiently)
    for i in range(GridInfo.nlat): rglat[i,:] = GridInfo.south + i*GridInfo.dlat
    for i in range(GridInfo.nlon): rglon[:,i] = GridInfo.west  + i*GridInfo.dlon
    
    return rglat, rglon
