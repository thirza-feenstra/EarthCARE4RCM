#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code to read in RACMO grid data.

@author: Willem Jan van den Berg
"""

import numpy as np
import Convert_From_To_RacmoGrid as CFTR


class GridInfoType:
    """
    """    
    def __init__(self):
        self.name  = ""
        self.nlon  = 0
        self.nlat  = 0
        self.nlev  = 0
        self.south = 0.
        self.west  = 0.
        self.north = 0.
        self.east  = 0.
        self.polon = 0.
        self.polat = 0.
        self.dlat  = 0.
        self.dlon  = 0.
        self.rglat = 0.
        self.rglon = 0.
        self.rwlat = 0.
        self.rwlon = 0.
        
    def add_RotatedCoordinates(self):
        [self.rglat, self.rglon] = CFTR.rgrid(self)
        
    def add_RealWorldCoordinates(self, log=False, ListEdges=False):
        if np.size(self.rglat) != self.nlon*self.nlat:
            [self.rglat, self.rglon] = CFTR.rgrid(self)
        if ListEdges:
            CentreDomain = CFTR.RotatedGrid2RealWorld(np.zeros([1,1]), 
                        np.zeros([1,1]), self)
            print( "The centre of domain {2:s} is at lon/lat: {0:8.3f} {1:8.3f}".format(
                CentreDomain[1][0,0], CentreDomain[0][0,0], self.name) )
        [self.rwlat, self.rwlon] = CFTR.RotatedGrid2RealWorld(self.rglat, 
                    self.rglon, self, log=log, ListEdges=ListEdges)
    
    def add_gridboxarea(self, log=False):
        self.gbarea = CFTR.get_area(self)
        
# read data
def read_Setting_data(domain, log=False, path="GridSettings"):
    file = path+"/SETTINGS_"+domain
    with open(file) as GSfile:
        lok = True
        linfo    = np.zeros(9, dtype = bool)
        GridInfo = GridInfoType()
        GridInfo.name = domain
        if log: print('read grid data domain '+domain+':')
        for line in GSfile:
            if line[:11]=="setenv NLON":
                GridInfo.nlon = int(line[12:])    
                if (log): print('nlon  = {0:5d}'.format(GridInfo.nlon))
                linfo[0] = True
            elif line[:11]=="setenv NLAT":
                GridInfo.nlat = int(line[12:])    
                if (log): print('nlat  = {0:5d}'.format(GridInfo.nlat))
                linfo[1] = True
            elif line[:11]=="setenv NLEV":
                GridInfo.nlev = int(line[12:])    
                if (log): print('nlev  = {0:5d}'.format(GridInfo.nlev))
                linfo[2] = True
            elif line[:12]=="setenv SOUTH":
                GridInfo.south = float(line[13:])    
                if (log): print('south = {0:9.3f}'.format(GridInfo.south))
                linfo[3] = True
            elif line[:11]=="setenv WEST":
                GridInfo.west = float(line[13:])    
                if (log): print('west  = {0:9.3f}'.format(GridInfo.west))
                linfo[4] = True
            elif line[:12]=="setenv NORTH":
                GridInfo.north = float(line[13:])    
                if (log): print('north = {0:9.3f}'.format(GridInfo.north))
                linfo[5] = True
            elif line[:11]=="setenv EAST":
                GridInfo.east = float(line[13:])    
                if (log): print('east  = {0:9.3f}'.format(GridInfo.east))
                linfo[6] = True
            elif line[:12]=="setenv POLAT":
                GridInfo.polat = float(line[13:])    
                if (log): print('polat = {0:9.3f}'.format(GridInfo.polat))
                linfo[7] = True
            elif line[:12]=="setenv POLON":
                GridInfo.polon = float(line[13:])    
                if (log): print('polon = {0:9.3f}'.format(GridInfo.polon))
                linfo[8] = True
            else:
                print('Read error: '+line[:(len(line)-1)])
                lok = False
    
    if not(lok) or any(linfo==False):
        print('Read of '+domain+' failed!')
        return

    GridInfo.dlat = (GridInfo.north-GridInfo.south)/(GridInfo.nlat-1)
    GridInfo.dlon = (GridInfo.east - GridInfo.west)/(GridInfo.nlon-1)
    if log: 
        print('dlat  = {0:9.3f}\ndlon  = {1:9.3f}'.format(GridInfo.dlat, GridInfo.dlon))
    
    return GridInfo


