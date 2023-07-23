#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configure settings and select parameters
"""
import pandas as pd
import numpy as np

global WOSM_rad

print('Loading configurations...')

# Select Season: winter, annual
season = "annual"

# Select method: CRG = closest reference glacier, AA = alpine average
method = 'CRG'

# Select Glacier Inventory: all, HKH
inventory = "all"
coarsen = False # If LAC is coarsened to GAC-resolution

# Define Calibrating period:
start = 1982
stop = 2019
allyears = np.arange(start, stop+1).astype('<U21')

# Set up input variables
min_R = 2 #WOSM start radius (2 km = WOSM of 5x5km)
if inventory == "HKH":
    unit = 0.05
    max_R = 100
elif coarsen:
    unit = 4e3
    max_R = 50
else:# CRS units (1000 m)
    unit = 1e3 
    max_R = 200
WOSM_rad = np.arange(min_R * unit,max_R* unit,unit) # List of WOSM radii
stepsize = 1

# Predefine Output Variables
allscores = [None]*len(WOSM_rad)
allgrouped = [None]*len(WOSM_rad)

print('done.')
