#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 17:36:11 2020

@author: nrietze
"""
import os

      
#%% ==========================================================================    
# Run config
exec(open("./code/config.py").read())

if not os.path.exists(f'./intermediate/avhrr/pso_summer_{band}.nc') or not os.path.exists(f'./intermediate/avhrr/pso_summer_{band}.nc'):
    resp = input('The seasonal snow datasets are missing, do you want to compute them? \n  y / n \n (WARNING: This will take a while!)')
    
    if resp == 'y':
        exec(open("./code/aggregate_netCDF.py"))
    if resp == 'n':
        print(f'Please create/move a folder called ./intermediate/avhrr/ into this parent directory and provide the seasonal datasets using the filenames: pso_summer_{band}.nc and pso_winter_{band}.nc')
        pass
    
# Run importer
exec(open("./code/importer.py").read())

#%% Compute altitudinal distribution of snow occurrence for each WOSM size
# exec(open("./code/getAltSnowDistr.py").read())    

#%% Calibrate Model:
exec(open("./code/main_calibrate.py").read())  

# %% Compute Summer MB
start = 1982
stop = 2019
allyears = np.arange(start, stop+1)
allyears = calper.astype('<U21')

df_annual = pd.read_csv('../../results/updated/annual_MB_allglaciers_Alps_V%s.csv' % (method),
                    index_col = 0,sep = ';')
df_winter = pd.read_csv('../../results/updated/winter_MB_allglaciers_Alps_V%s.csv' % (method),
                    index_col = 0,sep = ';')

df_summer = df_annual.copy()
df_summer.loc[:,allyears] = df_annual.loc[:,allyears] - df_winter.loc[:,allyears]
  

