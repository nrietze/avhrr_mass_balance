#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 17:36:11 2020

@author: nrietze
"""
import os

os.chdir('C:/Users/Nils Rietze/Documents/0_Uni Bern/MSc/MSc Thesis/code/final/')
      
#%% ==========================================================================    
# Run config
exec(open("config.py").read())

# Run importer
exec(open("importer.py").read())

#%% Compute altitudinal distribution of snow occurrence for each WOSM size
# exec(open("getAltSnowDistr.py").read())    

#%% Calibrate Model:
exec(open("main_calibrate.py").read())  

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
  

