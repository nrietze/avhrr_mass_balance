#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting 
"""
import matplotlib

import numpy as np
import pandas as pd


import os
os.chdir('C:/Users/Nils Rietze/Documents/0_Uni Bern/MSc/MSc Thesis/code/final/')

from functions_plots import *

# %% Load data
print('Loading data...')
df_GlacierAttrs = pd.read_csv('../../data/RGI/EUROPE/CentralEurope_ge02km2_List.csv',
                            sep=';',header=0,index_col = 'GlogemId')

df_MBobs_a = pd.read_csv('../../data/MB/insitu/all/all_insitu_annual.csv',
                         sep=';',header=0,index_col ='GLOGEM_ID')

DF_model_params_w = pd.read_csv('../../results/all/winter_model_parameters_allglaciers_Alps_VNN.csv',sep=';', lineterminator='\n',header=0,index_col = 0)
DF_model_params_a = pd.read_csv('../../results/all/annual_model_parameters_allglaciers_Alps_VNN.csv',sep=';', lineterminator='\n',header=0,index_col = 0)
DF_model_params_a_AA = pd.read_csv('../../results/all/annual_model_parameters_allglaciers_Alps_VAA.csv',sep=';', lineterminator='\n',header=0,index_col = 0)
DF_model_params_a_4km = pd.read_csv('../../results/all/annual_model_parameters_allglaciers_Alps_VAA_GAC.csv',sep=';', lineterminator='\n',header=0,index_col = 0)

print('done.')
#%% Plot results =============================================================
#  FIG 2 - METHOD PLOTS
PlotFigure2a(years,df_GlacierAttrs)
PlotFigure2b(years=years, df_MBobs=df_MBobs_a,df_ModelParams=DF_model_params_a)

matplotlib.rcParams.update(matplotlib.rcParamsDefault)

# FIG 3 - SCATTERPLOTS
exec(open("fig_3_scatterplots.py").read())

# FIG 4 - COMPARISON WITH GEODETIC
DF_MB_AVHRR = pd.read_csv('./results/all/annual_MB_allglaciers_Alps_VNN.csv',sep=';', 
                          header=0,index_col = 0)  
DF_Hugonnet = pd.read_csv('./data/MB/validation_data/Hugonnet/11_mb_glspec.dat', sep='\s+', 
                          header=2).set_index('RGI-ID')
PlotFigure4(years,DF_MB_AVHRR,DF_Hugonnet,df_GlacierAttrs)

# FIG 5
exec(open("fig_4_TimeseriesRegions.py".py").read())

# FIG 6 - ECDF
PlotFigure6(DF_model_params_w,
            DF_model_params_a,
            DF_model_params_a_AA,
            DF_model_params_a_4km)