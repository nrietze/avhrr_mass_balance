#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 09:40:58 2021

@author: nrietze

Create empirical cumulative distribution function plot for Figure 6.
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import os
os.chdir('C:/Users/Nils Rietze/Documents/0_Uni Bern/MSc/MSc Thesis')

# %%
print('Loading data...')
def ecdf(xdata):
    xdataecdf = np.sort(xdata)
    ydataecdf = np.arange(1, len(xdata) + 1) / len(xdata) * 100
    return xdataecdf, ydataecdf

DF_model_params_w = pd.read_csv('./results/all/winter_model_parameters_allglaciers_Alps_VNN.csv',sep=';', lineterminator='\n',header=0,index_col = 0)
DF_model_params_a = pd.read_csv('./results/all/annual_model_parameters_allglaciers_Alps_VNN.csv',sep=';', lineterminator='\n',header=0,index_col = 0)
DF_model_params_a_AA = pd.read_csv('./results/all/annual_model_parameters_allglaciers_Alps_VAA.csv',sep=';', lineterminator='\n',header=0,index_col = 0)
DF_model_params_a_4km = pd.read_csv('./results/all/annual_model_parameters_allglaciers_Alps_VAA_GAC.csv',sep=';', lineterminator='\n',header=0,index_col = 0)

# %% ECDF Plot
print('Plotting ECDF...')
_,ax = plt.subplots(figsize = (5,4),dpi = 150)

x,y = ecdf(DF_model_params_w['WOSM*'].div(.5e3))
ax.plot(x,y, lw = 1,label = 'winter CRG')
x,y = ecdf(DF_model_params_a['WOSM*'].div(.5e3))
ax.plot(x,y,c='k', lw = 1,label = 'annual CRG')
x,y = ecdf(DF_model_params_a_AA['WOSM*'].div(.5e3))
ax.plot(x,y,c='k',ls = '--', lw = 1,label = 'annual AA')
x,y = ecdf(DF_model_params_a_4km['WOSM*'].div(.5e3))
ax.plot(x,y, c = 'brown',ls = '--', lw = 1,label = 'annual 4km AA')

ax.axhline(50, c = 'gray', alpha = .3, ls = '--')
ax.text(x = -30, y = 50, s = '50',va = 'center',c = 'gray')

ax.set(xlabel = 'WOSM* side length (km)',
       ylabel = 'Fraction of studied glaciers (%)',
       ylim = [0,100],
       xlim = [0,400])
ax.legend()

print('Saving Figure...')
plt.savefig('./figures/final_paper/fig_6.png', bbox_inches = 'tight')

print('done.')