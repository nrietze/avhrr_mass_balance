# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 19:36:07 2021

@author: Nils Rietze
"""

import pandas as pd
import numpy as np

import os
os.chdir('C:/Users/Nils Rietze/Documents/0_Uni Bern/MSc/MSc Thesis/')

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


# %%% LOAD DATA

years = np.arange(1982,2020).astype(str)

method = "NN"
inventory = "all"

MB_s_AVHRR = pd.read_csv('./results/all/summer_MB_allglaciers_Alps_VNN.csv',sep=';',
                         lineterminator='\n',header=0,index_col = 0,
                         usecols=np.concatenate(([0],range(2,40))))
MB_s_AVHRR_Alps = MB_s_AVHRR.mean(axis=0).div(1e3)
std_s_Alps = MB_s_AVHRR.std(axis=0).div(1e3)

MB_w_AVHRR = pd.read_csv('./results/all/winter_MB_allglaciers_Alps_VNN.csv',sep=';',
                         lineterminator='\n',header=0,index_col = 0,
                         usecols=np.concatenate(([0],range(2,40))))
MB_w_AVHRR_Alps = MB_w_AVHRR.mean(axis=0).div(1e3)
std_w_Alps = MB_w_AVHRR.std(axis=0).div(1e3)

MB_a_AVHRR = pd.read_csv('./results/all/annual_MB_allglaciers_Alps_VNN.csv',sep=';',
                         lineterminator='\n',header=0,index_col = 0,
                         usecols=np.concatenate(([0],range(2,40))))
MB_a_AVHRR_Alps = MB_a_AVHRR.mean(axis=0).div(1e3)
std_a_Alps = MB_a_AVHRR.std(axis=0).div(1e3)

# Load AVHRR model parameters
# DF_s_model_params = pd.read_csv('./results/all/summer_model_parameters_allglaciers_Alps_VNN.csv',
#                               sep=';',lineterminator='\n',header=0,index_col = 0)  

#  Load Glacier List with Attributes 
Glacier_attrs = pd.read_csv('./data/RGI/EUROPE/CentralEurope_ge02km2_List.csv',
                            sep=';', lineterminator='\n',header=0,index_col = 'GlogemId')


# %% Plot AVHRR MB for all ISMSA regions 
print('Building Figure...')
# create objects 
gs_top = GridSpec(8, 2 , top=0.9) 
gs_base = GridSpec(8, 2, hspace=0.3,wspace = 0) 
fig = plt.figure(constrained_layout=True,figsize = (8,14), dpi = 150) 
axs = []

ylims = [-3.5,2.5]
colors = ['red','black','blue']

# Set up top figure
ax1 = fig.add_subplot(gs_top[0, :]) 
ax1.axhline(0,linewidth = .5,linestyle = '--')
ax1.set_xticks(np.arange(0,38,4))
ax1.set_xticklabels(years[::4])
ax1.set_ylim(ylims)
ax1.set_title('European Alps',fontweight = 'bold')
ax1.set(ylabel = 'MB (m w.e.)')

# Set up Subfigures for each region
row = 1; col = 0
for i in range(14):
    axi = fig.add_subplot(gs_base[row, col]) 
    axi.axhline(0,linewidth = .5,linestyle = '--')

    axi.set_xticks(np.arange(0,38,4))
    axi.set_xticklabels(years[::4],rotation = 45)
    axi.set_ylim(ylims)
    
    if (i % 2) == 0:
        col += 1
    else:
        row +=1 ; col -= 1

    axs.append(axi)
        

for jj,ssn in enumerate(['summer','annual','winter']):
    print("Plotting %s data..." % ssn)
    MB_AVHRR = pd.read_csv('./results/all/%s_MB_allglaciers_Alps_VNN.csv' % ssn ,sep=';',
                           header=0,index_col = 0,
                             usecols=np.hstack(('Glogem_ID',years)) )
    clr = colors[jj]
    
    # Alpine Average
    MB_AVHRR_Alps = MB_AVHRR.mean(axis=0).div(1e3)
    std_Alps = MB_AVHRR.std(axis=0).div(1e3)
    
    #  Add one main plot of all Alps MB
    ax1.plot(years , MB_AVHRR_Alps, c = clr)
    ax1.fill_between(years, 
                     MB_AVHRR_Alps - std_Alps, 
                     MB_AVHRR_Alps + std_Alps,
                     color = clr, alpha=0.3,
                     linewidth=0)
    
    # Compute regional averages
    regmeans = MB_AVHRR.groupby(Glacier_attrs.ISMSA_ID).mean().div(1e3)
    regstds = MB_AVHRR.groupby(Glacier_attrs.ISMSA_ID).std().div(1e3)
    regcounts = MB_AVHRR['2019'].groupby(Glacier_attrs.ISMSA_ID).count()
    
    # Iterate Plotting for all ISMSA Regions
    row = 1; col = 0
    for i in range(14):
        axi = axs[i]
        
        region = i+1 # Nr. of ISMSA Region in Map
        axi.plot(years,regmeans.loc[region,:], c = clr)
        axi.fill_between(years, 
                         regmeans.loc[region,:]-regstds.loc[region], 
                         regmeans.loc[region,:]+regstds.loc[region],
                         color = clr, alpha=0.3,
                         linewidth=0)
        
        axi.set_title(int(region),fontweight = 'bold')
        
        axi.text('1982',(ylims[1] - .6),'n = %i' % regcounts.loc[region], fontsize = 'small')
        
# fig.subplots_adjust(hspace=0.3,wspace = 0.05)
for ax in np.array(axs).flat:
    ax.set(ylabel = 'MB (m w.e.)')
    ax.label_outer()

plt.savefig('./figures/results/fig_5.png',bbox_inches='tight')

print('Done and saved.')