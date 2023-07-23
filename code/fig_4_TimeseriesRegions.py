#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 09:40:58 2021

@author: nrietze

Create comparison plot like Fig 3b in Hugonnet et al. (2021)
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import  HandlerPatch
import matplotlib.patches as mpatches

def make_legend_polygon(legend, orig_handle,
                      xdescent, ydescent,
                      width, height, fontsize):
    a=2*height
    p = mpatches.Polygon(np.array([[0,-a/2],[width,-a/2],[width,height+a/2],[0,height+a/2],[0,-a/2]]))
    return p

# %%
years = np.arange(1982,2020).astype(str)

# fig,axs = plt.subplots(1,2,figsize = (12,9), dpi = 150,
#                        sharey = True)
# plt.subplots_adjust(wspace=0.1)
fig,ax = plt.subplots(figsize = (12,9), dpi = 150)

# ALPS 
#----------
DF_MB_AVHRR = pd.read_csv('../../results/all/annual_MB_allglaciers_Alps_VNN.csv',sep=';', lineterminator='\n',header=0,index_col = 0)  
DF_Hugonnet = pd.read_csv('../../data/MB/validation_data/Hugonnet/11_mb_glspec.dat', sep='\s+', lineterminator='\n',header=2).set_index('RGI-ID')
Glacier_attrs = pd.read_csv('../../data/RGI/EUROPE/CentralEurope_ge02km2_List.csv', sep=';', lineterminator='\n',header=0).set_index('GlogemId')

rgi_ids = Glacier_attrs.RGIId
common_rgi_ids = np.isin(DF_Hugonnet.index,rgi_ids)

mb_AVHRR = DF_MB_AVHRR[years].mean(axis=0).div(1e3)
unc_AVHRR = 0.401

tlim_huss = [np.arange(1982,1990+1).astype(str),
               np.arange(1990,2000+1).astype(str),
               np.arange(2000,2010+1).astype(str)]

tlim_zemp = np.arange(2006,2016+1).astype(str)
tlim_sommer = np.arange(2000,2014+1).astype(str)
tlim_hugo = np.arange(2000,2019+1).astype(str)
list_tlim = [tlim_zemp,tlim_sommer,tlim_hugo]

mb_huss = [-0.4,
           -0.68,
           -0.99]

mb_zemp = -0.87
mb_sommer = -0.698 # entire Alps
mb_hugo = DF_Hugonnet.B.mean()
list_mbs = [mb_zemp,mb_sommer,mb_hugo]

unc_huss = 0.041

unc_zemp = 0.07
unc_sommer = 0.128
unc_hugo = sum(DF_Hugonnet[common_rgi_ids].Area * DF_Hugonnet[common_rgi_ids].errB) / sum(DF_Hugonnet[common_rgi_ids].Area) * 1.96
list_unc = [unc_zemp,unc_sommer,unc_hugo]

# List of colors (last one is Huss)
# list_colors = ['#3399FF','plum','#FF9966']
list_colors = ['#2C6700','#CC0000','#FF9966','#3333FF' ]
#----------

# ax = axs[0]
ax.axhline(0,lw = .7, c = 'k', ls = 'dashed')

# Plot AVHRR yearly results
for i,_ in enumerate(years[:-1]):
    ax.fill_between([years[i],years[i+1]] ,mb_AVHRR[i] - unc_AVHRR, mb_AVHRR[i] + unc_AVHRR,
                    color=plt.cm.Greys(0.9),alpha=0.4,linewidth=0.25,zorder = 0)
    ax.plot([years[i],years[i+1]] , [mb_AVHRR[i]] * 2 ,
            color=plt.cm.Greys(0.9),lw=.7,zorder = 0)

# Plot Huss (2012)
for ih,tlimh in enumerate(tlim_huss):
    ax.fill_between([tlimh[0],tlimh[-1]] ,mb_huss[ih] - unc_huss, mb_huss[ih] + unc_huss,
                    color=list_colors[3],alpha=0.4,linewidth=0.25)
    ax.plot(tlimh, [mb_huss[ih]] * len(tlimh) ,
            color=plt.cm.Blues(0.9),lw=.7)

# Plot other geodetic MB
for ii,tlim in enumerate(list_tlim):
    ax.fill_between([tlim[0],tlim[-1]] ,list_mbs[ii] - list_unc[ii], list_mbs[ii] + list_unc[ii] ,
                    color=list_colors[ii],alpha=0.4,linewidth=0.25,zorder = 10)
    ax.plot(tlim, [list_mbs[ii]] * len(tlim) 
            ,color=list_colors[ii],lw=.7,zorder = 10)

ax.set_ylim([-1.8,1])
ax.set_xlim([years[0],years[-1]])
ax.set_xticks(np.arange(0,38,4))
ax.set_xticklabels(years[::4])
ax.set_ylabel('Specific mass balance (m w.e yr$^{-1}$)',labelpad=0.25)
ax.set_xlabel('Year')
ax.tick_params(width=0.35,length=2.5)

# Create Legend
p1 = ax.plot([], [], color=list_colors[3], linewidth=0.35)
p2 = ax.fill([], [], color=list_colors[3], alpha=0.4,linewidth=0.25)

p3 = ax.plot([], [], color=list_colors[0], linewidth=0.5)
p4 = ax.fill([], [], color=list_colors[0], alpha=0.45,linewidth=0.25)
p5 = ax.plot([], [], color=list_colors[1], linewidth=0.5)
p6 = ax.fill([], [], color=list_colors[1], alpha=0.45,linewidth=0.25)
p7 = ax.plot([], [], color=list_colors[2], linewidth=0.5)
p8 = ax.fill([], [], color=list_colors[2], alpha=0.45,linewidth=0.25)

p9= ax.plot([], [], color=plt.cm.Greys(0.9), linewidth=1)
p10 = ax.fill([], [], color=plt.cm.Greys(0.9), alpha=0.4,linewidth=0.25)

hm = {p2[0]: HandlerPatch(patch_func=make_legend_polygon), 
      p4[0]: HandlerPatch(patch_func=make_legend_polygon), 
      p6[0]: HandlerPatch(patch_func=make_legend_polygon),
      p8[0]: HandlerPatch(patch_func=make_legend_polygon), 
      p10[0]: HandlerPatch(patch_func=make_legend_polygon)}
l = ax.legend([(p1[0],p2[0]),
                (p3[0],p4[0]),(p5[0],p6[0]),(p7[0], p8[0]),
                (p9[0],p10[0])], 
               ['Huss (2012) \n (decadal)',
                'Zemp et al.(2019)','Sommer et al. (2020)','Hugonnet et al. (2021)',
                'This study \n (annual)'],
               handlelength=0.75,
               framealpha=0.6,
               # loc=(0.35,0.725),
               loc = 'upper right',
               ncol=2,
               labelspacing=1.35,
               handler_map=hm,
               borderpad=0.6)
l.get_frame().set_linewidth(0.5)

plt.savefig('../../figures/final_paper/updated_MB/fig_4.png', bbox_inches = 'tight')