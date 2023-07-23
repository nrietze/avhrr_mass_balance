#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 19:19:41 2021

@author: nrietze
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator
import mpl_toolkits.axes_grid1.inset_locator as mpl_il
from matplotlib.gridspec import GridSpec

from scipy.stats.kde import gaussian_kde

from sklearn.metrics import r2_score

def rmse(Y_test, Y_pred):
    return np.sqrt(((Y_test - Y_pred) ** 2).mean())

def plot_refglac_seas(DF1,DF2,DF_c,diffs,valid_idx,season,color,cmap,ax,histbox,i):
    xy = np.vstack([DF_c.OBS[valid_idx],DF_c.AVHRR[valid_idx]])
    k = gaussian_kde(xy)(xy)

    nr = ['(d)','(f)']
    
    P = ax.scatter(DF_c.OBS[valid_idx]/1e3 , DF_c.AVHRR[valid_idx]/1e3 , c = k , marker='.',cmap = cmap)

    if i:
        pass
    else:
        ax.set_ylabel('MB from AVHRR (m w.e. a$^{-1}$)')
    textstr = '\n'.join((
        r'R$^2$ = {0:.2f}'.format( r2_score(DF1.loc[DF1.OBS.notna()],DF2.loc[DF1.OBS.notna()]) ),                                              
        r'RMSE = %.3f m w.e. a$^{-1}$' % (rmse(DF1.OBS,DF2.AVHRR)/1e3),
        r'N = {}'.format( DF_c.OBS.count() ) ))
    
    ax.text(pos[0],pos[1],r'{} balance'.format(season),
            verticalalignment='bottom',horizontalalignment = 'left',
            fontsize = 11,weight = 'bold')
    ax.text(pos[0],pos[1],textstr,
            verticalalignment='top',horizontalalignment = 'left',
            fontsize = 10)
    ax.text(0.08,0.98,nr[i],
        verticalalignment='top',horizontalalignment = 'right',
        fontsize = 14,weight = 'bold',
        transform=ax.transAxes)
            
    ax2 = mpl_il.inset_axes(ax, 
                            width='100%', 
                            height='100%',
                            bbox_to_anchor = histbox,
                            bbox_transform = ax.transAxes,
                            loc = 3)
    ax2.hist(diffs, 
             bins=20,
             histtype='step',
             color = color)
    ax2.axvspan(diffs.mean()-diffs.std(), diffs.mean()+diffs.std(), alpha=0.2, color='gray')
    ax2.axvline(ls = (0, (5, 10)),c='k',lw=.7)
    
    ax2.set_xlim([-2.1,2.1])
    ax2.margins(x=0.5)
    ax2.set_xlabel('$\Delta B_{}$'.format(season[0]),fontsize=8)
    ax2.set_ylabel("Count",labelpad = -3,fontsize=8)
    ax2.set_yticklabels("")
    ax2.tick_params(left=False,labelsize = 8)
    
def plot_nonrefglac_seas(DF1,DF2,DF_c,diffs,valid_idx,season,color,ax,histbox):
    
    P = ax.scatter(DF_c.OBS[valid_idx]/1e3 , DF_c.AVHRR[valid_idx]/1e3 , c = color , marker='.')

    ax.set_xlabel('MB from glaciological observations (m w.e. a$^{-1}$)')
        
    textstr = '\n'.join((
        r'R$^2$ = {0:.2f}'.format( r2_score(DF1.loc[DF1.OBS.notna()],DF2.loc[DF1.OBS.notna()]) ),                                              
        r'RMSE = %.3f m w.e. a$^{-1}$' % (rmse(DF1.OBS,DF2.AVHRR)/1e3),
        r'N = {}'.format( DF_c.OBS.count() ) ))
    
    ax.text(pos[0],pos[1],r'{} balance'.format(season),
            verticalalignment='bottom',horizontalalignment = 'left',
            fontsize = 11,weight = 'bold')
    ax.text(pos[0],pos[1],textstr,
            verticalalignment='top',horizontalalignment = 'left',
            fontsize = 10)
    ax.text(0.08,0.98,'(e)',
        verticalalignment='top',horizontalalignment = 'right',
        fontsize = 14,weight = 'bold',
        transform=ax.transAxes)
            
    ax2 = mpl_il.inset_axes(ax, 
                            width='100%', 
                            height='100%',
                            bbox_to_anchor = histbox,
                            bbox_transform = ax.transAxes,
                            loc = 3)
    ax2.hist(diffs, 
             bins=20,
             histtype='step',
             color = color)
    ax2.axvspan(diffs.mean()-diffs.std(), diffs.mean()+diffs.std(), alpha=0.2, color='gray')
    ax2.axvline(ls = (0, (5, 10)),c='k',lw=.7)
    
    ax2.set_xlim([-2.1,2.1])
    ax2.margins(x=0.5)
    ax2.set_xlabel('$\Delta B_{}$'.format(season[0]),fontsize=8)
    ax2.set_ylabel("Count",labelpad = -3,fontsize=8)
    ax2.set_yticklabels("")
    ax2.tick_params(left=False,labelsize = 8)
    
    
def plot_annual(DF1,DF2,DF_c,diffs,valid_idx,refglac,ax,i):
    if refglac:
        xy = np.vstack([DF_c.OBS[valid_idx],DF_c.AVHRR[valid_idx]])
        k = gaussian_kde(xy)(xy)
        
        P = ax.scatter(DF_c.OBS[valid_idx]/1e3 , DF_c.AVHRR[valid_idx]/1e3 , c = k , marker='.')
        
    else:
        P = ax.scatter(DF_c.OBS[valid_idx]/1e3 , DF_c.AVHRR[valid_idx]/1e3 , c = 'k' , marker='.')
     
    # Lines at x=y=0
    ax.axhline(0, c=".7",linewidth = .7,zorder=0)
    ax.axvline(0, c=".7",linewidth = .7,zorder=0)
    
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    
    # 1:1 Line
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".7",linewidth = .7)
    
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # For Plots on the right hide y label
    if i==0:
        ax.set_xlabel('')
        ax.set_ylabel('MB from AVHRR (m w.e. a$^{-1}$)')
        
        ax.text(pos[0],pos[1],r'Annual balance: closest ref. glacier',
            verticalalignment='bottom',horizontalalignment = 'left',
            fontsize = 11,weight = 'bold')
    elif i==1:
        ax.set_xlabel('MB from glaciological observations (m w.e. a$^{-1}$)')
        ax.set_ylabel('')
        
        ax.text(pos[0],pos[1],r'Annual balance: closest ref. glacier',
            verticalalignment='bottom',horizontalalignment = 'left',
            fontsize = 11,weight = 'bold')
    elif i==2:
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        ax.text(pos[0],pos[1],r'Annual balance: alpine average',
            verticalalignment='bottom',horizontalalignment = 'left',
            fontsize = 11,weight = 'bold')
        
    textstr = '\n'.join((
        r'R$^2$ = {0:.2f}'.format( r2_score(DF1.loc[DF1.OBS.notna()],DF2.loc[DF1.OBS.notna()]) ),                                              
        r'RMSE = %.3f m w.e. a$^{-1}$' % (rmse(DF1.OBS,DF2.AVHRR)/1e3) ,
        r'N = {}'.format( DF_c.OBS.count() ) ))
    
    nr = ['(a)','(b)','(c)','(d)']
    
    ax.text(pos[0],pos[1],textstr,
            verticalalignment='top',horizontalalignment = 'left',
            fontsize = 10)
    ax.text(0.08,0.98,nr[i],
        verticalalignment='top',horizontalalignment = 'right',
        fontsize = 14,weight = 'bold',
        transform=ax.transAxes)
            
    ax2 = mpl_il.inset_axes(plt.gca(), 
                            width='30%', 
                            height='30%', 
                            # loc=4,
                            # borderpad = 1.8,
                            bbox_to_anchor=(.014,-.5,1,.9), bbox_transform=ax.transAxes)
    ax2.hist(diffs, 
             bins=20,
             histtype='step',
             color = 'k')
    ax2.axvspan(diffs.mean()-diffs.std(), diffs.mean()+diffs.std(), alpha=0.2, color='gray')
    ax2.axvline(ls = (0, (5, 10)),c='k',lw=.7)
    
    ax2.set_xlim([-2.1,2.1])
    ax2.margins(x=0.5)
    ax2.set_xlabel('$\Delta B_a$',fontsize=8)
    ax2.set_ylabel("Count",labelpad = -3,fontsize=8)
    ax2.set_yticklabels("")
    ax2.tick_params(left=False,labelsize = 8)    
    
# %% ANNUAL MB
#  Preparations
years = np.arange(1982,2020).astype(str)

#  Load Glacier List with Attributes
Glacier_attrs = pd.read_csv('../../data/RGI/EUROPE/CentralEurope_ge02km2_List.csv', sep=';', header=0).set_index('GlogemId')

methods = ["NN","AA"]

gs = GridSpec(2, 3, hspace=0.15, wspace = 0.05) 
fig = plt.figure(figsize = (21,14), dpi = 300) 

print('Building grid & iterating methods...')

for i,mtd in enumerate(methods):
    print('Assembling plots for method: %s' mtd)
    DF_obs_MB = pd.read_csv('../../data/MB/insitu/all/all_insitu_annual.csv', sep=';', header=0,index_col ='GLOGEM_ID')

    DF_MB_AVHRR = pd.read_csv('../../results/updated/annual_MB_allglaciers_Alps_V%s.csv' % (mtd),sep=';', header=0,index_col = 0)  
    MB_AVHRR = DF_MB_AVHRR[years]
    
    DF_MBwinter_AVHRR = pd.read_csv('../../results/updated/winter_MB_allglaciers_Alps_V%s.csv' % (mtd),sep=';', header=0,index_col = 0)  
    
    limits = [-4,1.5]; pos = [-3.7,1]

    if mtd == 'NN':
        # ANNUAL MB
        # ===========
        #  Reference glaciers
        #----------
        #  Get Glogem Ids of Reference glaciers (the ones with >10yrs)
        refglac_ids = Glacier_attrs.index[Glacier_attrs.HubDist_10yrs==0]
        
        DF1 = DF_obs_MB.loc[refglac_ids,years].melt(value_name = 'OBS').set_index('variable',append=True)
        DF2 = MB_AVHRR.loc[refglac_ids,years].melt(value_name = 'AVHRR').set_index('variable',append=True)
        
        DF_c=pd.concat([DF1,DF2],axis=1)
        valid_idx = DF_c.OBS.notna()
        
        diffs = DF_c.OBS[valid_idx]/1e3 - DF_c.AVHRR[valid_idx]/1e3

        ax1 = fig.add_subplot(gs[0,0])
        plot_annual(DF1,DF2,DF_c,diffs,valid_idx,True,ax1,0)
        #----------
        
        #  Non - Reference glaciers
        #----------
        insitu_ids = Glacier_attrs.index[(Glacier_attrs.N_OBS_ANN>0) & (Glacier_attrs.N_OBS_ANN<10)]
        
        DF1 = DF_obs_MB.loc[insitu_ids,years].melt(value_name = 'OBS').set_index('variable',append=True)
        DF2 = MB_AVHRR.loc[insitu_ids,years].melt(value_name = 'AVHRR').set_index('variable',append=True)
        
        DF_c=pd.concat([DF1,DF2],axis=1)
        valid_idx = DF_c.OBS.notna()
        
        diffs = DF_c.OBS[valid_idx]/1e3 - DF_c.AVHRR[valid_idx]/1e3

        ax2 = fig.add_subplot(gs[0,1])
        plot_annual(DF1,DF2,DF_c,diffs,valid_idx,False,ax2,1)
        ax2.tick_params(labelleft = False)

        
    else:
        #  Plot Obs vs. AVHRR 
        #----------
        #  Get Glogem Ids of monitored glaciers 
        insitu_ids = DF_obs_MB.index
        
        DF1 = DF_obs_MB.loc[insitu_ids,years].melt(value_name = 'OBS').set_index('variable',append=True)
        DF2 = MB_AVHRR.loc[insitu_ids,years].melt(value_name = 'AVHRR').set_index('variable',append=True)
        
        DF_c=pd.concat([DF1,DF2],axis=1)
        valid_idx = DF_c.OBS.notna()
        
        diffs = DF_c.OBS[valid_idx]/1e3 - DF_c.AVHRR[valid_idx]/1e3
        
        ax3 = fig.add_subplot(gs[0,2])
        plot_annual(DF1,DF2,DF_c,diffs,valid_idx,True,ax3,2)
        ax3.tick_params(labelleft = False)
        #----------
        
    # SEASONAL MB
    # ===========
    limits = [-5,3]
    
    for season in ["winter","summer"]:
        DF_obs_MB = pd.read_csv('../../data/MB/insitu/all/all_insitu_%s.csv' % season, sep=';', lineterminator='\n',header=0,index_col ='GLOGEM_ID')
        
        if season == "summer":
            cmap = 'YlOrRd_r'
            color = "r"
            pos = [.2,-3.7]
            xbox,ybox = (.65,.27)
            
        else:
            cmap = 'PuBu_r'
            color = "b"
            pos = [-4.8,1.8]
            xbox,ybox = (.35,.75)
            
        histbox = (xbox,ybox,.25,.21)
    
        if mtd == 'NN':
            #  Reference glaciers
            #----------    
            refglac_ids = DF_obs_MB[DF_obs_MB.N_OBS >= 10].index
            
            if season == "summer":
                MB_AVHRR_seas =  MB_AVHRR.loc[refglac_ids,years] - DF_MBwinter_AVHRR.loc[refglac_ids,years]
            else:
                MB_AVHRR_seas =  DF_MBwinter_AVHRR.loc[refglac_ids,years]
            
            DF1 = DF_obs_MB.loc[refglac_ids,years].melt(value_name = 'OBS').set_index('variable',append=True)
            DF2 = MB_AVHRR_seas.loc[refglac_ids,years].melt(value_name = 'AVHRR').set_index('variable',append=True)
        
            DF_c=pd.concat([DF1,DF2],axis=1)
            valid_idx = DF_c.OBS.notna()
        
            diffs = DF_c.OBS[valid_idx]/1e3 - DF_c.AVHRR[valid_idx]/1e3
            
            ax = fig.add_subplot(gs[1, 0])
            ax.axhline(0, c=".7",linewidth = .7,zorder=0)
            ax.axvline(0, c=".7",linewidth = .7,zorder=0)
            
            ax.set_xlim(limits)
            ax.set_ylim(limits)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            
            ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".7",linewidth = .7)
            
            plot_refglac_seas(DF1,DF2,DF_c,diffs,valid_idx,season,color,cmap,ax,histbox,i)
            #---------- 
            
            #  Non - Reference glaciers
            #----------
            insitu_ids = DF_obs_MB.index[(DF_obs_MB.N_OBS>0) & (DF_obs_MB.N_OBS<10)]
            
            if season == "summer":
                MB_AVHRR_seas =  MB_AVHRR.loc[insitu_ids,years] - DF_MBwinter_AVHRR.loc[insitu_ids,years]
            else:
                MB_AVHRR_seas =  DF_MBwinter_AVHRR.loc[insitu_ids,years]
                
            DF1 = DF_obs_MB.loc[insitu_ids,years].melt(value_name = 'OBS').set_index('variable',append=True)
            DF2 = MB_AVHRR_seas.loc[insitu_ids,years].melt(value_name = 'AVHRR').set_index('variable',append=True)
            
            DF_c=pd.concat([DF1,DF2],axis=1)
            valid_idx = DF_c.OBS.notna()
            
            diffs = DF_c.OBS[valid_idx]/1e3 - DF_c.AVHRR[valid_idx]/1e3
            
            axr = fig.add_subplot(gs[1,1])
            axr.axhline(0, c=".7",linewidth = .7,zorder=0)
            axr.axvline(0, c=".7",linewidth = .7,zorder=0)
            
            axr.set_xlim(limits)
            axr.set_ylim(limits)
            axr.xaxis.set_major_locator(MaxNLocator(integer=True))
            axr.yaxis.set_major_locator(MaxNLocator(integer=True))
            
            axr.plot(axr.get_xlim(), axr.get_ylim(), ls="--", c=".7",linewidth = .7)
            
            plot_nonrefglac_seas(DF1,DF2,DF_c,diffs,valid_idx,season,color,axr,histbox)
            axr.tick_params(labelleft = False)
            
        else:
            #  Plot Obs vs. AVHRR AA
            # ----------
            #  Get Glogem Ids of monitored glaciers 
            insitu_ids = DF_obs_MB.index
            
            if season == "summer":
                MB_AVHRR_seas =  MB_AVHRR.loc[insitu_ids,years] - DF_MBwinter_AVHRR.loc[insitu_ids,years]
            else:
                MB_AVHRR_seas =  DF_MBwinter_AVHRR.loc[insitu_ids,years]
            
            DF1 = DF_obs_MB.loc[insitu_ids,years].melt(value_name = 'OBS').set_index('variable',append=True)
            DF2 = MB_AVHRR_seas.loc[insitu_ids,years].melt(value_name = 'AVHRR').set_index('variable',append=True)
        
            DF_c=pd.concat([DF1,DF2],axis=1)
            valid_idx = DF_c.OBS.notna()
        
            diffs = DF_c.OBS[valid_idx]/1e3 - DF_c.AVHRR[valid_idx]/1e3
            
            ax = fig.add_subplot(gs[1, 2])
            ax.axhline(0, c=".7",linewidth = .7,zorder=0)
            ax.axvline(0, c=".7",linewidth = .7,zorder=0)
            
            ax.set_xlim(limits)
            ax.set_ylim(limits)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            
            ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".7",linewidth = .7)
            
            plot_refglac_seas(DF1,DF2,DF_c,diffs,valid_idx,season,color,cmap,ax,histbox,i) 
            ax.tick_params(labelleft = False)
            print('Seasonal & Annual plots built.')
print('Saving Figure...')
plt.savefig('../../figures/final_paper/updated_MB/Fig_3.png', bbox_inches = 'tight')

print('done.')