#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 19:27:59 2021

@author: nrietze
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from tqdm import tqdm
from joblib import Parallel, delayed

os.chdir('C:/Users/Nils Rietze/Documents/0_Uni Bern/MSc/MSc Thesis/code/final')
from functions_model import map_snowaltdist, get_opt_parameters_and_MB

exec(open("config.py").read())
# %%
if inventory == "all" : 
    if season == "winter":
            pickldir = '../../data/AVHRR_snow_maps/winter_avgs/processed_altitudinal_distributions' 
    else:
        if coarsen:
            pickldir = '../../data/AVHRR_snow_maps/summer_avgs/processed_altitudinal_distributions_4km'
        else:
            pickldir = '../../data/AVHRR_snow_maps/summer_avgs/processed_altitudinal_distributions'
elif inventory == "HKH":
    pickldir = '../../data/AVHRR_snow_maps/summer_avgs/processed_altitudinal_distributions_HKH'
    
            
#%% CALIBRATE INDIVIDUAL GLACIERS ON NEAREST GLACIER
DF_obs_MB = pd.read_csv('../../data/MB/insitu/all/all_insitu_annual.csv', 
                     sep=';', lineterminator='\n',header=0,index_col = "RGI_ID") 

years = np.arange(1982,2020).astype(str)

gl_id = 1876

# Open altitudinal RSSO distribution file
f = open(pickldir+'/allGrDF_%s.pckl' % (gl_id), 'rb')
allgrouped = pickle.load(f)
f.close();del f

# Compute arithmetic mean MB over all insitus (even the ones with nobs<10)
# y = DF_obs_MB.loc[:,years].mean()

# Get MB of Grieglacier
y = DF_obs_MB.loc['RGI60-11.01876',years].astype(float)

P=[]
for i,t in enumerate(y.index[:-9].astype(int)):
    for d in range(9,len(y)-i):
        last = min(2019,(t+d))
        P.append(np.arange(t,last+1).astype(str))
  
# %%
M_out = np.array(range(8))

for calper in tqdm(P):
    ycal = y[calper]

    OptSnow = pd.DataFrame(Parallel(
            n_jobs = -2, # -1 = all CPUs, -2 = all but one CPU
            backend = 'loky',
            # verbose = 2
            )(delayed(map_snowaltdist)(pdSeries_ycal=ycal,t=tup) for tup in enumerate(allgrouped))
    )
    OptSnow.columns = ['R2','RMSE','alpha','beta','SSC']

    # Calibrate & Compute AVHRR MB to closest insitu MB or region-wide average MB
    wosm,ssc,alpha,beta,RMSE,R2,_ = get_opt_parameters_and_MB(OptSnow,ycal,allgrouped)
    M_out = np.vstack((M_out,np.array([wosm,ssc,alpha,beta,RMSE,R2,len(calper),int(min(calper)) ])))
    
DF_out = pd.DataFrame(data=M_out,
                      columns = ['wosm','ssc','alpha','beta','rmse','rsq','duration','start'],
                      dtype = float)
DF_out = DF_out.iloc[1:,:]
DF_out.to_csv('../../results/all/Gries_all_calpers_v1.csv',sep = ';')

#%%
import matplotlib.colors as colors

DF_allcalpers = pd.read_csv('../../results/all/Gries_all_calpers.csv',sep = ';')
# DF_allcalpers = pd.read_csv('../../results/all/Gries_all_calpers_v1.csv',sep = ';')
diff = []
mean_MB = []

for ii,calper in enumerate(P):
    ytest = y.loc[y.index.difference(calper)] /1e3
    
    wosm = int(DF_allcalpers.loc[ii,'wosm']/unit)
    ssc = int(DF_allcalpers.loc[ii,'ssc'])
    alpha = DF_allcalpers.loc[ii,'alpha']
    beta = DF_allcalpers.loc[ii,'beta']
    
    # get mean seasonal snow altitude for wosm and rsso
    Z = allgrouped[wosm-2].iloc[ssc,1:] 
    Z.index = years
    X = Z[ytest.index]

    # Out of sample validation
    #=================
    ypred = alpha * X + beta 
    ypred = ypred / 1e3
    mean_MB.append(np.mean(alpha * Z + beta) / 1e3)
    # rmse_out.append(rmse(ytest, ypred))
    diff.append((ypred - ytest).mean())  

DF_allcalpers['diff'] = diff
DF_allcalpers.rmse = DF_allcalpers.rmse.div(1e3)

DF_insample = DF_allcalpers.pivot(index='duration',columns='start',values='rmse')
DF_outofsample = DF_allcalpers.pivot(index='duration',columns='start',values='diff')
DF_outofsample.index = 38 - DF_outofsample.index
# DF_outofsample = pd.DataFrame(data=np.flipud(np.fliplr(DF_outofsample)),index = range(10,39),columns = years[-10::-1])


fig,axs = plt.subplots(1,2,figsize=(9,5),dpi=200)
fig.subplots_adjust(wspace = 0.01)

M=axs[0].matshow(DF_insample,
                 cmap = 'inferno',
                 vmin = 0,vmax = .4
                 )
cb = plt.colorbar(M,orientation = 'horizontal',ax = axs[0],pad = 0.12,label='m w.e. a$^{-1}$')
cb.set_ticks(np.linspace(0,0.4,5))
cb.set_ticklabels([0,.1,.2,.3,.4])

axs[0].set_title('$RMSE_{cal}$',pad = 20, y = -.2)

axs[0].set_xticks(np.arange(0,29,5))
axs[0].set_xticklabels(DF_insample.columns[::5])
axs[0].tick_params(axis="x", bottom=False, top=True)
axs[0].set_ylabel('Duration of calibration period (yr)')
axs[0].set_yticks(np.arange(0,29,5))
axs[0].set_yticklabels(DF_insample.index[::5])

axs[0].annotate('%.3f m w.e. a$^{-1}$' % (DF_insample.loc[38,1982]), xy=(0,28), xytext=(10,25),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )

N=axs[1].matshow(DF_outofsample,
                 cmap = 'bwr_r',
                  vmin=-1.5,vmax=1.5
                 )
plt.colorbar(N,orientation = 'horizontal',ax = axs[1],extend = 'both',pad = 0.12,label='m w.e. a$^{-1}$')

axs[1].set_title('Mean MB Difference',pad = 20, y = -.2)
axs[1].set_xticks(np.arange(0,29,5))
axs[1].set_xticklabels(DF_outofsample.columns[::5])
axs[1].tick_params(axis="x", bottom=False, top=True)

axs[1].set_ylabel('Duration of remaining period (yr)')

axs[1].set_yticks(np.arange(0,29,5))
axs[1].set_yticklabels(DF_outofsample.index[::5])
fig.tight_layout()
# plt.savefig('../../figures/Validation/annual/Fig_6.png', bbox_inches='tight' )

#%% Plot mean MB and RMSE for calpers from 1982
_,ax = plt.subplots(figsize=(8,4),sharex = 'all',dpi = 200)
lns1 = ax.plot(mean_MB[:28])

axx = ax.twinx()
lns2 = axx.plot(DF_allcalpers.rmse.iloc[:28],c = 'r')
lns3= axx.plot(DF_allcalpers.rmse_out.iloc[:28],c = 'g')

ax.axhline(y.mean(),linestyle = '--',c='k')


ax.set_xticks(range(28))
ax.set_xticklabels([len(per) for per in P[:28]])
ax.set_xlabel('Duration of calibration period (yr)')

ax.set_ylabel('mm w.e. a$^{-1}$')

lns = lns1+lns2+lns3
# labs = [l.get_label() for l in lns]
ax.legend(lns, ['mean MB pred','RMSE_is','RMSE_os'], loc=2)
ax.set_title('Griesgletscher results for calpers starting in 1982 ')

# plt.savefig('./figures/Validation/annual/Gries_1982_calpers.pdf' )
