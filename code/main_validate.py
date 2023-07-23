 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 13:18:39 2021

@author: nrietze
"""
import pandas as pd
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm  


from numpy.linalg import inv
from numpy.random import normal
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split,LeavePOut, LeaveOneOut, KFold, cross_val_score,cross_validate
from sklearn.utils import resample
from sklearn.preprocessing import PolynomialFeatures

# import os
# os.chdir('C:/Users/Nils Rietze/Documents/0_Uni Bern/MSc/MSc Thesis/code')

DF_model_params = pd.read_csv('../../results/updated/%s_model_parameters_allglaciers_Alps_%s.csv' % (season,method),
                              sep=';', lineterminator='\n',header=0,index_col = 1) 
DF_model_params.index

from functions_model import *

#%% Perform Cross Validation
for insitu_id in MB_obs.index:
    if not np.isin(insitu_id,DF_model_params.index) or MB_obs.N_OBS[insitu_id] < 10 :
        print('%s has too little MB data or is too small'% MB_obs.NAME[insitu_id])
        continue
    
    print("Performing Cross Validation on %s..." % MB_obs.NAME[insitu_id] )
    
    wosm = int(DF_model_params.loc[insitu_id,'WOSM*']/unit)
    rsso = DF_model_params.loc[insitu_id,'RSSO*']
    # rmse_insitu = float(DF_model_params.loc[insitu_id,'RMSE*'].replace(',','.'))
    rmse_insitu = DF_model_params.loc[insitu_id,'RMSE*']
    
        
    f = open(pickldir+'/allGrDF_%s.pckl'%(glogem_id), 'rb')
    allgrouped = pickle.load(f)
    f.close();del f
    
    # get mean seasonal snow altitude for wosm and rsso
    Z = allgrouped[wosm-2].iloc[rsso,1:] 
    Z.index = allyears
    
    del allgrouped
    
    y = MB_obs.loc[insitu_id,years].astype(float)
    y = y[y.notna()]
    X = Z[y.index].values.reshape((-1, 1))

    # Cross Validation
    #=================
    
    # Leave One Out CV
    # ----------------
    loo = LeaveOneOut()
    
    y_pred_loo, alpha_loo,std_alpha_loo, beta_loo,std_beta_loo,rsq_loo = cross_val(X, y, loo)
        
    rmse_loo = rmse(y, y_pred_loo); #print("RMSE of LOO_CV is {0:.2f} mm w.e. a-1".format(rmse_loo))
    
    # Reference model
    y_ref = []
    for train, test in loo.split(y):
        y_ref.append(y[train].mean()) # Average of all MB EXCEPT in year k
    y_ref = pd.Series(data=y_ref,index=y.index)

    rmse_loo_ref = rmse(y_ref, y_pred_loo); #print("RMSE of LOO_CV_REF is {0:.2f} mm w.e. a-1".format(rmse_loo_ref))
     
    ss_loo = skill_score(rmse_loo, rmse_loo_ref); print("SS of LOO_CV is {0:.2f}".format(ss_loo))
    
    # Leave Three Out CV
    # ------------------
    lpo = LeavePOut(3)
    
    y_pred_lpo, alpha_lpo,std_alpha_lpo, beta_lpo,std_beta_lpo,rsq_lpo = cross_val(X, y, lpo)
        
    rmse_lpo = rmse(y, y_pred_lpo); #print("RMSE of LPO_CV is {0:.2f} mm w.e. a-1".format(rmse_lpo))
    
    # Reference model
    rmse_lpo_ref = rmse(y_ref, y_pred_lpo); #print("RMSE of LPO_CV_REF is {0:.2f} mm w.e. a-1".format(rmse_lpo_ref))
    
    ss_lpo = skill_score(rmse_lpo, rmse_lpo_ref); print("SS of LPO_CV is {0:.2f}".format(ss_lpo))
    
    row_contents = [insitu_id,wosm,rsso,alpha_lpo,
                    std_alpha_lpo,beta_lpo,std_beta_lpo,
                    rmse_insitu,rmse_lpo,ss_lpo]
    append_list_as_row('../../results/updated/%s_crossvalidated_insitu_Alps.csv' % season,
                       row_contents)

    print("done.")

#%% Validate on glaciers with < 10 years of observations
DF_lt10yrs = pd.DataFrame(columns = ['RMSE','SS'],dtype = float)

for insitu_id in MB_obs.index:
    if not np.isin(insitu_id,DF_model_params.index) or insitu_id == 1834 or MB_obs.N_OBS[insitu_id] < 10 :
        print('Validating %s'% MB_obs.NAME[insitu_id])
        
        y = MB_obs.loc[insitu_id,years].astype(float)
        y = y[y.notna()]
        
        calper = y.index
        
        glogem_id = MB_obs.GLOGEM_ID[insitu_id]
        
        f = open(pickldir+'/allGrDF_%s.pckl'%(glogem_id), 'rb')
        allgrouped = pickle.load(f)
        f.close();del f

        rmse_insitu = DF_model_params.loc[insitu_id,'RMSE*']
        
        wosm = int(DF_model_params.loc[insitu_id,'WOSM*']/unit)
        rsso = DF_model_params.loc[insitu_id,'RSSO*']
        rmse_insitu = DF_model_params.loc[insitu_id,'RMSE*']
        
        # get mean seasonal snow altitude for wosm and rsso
        Z = allgrouped[wosm-2].iloc[rsso,1:] 
        Z.index=years
        
        del allgrouped
        
        X = Z[y.index].values.reshape((-1, 1))
        
        # Fit X_train to y_train (i.e. all years but test year)
        model = LinearRegression().fit(X,y) 
        
        # Predict MB in test year 
        ypred = model.predict(X) 
        
        # Get R2 of model
        R2 = model.score(X,y)  
        
        n = len(calper); p = 1
        R2_adj = 1-(1-R2)*(n-1)/(n-p-1)
        

        # RMSE btw modeled MB from the calibrated model and mean observed MB over all years
        DF_lt10yrs.loc[insitu_id,'RMSE'] = rmse(MB.loc[insitu_id,calper] , y)
        DF_lt10yrs.loc[insitu_id,'SS'] = skill_score(DF_lt10yrs.loc[insitu_id,'RMSE'], rmse(MB.loc[insitu_id,calper] , y.mean()))
    continue


#%% Sensitivity analysis
DF_CV = pd.read_csv('/backup/nrietze/MA/results/%s_crossvalidated_insitu_Alps.csv'%season,sep = ";",index_col='GlogemId')
DF_A = pd.concat([ DF_CV , MB_obs.loc[DF_CV.index,['N_OBS', 'OBS_DATE_MIN', 'OBS_DATE_MAX']] , DF_model_params.loc[DF_CV.index,['alpha*', 'beta*']]],axis=1)

for insitu_id in MB_obs.index:
    if not np.isin(insitu_id,DF_model_params.index) or MB_obs.N_OBS[insitu_id] < 10 :
        print('%s has too little MB data or is too small'% MB_obs.NAME[insitu_id])
        continue
    print("Copmuting skill score on %s..." % MB_obs.NAME[insitu_id] )
    
    rmse_insitu = DF_A.loc[insitu_id,'RMSE*']
    
    y = MB_obs.loc[insitu_id,years].astype(float)
    y = y[y.notna()]
    
    DF_A.loc[insitu_id,'std_obs'] = y.std()
    DF_A.loc[insitu_id,'std_cal'] = MB.loc[insitu_id,years].std()
    
    calper = y.index
    # RMSE btw modeled MB from the calibrated model and mean observed MB over all years
    rmse_cal_ref = rmse(MB.loc[insitu_id,calper] , y.mean())
     
    DF_A.loc[insitu_id,'ss_cal'] = skill_score(rmse_insitu , rmse_cal_ref)
    

#%% Plot Skill score vs. length of calibration period (i.e. # of observations)
fig, ax = plt.subplots(figsize = (10,5),dpi=200)

ax.scatter(DF_A.N_OBS,DF_A.ss_lpo, label = 'Cross validation' )
ax.scatter(DF_A.N_OBS,DF_A.ss_cal, label = 'Calibration')
ax.axhline(0,linestyle = '--',color = 'k')

ax.set_xlabel('Length of calibration/observation period (years)')
ax.set_ylabel('Skill score')
ax.legend()

plt.savefig('/cloud/filr/nrietze/MSc Thesis/figures/Validation/%s/SS_vs_NOBS.pdf'%season)

#%% Plot RMSE* vs. length of calibration period (i.e. # of observations)
fig, ax = plt.subplots(figsize = (10,5),dpi=200)

ax.scatter(DF_A.N_OBS,DF_A.rmse_lpo, label = 'Cross validation' )
ax.scatter(DF_A.N_OBS,DF_A['RMSE*'], label = 'Calibration')

ax.set_xlabel('Length of calibration/observation period (years)')
ax.set_ylabel('RMSE (mm w.e. a$^{-1}$)')
ax.legend()

plt.savefig('/cloud/filr/nrietze/MSc Thesis/figures/Validation/%s/RMSE_vs_NOBS.pdf'%season)
    
#%% Plot Skill score vs. RMSE
fig, ax = plt.subplots(figsize = (10,5),dpi=200)

ax.scatter(DF_A.rmse_lpo,DF_A.ss_lpo, label = 'Cross validation' )
ax.scatter(DF_A['RMSE*'],DF_A.ss_cal, label = 'Calibration')
ax.axhline(0,linestyle = '--',color = 'k')

ax.set_ylabel('Skill score')
ax.set_xlabel('RMSE (mm w.e. a$^{-1}$)')
ax.legend()

plt.savefig('/cloud/filr/nrietze/MSc Thesis/figures/Validation/%s/SS_vs_RMSE.pdf'%season)

#%% Plot alpha vs. RMSE
fig, ax = plt.subplots(figsize = (10,5),dpi=200)

ax.scatter(DF_A.rmse_lpo,DF_A.alpha_lpo, label = 'Cross validation' )
ax.scatter(DF_A['RMSE*'],DF_A['alpha*'], label = 'Calibration')
ax.axhline(0,linestyle = '--',color = 'k')

ax.set_ylabel('Alpha (mm w.e. a$^{-1}$ m$^{-1}$)')
ax.set_xlabel('RMSE (mm w.e. a$^{-1}$)')
ax.legend()

plt.savefig('/cloud/filr/nrietze/MSc Thesis/figures/Validation/%s/alpha_vs_RMSE.pdf'%season)

#%% Plot Skill score vs. Difference in Std and color by N_OBS
fig, ax = plt.subplots(figsize = (10,5),dpi=200)

P = ax.scatter((DF_A.std_obs-DF_A.std_cal),DF_A['ss_cal'],
               c=DF_A['N_OBS']>30 , cmap = 'bwr', label = DF_A['N_OBS']>30)
ax.axhline(0,linestyle = '--',color = 'k')

ax.set_xlabel('$\Delta$ $\sigma$ (mm w.e. a$^{-1}$)')
ax.set_title('Difference in mass balance standard deviation: Observation vs. AVHRR')
ax.set_ylabel('Skill score')
ax.legend(P.legend_elements()[0],['<30 years','>30 years'])

plt.savefig('/cloud/filr/nrietze/MSc Thesis/figures/Validation/%s/DiffStd_vs_SScal_NOBS_Vertical.pdf'%season)

#%% Plot Skill score vs. Difference in Std and color by Reference Glacier
fig, ax = plt.subplots(figsize = (10,5),dpi=200)

P = ax.scatter((DF_A.std_obs-DF_A.std_cal),DF_A['ss_cal'],
               c=DF_A.index , cmap = cm.get_cmap('jet',len(DF_A['OBS_DATE_MIN'].unique())), label = DF_A['OBS_DATE_MIN'])
ax.axhline(0,linestyle = '--',color = 'k')

ax.set_xlabel('$\Delta$ $\sigma$ (mm w.e. a$^{-1}$)')
ax.set_title('Difference in mass balance standard deviation: Observation vs. AVHRR')
ax.set_ylabel('Skill score')
# ax.legend(P.legend_elements()[0],['<30 years','>30 years'])
plt.colorbar(P,label= 'GloGEM ID of Reference Glacier')

# plt.savefig('/cloud/filr/nrietze/MSc Thesis/figures/Validation/%s/DiffStd_vs_SScal_referencgl_Vertical.pdf'%season)

#%% MORTERATSCH - Use different reference glaciers
morteratsch_id = 1946
camposett_id = 1930
silvretta_id = 804

MB_Morteratsch = pd.Series(index = np.arange(2002,2020).astype(str), data = [-780,-1492,-602,-1168,-1564,-1418,-1072,-702,-759,-924,-1152,-622,73,-1118,-743,-1613,-1337,-1536])

gl_ids = [camposett_id,silvretta_id,morteratsch_id]

f = open(pickldir+'/allGrDF_%s.pckl'%(morteratsch_id), 'rb')
allgrouped = pickle.load(f)
f.close();del f
    
DF_Morteratsch = pd.DataFrame(np.nan,index = gl_ids , columns = ['Rsq','RMSE','alpha','beta']+list(years) )

for gl_id in gl_ids:
    y = MB_obs.loc[gl_id,years].astype(float)
    y = y[y.notna()]
    
    wosm = int(DF_model_params.loc[gl_id,'WOSM*']/unit)
    rsso = DF_model_params.loc[gl_id,'RSSO*']
    rmse_insitu = DF_model_params.loc[gl_id,'RMSE*']
    
    # get mean seasonal snow altitude for wosm and rsso
    Z = allgrouped[wosm-2].iloc[rsso,1:] 
    Z.index=years
    
    X = Z[y.index].values.reshape((-1, 1))
    
    # Fit X_train to y_train (i.e. all years but test year)
    model = LinearRegression().fit(X,y) 
    
    # Predict MB in test year 
    DF_Morteratsch.loc[gl_id,years] = model.predict(Z.values.reshape((-1, 1))) 
    
    #Get coefficient of the fit
    DF_Morteratsch.loc[gl_id,"alpha"] = model.coef_ 
    
    #Get intercept of the fit
    DF_Morteratsch.loc[gl_id,"beta"] = model.intercept_ 
    
    # Get R2 of model
    DF_Morteratsch.loc[gl_id,"Rsq"] = model.score(X,y) 
    
    DF_Morteratsch.loc[gl_id,"RMSE"] = rmse(y,DF_Morteratsch.loc[gl_id,years])

# Plot MB from referencing for each of the 3 glaciers (morteratsch, silvretta, campo sett) vs. GLAMOS
main_plot.PlotOBSvsAVHRR(DF_Morteratsch.loc[camposett_id,years] , DF_Morteratsch.loc[silvretta_id,years] ,
                         wosm*1e3 , y.index , season , "Morteratsch" , outdir = '/cloud/filr/nrietze/MSc Thesis/figures/Validation/case_Morteratsch' , 
                         MB_Series3=DF_Morteratsch.loc[morteratsch_id,years],MB_Series4=MB_Morteratsch ,
                         Series1_Name = 'Campo Sett' , Series2_Name = 'Silvretta' , Series3_Name = 'Morteratsch',Series4_Name = 'GLAMOS')

insitu_years = MB_Morteratsch.index
main_plot.PlotOBSvsAVHRR_cumMB(DF_Morteratsch.loc[camposett_id,insitu_years] , DF_Morteratsch.loc[silvretta_id,insitu_years] ,
                         wosm*1e3 , MB_Morteratsch.index , season , "Morteratsch" , outdir = '/cloud/filr/nrietze/MSc Thesis/figures/Validation/case_Morteratsch' , 
                         MB_Series3=DF_Morteratsch.loc[morteratsch_id,insitu_years],MB_Series4=MB_Morteratsch ,
                         Series1_Name = 'Campo Sett' , Series2_Name = 'Silvretta' , Series3_Name = 'Morteratsch',Series4_Name = 'GLAMOS')

# # Plot MB from referencing for of the 2 glaciers ( silvretta, campo sett) vs. GLAMOS
# main_plot.PlotOBSvsAVHRR(DF_Morteratsch.loc[camposett_id,years] , DF_Morteratsch.loc[silvretta_id,years] ,
#                          wosm*1e3 , y.index , season , "Morteratsch" , outdir = '/cloud/filr/nrietze/MSc Thesis/figures/Validation/case_Morteratsch' , 
#                          MB_Series3=MB_Morteratsch ,Series1_Name = 'Campo Sett' , Series2_Name = 'Silvretta' , Series3_Name = 'GLAMOS')

# main_plot.PlotOBSvsAVHRR_cumMB(DF_Morteratsch.loc[camposett_id,MB_Morteratsch.index] , DF_Morteratsch.loc[silvretta_id,MB_Morteratsch.index] ,
#                          wosm*1e3 , y.index , season , "Morteratsch" , outdir = '/cloud/filr/nrietze/MSc Thesis/figures/Validation/case_Morteratsch' , 
#                          MB_Series3=MB_Morteratsch ,Series1_Name = 'Campo Sett' , Series2_Name = 'Silvretta' , Series3_Name = 'GLAMOS')
#%% Validate with Fischer et al. 2015
DF_MB_AVHRR_ann = pd.read_csv('/backup/nrietze/MA/results/%s_MB_allglaciers_Alps_VNN_4km.csv'%season, sep=';', lineterminator='\n',header=0,index_col = 0)

DF_Fischer = pd.read_csv('/backup/nrietze/MA/data/MB/validation_data/Fischer et al 2015/database_Fischer_et_al_2015_alle.csv',sep=';', lineterminator='\n',header=0,index_col = 0)  
fischer_years = np.arange(1982,2011).astype(str)

Glacier_attrs = pd.read_csv('/backup/nrietze/MA/data/RGI/EUROPE/CentralEurope_ge02km2_List.csv', sep=';', lineterminator='\n',header=0).set_index('GlogemId')
gl_ids_ch = Glacier_attrs.index[Glacier_attrs.ID_SGI2010.notna()]
ezgs = Glacier_attrs.Unique_ID[gl_ids_ch]

# convert Fischer to mm w.e. a-1 
sgi_ids = Glacier_attrs.ID_SGI2010[Glacier_attrs.ID_SGI2010.notna()]
MB_Fischer = DF_Fischer.loc[sgi_ids,'temporally_homogenized_geodetic_mass balance_1980-2010_ m w.e. yr-1'] * 1e3 
MB_Fischer.index = gl_ids_ch

Areas_Fischer = DF_Fischer.loc[sgi_ids,'average_area_km2_between_SGI1973_and_SGI2010']
Areas_Fischer.index = gl_ids_ch

Areas_Glogem = Glacier_attrs.Area[gl_ids_ch]

# A_diff = Areas_Fischer.sub(Areas_Glogem)
# A_diff[(A_diff < -5) | (A_diff > 5)]

# Area-weighted catchment mean
MB_Fischer_aw = DF_Fischer.loc[sgi_ids,'area-weighted_homogenized_geodetic_mass balance_1980-2010_ m w.e. yr-1'] *1e3
MB_Fischer_aw.index = gl_ids_ch
MB_Fischer_aw = MB_Fischer_aw.groupby(ezgs).sum()/Areas_Fischer.groupby(ezgs).sum()

MB_AVHRR_CH = DF_MB_AVHRR_ann.loc[gl_ids_ch,fischer_years]
MB_AVHRR_CH_allyears = DF_MB_AVHRR_ann.loc[gl_ids_ch,years]
mean_MB_AVHRR_CH = MB_AVHRR_CH.mean(axis=1);mean_MB_AVHRR_CH.name = 'mean_MB_AVHRR_82_10'

# Area-weighted catchment mean
mean_MB_AVHRR_CH_aw = mean_MB_AVHRR_CH * Areas_Glogem
mean_MB_AVHRR_CH_aw = mean_MB_AVHRR_CH_aw.groupby(ezgs).sum()/Areas_Glogem.groupby(ezgs).sum()

# Merge Fischer and AVHRR MB and 
DF_joined = pd.concat([MB_Fischer,mean_MB_AVHRR_CH, (mean_MB_AVHRR_CH-MB_Fischer),Areas_Fischer,Areas_Glogem,ezgs],axis=1)
DF_joined.columns =  ['MB_Fischer','MB_AVHRR','Diff','Area_Fischer','Area_Glogem','EZG']

print('Area-weighed mass change rate from AVHRR = %.2f mm w.e. a$^{-1}$' % mean_MB_AVHRR_CH_aw.mean())
print('Area-weighed mass change rate from Fischer et al. (2015) = %.2f mm w.e. a$^{-1}$' % MB_Fischer_aw.mean())

#%% Plot Difference (Error) AVHRR-FISCHER
plt.scatter(DF_joined.index,DF_joined.Diff,c = Glacier_attrs.loc[Glacier_attrs.ID_SGI2010.notna(),'Unique_ID'],cmap='jet')
# plt.ylim([-500,500])
plt.ylabel('MB Difference AVHRR - Fischer et al. (2015)')
plt.xlabel('Glacier Nr.')
plt.show()

#%% Analyse Mass Balance Errors btw Fischer and AVHRR
mbe = mean_MB_AVHRR_CH-MB_Fischer
mbe_abs = np.abs(mean_MB_AVHRR_CH-MB_Fischer)
std_mbe = mbe.std()

# Transformed MBE
z_mbe = (mbe - mbe.mean()) / std_mbe

asp = Glacier_attrs.loc[gl_ids_ch,'Aspect']
zmed = Glacier_attrs.loc[gl_ids_ch,'Zmed']


# Look at which reference glacier produces most over- or underestimations
A = pd.concat([Glacier_attrs.loc[gl_ids_ch,'HubName_10yrs'],mbe],axis=1)
A.HubName_10yrs = [int(s[-4:]) for s in Glacier_attrs.loc[gl_ids_ch,'HubName_10yrs']]

# Number of under-/overestimations per reference glacier (i.e. MBE >= 700 mm or <= 700 mm)
# N_ov_un_est = A.groupby('HubName_10yrs')[0].apply(lambda x: ((x>=700) | (x<= -700)).sum())
N_ov_un_est = A.groupby('HubName_10yrs')[0].apply(lambda x: ((x>=std_mbe) | (x<= -std_mbe)).sum())

# Compute percentage of glaciers that are underestimated
N_ov_un_est_rel = N_ov_un_est / A.groupby('HubName_10yrs').count()[0] * 100

# Number of overestimations
N_ov_est = A.groupby('HubName_10yrs')[0].apply(lambda x: ((x>=std_mbe)).sum())

# Compute percentage of glaciers that are underestimated
N_ov_est_rel = N_ov_est / A.groupby('HubName_10yrs').count()[0] * 100

# Number of underestimations
N_un_est = A.groupby('HubName_10yrs')[0].apply(lambda x: ((x<= -std_mbe)).sum())

# Compute percentage of glaciers that are underestimated
N_un_est_rel = N_un_est / A.groupby('HubName_10yrs').count()[0] * 100

# Analyze Aspect vs. MBE
B = pd.concat([Glacier_attrs.loc[gl_ids_ch,'HubName_10yrs'],mbe,asp],axis=1)
B.HubName_10yrs = [int(s[-4:]) for s in Glacier_attrs.loc[gl_ids_ch,'HubName_10yrs']]

# Get Aspects of glaciers where MBE >= 1 sigma
asp_ov = B.loc[B[0]>=std_mbe,'Aspect']

# Get Aspects of glaciers where MBE <= 1 sigma
asp_un = B.loc[B[0]<=-std_mbe,'Aspect']


#%% Polar Hisogram of over- & underestimations

ang = 220

fig, ax = plt.subplots(1,2,subplot_kw=dict(projection='polar'),figsize =(10,6))
# Visualise by radius of bins (wenn density = False ist nicht gut ersichtlich, dass bei 315° halb so viele sind, man nimmt an dass es weniger sind)
main_plot.circular_hist(ax[0], np.radians(asp_un), density=False)
ax[0].set_title('N = {}'.format(len(asp_un)),loc = 'right',pad = 20,fontsize = 12,weight="bold")
ax[0].set_rgrids(radii = np.arange(ax[0].get_rmin()+2,ax[0].get_rmax(),2),angle = ang)

ax[0].text(np.radians(ang),ax[0].get_rmax()+1,'Count',ha='left',va='top', fontsize=10)

main_plot.circular_hist(ax[1], np.radians(asp_ov), density=False)
ax[1].set_title('N = {}'.format(len(asp_ov)),loc = 'right',pad = 20,fontsize = 12,weight="bold")
ax[1].set_rgrids(radii = np.arange(ax[1].get_rmin()+2,ax[1].get_rmax(),2),angle = ang)

ax[1].text(np.radians(ang),ax[1].get_rmax()+1,'Count',ha='left',va='top', fontsize=10)
plt.suptitle("Histogram of underestimations (<= -1$\sigma$) and overestimations (>= 1$\sigma$)")
# plt.savefig('/cloud/filr/nrietze/MSc Thesis/figures/Validation/%s/Fischer/Hist_misest_vs_aspect_polar_Fischer_4km.pdf'%season)


# fig, ax = plt.subplots(subplot_kw=dict(projection='polar'),figsize =(6,6))
# # Visualise by area of bins (hier wird ersichtlich, dass z.B. bei 315° halb so viel sind wie bei 0°)
# circular_hist(ax, np.radians(asp_un))

#%% Polar Plot absolute MBE vs. Aspect
fig = plt.figure(figsize =(7,6)) 
ax = plt.subplot(111, projection='polar') 
  
# Plot  distance radially and color by MBE (negative = red, positive = blue)
# P = ax.scatter(np.radians(asp),Glacier_attrs.loc[gl_ids_ch,'HubDist_10yrs'],c =  mbe,cmap = 'bwr_r', vmax = 1000, vmin = -1000); plt.colorbar( P )
# P = ax.scatter(np.radians(asp),mbe_abs,c = [int(s[-4:]) for s in Glacier_attrs.loc[gl_ids_ch,'HubName_10yrs']],cmap = ''); plt.colorbar( P )

#Plot absoulute MBE radially
P = ax.scatter(np.radians(asp),mbe_abs,marker = ".")
circle = plt.Circle((0, 0), std_mbe, transform = ax.transData._b, fill=False, edgecolor='red', linewidth=2, zorder=10)
plt.gca().add_artist(circle)

ang = 220
ax.set_theta_zero_location('N')
ax.set_thetagrids(angles = [0,90,180,270], labels=['N','W','S','E'], fontsize=12, weight="bold", color="black")
ax.set_rgrids(radii = np.arange(500,2500,500),weight="bold",angle = ang)
ax.yaxis.set_zorder(10) # Moves r-axis labels to front

ax.legend((None,circle),('','1$\sigma$ = %.2f mm w.e. a$^{-1}$'%std_mbe) , loc = 'best', bbox_to_anchor=(1.25, 1.05))
ax.text(np.radians(ang),ax.get_rmax()+1e2,'|MBE| (mm w.e. a$^{-1}$)',ha='left',va='bottom', fontsize=10, weight="bold", color="black")
ax.set_title("Absolute MBE AVHRR - Fischer et al. (2015)",pad = 20)
# plt.savefig('/cloud/filr/nrietze/MSc Thesis/figures/Validation/%s/Fischer/MBE_vs_aspect_polar_Fischer_4km.pdf'%season)


# from windrose import WindroseAxes

# fig=plt.figure()
# rect=[0.5,0.5,0.4,0.4] 
# wa=WindroseAxes(fig, rect)
# fig.add_axes(wa)
# wa.bar(asp, mbe, normed=True, opening=0.8, edgecolor='white')

#%% Plot contribution to MBE and overall Mass loss of glaciers for a given range of surface areas 
# area-weighted MB sum for AVHRR and swiss glaciers in 1982-2010
cumMB_AVHRR_CH_aw = MB_AVHRR_CH_allyears.mul(Areas_Glogem,axis=0).sum(axis=1)

# Surface Area range
A_iter = np.arange(0,20,.5)

# cumulative MBE sum for glaciers  (and convert to m w.e. for readability)
mbe_column = 0
cumMBE = [mbe[Areas_Glogem < area].sum() / 1e3 for area in A_iter]

A_90pct_cumMBE = A_iter[np.where(cumMBE<np.quantile(cumMBE,0.9))[0].min()]

cumPct_MBshare = [(cumMB_AVHRR_CH_aw[Areas_Glogem < area].sum() / cumMB_AVHRR_CH_aw.sum())*100 for area in A_iter]
cumPct_MBshare_Fischer = [((MB_Fischer[Areas_Fischer < area] * Areas_Fischer).sum() / (MB_Fischer * Areas_Fischer).sum())*100 for area in A_iter]

meanMBE_perArea = [mbe[Areas_Glogem < area].mean() / 1e3 for area in A_iter]
count = [mbe[Areas_Glogem < area].count()  for area in A_iter]
# Compute MBE contribution independent of # of glaciers 
z = np.array(cumMBE)
zc = np.array(count)

# computes how much every collection of glaciers < certain Area contributes to change in cumulative MBE (e.g. at position 0 contribution is 0, 
# because no glacier is < 0km2, at position 1 contribution is -3.04 m w.e. from each glacier because we have N glaciers < 0.5 km2)
z[1:] -= z[:-1]
zc[1:] -= zc[:-1]

cumArea = [Areas_Glogem[Areas_Glogem < area].sum() for area in A_iter]
D = pd.DataFrame(data=[count,zc, cumMBE,z,cumArea],index = ['N','dN','cumMBE','change_per_Area','cumArea'],columns=A_iter).T

# Computes contribution of all new glaciers added per Area-bin to the change in cumulative MBE 
# (i.e. glaciers<0.5 contribute -0.14 mwe per glacier added, for all new glaciers <1km2 added, they contribute -.11)
contrMBE_newbin = D.change_per_Area.div(D.dN)

#--------
fig,(ax1,axh,axb) = plt.subplots(nrows = 3,ncols = 1,figsize = (10,7),dpi=150,
                                 sharex='all', gridspec_kw={'height_ratios': [3, 1,1.5]})
fig.subplots_adjust(hspace=0)
# Plot cumulative MBE on 1st y-axis
ax1.plot(A_iter,cumMBE)

# Vertical line at 90% quantile 
ax1.axvline(A_90pct_cumMBE,linestyle='--',c = 'k')

ax1.set_ylabel('Cumulative MBE (m w.e. a$^{-1}$)', color = '#1f77b4')
ax1.tick_params('y', colors = '#1f77b4')
ax1.set_xlim([A_iter.min(),A_iter.max()+.5])

# Plot contribution to overall mass loss on 2nd y-axis
ax2 = ax1.twinx()
ax2.plot(A_iter,cumPct_MBshare,'r--')
ax2.plot(A_iter,cumPct_MBshare_Fischer,'g--')

ax2.set_ylabel('Contribution to overall mass loss (%)', color = 'r')
ax2.tick_params('y', colors = 'r')
ax2.legend(('AVHRR','Fischer et al. (2015)'),loc = 'center right')

ax1.set_title('Accumulated MBE AVHRR-Fischer for glaciers smaller than a certain area', pad = 20)

#  Plot Histogram of glacier areas
axh.hist(Areas_Glogem [ Areas_Glogem <= A_iter.max()],bins=40,color = 'gray')
axh.axvline(A_90pct_cumMBE,linestyle='--',c = 'k')

axh.set_xlim([A_iter.min(),A_iter.max()+.5])
axh.set_ylabel('Count')

# Plot bars of decomposed contribution to MBE
color = mbe.loc[Areas_Glogem[Areas_Glogem<20].sort_values().index]<0

# Plot every mbe
axb.bar(Areas_Glogem [Areas_Glogem<20].sort_values(),mbe.loc[Areas_Glogem[Areas_Glogem<20].sort_values().index],
        color = color.map({True: '#fb9a99', False: '#a6cee3'}) , width = .05)
axb.axvline(A_90pct_cumMBE,linestyle='--',c = 'k')

# axb.bar(A_iter,contrMBE_newbin,
#         color = color.map({True: 'r', False: 'g'}) , width = 0.5)

axb.set_xlim([A_iter.min(),A_iter.max()+.5])
axb.set_xlabel('Glacier Area (km$^2$)')
axb.set_ylabel('MBE (m w.e. a$^{-1}$)')

# plt.savefig('/cloud/filr/nrietze/MSc Thesis/figures/Validation/%s/Fischer/cumMBE_cumPct_Hist_MBE_for_Areas.png'%season)

#%% Boxplot of MBE for every reference glacier
DF = pd.concat([Areas,mbe],axis=1)
DF['hubID'] = [int(s[-4:]) for s in Glacier_attrs.loc[gl_ids_ch,'HubName_10yrs']]
maxs = DF.groupby(by='hubID').max()[0]
nglac = DF.groupby(by='hubID').count()[0]
insitus = np.unique([int(s[-4:]) for s in Glacier_attrs.loc[gl_ids_ch,'HubName_10yrs']])
insitunames = Glacier_attrs.Name[insitus]
pos = range(len(nglac))

fig,ax = plt.subplots(figsize = (10,7))
BP = DF.boxplot(column=0,by='hubID',ax=ax,
           return_type='both',
           grid = False, sym = '+',color = dict(medians="k"),
           patch_artist = True)
ax.axhline(0,linestyle='-',c='k',lw = .5)
ax.axhline(mbe.mean(),linestyle='--',c='r')

ax.set_xlabel('')
ax.set_ylabel('MBE (mm w.e. a${^-1}$)')
ax.set_title('Mass balance error AVHRR-Fischer for each reference glacier',pad = 20)

ax.text(-1,mbe.mean(),
        '{0:.2f}'.format(mbe.mean()),
        color = 'r')

plt.suptitle('')

ax.set_xticklabels(insitunames,rotation=45, ha="right")

for tick,label in zip(pos,ax.get_xticklabels()):
    ax.text(pos[tick]+1,
            maxs.iloc[tick]+20,
            nglac.iloc[tick],
            horizontalalignment='center',
            size='medium',
            color='k',
            weight='semibold')
# plt.savefig('/cloud/filr/nrietze/MSc Thesis/figures/Validation/%s/Fischer/Boxplot_MBE_referenceglac.pdf'%season)

#%% Plot MBE vs. different altitudes
n =  len(Glacier_attrs.loc[gl_ids_ch,'HubName_10yrs'].unique())
plt.scatter(mbe_abs,Glacier_attrs.loc[gl_ids_ch,'Zmed'],
            c = [int(s[-4:]) for s in Glacier_attrs.loc[gl_ids_ch,'HubName_10yrs']],cmap = cm.get_cmap('jet',n))
plt.title("Absolute MBE Fischer - AVHRR")
plt.xlabel('|MBE| (mm w.e. a${^-1}$)')
plt.ylabel("Median Altitude (m a.s.l.)")
plt.colorbar(label = "Reference ID" )
# plt.savefig('/cloud/filr/nrietze/MSc Thesis/figures/Validation/%s/Fischer/absMBE_vs_zmed_Fischer_4km.pdf'%season)

#%% Plot Distance to nearest reference vs. error
x_centr = Glacier_attrs.x_centr
y_centr = Glacier_attrs.y_centr

x_centr_ref = pd.Series(np.empty_like(x_centr),index = x_centr.index)
y_centr_ref = pd.Series(np.empty_like(x_centr),index = x_centr.index)
for ii,id_ref in enumerate(Glacier_attrs.HubName_10yrs):
    x_centr_ref.iloc[ii] = Glacier_attrs.x_centr[Glacier_attrs.RGIId==id_ref].item()
    y_centr_ref.iloc[ii] = Glacier_attrs.y_centr[Glacier_attrs.RGIId==id_ref].item()

# Cartesian Distance to nearest reference in km
dist2Hub = np.sqrt((x_centr - x_centr_ref)**2 + (y_centr - y_centr_ref)**2) / 1e3
dist2Hub = dist2Hub.loc[Glacier_attrs.ID_SGI2010.notna()].sort_values()

# Glacier_attrs.loc[Glacier_attrs.ID_SGI2010.notna(),'HubDist_10yrs'].sort_values()
plt.scatter(dist2Hub,DF_joined.Diff[dist2Hub.index])
# plt.ylim([-500,500])
plt.xlabel('Distance to closest reference glacier [km]')
plt.ylabel('MB Difference AVHRR - Fischer et al. (2015)')
print("RMSE AVHRR - Fischer et al. (2015): %.2f mm w.e. a-1" % rmse(MB_Fischer,mean_MB_AVHRR_CH))

bplot = DF_joined.boxplot(column='Diff',by='EZG',
                                return_type='both',
                                grid = False, sym = '+',color = dict(medians="k"),
                                patch_artist = True,figsize = (15,10))
bplot.Diff.ax.set_title('')
color_key = Glacier_attrs.groupby('Unique_ID').sum().Monitored.gt(0).astype(int)
colors = np.where(color_key==0,'g','r')
for row_key, (ax,row) in bplot.iteritems():
    # ax.set_xlabel('')
    for i,box in enumerate(row['boxes']):
        box.set_color(colors[i])
        box.set_facecolor(colors[i])
        box.set_alpha(0.7)

# plt.savefig('/cloud/filr/nrietze/MSc Thesis/figures/Validation/Boxplot_Diff_AVHRR_Fischer_4km.pdf')

#%% Compare to newset, unpublished MB 
DF_Hugonnet = pd.read_csv('/backup/nrietze/MA/data/MB/validation_data/Hugonnet/11_mb_glspec.dat', sep='\s+', lineterminator='\n',header=2).set_index('RGI-ID')
rgi_ids = Glacier_attrs.RGIId
common_rgi_ids = np.isin(DF_Hugonnet.index,rgi_ids)

# Convert to mm w.e. a-1
MB_Hugonnet = DF_Hugonnet.B[common_rgi_ids] * 1e3
# Change index to Glogem ID
MB_Hugonnet.index = [int(id[-4:]) for id in MB_Hugonnet.index]

Areas = DF_Hugonnet.Area[common_rgi_ids]
Areas.index = MB_Hugonnet.index

# Years of Hugonnet data set
hugo_years = np.arange(2000,2019).astype(str)

MB_AVHRR = DF_MB_AVHRR_ann[hugo_years].mean(axis=1)
# Merge AVHRR and Hugonnet MB
DF_joined = pd.concat([MB_Hugonnet,MB_AVHRR,(MB_AVHRR-MB_Hugonnet),Areas,DF_MB_AVHRR_ann.EZG_Nr ],axis=1)
DF_joined.columns =  ['MB_Hugonnet','MB_AVHRR','Diff','Area','EZG']

# Area-weighted catchment mean
DF_joined_aw = DF_joined
DF_joined_aw.loc[:,('MB_Hugonnet','MB_AVHRR')] = DF_joined.loc[:,('MB_Hugonnet','MB_AVHRR')].multiply(DF_joined.Area,axis=0)

# Copmute catchment-wise area-weighted MB ...
DF_aw_ezg = DF_joined_aw.loc[:,('MB_Hugonnet','MB_AVHRR')].groupby(DF_joined_aw.EZG).sum().divide( DF_joined_aw.Area.groupby(DF_joined_aw.EZG).sum() , axis=0)

#  ...and Differences
DF_aw_ezg.loc[:,'Diff'] = DF_aw_ezg.diff(axis=1).iloc[:,1]

print('Area-weighed mass change rate from AVHRR = %.2f mm w.e. a$^{-1}$' % DF_aw_ezg.mean()[1])
print('Area-weighed mass change rate from Hugonnet et al. (in press) = %.2f mm w.e. a$^{-1}$' % DF_aw_ezg.mean()[0])

#%% Plot area-weighted and arithmetic mean cathcment MB
fig, ax = plt.subplots(figsize = (10,5),dpi=200)
ax.plot(DF_aw_ezg.Diff,'^')
ax.plot(DF_joined.Diff.groupby(DF_joined.EZG).mean(),'.')

ax.set_xlabel('catchment ID')
ax.set_ylabel('$\Delta$MB modeled - geodetic (mm w.e. a$^{-1}$)')
ax.set_title('Catchment wide differences of %s MB (%s - %s)' % (season , hugo_years[0], hugo_years[-1]))
ax.legend(('area-weighted','arithmetic'),loc='best')

# plt.savefig('/cloud/filr/nrietze/MSc Thesis/figures/Validation/%s/Hugonnet/EZG_wise_mean_MB_Diff.pdf'%season)

#%% Boxplot all catchments
bplot = DF_joined.boxplot(column='Diff',by='EZG',
                                return_type='both',
                                grid = False, sym = '+',color = dict(medians="k"),
                                patch_artist = True,figsize = (15,10))

plt.xticks(ha='right')
bplot.Diff.ax.set_title('')
color_key = Glacier_attrs.groupby('Unique_ID').sum().Monitored.gt(0).astype(int)
colors = np.where(color_key==0,'g','r')
for row_key, (_,row) in bplot.iteritems():
    # ax.set_xlabel('')
    for i,box in enumerate(row['boxes']):
        box.set_color(colors[i])
        box.set_facecolor(colors[i])
        box.set_alpha(0.7)