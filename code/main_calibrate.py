#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model seasonal glacier mass balance using AVHRR-derived seasonal snowline elevations.
"""
from functions_model import *
import pickle
from tqdm import tqdm
from joblib import Parallel, delayed

global ycal

print('Running %s MB model for %s data...'% (method, season))

#%% Loop through all glaciers:
for j,row in Glacier_attrs.iterrows():

    # Get coordinates and ID of that glacier
    x_center = row.x_centr
    y_center = row.y_centr
    rgi_id = row.RGIId
    glogem_id = row.name
    
    print("\n Processing %s ..." % rgi_id)
    
    # Set path to AVHRR snowline altitudes 
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
           
    if method == 'CRG':
        # RGI-ID of nearest reference glacier
        if season =="winter":
            hubname = row.HubName_10yrs_winter
        else:
            hubname = row.HubName_10yrs
                
        # In situ MB of that reference glacier
        y = MB_obs.loc[hubname,years]
    
        # Drop nan in that series
        ycal = y[y.notna()]
        
        # Define Calibration Period for that glacier
        calper = np.array(ycal.index[np.isin(ycal.index,years)]).astype(int)
    else:
        # Average mass balance
        ycal = MB[years].mean()
        
        # Define Calibration Period for that glacier
        calper = years.astype(int)
        
    f = open(pickldir+'/allGrDF_%s.pckl'%(glogem_id), 'rb')
    allgrouped = pickle.load(f)
    f.close();del f

    # Calibrate & Compute AVHRR MB to closest insitu MB or region-wide average MB:
    # ---- use map (slower than joblib Parallel)
    """
    OptSnow = pd.DataFrame(
        map(
            lambda tup: map_snowaltdist(tup,ycal), tqdm(enumerate(allgrouped))
            )
        )
    """
    # ---- use joblib Parallel to parallelize
    OptSnow = pd.DataFrame(Parallel(
            n_jobs = -2, # -1 = all CPUs, -2 = all but one CPU
            backend = 'loky',
            verbose = 2
            )(delayed(map_snowaltdist)(pdSeries_ycal=ycal,t=tup) for tup in tqdm(enumerate(allgrouped)))
    )

    OptSnow.columns = ['R2','RMSE','alpha','beta','SSC']
    # OptSnow = calibration_fct(allgrouped,WOSM_rad,ycal)    
    wosm,ssc,alpha,beta,RMSE,R2,MB = get_opt_parameters_and_MB(OptSnow,ycal,allgrouped, WOSM_rad)
        
    
    #%% Append MB to csv
    row_contents = [glogem_id] + MB.to_list()
    if inventory == "HKH":
        append_list_as_row('../../results/updated/annual_MB_allglaciers_HKH_VNN.csv',row_contents)
    else:
        append_list_as_row('../../results/updated/%s_MB_allglaciers_Alps_V%s.csv' % (season,method),row_contents)
        
    #%% Append Parameters to csv
    row_contents = [glogem_id, rgi_id,
                    wosm,ssc,
                    alpha,beta,RMSE,R2,
                    row.Area, row.Slope, 
                    row.Zmin,row.Zmed, row.Zmax,
                    row.Lmax, x_center, y_center]
    if inventory == "HKH":
        append_list_as_row('../../results/updated/annual_model_parameters_allinsitus_HKH_VNN.csv',row_contents)
    else:
        append_list_as_row('../../results/updated/%s_model_parameters_allglaciers_Alps_%s.csv' % (season,method),row_contents)
        
print('done.')