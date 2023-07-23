#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 7 08:37:24 2020

@author: Nils Rietze, nils.rietze@uzh.ch
"""

import os, os.path
import sys
import xarray as xr

import numpy as np
import time
import pandas as pd
import pickle

def group_PdSeries(PdSeries_PSO,PdSeries_Elev):
    return PdSeries_Elev.groupby(PdSeries_PSO).mean()

def getAltitude(DF,snowprob):
    
    # Version 2:
    T = DF.where(DF.ge(snowprob) & DF.lt(snowprob+1)).reset_index() # Find where snowprob is within range
    Rep = pd.DataFrame([DF.index.values]*DF.shape[1],index=T.columns[1:]).T # Create DF with replica of elevation index with n rows
    Out = T.iloc[:,1:].where(T.iloc[:,1:].isna(),Rep) # Replace values where snowprob is within range with elevation value
    
    # Version1 1:
    # L = [DF[col].loc[DF[col].ge(snowprob) & DF[col].lt(snowprob+1)].index.values.mean() for col in DF.columns]
    # L = [DF[col].loc[DF[col].isin(list(np.arange(snowprob,snowprob+1,.1).round(1)))].index.values.mean() for col in DF.columns]
    # return np.squeeze(L)
    return Out.mean(axis=0).values

def DFobj2float(DF,columns):
    DF_out=DF
    for col in columns:
        DF_out[col] = DF[col].astype(float)
    return DF_out


def netCDF2SnowPerAlt(xrDataArray,DEM):    
    """
    This function computes the mean altitude at which a probability of snow occurrence
    (PSO) of a certain percentage is found, e.g., pixels with 57 % of seasonal snow cover 
    in 1981 are found on 2109 m. 

    Parameters
    ----------
    xrDataArray : xr.DataArray
        The PSO (yearly values) dataset sliced to the extents of the WOSM and specific band.
    DEM : TYPE
        The DEM in the same grid format as the xrDataArray .

    Returns
    -------
    GrDF : pd.DataFrame
        A table containing the mean elevations per 1% PSO for each year.
        THe first column is the PSO percentage, values are in m a.s.l.

    """
    # Radius already converted to length unit (i.e. 200'000 m input instead of 200 km)
    years = xrDataArray.time.dt.year.values
    
    # Convert to DataFrame with MultiIndex Lat and Lon and Years as Columns
    DF = xrDataArray.to_dataframe('PSO').reset_index().pivot_table(values='PSO',
                                                                   index=['y','x'],
                                                                   columns='time')                    
    
    D = DEM.sel(
        x = xrDataArray.x,
        y = xrDataArray.y,
        method='nearest')    
    D = D.to_dataframe(name='elev').reset_index('band',drop=True)
    
    # DF = pd.DataFrame(A).T
    DF.columns = years
    DF = DF.join(D['elev'], on=['y','x'])
    # DF.insert(loc = 0, column = 'elev', value = D['elev'])

    # Group snow occurence by altitude and fill missing values with previous value
    grouped = DF.groupby('elev').mean().round(1).fillna(method='ffill')
    grouped = grouped.iloc[1:,:] # Remove 1st row with elev = -9999
    
    # Version 3: Uses np.floor > 20.1 floor = 20.0, groupby then works
    T = np.floor(grouped.reset_index())
    
    # Apply rollingaverage over altitudes through all snow probabilities
    # if radius > 50:
    #     grouped = grouped.rolling(10).mean() #Apply rolling average for smoothing
       
    stepsize = 1
    GrDF = pd.DataFrame(columns = years)
    GrDF.insert(loc = 0, column = 'snow', value = np.arange(0,100+stepsize,stepsize).round(2))
               
    # start_time = time.time()
    GrDF.iloc[:,1:] = T.iloc[:,1:].apply(group_PdSeries,PdSeries_Elev=T.elev,axis=0)
    # Using Version 1 or 2 of getAltitude
    # for j,val in enumerate(np.arange(0,100+stepsize,stepsize).round(2)):
    #     GrDF.iloc[j,1:] = getAltitude(grouped,val)
    # print("Done: --- %s seconds ---" % (time.time() - start_time))
        
    # GrDF=DFobj2float(GrDF,GrDF.columns[1:]) # convert dtype object to float
    GrDF = GrDF.astype(float)
    
    # Interpolate missing mean seasonal elevations 
    for column in GrDF.columns[1:]:
        GrDF[column] = GrDF[column].interpolate()
        
    # GrDF = GrDF.fillna(method='ffill') # fills na with previous values
    GrDF = GrDF.fillna(method='bfill') # fills na after interpolation with following values (only important for low snow occurences)
    
    return GrDF
    
#%% ==========================================================================    
# Run config
exec(open("./code/config.py").read())

# Run importer
exec(open("./code/importer.py").read())

if season == "winter":
    OUT_PATH = './intermediate/winter_avgs/processed_altitudinal_distributions/'
else:
    OUT_PATH = './intermediate/summer_avgs/processed_altitudinal_distributions/'
 
# create output directory 
try:
    os.makedirs(OUT_PATH)
except:
    pass

Rmax = max(WOSM_rad)

#%% Run Loop
for i,_ in enumerate(Glacier_List.index):
        
    glacier_id = glacier_ids[i]
    glacier_name = glacier_names[i]
    x_center = x_centr[i]
    y_center = y_centr[i]
    
    if os.path.isfile(OUT_PATH + '/allGrDF_%s.pckl' % glacier_id):
        print ("File exists")
        continue
    else:       
        print("Processing Glacier Nr. %s ... \n" % glacier_id)
        
        # Compute altitudinal distribution of snow occurrence for each WOSM size
        start_time = time.time()
        
        DA_tot = DS_SeasAvg.sel(
            x = np.arange(x_center-Rmax , x_center+Rmax+unit,unit),
            y = np.arange(y_center-Rmax , y_center+Rmax+unit,unit),
            method='nearest')

        for j,R in enumerate(WOSM_rad):
        
            print('\r Computing for WOSM side length %s km' % int((2*R+unit)/unit), end='') 
            # allgrouped[i] = netCDF2SnowPerAlt_old(DS_SeasAvg['band 1'],DEM,x_center,y_center,unit,R)
                
            r_sub = int(Rmax/unit - R/unit) 
            
            if r_sub == 0:
                allgrouped[j] = netCDF2SnowPerAlt(DA_tot[band], DEM)
            else:
                # WOSM Pixel selection (starting from 400 x 400 km WOSM and subtracting)
                allgrouped[j] = netCDF2SnowPerAlt(DA_tot[band][:,r_sub:-r_sub,r_sub:-r_sub] , DEM)

        print("Done: --- %s seconds ---" % (time.time() - start_time))
        
        # Store List of GrDF to pickle file in corresponding folder, one file per glacier
        f = open(OUT_PATH+'/allGrDF_%s.pckl'%glacier_id, 'wb')
        pickle.dump(allgrouped, f)
        f.close()
    
    allgrouped = [None]*len(WOSM_rad)
# else:
#     exec(open("getAltSnowDistr.py").read())    
    
sys.exit(1)


"""
# %% WITH Joblib
from joblib import Parallel, delayed

def joblib_fct(DS,WOSM_rad,unit,OUT_PATH,row):            
    glacier_id = row[0]
    glacier_name = row[1]
    x_center = row[2]
    y_center = row[3]
    
    if os.path.isfile(OUT_PATH+'/allGrDF_%s.pckl'%glacier_id):
        print ("File exist")
        pass
    else:       
        print("Processing Glacier Nr. %s ... \n"%glacier_id)
        
        # Compute altitudinal distribution of snow occurrence for each WOSM size
        start_time = time.time()
        
        allgrouped = [None]*len(WOSM_rad)
        
        DA_tot = DS.sel(
                        Lon = np.arange(x_center-Rmax , x_center+Rmax+unit,unit),
                        Lat = np.arange(y_center-Rmax , y_center+Rmax+unit,unit),
                        method='nearest')

        for i,R in enumerate(WOSM_rad):

            print('\r Computing for WOSM side length %s km'%int((2*R+unit)/unit), end='') 
            # allgrouped[i] = netCDF2SnowPerAlt_old(DS_SeasAvg['band 1'],DEM,x_center,y_center,unit,R)
                
            r_sub = int(Rmax/1e3 - R/1e3) 
            
            if r_sub == 0:
                allgrouped[i] = netCDF2SnowPerAlt(DA_tot , DEM)
            else:
                # WOSM Pixel selection (starting from 400 x 400 km WOSM and subtracting)
                allgrouped[i] = netCDF2SnowPerAlt(DA_tot[:,r_sub:-r_sub,r_sub:-r_sub] , DEM)
            
        print("Done: --- %s seconds ---" % (time.time() - start_time))

    # Store List of GrDF to pickle file in corresponding folder, one file per glacier
            
        f = open(OUT_PATH+'/allGrDF_%s.pckl'%glacier_id, 'wb')
        pickle.dump(allgrouped, f)
        f.close()

Glacier_List = pd.read_csv('data/RGI/EUROPE/parallel_ge02km2_baselist.csv',sep = ';')
glacier_ids = Glacier_List.iloc[:,0] 
glacier_names = Glacier_List.iloc[:,1]
x_centr = Glacier_List.iloc[:,2]
y_centr = Glacier_List.iloc[:,3]  

ids_to_proc = np.logical_not([os.path.isfile(OUT_PATH+'/allGrDF_%s.pckl'%gid) for gid in glacier_ids])
Rmax = WOSM_rad.max()

n_cores = -1 # -1 = all are used, or os.cpu_count()

Parallel(n_jobs=n_cores, backend = 'multiprocessing')(delayed(joblib_fct)
                                (DS_SeasAvg["band 1"],WOSM_rad,unit,OUT_PATH,row) for _,row in Glacier_List[ids_to_proc].iterrows())
"""
