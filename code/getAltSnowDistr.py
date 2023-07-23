#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 19:07:00 2020

@author: nrietze
"""
import numpy as np
import xarray as xr
import time
import pandas as pd
import pickle
import os

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
        # Radius already converted to length unit (i.e. 200'000 m input instead of 200 km)
        years = xrDataArray.time.dt.year.values
        
        A = xrDataArray

        # Convert to DataFrame with MultiIndex Lat and Lon and Years as Columns
        DF = A.to_dataframe('PSO').reset_index().pivot_table(values='PSO',
                                                             index=['Lat','Lon'],
                                                             columns='time')                    
        
        D = DEM.sel(
            x =  A.Lon,
            y = A.Lat,
            method='nearest')    
        D = D.to_dataframe(name='elev').reset_index('band',drop=True)
        
        D.index = D.index.rename(['Lon','Lat'])
        
        # DF = pd.DataFrame(A).T
        DF.columns = years
        DF = DF.join(D['elev'],on=['Lat','Lon'])
        # DF.insert(loc = 0, column = 'elev', value = D)

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

def netCDF2SnowPerAlt_old(xrDataArray,DEM,xcentr,ycentr,lengthunit,radius):    
        # Radius already converted to length unit (i.e. 200'000 m input instead of 200 km)
        years = xrDataArray.time.dt.year.values
        
        A = xrDataArray.sel(
                Lon = np.arange(xcentr-radius,xcentr+radius+lengthunit,lengthunit),
                Lat = np.arange(ycentr-radius,ycentr+radius+lengthunit,lengthunit),
                method='nearest')
        
        # A = [array.flatten().round(2) for array in
            # xrDataArray.sel(
            #     Lon = np.arange(xcentr-radius,xcentr+radius+lengthunit,lengthunit),
            #     Lat = np.arange(ycentr-radius,ycentr+radius+lengthunit,lengthunit),
            #     method='nearest')
        #     .values]

        # Convert to DataFrame with MultiIndex Lat and Lon and Years as Columns
        DF = A.to_dataframe('PSO').reset_index().pivot_table(values='PSO',
                                                             index=['Lat','Lon'],
                                                             columns='time')                    
        
        D = DEM.sel(
            x =  A.Lon,
            y = A.Lat,
            method='nearest')    
        D = D.to_dataframe(name='elev').reset_index('band',drop=True)
        
        D.index = D.index.rename(['Lon','Lat'])
        
        # DF = pd.DataFrame(A).T
        DF.columns = years
        DF = DF.join(D['elev'],on=['Lat','Lon'])
        # DF.insert(loc = 0, column = 'elev', value = D)

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

#%%
start_time = time.time()

DA_tot = DS_SeasAvg["band 1"].sel(
                Lon = np.arange(x_center-Rmax , x_center+Rmax+unit,unit),
                Lat = np.arange(y_center-Rmax , y_center+Rmax+unit,unit),
                method='nearest')

OUT_PATH

for i,R in enumerate(WOSM_rad):

    print('\r Computing for WOSM side length %s km'%int((2*R+unit)/unit), end='') 
    
    # allgrouped[i] = netCDF2SnowPerAlt_old(DS_SeasAvg['band 1'],DEM,x_center,y_center,unit,R)
        
    r_sub = int(max_R - R/1e3) 
    
    if r_sub == 0:
        DA = DA_tot
    else:
        # WOSM Pixel selection (starting from 400 x 400 km WOSM and subtracting)
        DA = DA_tot[:,r_sub:-r_sub,r_sub:-r_sub]
    
    print('\r Computing for WOSM side length %s km'%int(DA.Lon.shape[0]), end='') 
    
    allgrouped[i] = netCDF2SnowPerAlt(DA,DEM)
        
print("Done: --- %s seconds ---" % (time.time() - start_time))

# Store List of GrDF to pickle file in corresponding folder, one file per glacier
        
f = open(OUT_PATH + '/allGrDF_%s.pckl'%glacier_id, 'wb')
pickle.dump(allgrouped, f)
f.close()

# %%
"""
def netCDF2SnowPerAlt_old(xrDataArray,DEM,xcentr,ycentr,lengthunit,radius):    
        # Radius already converted to length unit (i.e. 200'000 m input instead of 200 km)
        years = xrDataArray.time.dt.year.values
        
        A = xrDataArray.sel(
                Lon = np.arange(xcentr-radius,xcentr+radius+lengthunit,lengthunit),
                Lat = np.arange(ycentr-radius,ycentr+radius+lengthunit,lengthunit),
                method='nearest')
        
        # A = [array.flatten().round(2) for array in
            # xrDataArray.sel(
            #     Lon = np.arange(xcentr-radius,xcentr+radius+lengthunit,lengthunit),
            #     Lat = np.arange(ycentr-radius,ycentr+radius+lengthunit,lengthunit),
            #     method='nearest')
        #     .values]

        # Convert to DataFrame with MultiIndex Lat and Lon and Years as Columns
        DF = A.to_dataframe('PSO').reset_index().pivot_table(values='PSO',
                                                             index=['Lat','Lon'],
                                                             columns='time')                    
        
        D = DEM.sel(
            x =  A.Lon,
            y = A.Lat,
            method='nearest')    
        D = D.to_dataframe(name='elev').reset_index('band',drop=True)
        
        D.index = D.index.rename(['Lon','Lat'])
        
        # DF = pd.DataFrame(A).T
        DF.columns = years
        DF = DF.join(D['elev'],on=['Lat','Lon'])
        # DF.insert(loc = 0, column = 'elev', value = D)

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
"""