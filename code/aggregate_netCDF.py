"""
Use this script to compute seasonal "Probability of snow occurrence (PSO)"
from 10-day composite snow maps from AVHRR LAC data. 

The resulting netCDFs contain the seasonally aggregated PSO, which indicates in %
out of how many days in the season a pixel was snow covered, i.e., if the summer
season counts 181 days and a pixel was covered in snow during 181 days, the PSO
is 100 %.

Created on Sun Jul 23 2023

@author: Nils Rietze, nils.rietze@uzh.ch

"""

#%% Config
import xarray as xr
import numpy as np
import datetime as dt
import time
from glob import glob
import pandas as pd

from functools import partial

# Function to slice netCDF before loading
def _preprocess(x, lon_bnds, lat_bnds,band):
    return x.sel(x=slice(*lon_bnds), y=slice(*lat_bnds))[band]

# %% Read data
PATH = './data/avhrr/10_d_snowmasks/'

# list all 10-day composite netCDF files
FILES = glob(PATH + '*.nc')

# Extract date from filename
dates = [dt.datetime.strptime(s[52:60],'%Y%m%d') for s in FILES]

years = range(dates[0].year,dates[-1].year+1)
summer_start_pos = []
summer_end_pos = []
winter_start_pos = []
winter_end_pos = []

# dataframe to store number of scenes per year & season
df_nscenes = pd.DataFrame(index = years,columns = ['Nwinter','Nsummer'])
df_nscenes.index.name = 'Year'

# Split dates into hydrological summer or winter
for year in years:
    try:
        # summer: 1.5. - 30.9.
        summer_start = min(date for date in dates if dt.datetime(year,5,1) <= date <= dt.datetime(year,9,30))
        summer_end = max(date for date in dates if dt.datetime(year,5,1) <= date <= dt.datetime(year,9,30))
    
        summer_start_pos.append(dates.index(summer_start))
        summer_end_pos.append(dates.index(summer_end) +1 )
        
        df_nscenes.loc[year,'Nsummer'] = dates.index(summer_end) - dates.index(summer_start)+1
        
        # winter: 1.10. -30.4.
        winter_start = min(date for date in dates if dt.datetime(year,10,1) <= date <= dt.datetime(year+1,4,30))
        winter_end = max(date for date in dates if dt.datetime(year,10,1) <= date <= dt.datetime(year+1,4,30))
        
        winter_start_pos.append(dates.index(winter_start)) 
        winter_end_pos.append(dates.index(winter_end) +1 ) 
        
        df_nscenes.loc[year,'Nwinter'] = dates.index(winter_end) - dates.index(winter_start)+1    
        
        print('Hydrological Summer covers: %s to %s'%(summer_start,summer_end))
        print('Hydrological Winter covers: %s to %s'%(winter_start,winter_end))
    
    except:
        print('This year is incomplete. Ignored.')
        continue
    
# Save dataframe to csv
df_nscenes.to_csv('./intermediate/avhrr_scenes_per_season.csv',sep = ';')
#%% AGGREGATE SEASON AVERAGES
# Outline coordinates of Alps (xmin, xmax) (ymax,ymin)
lon_ext, lat_ext = (-500000, 499000), (6100000, 5101000)

# Snow cover product/dataset band to use
band = 'scfg'

# output directory
OUT_PATH = "./intermediate/avhrr/"

# Process summer:

# Create empty list to store seasonal datasets:
dsl = [None] * len(summer_start_pos)

for i, (start,end) in enumerate(zip(summer_start_pos,summer_end_pos)):
    start_time = time.time()
    print("Processing summer period %s ..." % dates[start].year)
    
    # Subset files for summer in that year
    FILES_SUMMER = FILES[start:end]
    
    # Create time coordinate for output dataset
    datelist = xr.Variable('time', list(dates[start:end]) )
    
    # Preselect data within extent before loading
    partial_func = partial(_preprocess, 
                           lon_bnds=lon_ext, lat_bnds=lat_ext, 
                           band = band)
    
    # Load all netCDF of that summer
    ds = xr.open_mfdataset(
        FILES_SUMMER, concat_dim="time", combine = 'nested',
        preprocess=partial_func
    )      
    
    # Aggregate composite, daily datasets to seasonal range
    dsl[i] = xr.where(ds <= 100, ds, other=np.nan).mean(
            dim='time').assign_coords(
                {'time': datelist.data.max()})
    
    print("--- %s seconds ---" % (time.time() - start_time))
    print("done.")
    
ds_seas = xr.concat(dsl,dim = 'time')
ds_seas.to_netcdf(OUT_PATH + f"pso_summer_{band}.nc")

#%% Process Winter:
dwl = [None] * len(winter_start_pos)

for i, (start,end) in enumerate(zip(winter_start_pos,winter_end_pos)):
    start_time = time.time()
    print("Processing winter period %s ..." % dates[start].year)
    
    # Subset files for winter in that year
    FILES_WINTER = FILES[start:end]
    
    # Create time coordinate for output dataset
    datelist = xr.Variable('time', list(dates[start:end]) )
    
    # Preselect data within extent before loading
    partial_func = partial(_preprocess, 
                           lon_bnds=lon_ext, lat_bnds=lat_ext, 
                           band = band)
    
    # Load all netCDF of that winter
    ds = xr.open_mfdataset(
        FILES_WINTER, concat_dim="time", combine = 'nested',
        preprocess=partial_func
    )      
    
    # Aggregate composite, daily datasets to seasonal range
    dwl[i] = ds.where(
        ds <= 100,drop = True).mean(
            dim='time').assign_coords(
                {'time': datelist.data.max()})
    
    print("--- %s seconds ---" % (time.time() - start_time))
    print("done.")
    
ds_seas = xr.concat(dwl,dim = 'time')
ds_seas.to_netcdf(OUT_PATH + f"pso_winter_{band}.nc")
