#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Import Files and prepare for processing
"""
import pandas as pd
import xarray as xr
import datetime as dt
import os

print('Loading data...')

# %%
DF_nscenes = pd.read_csv('../../data/AVHRR_snow_maps/nrScenesPerSeason.csv',sep = ';',index_col = 0)

# Define minimal Nr. of scenes for a valid season:
q = .1
thr_winter = DF_nscenes.Nwinter.quantile(q).astype(int)
thr_summer = DF_nscenes.Nsummer.quantile(q).astype(int)

if season == 'annual':
    calper = DF_nscenes.Nsummer.loc[DF_nscenes.Nsummer >= thr_summer].index.values
else:
    calper = DF_nscenes.Nwinter.loc[DF_nscenes.Nwinter >= thr_winter].index.values[1:]
    
years = calper.astype('<U21')

# %% Plot decision for Nr. of scenes
"""
fig,ax = plt.subplots(figsize=(10,8))
DF_nscenes.Nsummer.plot(ax=ax)
ax.axhline(DF_nscenes.Nsummer.quantile(.1), c = 'k',ls = '--')
ax.text(2016,DF_nscenes.Nsummer.quantile(.1) + 5 , 'q10%')
ax.axhline(100, c = 'gray',ls = '-.')
ax.text(2016,100 - 5 ,  '100')
ax.set(title = 'Summer',
xlabel = '',ylabel = 'Number of composite scenes')
plt.savefig('../../figures/Method/summer_scenes.png')
plt.show()

x,y = ecdf(DF_nscenes.Nwinter)
ax = plt.gca()
DF_nscenes.Nwinter.plot(ax=ax)
ax.axhline(DF_nscenes.Nwinter.quantile(.1), c = 'k',ls = '--')
ax.text(2016,DF_nscenes.Nwinter.quantile(.1) + 5 ,  'q10%')
ax.axhline(150, c = 'gray',ls = '-.')
ax.text(2016,150 - 15 ,  '150')
ax.set(title = 'Winter',
xlabel = 'Number of scenes',ylabel = 'Cumulative distribution')
plt.savefig('../../figures/Method/winter_scenes.png')
"""
# %%

# Load DEM
if inventory == "HKH":
    DEM = xr.open_rasterio('../../data/DEM/SRTM30_HKH_cubic.tif')
else:
    DEM = xr.open_rasterio('../../data/DEM/w020n90/SRTM30_EU_warped.tif')

# Load Glacier attributes
if inventory == "HKH":
    Glacier_attrs = pd.read_csv('../../data/MB/GloGEM/thick_centralasiaS_5yrs.dat', 
                 sep='\t', lineterminator='\n',header=0).set_index('ID')
    
else: 
    Glacier_attrs = pd.read_csv('../../data/RGI/EUROPE/CentralEurope_ge02km2_List.csv', 
                 sep=';', lineterminator='\n',header=0).set_index('GlogemId')
    Area = Glacier_attrs['Area']
    Length = Glacier_attrs['Lmax']
    Zmin = Glacier_attrs['Zmin']
    Zmed = Glacier_attrs['Zmed']
    Zmax = Glacier_attrs['Zmax']
    Slope = Glacier_attrs['Slope']    

# Load Mass Balance data
if inventory=="GLAMOS": 
    MB_obs = pd.read_csv('../../data/GLAMOS/massbalance_fixdate_ed.csv', 
                     sep=',', lineterminator='\n',header=0)
    # MB_obs = pd.read_csv('../../data/GLAMOS/gries_fixdate.txt', 
    #                  sep='\t', lineterminator='\n',header=0)
    
    MB_obs_index = [dt.datetime.strptime(row,'%d.%m.%Y').year for row in MB_obs['end date of observation']]
    MB_obs.index = MB_obs_index
    # glacier_ids = MB_obs['glacier id']
    # glacier_ids = glacier_ids.min().replace('/','_')
    
    Glacier_List = pd.read_csv('../../data/RGI/CH_glac_inGLAMOS_2km.csv',
                  sep=',', lineterminator='\n',header=0)
    glacier_ids = Glacier_List['GLAMOS_id'].unique() 
    glacier_names = Glacier_List['Name'].unique() 
    x_centr = Glacier_List['x_centr']
    y_centr = Glacier_List['y_centr']
    # x_gries = x_centr[6]
    # y_gries = y_centr[6]
    
elif inventory=="GloGem":
    if season=="annual":
        MB_obs = pd.read_csv('../../data/MB/GloGEM/centraleurope_Annual_Balance_sfc_r1.dat', 
                     sep='\s+', lineterminator='\n',header=0)
        # Convert m w.e. to mm w.e.
        MB_obs.iloc[:,1:] = MB_obs.iloc[:,1:]*1000
    if season=="summer":
        MB_obs = pd.read_csv('../../data/MB/GloGEM/centraleurope_Summer_Balance_sfc_r1.dat', 
                     sep='\s+', lineterminator='\n',header=0)
        # Convert m w.e. to mm w.e.
        MB_obs.iloc[:,1:] = MB_obs.iloc[:,1:]*1000
    elif season=="winter":
        MB_obs = pd.read_csv('../../data/MB/GloGEM/centraleurope_Winter_balance_sfc_r1.dat', 
                     sep='\s+', lineterminator='\n',header=0)
        # Convert m w.e. to mm w.e.
        MB_obs.iloc[:,1:] = MB_obs.iloc[:,1:]*1000

    # Glacier_List = pd.read_csv('../../data/glogem_centerpoints/GlacierList_2qkm.csv',
    #               sep=',', lineterminator='\n',header=0)
    Glacier_List = pd.read_csv('../../data/glogem_centerpoints/GlacierList_05qkm.csv',
                  sep=',', lineterminator='\n',header=0)
    glacier_ids = Glacier_List['ID'].unique() 
    glacier_names = Glacier_List['Name'].unique() 
    x_centr = Glacier_List['x_centr']
    y_centr = Glacier_List['y_centr']
    
elif inventory =="WGMS":
        MB_obs = pd.read_csv('../../data/MB/insitu/WGMS/CentrEU/all_glaciers_noCH.csv', 
                     sep=';', lineterminator='\n',header=13)
        MB_obs.index = MB_obs['SURVEY_YEAR']
        
elif inventory == "all":
    if season=="annual":
        MB_obs =  pd.read_csv('../../data/MB/insitu/all/all_insitu_annual.csv', 
                     sep=';', lineterminator='\n',header=0)
        MB_obs.index = MB_obs['RGI_ID']
    if season=="summer":
        MB_obs =  pd.read_csv('../../data/MB/insitu/all/all_insitu_summer.csv', 
                     sep=';', lineterminator='\n',header=0)
        MB_obs.index = MB_obs['RGI_ID']

    if season=="winter":
        MB_obs =  pd.read_csv('../../data/MB/insitu/all/all_insitu_winter.csv', 
                     sep=';', lineterminator='\n',header=0)
        MB_obs.index = MB_obs['RGI_ID']

    
    Glacier_List = MB_obs
    glacier_ids = Glacier_List['GLOGEM_ID']
    glacier_names = Glacier_List['NAME']     
    x_centr = Glacier_List['x_centr']
    y_centr = Glacier_List['y_centr']

elif inventory =="HKH":
    MB_obs = pd.read_csv('../../data/MB/insitu/WGMS/HKH/HKH_all_insitu.csv', 
                 sep=',', lineterminator='\n',header=13)
    MB_obs.index = MB_obs['SURVEY_YEAR']
    
    Glacier_List = pd.read_csv('../../data/MB/insitu/WGMS/HKH/parallel.txt', 
                 sep='\t', lineterminator='\n',header=None)
    glacier_ids = Glacier_List[1]
    glacier_names = Glacier_List[0]     
    x_centr = Glacier_List[3]
    y_centr = Glacier_List[2]
        
# Set season
if season == "winter":    
    if inventory == "GLAMOS":
        all_glaciers_y =  MB_obs[['glacier name','glacier id','winter mass balance']].reset_index().pivot_table(values='winter mass balance',
                                                                                          index=['glacier name','glacier id'], 
                                                                                          columns='index')
    elif inventory == "WGMS":
        all_glaciers_y =  MB_obs[['NAME','WGMS_ID','WINTER_BALANCE']].reset_index().pivot_table(values='WINTER_BALANCE',
                                                                                          index=['NAME','WGMS_ID'], 
                                                                                          columns='SURVEY_YEAR')
                             
    elif inventory == "GloGem":
        all_glaciers_y = MB_obs.set_index('ID')
    
    # Load band 1 winter averages
    DS_SeasAvg = xr.open_dataset('../../data/AVHRR_snow_maps/winter_avgs/v05_winter_avg_band1.nc')
    
elif season == "annual":
    if inventory == "GLAMOS":
        all_glaciers_y =  MB_obs[['glacier name','glacier id','annual mass balance']].reset_index().pivot_table(values='annual mass balance',
                                                                                          index=['glacier name','glacier id'], 
                                                                                          columns='index')
    elif inventory == "WGMS":
        all_glaciers_y =  MB_obs[['NAME','WGMS_ID','ANNUAL_BALANCE']].reset_index().pivot_table(values='ANNUAL_BALANCE',
                                                                                          index=['NAME','WGMS_ID'], 
                                                                                          columns='SURVEY_YEAR')
    elif inventory == "GloGem":
        all_glaciers_y = MB_obs.set_index('ID')
        
    elif inventory == "all":
        all_glaciers_y = MB_obs.set_index(['GLOGEM_ID','NAME'])[MB_obs.columns[7:-1]]
        
    elif inventory == "HKH":
        all_glaciers_y =  MB_obs[['NAME','WGMS_ID','ANNUAL_BALANCE']].reset_index().pivot_table(values='ANNUAL_BALANCE',
                                                                                          index=['NAME','WGMS_ID'], 
                                                                                          columns='SURVEY_YEAR')
    
    # Load band 1 summer averages
    if inventory == "HKH":
        DS_SeasAvg = xr.open_dataset('../../data/AVHRR_snow_maps/summer_avgs/CCI_summer_avg_band1.nc')
    else:
        DS_SeasAvg = xr.open_dataset('../../data/AVHRR_snow_maps/summer_avgs/v05_summer_avg_band1.nc')

print('done.')
