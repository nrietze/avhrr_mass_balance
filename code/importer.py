#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Import Files and prepare for processing
"""
import pandas as pd
import xarray as xr

print('Loading data...')

# %% Config number of scenes
df_nscenes = pd.read_csv('./intermediate/avhrr_scenes_per_season.csv',
                         sep = ';',index_col = 0)

# Define minimal Nr. of scenes for a valid season:
q = .1
thr_winter = df_nscenes.Nwinter.quantile(q).astype(int)
thr_summer = df_nscenes.Nsummer.quantile(q).astype(int)

if season == 'annual':
    calper = df_nscenes.Nsummer.loc[df_nscenes.Nsummer >= thr_summer].index.values
else:
    calper = df_nscenes.Nwinter.loc[df_nscenes.Nwinter >= thr_winter].index.values[1:]
    
years = calper.astype('<U21')

#  Plot decision for threshold for number of scenes
if False:
    import matplotlib.pyplot as plt

    fig,ax = plt.subplots(figsize=(10,8))
    df_nscenes.Nsummer.plot(ax=ax)
    ax.axhline(df_nscenes.Nsummer.quantile(.1), c = 'k',ls = '--')
    ax.text(2016,df_nscenes.Nsummer.quantile(.1) + 5 , 'q10%')
    ax.axhline(100, c = 'gray',ls = '-.')
    ax.text(2016,100 - 5 ,  '100')
    ax.set(title = 'Summer',
    xlabel = '',ylabel = 'Number of composite scenes')
    plt.savefig('./output/figures/number_summer_scenes.png')
    plt.show()
    
    ax = plt.gca()
    df_nscenes.Nwinter.plot(ax=ax)
    ax.axhline(df_nscenes.Nwinter.quantile(.1), c = 'k',ls = '--')
    ax.text(2016,df_nscenes.Nwinter.quantile(.1) + 5 ,  'q10%')
    ax.axhline(150, c = 'gray',ls = '-.')
    ax.text(2016,150 - 15 ,  '150')
    ax.set(title = 'Winter',
    xlabel = 'Number of scenes',ylabel = 'Cumulative distribution')
    plt.savefig('./output/figures/number_winter_scenes.png')

# %% Load auxillary data:

# Load DEM
DEM = xr.open_rasterio('./data/geodata/dem_europe.tif')

# Load glacier attributes (based on RGI)
Glacier_attrs = pd.read_csv('./data/geodata/CentralEurope_ge02km2_List.csv', 
             sep=';', lineterminator='\n',header=0).set_index('GlogemId')
Area = Glacier_attrs['Area']
Length = Glacier_attrs['Lmax']
Zmin = Glacier_attrs['Zmin']
Zmed = Glacier_attrs['Zmed']
Zmax = Glacier_attrs['Zmax']
Slope = Glacier_attrs['Slope']    

# Load seasonal data
if season=="annual":
    # Load WGMS data
    MB_obs =  pd.read_csv('./data/insitu/all_insitu_annual.csv', 
                 sep=';', lineterminator='\n',header=0)
    MB_obs.index = MB_obs['RGI_ID']
    
    # extract annual mass balances only (without attributes)
    all_glaciers_y = MB_obs.set_index(['GLOGEM_ID','NAME'])[MB_obs.columns[7:-1]]
    
    # Load avhrr summer dataset
    DS_SeasAvg = xr.open_dataset(f'./intermediate/avhrr/pso_summer_{band}.nc')
    
if season=="summer":
    # Load WGMS data
    MB_obs =  pd.read_csv('./data/insitu/all_insitu_summer.csv', 
                 sep=';', lineterminator='\n',header=0)
    MB_obs.index = MB_obs['RGI_ID']

    # Load avhrr summer dataset
    DS_SeasAvg = xr.open_dataset(f'./intermediate/avhrr/pso_summer_{band}.nc')

if season=="winter":
    # Load WGMS data
    MB_obs =  pd.read_csv('./data/insitu/all_insitu_winter.csv', 
                 sep=';', lineterminator='\n',header=0)
    MB_obs.index = MB_obs['RGI_ID']
    
    # Load avhrr winter dataset
    DS_SeasAvg = xr.open_dataset(f'./intermediate/avhrr/pso_winter_{band}.nc')

Glacier_List = MB_obs
glacier_ids = Glacier_List['GLOGEM_ID']
glacier_names = Glacier_List['NAME']     
x_centr = Glacier_List['x_centr']
y_centr = Glacier_List['y_centr']

print('done.')
