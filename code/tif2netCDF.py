import xarray as xr
import pandas as pd
import datetime as dt
import time

#%% FIND SEASON END & START POSITIONS
version = 'v05'

basedir = '../data/avhrr/10d_snowmasks'
v0x_list = pd.read_csv('/backup/nrietze/MA/data/AVHRR_snow_maps/%s_filelist.txt'%version,header=None)
dates = [dt.datetime.strptime(s[51:59],'%Y%m%d') for s in v0x_list[0]]

years = range(dates[0].year,dates[-1].year+1)
summer_start_pos = []
summer_end_pos = []
winter_start_pos = []
winter_end_pos = []

"""
winter: 1.10. -30.4.
summer: 1.5. - 30.9.
"""

for year in years:
    summer_start = min(date for date in dates if dt.datetime(year,5,1) <= date <= dt.datetime(year,9,30))
    summer_end = max(date for date in dates if dt.datetime(year,5,1) <= date <= dt.datetime(year,9,30))

    summer_start_pos.append(dates.index(summer_start))
    summer_end_pos.append(dates.index(summer_end))
    
    winter_start = min(date for date in dates if dt.datetime(year,10,1) <= date <= dt.datetime(year+1,4,30))
    winter_end = max(date for date in dates if dt.datetime(year,10,1) <= date <= dt.datetime(year+1,4,30))
    
    winter_start_pos.append(dates.index(winter_start)) 
    winter_end_pos.append(dates.index(winter_end)) 
    # winter_start_pos.append(dates.index(summer_end)+1) 
    # if year != 1981:
    #     winter_end_pos.append(dates.index(summer_start)-1) 

    print('Hydrological Summer covers: %s to %s'%(summer_start,summer_end))
    print('Hydrological Winter covers: %s to %s'%(winter_start,winter_end))

#%% AGGREGATE SEASON AVERAGES
global lat_corr, lon_corr

def tifs2xrDS(filelist, band, timecoord, ext):
    """
    ext = [xmin,ymin,xmax,ymax]
    """
    print("Converting list of tifs to xr.DataSet...")
    # Koordinatesysteme ändern sich im Laufe der Zeitreihe --> immer richtige Koordinaten wählen
    f = xr.open_rasterio(filelist.min())  # dafür ein tif der aktuellen liste wählen
    lon = xr.Variable('Lon',f.x.values[ext[0]:ext[2]])
    lat = xr.Variable('Lat',f.y.values[ext[1]:ext[3]])
    
    ds = xr.Dataset(data_vars = {'band %s'%band: (('time','Lat', 'Lon'), 
                                            [xr.open_rasterio(file).sel(
                                                band = band,
                                                y = lat,
                                                x = lon).values for file in filelist] )},
            coords={'Lon':  lon_corr,
                      'Lat': lat_corr,
                      'time': timecoord})
    print("done.")
    return(ds)

# Zuerst richtige Koordinaten definieren
ext = [1200,2500,2200,3500] # Alps within these pixels
bnd = 1

# Version 05 hat richtiges CRS ab 1990, v06 nie
f = xr.open_rasterio('/data/sensor/avhrr/eu1km/v03/snow/2015/02/multiday/20150201_----_------_avhrr-_comp7dcl_v05.tif')  
lat_corr = xr.Variable('Lat',f.y.values[ext[1]:ext[3]])
lon_corr = xr.Variable('Lon',f.x.values[ext[0]:ext[2]])
del f 

# Process summer
dsl = [None] * len(summer_start_pos)
summerdir = "/backup/nrietze/MA/data/AVHRR_snow_maps/summer_avgs/"
for i in range(len(summer_start_pos)):
    start_time = time.time()
    print("Processing summer period %s ..."%dates[summer_start_pos[i]].year)
    tiflist = v0x_list[summer_start_pos[i]:summer_end_pos[i]+1]
    
    datelist = xr.Variable('time', list(dates[summer_start_pos[i]:summer_end_pos[i]+1]))
    ds = tifs2xrDS(tiflist[0],bnd, datelist, ext)
    dsl[i] = ds.where(ds['band %s'%bnd]<=100,drop = True).mean(dim='time').assign_coords({'time': datelist.max()})
    print("--- %s seconds ---" % (time.time() - start_time))
    print("done.")
ds_seas = xr.concat(dsl[1:],dim = 'time')
ds_seas.to_netcdf(summerdir+"%s_summer_avg_band%s.nc"%(version,bnd))

#%% Process Winter
dwl = [None] * len(winter_start_pos)
winterdir = "/backup/nrietze/MA/data/AVHRR_snow_maps/winter_avgs/"
for i,_ in enumerate(winter_start_pos):
    start_time = time.time()
    print("Processing winter period %s ..."%dates[winter_start_pos[i]].year)
    tiflist = v0x_list[winter_start_pos[i]:winter_end_pos[i]+1]
    datelist = xr.Variable('time', list(dates[winter_start_pos[i]:winter_end_pos[i]+1]))
    
    if i==8 and version=="v05":
        dw_1 = tifs2xrDS(tiflist[0][:80],bnd, datelist[:80], ext)
        dw_2 = tifs2xrDS(tiflist[0][81:],bnd, datelist[81:], ext)
        dw = xr.concat([dw_1,dw_2],dim = 'time')
        dwl[i] = dw.where(dw['band %s'%bnd]<=100,drop = True).mean(dim='time').assign_coords({'time': datelist[-1]})      
        del(dw_1,dw_2)
    elif i==31 and version=="v05":
        dw_1=tifs2xrDS(tiflist[0][:89],bnd, datelist[:89], ext)
        dw_2=tifs2xrDS(tiflist[0][90:],bnd, datelist[90:], ext)
        dw = xr.concat([dw_1,dw_2],dim = 'time')
        dwl[i] = dw.where(dw['band %s'%bnd]<=100,drop = True).mean(dim='time').assign_coords({'time': datelist[-1]})      
        del(dw_1,dw_2)
    elif i==32 and version=="v05" or i==35 and version=="v05":
        dw_1=tifs2xrDS(tiflist[0][:92],bnd, datelist[:92], ext)
        dw_2=tifs2xrDS(tiflist[0][93:],bnd, datelist[93:], ext)
        dw = xr.concat([dw_1,dw_2],dim = 'time')
        dwl[i] = dw.where(dw['band %s'%bnd]<=100,drop = True).mean(dim='time').assign_coords({'time': datelist[-1]})      
        del(dw_1,dw_2)          
    else:
        dw = tifs2xrDS(tiflist[0],bnd, datelist, ext)
        dwl[i] = dw.where(dw['band %s'%bnd]<=100,drop = True).mean(dim='time').assign_coords({'time': datelist.max()})

    print("--- %s seconds ---" % (time.time() - start_time))
    print("done.")
dw_seas = xr.concat(dwl[:-1],dim = 'time')
dw_seas.to_netcdf(winterdir+"%s_winter_avg_band%s.nc"%(version,bnd))

#%% PLOT RESULTING SEASONAL SNOW DISTRIBUTION
"""
cmap = plt.cm.Blues
cmap.set_bad(color='gray')
cmap.set_under(color = 'green')
ds_seas['band 1'].sel(
    time = slice('1981-09-30,1984-09-30'),
    Lat = slice(5750000,5630000),
    Lon = slice(-160000,-90000)).plot(col='time',vmin = 1,col_wrap=2, cmap=cmap)

from mpl_toolkits.axes_grid1 import ImageGrid
f = xr.open_rasterio(v0x_list[0][2560])
fig = plt.figure(figsize=(20,25))
grid = ImageGrid(fig, 111, (2,3),
                 axes_pad=.1,
                 cbar_location='bottom',
                 cbar_mode='each',
                 cbar_pad=0.1,
                 aspect=False,
                 cbar_size='3%')

for (lab,data), ax, cax in zip(f.groupby('band'), grid, grid.cbar_axes):
    im = ax.imshow(data)
    ax.set_title(lab)
    cax.colorbar(im)
"""