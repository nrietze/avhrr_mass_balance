#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting 
"""
import pickle

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

import xarray as xr
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# %%==========================================================================
def PlotFigure2a(years: np.array,
                 df_GlacierAttrs: pd.DataFrame,
                 SSC_PATH = './data/AVHRR_snow_maps/summer_avgs/v05_summer_avg_band1.nc',
                 DEM_PATH= './data/DEM/w020n90/SRTM30_CH_warped.tif'
                 ):
    """
    Parameters
    ----------
    years : array/np.array
        Contains all years analyzed in this study as strings (e.g. '1982', '1983',...).
    df_GlacierAttrs: pd.DataFrame
        DataFrame containing the centroid coordinates of the glaciers in Cartesian coordinates (Lambert Conformal Conic)
    SSC_PATH & DEM_PATH : str
        Path to the DEM and seasonal snow cover maps to derive altitudinal snowline distributions.

    Returns
    -------
    A saved Figure 2a in the folder './Figures/final_paper'.

    """

    print('Plotting Figure 2 subplot a...')
    
    # get mean seasonal snow altitude for wosm radius = 25 km (51 x 51 km)
    SSC = xr.open_dataset(SSC_PATH)
    DEM = xr.open_rasterio(DEM_PATH)
    
    R = 25e3
    unit = 1e3
    
    x_rhone = df_GlacierAttrs.loc[rhone_id].x_centr
    y_rhone = df_GlacierAttrs.loc[rhone_id].y_centr
    
    A = [array.flatten().round(2) for array in 
            SSC['band 1'].sel(
                Lon = np.arange(x_rhone-R,x_rhone+R+unit,unit),
                Lat = np.arange(y_rhone-R,y_rhone+R+unit,unit),
                method='nearest')
            .values]
    D = DEM.sel(
        x =  np.arange(x_rhone-R,x_rhone+R+unit,unit),
        y = np.arange(y_rhone-R,y_rhone+R+unit,unit),
        method='nearest').values[0].flatten()          
    
    DF = pd.DataFrame(A).T
    DF.columns = years
    DF.insert(loc = 0, column = 'elev', value = D)
    
    # Group snow occurence by altitude and fill missing values with previous value
    grouped = DF.groupby('elev').mean().round(0).fillna(method='ffill')
    grouped_rm = grouped.rolling(50).mean()

    # Set up Figure
    plt.rc('font', size=25) 
    fig, ax = plt.subplots(figsize = (10,5),dpi=200)
    
    ax.plot(grouped_rm,
            c = 'grey', lw = .2)
    ax.axhline(80,ls = '--',c = 'r',lw = .8)
    
    ax.set_xlabel('Altitude (m)')
    ax.set_ylabel('Seasonal Snow Cover (%)')
    ax.title.set_size(9) 
    
    # inset axes....
    axins = ax.inset_axes([0.05, 0.5, 0.3, 0.47])
    axins.plot(grouped_rm,
            c = 'grey', lw = .2)
    axins.axhline(80,ls = '--',c = 'r',lw = .8)
    intersects = [grouped_rm[column].iloc[min(np.argwhere(np.isclose(grouped_rm[column],80,atol=.8)))].index.item() for column in grouped_rm.columns]
    axins.plot(intersects,np.repeat(80,38),'ko')
    
    # sub region of the original plot
    x1, x2, y1, y2 = 2800, 3300, 70, 90
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels('')
    axins.set_yticklabels('')
    
    ax.indicate_inset_zoom(axins, edgecolor="black")
    
    plt.savefig('./figures/final_paper/Fig_2a.png',bbox_inches='tight')
    print('done.')
    
# %%==========================================================================
def PlotFigure2b(years: np.array, df_MBobs: pd.DataFrame, 
                df_ModelParams: pd.DataFrame, 
                pickldir = './data/AVHRR_snow_maps/summer_avgs/processed_altitudinal_distributions'
                ):
    """
    Parameters
    ----------
    years : array/np.array
        Contains all years analyzed in this study as strings (e.g. '1982', '1983',...).
    df_MBobs : pd.DataFrame
        DataFrame with all ANNUAL in situ observations, indexed by the GloGEM ID.
    df_ModelParams : pd.DataFrame
        DataFrame with model coefficients, intercepts, scores etc. Only alpha & beta will be used here.
    pickldir : str, optional
        Path to the pickl-files containing the list of DataFrames with the altitudinal snowline distributions.
        The default is './data/AVHRR_snow_maps/summer_avgs/processed_altitudinal_distributions'.

    Returns
    -------
    A saved Figure 2b in the folder './Figures/final_paper'.

    """
    print('Plotting Figure 2 subplot b...')
    
    # Use Rhone glacier as example
    rhone_id = 1238

    # Select Rhone in situ observations
    MB_obs = df_MBobs.loc[rhone_id,years]/1e3
    
    # Load DataFrames of Rhone glacier from the snow altitude distributions
    f = open(pickldir+'/allGrDF_%s.pckl' % (rhone_id), 'rb')
    allgrouped = pickle.load(f)
    f.close();del f

    # Choose random WOSM radius 
    R = 195
    
    # Select snowline elvations at that radius for 60% snow cover (just for illustrative means)
    Z = allgrouped[R-1].loc[60,years.astype(int)]
    
    # Get coefficient and intercept of linear model
    alpha = df_ModelParams.loc[rhone_id,'alpha*']/1e3	
    beta = df_ModelParams.loc[rhone_id,'beta*']/1e3
    mdl = alpha * Z + beta
    
    plt.rc('font', size=25) 
    
    fig, ax = plt.subplots(figsize = (10,5),dpi=200)
    ax.scatter(Z,MB_obs ,
               c = 'k')
    ax.plot(Z,mdl, c = 'k',lw = 0.6)
    ax.axhline(0,ls = '--',c = 'k',lw = 0.6)
    ax.set(ylim = [-2,.5],
           xlim = [2950,3200],
           ylabel='MB$_{obs}$ (m w.e. a$^{-1}$)',
           xlabel='Z (m)')
    plt.savefig('./figures/final_paper/Fig_2b.png',bbox_inches='tight')

# %%==========================================================================
def ecdf(xdata):
    xdataecdf = np.sort(xdata)
    ydataecdf = np.arange(1, len(xdata) + 1) / len(xdata) * 100
    return xdataecdf, ydataecdf

def PlotFigure6(DF_model_params_w: pd.DataFrame,
                DF_model_params_a: pd.DataFrame,
                DF_model_params_a_AA: pd.DataFrame,
                DF_model_params_a_4km: pd.DataFrame
                ):
    """
    Parameters
    ----------
    DF_model_params_w : pd.DataFrame
        DataFrame containing (amongst others) the optimized WOSM size of the winter MB model (CRG method). Named 'WOSM*'
    DF_model_params_a : pd.DataFrame
        Same as above but with annual CRG MB.
    DF_model_params_a_AA : pd.DataFrame
        Same as above but with annual AA MB.
    DF_model_params_a_4km : pd.DataFrame
        Same as above but with coarsened (4x4km "GAC" snow cover maps upscaled from LAC maps) of annual MB.

    Returns
    -------
    A saved Figure 6 in the folder './Figures/final_paper'.

    """
    # ECDF Plot
    print('Plotting Figure 6: ECDF...')
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)

    _,ax = plt.subplots(figsize = (5,4), dpi = 150)
    
    x,y = ecdf(DF_model_params_w['WOSM*'].div(.5e3))
    ax.plot(x,y, lw = 1,label = 'winter CRG')
    x,y = ecdf(DF_model_params_a['WOSM*'].div(.5e3))
    ax.plot(x,y,c='k', lw = 1,label = 'annual CRG')
    x,y = ecdf(DF_model_params_a_AA['WOSM*'].div(.5e3))
    ax.plot(x,y,c='k',ls = '--', lw = 1,label = 'annual AA')
    x,y = ecdf(DF_model_params_a_4km['WOSM*'].div(.5e3))
    ax.plot(x,y, c = 'brown',ls = '--', lw = 1,label = 'annual 4km AA')
    
    ax.axhline(50, c = 'gray', alpha = .3, ls = '--')
    ax.text(x = -30, y = 50, s = '50',va = 'center',c = 'gray')
    
    ax.set(xlabel = 'WOSM* side length (km)',
           ylabel = 'Fraction of studied glaciers (%)',
           ylim = [0,100],
           xlim = [0,400])
    ax.legend()
    
    plt.savefig('./figures/final_paper/Fig_6.png', bbox_inches = 'tight')
    
    print('done.')
    
# %%==========================================================================
from matplotlib.legend_handler import  HandlerPatch
import matplotlib.patches as mpatches

def make_legend_polygon(legend, orig_handle,
                      xdescent, ydescent,
                      width, height, fontsize):
    a=2*height
    p = mpatches.Polygon(np.array([[0,-a/2],[width,-a/2],[width,height+a/2],[0,height+a/2],[0,-a/2]]))
    return p

def PlotFigure4(years: np.array,
                df_annualMB_AVHRR: pd.DataFrame,
                df_MBHugonnet: pd.DataFrame,
                df_GlacierAttrs: pd.DataFrame):
    """
    Parameters
    ----------
    years : array/np.array
        Contains all years analyzed in this study as strings (e.g. '1982', '1983',...).
    df_annualMB_AVHRR : pd.DataFrame
        DataFrame with estimated annual MB from the AVHRR snowmap method (CRG).
    df_MBHugonnet : pd.DataFrame
        DataFrame of Hugonnet et al. (2021) MB.
    df_GlacierAttrs: pd.DataFrame
        DataFrame containing the centroid coordinates of the glaciers in Cartesian coordinates (Lambert Conformal Conic)

    Returns
    -------
    A saved Figure 4 in the folder 'Figures/final_paper/'.

    """
    print('Plotting Figure 4 - Comparison with geodetic MB...')
    
    fig,ax = plt.subplots(figsize = (12,9), dpi = 150)

    #----------
    rgi_ids = df_GlacierAttrs.RGIId
    common_rgi_ids = np.isin(df_MBHugonnet.index,rgi_ids)
    
    mb_AVHRR = df_annualMB_AVHRR[years].mean(axis=0).div(1e3)
    unc_AVHRR = 0.401
    
    tlim_huss = [np.arange(1982,1990+1).astype(str),
                   np.arange(1990,2000+1).astype(str),
                   np.arange(2000,2010+1).astype(str)]
    
    tlim_zemp = np.arange(2006,2016+1).astype(str)
    tlim_sommer = np.arange(2000,2014+1).astype(str)
    tlim_hugo = np.arange(2000,2019+1).astype(str)
    list_tlim = [tlim_zemp,tlim_sommer,tlim_hugo]
    
    mb_huss = [-0.4,
               -0.68,
               -0.99]
    
    mb_zemp = -0.87
    mb_sommer = -0.698 # entire Alps
    mb_hugo = df_MBHugonnet.B.mean()
    list_mbs = [mb_zemp,mb_sommer,mb_hugo]
    
    unc_huss = 0.041
    
    unc_zemp = 0.07
    unc_sommer = 0.128
    unc_hugo = sum(df_MBHugonnet[common_rgi_ids].Area * df_MBHugonnet[common_rgi_ids].errB) / sum(df_MBHugonnet[common_rgi_ids].Area) * 1.96
    list_unc = [unc_zemp,unc_sommer,unc_hugo]
    
    # List of colors (last one is Huss)
    # list_colors = ['#3399FF','plum','#FF9966']
    list_colors = ['#2C6700','#CC0000','#FF9966','#3333FF' ]
    #----------
    
    # ax = axs[0]
    ax.axhline(0,lw = .7, c = 'k', ls = 'dashed')
    
    # Plot AVHRR yearly results
    for i,_ in enumerate(years[:-1]):
        ax.fill_between([years[i],years[i+1]] ,mb_AVHRR[i] - unc_AVHRR, mb_AVHRR[i] + unc_AVHRR,
                        color=plt.cm.Greys(0.9),alpha=0.4,linewidth=0.25,zorder = 0)
        ax.plot([years[i],years[i+1]] , [mb_AVHRR[i]] * 2 ,
                color=plt.cm.Greys(0.9),lw=.7,zorder = 0)
    
    # Plot Huss (2012)
    for ih,tlimh in enumerate(tlim_huss):
        ax.fill_between([tlimh[0],tlimh[-1]] ,mb_huss[ih] - unc_huss, mb_huss[ih] + unc_huss,
                        color=list_colors[3],alpha=0.4,linewidth=0.25)
        ax.plot(tlimh, [mb_huss[ih]] * len(tlimh) ,
                color=plt.cm.Blues(0.9),lw=.7)
    
    # Plot other geodetic MB
    for ii,tlim in enumerate(list_tlim):
        ax.fill_between([tlim[0],tlim[-1]] ,list_mbs[ii] - list_unc[ii], list_mbs[ii] + list_unc[ii] ,
                        color=list_colors[ii],alpha=0.4,linewidth=0.25,zorder = 10)
        ax.plot(tlim, [list_mbs[ii]] * len(tlim) 
                ,color=list_colors[ii],lw=.7,zorder = 10)
    
    ax.set_ylim([-1.8,1])
    ax.set_xlim([years[0],years[-1]])
    ax.set_xticks(np.arange(0,38,4))
    ax.set_xticklabels(years[::4])
    ax.set_ylabel('Specific mass balance (m w.e yr$^{-1}$)',labelpad=0.25)
    ax.set_xlabel('Year')
    ax.tick_params(width=0.35,length=2.5)
    
    # Create Legend
    p1 = ax.plot([], [], color=list_colors[3], linewidth=0.35)
    p2 = ax.fill([], [], color=list_colors[3], alpha=0.4,linewidth=0.25)
    
    p3 = ax.plot([], [], color=list_colors[0], linewidth=0.5)
    p4 = ax.fill([], [], color=list_colors[0], alpha=0.45,linewidth=0.25)
    p5 = ax.plot([], [], color=list_colors[1], linewidth=0.5)
    p6 = ax.fill([], [], color=list_colors[1], alpha=0.45,linewidth=0.25)
    p7 = ax.plot([], [], color=list_colors[2], linewidth=0.5)
    p8 = ax.fill([], [], color=list_colors[2], alpha=0.45,linewidth=0.25)
    
    p9= ax.plot([], [], color=plt.cm.Greys(0.9), linewidth=1)
    p10 = ax.fill([], [], color=plt.cm.Greys(0.9), alpha=0.4,linewidth=0.25)
    
    hm = {p2[0]: HandlerPatch(patch_func=make_legend_polygon), 
          p4[0]: HandlerPatch(patch_func=make_legend_polygon), 
          p6[0]: HandlerPatch(patch_func=make_legend_polygon),
          p8[0]: HandlerPatch(patch_func=make_legend_polygon), 
          p10[0]: HandlerPatch(patch_func=make_legend_polygon)}
    l = ax.legend([(p1[0],p2[0]),
                    (p3[0],p4[0]),(p5[0],p6[0]),(p7[0], p8[0]),
                    (p9[0],p10[0])], 
                   ['Huss (2012) \n (decadal)',
                    'Zemp et al.(2019)','Sommer et al. (2020)','Hugonnet et al. (2021)',
                    'This study \n (annual)'],
                   handlelength=0.75,
                   framealpha=0.6,
                   # loc=(0.35,0.725),
                   loc = 'upper right',
                   ncol=2,
                   labelspacing=1.35,
                   handler_map=hm,
                   borderpad=0.6)
    l.get_frame().set_linewidth(0.5)
    
    plt.savefig('./figures/final_paper/fig_4.png', bbox_inches = 'tight')
    print('done.')