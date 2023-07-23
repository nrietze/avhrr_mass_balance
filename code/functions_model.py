#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 19:27:59 2021

@author: nrietze
"""
import pandas as pd
import numpy as np

#%%    
from csv import writer
def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj,delimiter=";")
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


def DFobj2float(DF,columns):
    DF_out=DF
    for col in columns:
        DF_out[col] = DF[col].astype(float)
    return DF_out

#%% Statistics
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

from sklearn.linear_model import LinearRegression
def linReg(y,x):
    model = LinearRegression().fit(x,y)
    R2 = model.score(x,y)
    RMSE = rmse(model.predict(x),y)
    alpha = model.coef_[0]
    beta = model.intercept_
    return [R2,RMSE,alpha,beta]

def weighted_mean(df, values, weights, groupby):
    df = df.copy()
    grouped = df.groupby(groupby)
    df['weighted_average'] = df[values] / grouped[weights].transform('sum') * df[weights]
    return grouped['weighted_average'].sum(min_count=1) #min_count is required for Grouper objects

def skill_score(rmse_cv,rmse_ref):
    return 1-(rmse_cv**2)/(rmse_ref**2) # Uses MSE instead of RMSE

def cross_val(X,y,cv):
    alpha = np.empty(len(X)) ; beta = np.empty(len(X)) 
    y_pred = np.empty(len(X)) ; R2 = np.empty(len(X))
    
    for train, test in cv.split(X):
        X_train,y_train= X[train], y[train]
        
        # Fit X_train to y_train (i.e. all years but test year)
        model = LinearRegression().fit(X_train,y_train) 
        
        # Predict MB in test year 
        y_pred[test] = model.predict(X[test]) 
        
        #Get coefficient of the fit
        alpha[test] = model.coef_ 
        
        #Get intercept of the fit
        beta[test] = model.intercept_ 
        
        # Get R2 of model
        R2[test] = model.score(X_train,y_train) 
    
    # Compute Mean over all N 
    R2 = R2.mean()
    Alpha = alpha.mean()
    std_alpha = np.std(alpha)
    Beta = beta.mean()
    std_beta = np.std(beta)
    
    return(y_pred,Alpha,std_alpha, Beta,std_beta, R2)
#%%
def map_linReg(row,ycal):
    xcal = row.values.reshape((-1, 1))
    return pd.Series(
        linReg(ycal,xcal)
        )

# ============================================================================

def map_snowaltdist(t:tuple, pdSeries_ycal):
    """
    This function performs the fitting for each WOSM radius and SSC.

    Parameters
    ----------
    t : tuple
        contains index and DataFrame.

    Returns
    -------
    series : pd.Series
        optimal parameters from each iteration.

    """
    i = t[0]
    # Fetch data in GrDF during calibration period
    df = t[1][pdSeries_ycal.index.astype(int)]
    
    # Compute linear regression (observed - AVHRR) for all 
    scores = df.apply(map_linReg, ycal = pdSeries_ycal, axis=1, result_type='expand')
    scores.columns = ['R2','RMSE','alpha','beta']

    # opt_idx = scores.RMSE[20:].idxmin() # optimal snow occurrence (only if > 20%) minimizing RMSE in the DataFrame 
    opt_idx = scores.RMSE.idxmin()
    
    # Store optimal snow occurrence (only if > 20%) minimizing RMSE in the DataFrame 
    series = scores.loc[opt_idx,:]
    series['SSC'] = opt_idx
    
    return series.T.to_numpy()

# ============================================================================

def get_opt_parameters_and_MB(pdDF_OptSnow,pdSeries_ycal,allgrouped, WOSM_rad):
    # Get Optimal WOSM size & seasonal snow cover
    opt_idx = int(pdDF_OptSnow.RMSE.idxmin())
    
    SSC_opt = int(pdDF_OptSnow.loc[opt_idx,'SSC'])
    
    WOSM_opt = WOSM_rad[opt_idx]
    
    RMSE_Opt = pdDF_OptSnow.RMSE.min()
    
    R2_Opt = pdDF_OptSnow.R2[opt_idx]
    
    alpha_opt = pdDF_OptSnow.alpha[opt_idx]
    
    beta_opt = pdDF_OptSnow.beta[opt_idx]
    
    OptElev = allgrouped[opt_idx].iloc[SSC_opt,1:] # get mean seasonal snow altitude for WOSM_opt and SSC_opt
    
    x = OptElev.values.reshape((-1,1)) # Predict MB over all years
    # x = OptElev.values.reshape((-1,1))[years.astype(int)] # predict over years with enough data only
    
    calper = pdSeries_ycal.index.values.astype(int)
    xcal = OptElev.loc[calper].values.reshape((-1, 1))
        
    # LR minimizing RMSE
    model = LinearRegression().fit(xcal,pdSeries_ycal)
    
    MB_AVHRR = pd.Series(model.predict(x),index=OptElev.index)
    
    return(WOSM_opt,SSC_opt,alpha_opt,beta_opt,RMSE_Opt,R2_Opt,MB_AVHRR)

# ============================================================================

def calibration_fct(allgrouped,WOSM_rad,pdSeries_ycal):
    """
    (deprecated) This function fits the observed MB series to AVHRR snowlines and returns the 
    collection of model scores & parameters. 

    Parameters
    ----------
    allgrouped : list of DataFrames
        The List containing all 198 DataFrames that hold the altitudinal snow cover distributions. The DataFrame 
        has years as columns and rows are the snow cover classes from 0-100 % snow cover.
    WOSM_rad : np.array
        Array of all WOSM radii (in m).
    pdSeries_ycal : pd.Series
        The reference MB series that we are fitting our snowlines to. Can either be from an individual glacier with 
        incomplete series (for the CRG method) or the entire time series (for the AA method)

    Returns
    -------
    OptSnow : pd.DataFrame
        Collection of model scores and parameters. Columns: 'SSC','R2','RMSE','alpha','beta'

    """
    OptSnow = pd.DataFrame(np.nan,index =  np.arange(min_R,max_R),
                           columns = ['SSC','R2','RMSE','alpha','beta']) # DF storing optimal parameters for each WOSM radius

    calper = pdSeries_ycal.index.astype(int)
    
    for i,GrDF in enumerate(tqdm(allgrouped)):    
        R = WOSM_rad[i]
        print('\r Computing for WOSM side length %s km'%int((2*R+unit)/unit), end='') 

        # Loop Probabilities of of snow occurrence and compute linear regression observed - AVHRR
        scores = GrDF.apply(map_linReg,axis=1, result_type='expand')
        
        # Convert dtype obj to float
        scores = DFobj2float(scores,scores.columns)
        scores.columns = ['R2','RMSE','alpha','beta']
        
        # opt_idx = scores.RMSE[20:].idxmin() # optimal snow occurrence (only if > 20%) minimizing RMSE in the DataFrame 
        opt_idx = scores.RMSE.idxmin()
        
        # Store optimal snow occurrence (only if > 20%) minimizing RMSE in the DataFrame 
        OptSnow['SSC'].iloc[i] = opt_idx
        OptSnow['R2'].iloc[i] = scores.R2[opt_idx]
        OptSnow['RMSE'].iloc[i] = scores.RMSE[opt_idx]
        OptSnow['alpha'].iloc[i] = scores.alpha[opt_idx]
        OptSnow['beta'].iloc[i] = scores.beta[opt_idx]
    return OptSnow