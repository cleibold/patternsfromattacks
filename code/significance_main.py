#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 16:37:06 2021

@author: athanasiadis
"""

import numpy as np
import time
from joblib import Parallel, delayed
import multiprocessing
import mkl
import tqdm
num_cores = multiprocessing.cpu_count()
mkl.set_num_threads(num_cores) 


import significance_shuffle_labels as shlab
import significance_modelfunctions as md
import significance_pvalue as pval

def main(sig_par):

    """
    main
    ______________________
    
    This is the main permutation test function that calls all helper functions 
    
    Input:  
            - several functions needed for the permutation test
                
    Output: 
            - python dictionary holding the results of the test               
                  
    """
    
    # Unzip the parameters for the permutation tests
    subsamplings        = sig_par['parameters']['nsubsamplings'] 
    model_flag          = sig_par['parameters']['model_flag']
    lr                  = sig_par['parameters']['lr_epochs'][0]
    epochs              = sig_par['parameters']['lr_epochs'][1] 
    nshuffles           = sig_par['parameters']['nshuffles']
    shuffletype         = sig_par['parameters']['shuffle_type']
    x_data              = sig_par['original_data']
    y_labels            = sig_par['original_labels']
    trial_labels        = sig_par['trial_labels']
    trial_contributions = sig_par['trial_contributions']
    setcores            = sig_par['computation_cores']

    # Create the labels for each permutation based on the shuffletype
    if shuffletype=='block':
        y_labels, nshuffles = shlab.block(y_labels,nshuffles,trial_labels,trial_contributions)    
    if shuffletype=='full':
        y_labels, nshuffles = shlab.full(y_labels,nshuffles)
    # Update the nshuffles dictionary
    sig_par['parameters']['nshuffles'] = nshuffles 
        
    # Grab the dimensions
    dimensions = x_data.shape[1]
    # Split the shuffles into the physical computational cores
    runs = range(0,nshuffles+1)    
    num_cores = multiprocessing.cpu_count()
    if setcores==-1:
        num_cores = num_cores
    else:
        if setcores<=num_cores:
            num_cores = setcores
        else:
            raise ValueError('Not enough physical cores available. Try agian with a different amount.')
    
    ccrs = Parallel(n_jobs=num_cores,timeout=None)(delayed(md.train_test_model)(np.copy(x_data),np.copy(y_labels[nperm]),dimensions,lr,epochs,subsamplings,model_flag) for nperm in runs)    
    
    # Initiallize the dict that will hold the CCR data
    ccr_data = {}    
    # Reshape data based on the shufle indices into a new dictionary
    for nsh in range(nshuffles+1):
        sh_key = 'shuffle_{}'.format(nsh)
        if nsh==0:
            sh_key = 'real_data'
        ccr_data[sh_key] = {}
        grab_i = ccrs[nsh]
        ccr_data[sh_key]['ccrs'] = grab_i[0]
        ccr_data[sh_key]['bin_ccrs'] = grab_i[1]
    
    # Add the parameters used in the dictionary
    ccr_data['parameters'] = {}
    ccr_data['parameters'] = sig_par['parameters']
    
    # Compute the pvalue of significance and the zscored ccr value for the dataset
    ccr_data = pval.computer(ccr_data)
     
    return ccr_data