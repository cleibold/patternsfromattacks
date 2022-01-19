#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 14:53:06 2022

@author: athanasiadis
"""

import numpy as np


def computer(sig_res):

    """
    computer
    ______________________
    
    This is a function that computes the results of the permutation test.
    
    Input:  
            - the dictionary holding the results of the permutation test.
                
    Output: 
            - the same dictionary with 3 new entries : the pvalue, 
              the z-scored ccr value and the z-scored ccr bin vector.
               
                  
    """
    
    # Get the real mean ccr
    real_ccr = np.round(np.mean(sig_res['real_data']['ccrs']),3)                  
    real_bin_ccr = sig_res['real_data']['bin_ccrs']
    # Get the shuffle crrs distribution
    shuffle_bin_ccr = []
    shuffle_ccr = []
    shuffles = len(sig_res) - 2
    for nsh in range(1,shuffles+1):
        sh_key = 'shuffle_{}'.format(nsh)
        dummy_real_ccrs = np.round(np.mean(sig_res[sh_key]['ccrs']),3)
        shuffle_ccr.append(dummy_real_ccrs)     
        shuffle_bin_ccr.append(sig_res[sh_key]['bin_ccrs'])
    shuffle_bin_ccr = np.concatenate(shuffle_bin_ccr,axis=0)

    # Get the zccr per bin
    bin_std  = np.std(shuffle_bin_ccr,axis=0)
    bin_mean = np.mean(shuffle_bin_ccr,axis=0)
    bin_std  = np.reshape(bin_std,(1,len(bin_std)))
    bin_mean  = np.reshape(bin_mean,(1,len(bin_mean)))
    binzccr  =  ( real_bin_ccr - bin_mean ) / bin_std
    binzccr  = [x for x in binzccr[0,:] if np.isnan(x)==False]
    
    # Compute the pvalue
    pvalue = len([1 for x in shuffle_ccr if real_ccr<x]) / len(shuffle_ccr)
    
    # Get the z scored ccr value                
    mean = np.mean(shuffle_ccr)
    std  = np.std(shuffle_ccr)                                        
    if std!=0:
        zccr = ( real_ccr - mean )/ std
    else:
        zccr = np.nan
    
    sig_res['permutation_results'] = {}
    sig_res['permutation_results']['pvalue']   = pvalue
    sig_res['permutation_results']['zccr']     = zccr
    sig_res['permutation_results']['binszccr'] = binzccr
    
    return sig_res