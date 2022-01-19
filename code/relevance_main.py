#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 14:14:57 2022

@author: athanasiadis
"""

import numpy as np
import pickle
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

import relevance_id as idf


def main(att_res):

    """
    main
    ______________________
    
    This is the main relevance test function that calls all helper functions. 
    
    Input:  
            - several functions needed for the relevance test.
                
    Output: 
            - python dictionary holding the analysis results.              
                  
    """    
    
    # Unpack the classifier parameters
    perc_up   = att_res['relevance']['parameters']['confidence_interval'][0]
    perc_down = att_res['relevance']['parameters']['confidence_interval'][1]
    setcores = att_res['relevance']['parameters']['num_cores']
    # Check if the number of physical cores is acceptable
    num_cores = multiprocessing.cpu_count()
    if setcores==-1:
        num_cores = num_cores
    else:
        if setcores<=num_cores:
            num_cores = setcores
        else:
            raise ValueError('Not enough physical cores available. Try agian with a different amount.')

    # Unpack the entries needed from the analysis dictionary
    projections     = att_res['projections']
    global_vectors  = att_res['global_vectors']         
    x_data          = att_res['original_data']
    gravity_centers = att_res['gravity_centers']       
    attackinfo      = att_res['parameters']['model_attack_savepaths'][1]

    dimensions      = x_data.shape[1]

    # Initiallize a dictionary tha will hold the relevance results 
    rel_res = {}
              
    # Loop over the identified MIPs that exceed the consistency threshold
    nvectors = len(global_vectors)
    for nvec in range(nvectors):
                       
        # Grab the projection and the global vector
        global_vector_i = global_vectors[nvec]
        center_i        = gravity_centers[nvec]            
        projection_i    = projections[nvec]
        
        # Grab the corrected data
        corrected_data = []
        for i in range(x_data.shape[0]):
            data_i = x_data[i,:]
            data_i = np.reshape(data_i,(1,len(data_i)))
            data_i = data_i - center_i
            data_i = data_i /np.linalg.norm(data_i)
            corrected_data.append(data_i)
        corrected_data = np.concatenate(corrected_data,axis=0)
                                                 

        # Shuffle the indices of the cells 1000 times
        nshuffles, perms = idf.shuffle(dimensions)
        if nshuffles>10:
            # Compute 1000 shuffled projections, while shuffling the order of cells in the corrected data
            runs = range(0,nshuffles)    
            shuffled_projection = Parallel(n_jobs=num_cores,timeout=None)(delayed(idf.dot)(corrected_data,perms[nsh],global_vector_i) for nsh in runs)
            # Identify the relevant datapoints indices
            relevant_indices,irelevant_indices,relevant_plus_indices,relevant_minus_indices = idf.find(shuffled_projection,runs,projection_i,perc_up,perc_down)
    
        # Innitiallize an entry per pattern and store the indices found to be relevant
        patternkey = 'pattern_{}'.format(nvec+1)
        rel_res[patternkey] = {}
        rel_res[patternkey]['relevant_indices']       = relevant_indices
        rel_res[patternkey]['irelevant_indices']      = irelevant_indices
        rel_res[patternkey]['plus_relevant_indices']  = relevant_plus_indices
        rel_res[patternkey]['minus_relevant_indices'] = relevant_minus_indices
            
    # Store the relevance test results
    att_res['relevance']['results'] = rel_res    

    # Save 
    att_res_name =  attackinfo   +'analysis_results'           
    with open(att_res_name,'wb') as f:
        pickle.dump(att_res, f) 
    
    return 