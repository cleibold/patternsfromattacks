#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 10:22:38 2021

@author: athanasiadis
"""

import numpy as np
import pickle
from joblib import Parallel, delayed
import multiprocessing
import mkl
num_cores = multiprocessing.cpu_count()

import attack_modelfunctions as md
import attack_neighboorhood as neigh
import attack_eigendecomposition as eigde
import attack_inference as infer
import attack_clustering as cl
import attack_unite_clusters as unt
import attack_projections as proj
import attack_gravity as gravity
#import attack_ica as icadec



def clf(att_res):

    """
    clf
    ______________________
    
    This is the main adversarial decoder function that calls all helper functions. 
    
    Input:  
            - several functions needed for the adversarial decoding.
                
    Output: 
            - python dictionary holding the decoding results.              
                  
    """    

    # Unpack the classifier parameters
    modelinfo    = att_res['parameters']['model_attack_savepaths'][0]
    attackinfo   = att_res['parameters']['model_attack_savepaths'][1]
    subsamplings = att_res['parameters']['nsubsamplings']
    model_flag   = att_res['parameters']['model_flag'] 
    lr           = att_res['parameters']['lr_epochs'][0]
    epochs       = att_res['parameters']['lr_epochs'][1]
    ring_start   = att_res['parameters']['ring_domain'][0]
    ring_stop    = att_res['parameters']['ring_domain'][1]
    min_samples  = att_res['parameters']['min_samples_epsilon'][0]
    epsilon      = att_res['parameters']['min_samples_epsilon'][1]
    setcores     = att_res['computation_cores']
    #consistency_threshold = att_res['parameters']['consistency_threshold']
    x_data   = att_res['original_data'] 
    y_labels = att_res['original_labels']
    dim_i    = x_data.shape[1] 

    # Split the shuffles into the physical computational cores
    runs = range(1,subsamplings+1)    
    num_cores = multiprocessing.cpu_count()
    if setcores==-1:
        num_cores = num_cores
    else:
        if setcores<=num_cores:
            num_cores = setcores
        else:
            raise ValueError('Not enough physical cores available. Try agian with a different amount.')
    # Call the training function
    att_res_dummy = Parallel(n_jobs=num_cores,timeout=None)(delayed(md.train_test_model)(np.copy(x_data),np.copy(y_labels),dim_i,lr,epochs,nsub,modelinfo,model_flag) for nsub in runs)    

    # Retrieve/reshape data form the joblib output            
    for n_subs in range(subsamplings):
        att_res['Subsampling_{}'.format(n_subs+1)] = {}           
        att_res['Subsampling_{}'.format(n_subs+1)] = att_res_dummy[n_subs]['Subsampling_{}'.format(n_subs+1)]     
        # Stack the weights from each subsampling if the model was linear
        if model_flag=='linear':
            weights_i = att_res_dummy[n_subs]['Subsampling_{}'.format(n_subs+1)]['weights']
            weights_i = np.reshape(weights_i,(1,len(weights_i)))
            if n_subs==0:
                weights = weights_i
            else:
                weights = np.concatenate((weights,weights_i),axis=0)

    # Get the mean/normalized weights over all subsamplings if the model was linear
    if model_flag=='linear':
        weights = np.mean(weights,axis=0)
        weights = weights / np.linalg.norm(weights)
        att_res['weights'] = weights
    
    # Save 
    att_res_name =  attackinfo   +'analysis_results'           
    with open(att_res_name,'wb') as f:
        pickle.dump(att_res, f)


    # For each subsampling grab the adversarial data
    for nsub in range(1,subsamplings+1):                          
        # Get subsampling key
        subsampling_key  = 'Subsampling_{}'.format(nsub)
        adversarial_data = att_res[subsampling_key]['adversarial_data']
        nvalues          = att_res[subsampling_key]['useless_features']
        
        # Compute a local neighborhood for each adversarial datapoint 
        ind = neigh.computer(adversarial_data,ring_start,ring_stop,dim_i,num_cores)
  
        # Compute the weight vectors from the attack                                   
        eigenvectors,usefull_indices = eigde.computer(ind,adversarial_data,num_cores)
        
        # Make sure at least a few local normal vectors are computed
        if len(eigenvectors)!=0:           
             
            # Infer the labels of the local normal vectors in order to adjust for their direction
            eigenvectors = infer.computer_local(eigenvectors,modelinfo,nsub,nvalues,model_flag)

            # Cluster the resulting weight vectors from the attack
            local_clusters_data = cl.computer_local(eigenvectors,min_samples,epsilon,nvalues)      

            # Store the clusters data on the attack dataset
            att_res[subsampling_key]['local_clustering_data'] = local_clusters_data
            att_res[subsampling_key]['usefull_indices']       = usefull_indices
    
    # Save 
    att_res_name =  attackinfo   +'analysis_results'           
    with open(att_res_name,'wb') as f:
        pickle.dump(att_res, f)   

    # Combine the results from all subsamplings and get the mean and the amount value per cluster
    meaned_clusters, amount, subs_ids,meaned_clusters_indices = unt.computer(subsamplings,att_res)

    # Train and flip the labels of the global vectors 
    meaned_clusters = infer.computer_global(meaned_clusters,x_data,y_labels,dim_i,lr,epochs,modelinfo,model_flag)
    
    # Cluster the mean resultant vectors and compute total amount of local weight vectors and consistency value
    global_vectors, amount, consistency,global_clusters_indices = cl.computer_global(meaned_clusters,amount,subsamplings,subs_ids,min_samples,epsilon) 
    
    # Unpdate the dictionary that holds the data from the second attack
    att_res['global_vectors']         = global_vectors
    att_res['global_vectors_indices'] = global_clusters_indices
    att_res['local_sub_ids']          = subs_ids
    att_res['local_vectors_indices']  = meaned_clusters_indices
    att_res['consistency']            = consistency
    att_res['amount']                 = amount
    
    # Save 
    att_res_name =  attackinfo   +'analysis_results'           
    with open(att_res_name,'wb') as f:
        pickle.dump(att_res, f)  

    # Compute the gravity centers for each of the identified global vectors
    gravity_centers = gravity.computer(att_res,global_clusters_indices,subs_ids,meaned_clusters_indices)
    att_res['gravity_centers'] = gravity_centers

    # Compute the projections vectors q 
    projections = proj.computer(x_data,global_vectors,gravity_centers)
    att_res['projections'] = projections

    # Save 
    att_res_name =  attackinfo   +'analysis_results'           
    with open(att_res_name,'wb') as f:
        pickle.dump(att_res, f) 

    return 
