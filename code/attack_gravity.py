#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 22:24:58 2021

@author: athanasiadis
"""


import numpy as np

def computer(att_res,global_clusters_indices,subs_ids,meaned_clusters_indices):

    """
    computer
    ______________________
    
    This is a function that computes the gravity center for each of 
    the identified gloval vectors / MIPs.
    
    Input:  
            - the indices of the contributors for each of the global
              and local clusters.
            - the adversarial data from each of the subsamplings
            
    Output: 
            - a gravity center per identified global vector/ MIP.

    """    
    
    # Initialize a list that will hold the centers of gravity per global vector
    gravity_centers = []

    # Loop over the global vectors
    for nvec in range(len(global_clusters_indices)):
        
        # Grab the indices of the local normals used for this cluster
        global_indices_i = global_clusters_indices[nvec]
        
        # Initialize a list that will hold the adversarial data
        adversarials = []
        
        # Loop over the local normals used 
        for nloc in global_indices_i:
            
            # Grab the subsampling id
            sub_id = subs_ids[nloc]
            # Create the subsampling key
            subsampling_key = 'Subsampling_{}'.format(sub_id)
            # Grab the indices of eigenvectors that created this local normal vector
            eig_inds = meaned_clusters_indices[nloc]
            # Grab the indices of the adversarial starting point for each eigenvector
            usefull_inds = att_res[subsampling_key]['usefull_indices']
            nvalues      = att_res[subsampling_key]['useless_features']
            # Find out if any features were deleted
            deleted_features = [0 if x==1 else 1 for x in nvalues]            
            # Grab the adversarials for the respective subsampling
            adversarials_i = att_res[subsampling_key]['adversarial_data']
            # Fill in the missing dimensions
            dummy_adversarials = []
            indexer = 0
            for ndim in range(len(nvalues)):
                if deleted_features[ndim]==1:
                    dummy_adversarials_i = adversarials_i[:,indexer]
                    dummy_adversarials_i = np.reshape(dummy_adversarials_i,(len(dummy_adversarials_i),1))
                    indexer = indexer + 1        
                else:
                    dummy_adversarials_i = np.zeros((adversarials_i.shape[0],1))
                dummy_adversarials.append(dummy_adversarials_i)
            adversarials_i = np.concatenate(dummy_adversarials,axis=1)  
            
            
            
            # Loop over the eigenvectors that created the respective local normal vector
            for neig in eig_inds:

                # Grab the center id for each eigenvector
                center_id = usefull_inds[neig]

                # Grab the center datapoint for the respective eigenvector
                center_i = adversarials_i[center_id,:]
                center_i = np.reshape(center_i, (1,len(center_i)))
                
                # Store the adversarials / centers for this global vector
                adversarials.append(center_i)
                
        adversarials = np.concatenate(adversarials,axis=0)
        
        # Create the center of gravity
        center_gravity_i = np.mean(adversarials,axis=0)
        center_gravity_i = np.reshape(center_gravity_i,(1,len(center_gravity_i)))
        # Store it 
        gravity_centers.append(center_gravity_i)


    return gravity_centers