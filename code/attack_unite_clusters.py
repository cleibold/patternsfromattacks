#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 21:44:54 2021

@author: athanasiadis
"""




import numpy as np


def computer(subsamplings,att_res):

    """
    computer
    ______________________
    
    This is a function that averages over all patterns within a cluster.
    Then it combined all the averaged patterns into a single array.
    
    Input:  
            - the clustering results.
            
    Output: 
            - array of averaged patterns.
            - the "amount" metric.
              This tells us how many eigenvectors belonged to each
              of the averaged patterns.

    """       
    
    # Create a list to hold the amount of each mean resultant weight vector
    amount = []
    meaned_clusters = []
    meaned_clusters_indices = []
    meaned_clusters_normed = []
    # Also initiallize a list that will hold the subsampling ID from which each mean cluster originates from
    subs_ids = []
    # Loop over the subsamplings
    for n_subs in range(1,subsamplings+1):      
        # Get subsampling key
        subsampling_key = 'Subsampling_{}'.format(n_subs)
        
        # Check if this subsampling had any clustering results
        keys = list(att_res[subsampling_key].keys())
        if 'local_clustering_data' in keys:
            # Get the clusters of the subsampling from the second attack for each subsampling        
            local_clusters_data = att_res[subsampling_key]['local_clustering_data']
        
            # Loop over the local clusters
            for cluster_key in local_clusters_data:
                # Get the data for the cluster
                cluster_i = local_clusters_data[cluster_key]['data']
                cluster_i_indices = local_clusters_data[cluster_key]['indices']

                
                # Count the local normals composing the cluster
                amount.append(cluster_i.shape[0])   
                # Store the subsampling ID
                subs_ids.append(n_subs)
                
                # Find a mean resultant vector per cluster     
                cluster_mean = np.reshape(np.mean(cluster_i, axis=0),(1,cluster_i.shape[1]))
                # Gather all mean resultant vectors across subsamplings                
                meaned_clusters.append(cluster_mean)
                meaned_clusters_indices.append(cluster_i_indices)
                meaned_clusters_normed.append(cluster_mean / np.linalg.norm(cluster_mean))
    meaned_clusters = np.concatenate(meaned_clusters,axis=0)
    meaned_clusters_normed = np.concatenate(meaned_clusters_normed,axis=0)

    # Initiallize an array that will check if the indice has been flipped already or not
    flip_flags    = np.zeros((meaned_clusters.shape[0],))
    flip_flags[:] = False
    
    # Loop over the meaned vectors
    for nvec in range(meaned_clusters_normed.shape[0]):
        # Check if the vector has already been flipped
        if flip_flags[nvec]==False:
            # Grab the respective vector
            vec_i = meaned_clusters_normed[nvec,:]
            vec_i = np.reshape(vec_i,(1,len(vec_i)))            
            # Compute the dot product with each of the other vectors
            dots  = np.dot(vec_i,meaned_clusters_normed.T)
            # Find which of the vector indices have a dot product close to  1 and -1
            inds  = [ind for ind,x in enumerate(dots[0,:]) if (x<-0.95) and (ind>nvec)]
            # For each of these indices check which ones are not the same vector and have not yet been flipped
            dummy_flip_flags = flip_flags[inds]
            flag_inds = [ind for ind,x in enumerate(dummy_flip_flags) if x==False]
            flip_inds = [inds[x] for x in flag_inds]
            # Loop over the indices and flip the sign in the real array
            for nflip in flip_inds:
                flip_flags[nflip] = True

    # Normalize the vectors
    flipped_clusters = []
    for neig in range(meaned_clusters.shape[0]):
        vec_i = meaned_clusters[neig,:]
        if flip_flags[neig]==True:
            vec_i = vec_i * (-1)
        vec_i = np.reshape(vec_i,(1,len(vec_i)))
        flipped_clusters.append(vec_i)
    meaned_clusters = np.concatenate(flipped_clusters,axis=0)    

     
    return meaned_clusters,amount, subs_ids,meaned_clusters_indices