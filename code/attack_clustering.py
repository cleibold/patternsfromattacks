#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 21:39:56 2021

@author: athanasiadis
"""

from sklearn.cluster import DBSCAN
import numpy as np

def computer_local(eigenvectors,min_samples,epsilon,nvalues):

    """
    computer_local
    ______________________
    
    This is a function that clusters the identified eigenvectors
    based on their direction in space. We use DBSCAN as our 
    unsupervised learning clustering algorithm.
    
    Input:  
            - the identified eigenvectors.
            - the parameters of the clustering needed for DBSCAN. 
            
    Output: 
            - a dictionary holding the clustered data as well as 
              the eigenvector indices for each of the identified
              clusters.

    """           

    # Find out if any features were deleted
    deleted_features = [0 if x==1 else 1 for x in nvalues]
    # Find out the original dimensions
    dimensions = len(nvalues)   
    
    # Set epsilon to be equal to ...
    epsilon = epsilon
    # Set min_samples to be equal to 5% of the dataset size
    min_samples = int((min_samples/100)*eigenvectors.shape[0])   
    flag_p = False
    while flag_p==False:
    
        clustering = DBSCAN(eps=epsilon,min_samples=min_samples,metric='euclidean',n_jobs=-1).fit(eigenvectors)
        predictions = clustering.labels_    

        # Check if the new clustering gives more or less clusters than the old one
        n_clusters = max(predictions) + 1
        if n_clusters>=1:
            flag_p = True
        else:
            epsilon = epsilon + 0.1


    # Initially a dictionary to hold the separated clusters data
    local_clusters_data = {}        
    # Loop over all the identified cluster IDs (= labels) a
    for nid in range(0,n_clusters):
        # Grab the indices for the ID
        indices = [ind for ind,x in enumerate(predictions) if x==nid]
        # Grab the local normals for these indices
        local_normals = np.copy(eigenvectors[indices,:])
        # Restore the missing "useless dimensions"
        dummy_local_normals = []   
        # Fill in the missing dimensions
        indexer = 0
        for ndim in range(dimensions):
            if deleted_features[ndim]==1:
                dummy_local_normals_i = local_normals[:,indexer]
                dummy_local_normals_i = np.reshape(dummy_local_normals_i,(len(dummy_local_normals_i),1))
                indexer = indexer + 1        
            else:
                dummy_local_normals_i = np.zeros((local_normals.shape[0],1))
            dummy_local_normals.append(dummy_local_normals_i)
        local_normals = np.concatenate(dummy_local_normals,axis=1)        
        # Store the local normals in the dictionary
        cluster_key = 'Cluster_{}'.format(nid+1)
        local_clusters_data[cluster_key] = {} 
        # Save the local normals and the indices that lead to the clustering
        local_clusters_data[cluster_key]['data'] = local_normals   
        local_clusters_data[cluster_key]['indices'] = indices   
                           
    return local_clusters_data


def computer_global(meaned_clusters,local_amount,subsamplings,subs_ids,min_samples,epsilon):

    """
    computer_global
    ______________________
    
    This is a function that clusters the averaged patterns
    that are produced from separate subsamplings. The clustering
    is based on their direction in space. We use DBSCAN as our 
    unsupervised learning clustering algorithm.
    
    Input:  
            - the averaged patterns.
            - the parameters of the clustering needed for DBSCAN.
            - the "amount" metric.
            
    Output: 
            - a dictionary holding the clustered data as well as 
              the averaged pattern indices for each of the identified
              clusters.
            - The new "amount" metric based on the global clustering.  
            - The "consistency" metric which tells how many of the 
              subsamplings contribute averaged patterns to each of the
              identified clusters.

    """           
        
    # Set epsilon to be equal to ...
    epsilon = epsilon
    # Set min_samples to be equal to 5% of the dataset size
    min_samples = int((min_samples/100)*meaned_clusters.shape[0])   
    flag_p = False
    while flag_p==False:
    
        clustering = DBSCAN(eps=epsilon,min_samples=min_samples,metric='euclidean',n_jobs=-1).fit(meaned_clusters)
        predictions = clustering.labels_    

        # Check if the new clustering gives more or less clusters than the old one
        n_clusters = max(predictions) + 1
        if n_clusters>=1:
            flag_p = True
        else:
            epsilon = epsilon + 0.1

    # Identify what is the signal/noise 
    #noise = [ind for ind,x in enumerate(predictions) if x==-1]
    #signal = len(predictions)-len(noise)
    # Find all indices that are not -1 in the predictions and then sum the local amounts values
    amount_indices = [ind for ind,x in enumerate(predictions) if x!=-1]
    amount_signal = [local_amount[x] for x in amount_indices]
    amount_signal = np.sum(amount_signal)
    # Get the subsampling IDs from which actual data is clustered
    subs_used = [subs_ids[x] for x in amount_indices]
    # ID how many are the subsamplings used
    nsubs_used = len(np.unique(subs_used))
    nsubs_used = 100

    # Initiallize two lists one for the amount values and one for the consistency
    amount = []
    consistency = []
    # Initially a dictionary to hold the separated clusters data
    global_clusters_data = []
    global_clusters_indices = []
    # Loop over all the identified cluster IDs (= labels) a
    for nid in range(0,n_clusters):
        # Grab the indices for the ID
        indices = [ind for ind,x in enumerate(predictions) if x==nid]
        # Grab the amount values per meaned cluster
        amount_i = [local_amount[x] for x in indices]
        amount_i = np.sum(amount_i)/amount_signal
        amount.append(np.round(100*amount_i,2))
        # Grab the consistency values per meaned cluster
        consistency_i = [subs_ids[x] for x in indices]
        consistency_i = len(np.unique(consistency_i))
        consistency_i = len(indices)
        consistency_i = consistency_i/nsubs_used
        if consistency_i>1:
            consistency_i = 1
        consistency.append(np.round(100*consistency_i,2))
                
        # Grab the global normals for these indices
        global_normals = np.copy(meaned_clusters[indices,:])
        global_normals = np.mean(global_normals, axis=0)
        global_normals = np.reshape(global_normals,(1,len(global_normals)))        
        # Normalize the global vector to length 1
        global_normals = global_normals/np.linalg.norm(global_normals)
        # Store the global normals in a list 
        global_clusters_data.append(global_normals)   
        global_clusters_indices.append(indices)

    # Grab the sorting indices based on the consistency values
    sorting_indices = np.flip(np.argsort(consistency))
    #  Sort out the consistency, amount and global clusters based on the sorting indices
    amount = [amount[x] for x in sorting_indices]
    consistency = [consistency[x] for x in sorting_indices]
    global_clusters_data = [global_clusters_data[x] for x in sorting_indices]

    return global_clusters_data, amount, consistency, global_clusters_indices