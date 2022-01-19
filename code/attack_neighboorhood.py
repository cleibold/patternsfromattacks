#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 21:08:57 2021

@author: athanasiadis
"""


from sklearn.neighbors import NearestNeighbors
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import mkl



def sorter(indexer,distancer,r_start,r_stop):

    """
    sorter
    ______________________
    
    This is a function that identifies the neighboors that are included within a ring shaped domain 
    around each of the adversarial patterns.
    
    Input:  
            - the distances of all other patterns from each adversarial pattern.
            - the ring domain dimensions (start and edge).
            
    Output: 
            - a list of indices per adversarial pattern. 
              These indices declare the neighbors that belong within the ring shaped domain. 
                  
    """    
    
    keeper = [(index) for (distance,index) in zip(distancer,indexer) if distance>r_start and distance<r_stop]    
    return keeper

def computer(adversarial_data,ring_start,ring_stop,dimensions,num_cores):

    """
    computer
    ______________________
    
    This is a function that identifies the closest neighbors 
    of each adversarial pattern and consequtivelly forms a ring
    space domaim of specified dimensions around each of 
    the adversarial patterns.
    
    Input:  
            - a numpy array holding the adversarial patterns.
            - the ring domain dimensions (start and edge).
            - the dataset dimensionality.
            
    Output: 
            - a list of indices per adversarial pattern. 
              These indices declare the neighbors that belong within the ring shaped domain. 
                  
    """    

    # Locate Closest neighboors for each datapoint    
    neigh = NearestNeighbors(n_neighbors=adversarial_data.shape[0],n_jobs=-1)
    neigh.fit(adversarial_data)
    dist,ind = neigh.kneighbors(adversarial_data, adversarial_data.shape[0], return_distance=True)
          
    # Find the closest neighbor with a non-zero distance for each datapoint
    min_dist = []
    max_dist = []
    for n_data in range(len(dist)):
        dist_i = dist[n_data]
        min_dist.append(next((value for index,value in enumerate(dist_i) if value != 0), 0))
        max_dist.append(dist_i[-1]) 
    
    # Turn into array n_datapointsx1
    min_dist = np.reshape(min_dist,(len(min_dist),1))
    # Turn into array n_datapointsx1    
    max_dist = np.reshape(max_dist,(len(max_dist),1))
    
    # Get the difference between min and max distance per datapoint
    delta_dist =  max_dist[:] - min_dist[:]
    
    # Decide how big the ring shaped local neighboorhood will be as a percentage of distance difference
    r_start = min_dist[:] + (delta_dist[:]*ring_start)/100 
    r_stop  = min_dist[:] + (delta_dist[:]*ring_stop)/100     
        
    # Set the parallel runs to be equal to the amount of adversarial datapoints and split them in all availabel cores of the server
    mkl.set_num_threads(num_cores) 
    runs = range(adversarial_data.shape[0])    
    keeper = Parallel(n_jobs=num_cores)(delayed(sorter)(ind[n_data],dist[n_data],r_start[n_data],r_stop[n_data]) for n_data in runs)

    return keeper