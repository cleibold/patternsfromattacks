#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 14:45:59 2021

@author: athanasiadis
"""

import numpy as np


def computer(x_data,global_vectors,gravity_centers):

    """
    computer
    ______________________
    
    This is a function that corrects the input dataset based on the 
    gravity center for each identified global vector/ MIP. Then 
    the projection of the MIP with the corrected dataset is computed.
    
    Input:  
            - the dataset.
            - the identified global vector/ MIPs.
            - the gravity centers.
            
    Output: 
            - the projections of the MIPs to the dataset.

    """    
    
    # Initiallize the array that will hold the projection vectors
    q_vectors = []
    n_vectors = len(global_vectors)
    for nvec in range(n_vectors):    
        
        # Grab the projection and the global vector
        global_vector_i = global_vectors[nvec]
        center_i = gravity_centers[nvec]
        
        # Corrected  the data
        corrected_data = []
        x_data_i = np.copy(x_data)
        for i in range(x_data_i.shape[0]):
            data_i = x_data_i[i,:]
            data_i = np.reshape(data_i,(1,len(data_i)))
            data_i = data_i - center_i
            data_i = data_i /np.linalg.norm(data_i)
            corrected_data.append(data_i)
        corrected_data = np.concatenate(corrected_data,axis=0)
    
        # Grab the projection       
        projection_i = np.dot(corrected_data,global_vector_i.T)                         
        # Store the projections
        q_vectors.append(projection_i)
    
    return  q_vectors