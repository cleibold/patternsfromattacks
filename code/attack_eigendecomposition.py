#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 21:13:59 2021

@author: athanasiadis
"""



import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import mkl



def eigendecomposition(cov_mat):

    """
    eigendecomposition
    ______________________
    
    This is a function that performs eigendecomposition on 
    the covariance matrix to identify the eigenvector with 
    the lowest eigenvalue.
    
    Input:  
            - a covariance matrix.

            
    Output: 
            - the eigenvector with the min eigenvalue.

    """            

    # Compute the weight vectors by solving eigenvalue/eigenvector problem (min eigenvalue) : eigenvector=weight vector
    covariance_matrix = cov_mat
    
    # Perform the eigendecomposition
    value,vector = np.linalg.eig(covariance_matrix)

    # Get the index of the minimum eigenvalue
    #value_min_ind = np.argmin(np.abs(value))
    value_min_ind = np.argmin(value)    
    
    # Get the eigenvector that corresponds to the minimum eigenvalue
    vector_min = vector[:,value_min_ind]
    # Get the real part from each chosen eigenvector and group them
    vector_min = [c.real for c in vector_min]

    # Create list that holds all the eigenvectors for further analysis
    vector_min = np.reshape(vector_min,(1,len(vector_min)))
    
    return vector_min

def computer(ind,adversarial_data,num_cores):

    """
    computer
    ______________________
    
    This is a function that creates a distance covariance matrix 
    for each of the adversarial patterns and then performs 
    eigendecomposition on this matrix to identify the eigenvector
    with the min eigenvalue. This eigenvector would be the 
    one that is normal to the trained models decision boundary 
    since it would represent the direction with the lowest 
    data variance.
    
    Input:  
            - a numpy array holding the adversarial patterns.
            - a list of indices per adversarial pattern. 
              These indices declare the neighbors that belong within the ring shaped domain.
            
    Output: 
            - an eigenvector per adversarial pattern.
            - the indices of the patterns for which we are able to compute an eigenvector.
              This information is usefull for computing the gravity centers onwards.
    """        
    mkl.set_num_threads(num_cores) 
    # Initiallize dictionary A which holds the covariance matrix per datapoint
    Amat = {}
    indexmat = 0
    usefull_indices = []
    # Compute a covariance matrix per datapoint from which the eigenvector will be extracted    
    for n_data in range(adversarial_data.shape[0]):
        # Check if any neighboors exist in the ring       
        if len(ind[n_data])!=0:
            # Set the dictionary key
            covariance_key = '{}{}'.format('Covariance_Datapoint_',indexmat+1) 
            indexmat = indexmat + 1
            # Distance Covariance matrix : n_usefull_neighboorsxdimensions per datapoint
            # For all neighboors included in the local neighborhood of datapoint di
            # Compute the difference di-dj (one matrix dij per datapoint) 
            dij_matrix = adversarial_data[n_data,:] -  adversarial_data[ind[n_data][:],:]
            # Grab the covariance matrix = dij.T x dij
            covariance_matrix = np.matmul(np.transpose(dij_matrix),dij_matrix)
            # Hold all covariance matrices
            Amat[covariance_key] = covariance_matrix
            # Hold the datapoint id for which we compute a local normal
            usefull_indices.append(n_data)
            
            
    # Initiallize the list which will hold the eigenvectos
    eigenvectors = []
    # Compute the eigenvector per datapoint
    if len(Amat)!=0:
        # Set the parallel runs to be equal to the amount of adversarial datapoints and split them in all available cores of the server
        runs = range(len(Amat))    
        eigenvectors = Parallel(n_jobs=num_cores)(delayed(eigendecomposition)(Amat['Covariance_Datapoint_{}'.format(n_data+1)]) for n_data in runs)

    return eigenvectors,usefull_indices