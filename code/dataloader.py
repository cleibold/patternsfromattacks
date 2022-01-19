#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 14:33:28 2021

@author: athanasiadis
"""

import numpy as np
import pickle

def loader(mainpath,dataname):

    """
    loader
    ______________________
    
    This is a function that loads and prepares the desired dataset 
    for the permutation test.
    
    Input:  
            - the main path and the name of the dataset
    
            - the dataset SHOULD be a python dictionary
              with the following entries:
                  - ['data']   a list of numpy arrays (nxD)
                  - ['labels']  a list of integer values                 
            
    Output: 
            - the dataset as a numpy array (N,D)
                  - N : patterns #
                  - D : dimensions #
            
            - the dataset labels as a numpy array (N,)
    
            - the label of each trial as a list (T)
                  - T : trials #
              
            - the contribution of each trial 
              (meaning the patterns #) as a list (T)
              
    """
    
    # Load the python dictionary holding the dataset/information
    with open(mainpath+dataname, 'rb') as f:
        dataset = pickle.load(f)
    data   = dataset['data']
    labels = dataset['labels']
    
    # Initiallize the lists that will hold the 
    x_data              = []
    y_data              = []
    trial_labels        = []
    trial_contributions = []
    # Loop over the data trials
    ntrials = len(data)     
    for trial_i in range(ntrials):
        
        # Get the data and the label of the trial
        data_i  = data[trial_i]
        label_i = labels[trial_i] 
        # Get the contribution of the trial
        contribution_i = data_i.shape[0]
        # Create the label trials
        labels_dummy     = np.zeros((contribution_i,))
        labels_dummy[:,] = label_i
        # Stack the data and the labels
        x_data.append(data_i)
        y_data.append(labels_dummy)
        trial_contributions.append(contribution_i)
        trial_labels.append(label_i)
    
    # Turn the lists into numpy arrays
    x_data   = np.concatenate(x_data,axis=0)
    y_data   = np.concatenate(y_data,axis=0)
    # Adjust the two labels into 0 and 1 which will be used by the model
    values   = np.unique(y_data)
    for ind,nel in enumerate(y_data):
        if nel==values[0]:
            y_data[ind] = 0
        else:
            y_data[ind] = 1
    for ind,nel in enumerate(trial_labels):
        if nel==values[0]:
            trial_labels[ind] = 0
        else:
            trial_labels[ind] = 1
            
    return x_data,y_data, trial_labels, trial_contributions


def pv_loader(mainpath,dataname):

    """
    pv_loader
    ______________________
    
    This is a function that loads the calculated pvalue 
    from the permutation test.
    
    Input:  
            - the main path and the name of the dataset
                           
    Output: 
            - a float between 0 and 1 
            
    """
   
    loadpath = mainpath + '/{}_results/data_info/'.format(dataname) + 'permutation_test_results'    
    # Load the python dictionary holding the dataset/information
    with open(loadpath, 'rb') as f:
        sig_res = pickle.load(f)

    pvalue = sig_res['permutation_results']['pvalue']  

    return pvalue

def attack_loader(mainpath,dataname):

    """
    attack_loader
    ______________________
    
    This is a function that loads the adversarial classifier results.
    
    Input:  
            - the main path and the name of the dataset
                           
    Output: 
            - a python dictionary holding the adversarial classifier results
            
    """
   
    loadpath = mainpath + '/{}_results/data_info/'.format(dataname) + 'attack_results'    
    # Load the python dictionary holding the dataset/information
    with open(loadpath, 'rb') as f:
        att_res = pickle.load(f)

    return att_res