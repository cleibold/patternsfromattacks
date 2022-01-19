#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 16:18:49 2021

@author: athanasiadis
"""

"""
significance
______________________

This is a script that runs a permutation test on a given dataset. 

Input:  
        - several functions needed for the permutation test.
            
Output: 
        - python dictionary holding the results of the test
          with the following entries : 
                
                - A list with the mean ccr value from each subsampling without shuffling.
                - A vector with a ccr value per pattern over the subsamplings without shuffling.
                - A list with the mean ccr value from each subsampling with each shuffle.
                - A vector with a ccr value per pattern over the subsamplings with each shuffle.   
                - A dictionary holding the permutation test parameters.
                - A dictionary holding the permutation test results (pvalue,zscored ccr etc).
                
"""

# ================================================================== #
#       ____Imports_____                                             #
# ================================================================== #

import time
start_time = time.time() 
import warnings
import pathlib
warnings.filterwarnings('ignore')
import pickle
import os

# ================================================================== #
#       ____Functions_____                                           #
# ================================================================== #

import timer as tm
import dataloader as dtl
import significance_main as sign

# ================================================================== #
#       ____Main_____                                                #
# ================================================================== #

# Ask for inputs about type of permutation test
paramertersname = input('Enter the parameters file name:\n')

# Import the parameters .txt file
with open (paramertersname, "r") as myfile:
    file=myfile.readlines()
    
parameters = {}
for line in file:
    line = line.strip()
    key_value = line.split(':')
    if len(key_value)==2:
        parameters[key_value[0].strip()] = key_value[1].strip()

nice_id = parameters['nice id']
if nice_id!='n':
    os.nice(int(nice_id))
    
setcores     = int(parameters['core id'])
mainpath     = parameters['main path']
dataname     = parameters['dataset name'] 
subsamplings = int(parameters['subsamplings #']) 
nshuffles    = int(parameters['shuffle #']) 
shuffletype  = parameters['shuffle id']
model_flag   = parameters['model id']
lr           = float(parameters['learning rate'])
epochs       = int(parameters['epochs #'])
myfile.close()
# Load the data 
x_data,y_labels, trial_labels, trial_contributions = dtl.loader(mainpath,dataname)
    
# Check if there are enough patterns per label to start the process
ones = len([1 for x in y_labels if x==1])
zeros = len(y_labels) - ones
# Skip the session if there is only one active cell left or not enough patterns
if x_data.shape[1]<=1 or ones<=2 or zeros<=2:
    raise ValueError('Not enough patterns per label or not enough dimensions. Try a different dataset.')
else:       
    # Initiallize a dictionary that will hold the parameters from the permutation tests
    sig_par = {}
    # Store the needed parameters for the classifier
    sig_par['parameters']                  = {}
    sig_par['parameters']['nsubsamplings'] = subsamplings
    sig_par['parameters']['model_flag']    = model_flag
    sig_par['parameters']['lr_epochs']     = [lr,epochs]
    sig_par['parameters']['nshuffles']     = nshuffles
    sig_par['parameters']['shuffle_type']  = shuffletype
    # Store initialliy needed data
    sig_par['original_data']               = x_data
    sig_par['original_labels']             = y_labels
    sig_par['trial_labels']                = trial_labels
    sig_par['trial_contributions']         = trial_contributions
    sig_par['computation_cores']           = setcores
    
    # Apply the permutation test on the dataset
    ccr_results = sign.main(sig_par)

    # Set the saving path and name
    savepath = mainpath + '/{}_results/data_info/'.format(dataname)
    pathlib.Path(savepath).mkdir(parents=True, exist_ok=True)
    savename = savepath + 'permutation_test_results'

    # Save the results that the ccr_data dictionary holds
    with open(savename,'wb') as f:
        pickle.dump(ccr_results, f)
        
# Print eclapsed time 
tm.timer(start_time)
