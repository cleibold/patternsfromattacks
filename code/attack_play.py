#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 15:36:51 2021

@author: athanasiadis
"""

"""
attack
______________________

This is a script that runs the adversarial decoder on the dataset of choise.

Input:  
        - several functions needed for the adversarial decoder.
            
Output: - python dictionary holding the results of the adversarial decoding
          with the following entries : 
                
                - The indices of the data used in the training/testing set per subsampling. Also the testing set predictions and CCR values.
                  The adversarial patterns and labels per subsampling. 
                 
                - The identified global vectors / MIPs from the decoding process. The gravity center per MIP as well as it's projection to the
                  corrected original dataset.
                
                - The "amount" and "consistency" metric which describe the dominance of each identified MIP.
                
"""

# ================================================================== #
#       ____Imports_____                                             #
# ================================================================== #

import time
start_time = time.time() 
import warnings
import pathlib
warnings.filterwarnings('ignore')
import os

    
# ================================================================== #
#       ____Functions_____                                           #
# ================================================================== #

import timer as tm
import dataloader as dtl
import attack_main as adv

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
model_flag   = parameters['model id']
lr           = float(parameters['learning rate'])
epochs       = int(parameters['epochs #'])
ring         = parameters['ring domain']
ring         = ring[1:-1].split(',')
ring_start   = int(ring[0])
ring_stop    = int(ring[1]) 
clustering   = parameters['clustering id']
clustering   = clustering[1:-1].split(',')
min_samples  = int(clustering[0])
epsilon      = float(clustering[1])
myfile.close()
# Load the permutation test calculated pvalue for the specific dataset
pvalue = dtl.pv_loader(mainpath,dataname)

# Skip the session if the p value exceeds the 5th percentile
if pvalue>=0.5:
    raise ValueError('p-value below significance threshold. Try a different dataset.')
else:   

    # Load the data 
    x_data,y_labels, _, _ = dtl.loader(mainpath,dataname)

    # Set the saving path and name
    attackinfo =  mainpath + '/{}_results/data_info/'.format(dataname)
    modelinfo  =  mainpath + '/{}_results/model_info/'.format(dataname)
    pathlib.Path(modelinfo).mkdir(parents=True, exist_ok=True)
    pathlib.Path(attackinfo).mkdir(parents=True, exist_ok=True)
                  
    # Check if there are enough patterns per label to start the process
    ones = len([1 for x in y_labels if x==1])
    zeros = len(y_labels) - ones
    # Skip the session if there is only one active cell left or not enough patterns
    if x_data.shape[1]<=1 or ones<=2 or zeros<=2:
        raise ValueError('Not enough patterns per label or not enough dimensions. Try a different dataset.')
    else:                 

        # Initiallize a dictionary that will hold the results from the subsamplings
        att_res = {}
        # Store the needed parameters for the adversarial classifier
        att_res['parameters'] = {}
        att_res['parameters']['model_attack_savepaths'] =[modelinfo,attackinfo]
        att_res['parameters']['nsubsamplings']         = subsamplings
        att_res['parameters']['model_flag']            = model_flag
        att_res['parameters']['lr_epochs']             = [lr,epochs]
        att_res['parameters']['ring_domain']           = [ring_start,ring_stop]
        att_res['parameters']['min_samples_epsilon']   = [min_samples,epsilon]
        #att_res['parameters']['consistency_threshold'] = consistency_threshold
        # Store initialliy needed data
        att_res['original_data'] = x_data
        att_res['original_labels'] = y_labels
        att_res['computation_cores'] = setcores

         
        # Use the adversarial classifier on the data
        adv.clf(att_res)

# Print eclapsed time 
tm.timer(start_time)
