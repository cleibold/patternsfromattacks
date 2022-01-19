#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 12:03:01 2021

@author: athanasiadis
"""

"""
relevance
______________________

This is a script that identifies the highly task relevant patterns.

Input:  
        - several functions needed for the relevance test and the results of the adversarial decoder.
            
Output: - python dictionary holding the results of the adversarial decoding
          with addition to the identified relevant indices per pattern : 
                
            - lists containing the indices of the patterns found to be relevant, irelevant
            and plus/minus relevant (this indicates the relevance towards a specific class).
                
"""

# ================================================================== #
#       ____Imports_____                                             #
# ================================================================== #

import os
import time
start_time = time.time() 
import warnings
warnings.filterwarnings('ignore')

# ================================================================== #
#       ____Functions_____                                           #
# ================================================================== #

import timer as tm
import relevance_main as rlv
import dataloader as dtl

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

setcores            = int(parameters['core id'])
mainpath            = parameters['main path']
dataname            = parameters['dataset name'] 
confidence_interval = parameters['conficence interval']
confidence_interval = confidence_interval[1:-1].split(',')
perc_up             = float(confidence_interval[0])
perc_down           = float(confidence_interval[1]) 

myfile.close()
# Load the permutation test calculated pvalue for the specific dataset
pvalue  = dtl.pv_loader(mainpath,dataname)


# Skip the session if the p value exceeds the 5th percentile
if pvalue>=0.5:
    raise ValueError('p-value below significance threshold. Try a different dataset.')
else:   
    
    # Load the adversarial classifier results 
    att_res = dtl.attack_loader(mainpath,dataname)    

    # Store the relevance test parameters
    att_res['relevance'] = {}
    att_res['relevance']['parameters'] = {}
    att_res['relevance']['parameters']['confidence_interval'] = [perc_down,perc_up]
    att_res['relevance']['parameters']['num_cores']           = setcores
    
    # Call the main relevance test function
    rlv.main(att_res)

    # Save the python dictionary holding the results



# Print the eclapsed time
tm.timer(start_time)