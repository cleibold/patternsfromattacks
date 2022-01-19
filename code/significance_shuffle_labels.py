#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 16:53:51 2021

@author: athanasiadis
"""

import numpy as np
import itertools

def block(y_labels,nshuffles,label_values,contribution):

    """
    block
    ______________________
    
    This is a function that shuffles the labels in a 
    block/trial manner.
    
    Input:  
            - a numpy array holding all the labels the labels.
    
            - an integer value equal to the amount of shuffles.

            - a list holding the label per trial.

            - a list holding the contribution per trial.                 
            
    Output: 
            - an integer equal to the updated amount of shuffles
              adjusted for the case we have very few trials.
              
            - a list of numpy arrays holding the new labels
              per shuffle.
              
    """
    
    # Create an array with the enumeration of the trials
    trials = list(np.arange(0,len(label_values),1))
    
    # Now compute the shuffles of the labels
    # First check how many trials we have and adjust the amount of shuffles
    if len(trials)<=6:
        new_nshuffles = np.math.factorial(len(trials))
        nshuffles = np.min([nshuffles,new_nshuffles])
    # Get the first permutation which is the unshuffled indices
    perms = [] 
    perms.append(trials)
    if nshuffles!=np.math.factorial(len(trials)):
        # Repeat the process untill you have at least "nshuffles" unique permutations
        while len(perms)<=nshuffles:
            # Generate a random permutation
            perm_i = list(np.random.permutation(trials))
            # Check if the permutation is unique so far
            flags = [ind for ind,x in enumerate(perms) if x==perm_i]
            if len(flags)==0:
                perms.append(perm_i)  
    else:
        perms = list(itertools.permutations(trials))
        perms.insert(0,trials)
    # Now make a list that stores the actual labels per shuffle
    y_perms = []
    # Loop over the shuffles
    for nsh in range(nshuffles+1):
        # Grab the permutations labels
        perm_i = perms[nsh]
        # Loop over the trials
        for ntr in range(len(trials)):
            # Grab the trials contribution
            cont = contribution[ntr]
            # Make an array for the labels of each trial
            dummy_trial = np.zeros((cont,))
            # Assign the proper labels
            if label_values[perm_i[ntr]]==1:
                dummy_trial[:] = 1
            else:
                dummy_trial[:] = 0
            # Stack the trials
            if ntr==0:
                labelsaver = dummy_trial
            else:
                labelsaver = np.concatenate((labelsaver,dummy_trial),axis=0)
        # Store the labels per shuffle
        y_perms.append(labelsaver)
            
    return y_perms,nshuffles



def full(y_labels,nshuffles):

    """
    full
    ______________________
    
    This is a function that shuffles the labels in a 
    full/pattern manner.
    
    Input:  
            - a numpy array holding all the labels the labels.
    
            - an integer value equal to the amount of shuffles.
                
            
    Output: 
            - an integer equal to the updated amount of shuffles
              adjusted for the case we have very few trials.
              
            - a list of numpy arrays holding the new labels
              per shuffle.
              
    """    

    # Create an array of the indices values
    indices = np.arrange(0,len(y_labels),1)
    # Now compute the shuffles of the labels
    # First check how many trials we have and adjust the amount of shuffles
    if len(indices)<=6:
        new_nshuffles = np.math.factorial(len(indices))
        nshuffles = np.min([nshuffles,new_nshuffles])    
        
    # Get the first permutation which is the unshuffled indices
    perms = [] 
    perms.append(list(indices))
    # Repeat the process untill you have at least 200 unique permutations
    while len(perms)<=nshuffles:
        # Generate a random permutation
        perm_i = list(np.random.permutation(indices))
        # Check if the permutation is unique so far
        flags = [ind for ind,x in enumerate(perms) if x==perm_i]
        if len(flags)==0:
            perms.append(perm_i)  

    # Now make a list that stores the actual labels per shuffle
    y_perms = []
    # Loop over the shuffles
    for nsh in range(nshuffles+1):
        # Grab the permutations labels
        perm_i = perms[nsh]
        # Grab the labels for this particular shuffle
        labelsaver = list(np.copy(y_labels[perm_i]))
        # Adjust the labels that are -1
        labelsaver_adj = [1 if x==1 else 0 for x in labelsaver]
        labelsaver_adj = np.reshape(labelsaver_adj,(len(labelsaver),))
        # Store the labels per shuffle
        y_perms.append(labelsaver_adj)
                
    return y_perms,nshuffles