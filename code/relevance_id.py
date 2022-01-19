#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 15:39:17 2021

@author: athanasiadis
"""


import itertools
import numpy as np

def shuffle(dimensions):
    
    """
    shuffle
    ______________________
    
    This is a function that generated permutations shuffling the order of the dimensions for
    the input dataset.
    
    Input:  
            - the dataset dimensionality.
            
    Output: 
            - the updated amount of shuffles. 
            - the dimensions shuffled indices per permutation. 
                  
    """     
    
    
    # Create an array that holds the indices of each cell
    shuffle_inds = np.arange(0,dimensions)
    # Set the amount of shuffles
    nshuffles = 1000
    if dimensions<=6:
        new_nshuffles = np.math.factorial(dimensions)
        nshuffles = np.min([1000,new_nshuffles])
    #nshuffles = 1000
    # Get the first permutation which is the unshuffled indices
    perms = [] 
    if nshuffles!=np.math.factorial(dimensions):
        # Repeat the process untill you have at least "nshuffles" unique permutations
        while len(perms)<=nshuffles:
            # Generate a random permutation
            perm_i = list(np.random.permutation(shuffle_inds))
            #perms.append(perm_i)
            # Check if the permutation is unique so far
            flags = [ind for ind,x in enumerate(perms) if x==perm_i]
            if len(flags)==0:
                perms.append(perm_i)  
    else:
        perms = list(itertools.permutations(shuffle_inds))
        perms = perms[1:]
    
    return nshuffles, perms
    
    

def dot(corrected_data,shuffler,pattern_i):

    """
    dot
    ______________________
    
    This is a function that computes the projection vector for each of the permutations.
    
    Input:  
            - the gravity center adjusted dataset and the global vector / MIP.
            - the shuffle indices.
            
    Output: 
            - a projection vector per shuffle.
                  
    """ 
    
    # Shuffle the corrected data
    shuffler_i    = corrected_data[:,shuffler]
    # Compute the shuffle projection by dot product of the shuffled data with the pattern_i
    sh_projection = np.dot(shuffler_i,pattern_i.T)
    sh_projection = np.reshape(sh_projection,(1,len(sh_projection)))   
    
    return sh_projection




def find(sh_projection,runs,projection_i,up,down):

    """
    find
    ______________________
    
    This is a function that identifies the relevant patters for each global vector / MIP.
    
    Input:  
            - the shufled projections.
            - the real projection
            - the confidence interval.
            
    Output: 
            - lists containing the indices of the patterns found to be relevant, irelevant
            and plus/minus relevant (this indicates the relevance towards a specific class).
                  
    """     

    # Concatenate the projections and the shuffle inds into a matrix proj_dist = (1000,timebins)
    dummy_proj_dist = np.concatenate(sh_projection,axis=0)
    # Flatten the projection values into a 1D vector and make a histogram out of them
    dummy_proj_dist = np.ndarray.flatten(dummy_proj_dist)
    
    # Identify the outlier 5%tile values of the distribution
    perc5 = np.percentile(dummy_proj_dist,down)
    perc95 = np.percentile(dummy_proj_dist,up)

    # Use these outlier values to identify relevant plus/minus and irrelevant datapoints    
    relevant_indices = [ind for ind,x in enumerate(projection_i[:,0]) if(x>perc95 or x<perc5)]
    irelevant_indices = [ind for ind,x in enumerate(projection_i[:,0]) if(x<=perc95 and x>=perc5)]
    relevant_plus_indices = [ind for ind,x in enumerate(projection_i[:,0]) if(x>perc95)]
    relevant_minus_indices = [ind for ind,x in enumerate(projection_i[:,0]) if(x<perc5)]
                    
    return relevant_indices,irelevant_indices,relevant_plus_indices,relevant_minus_indices