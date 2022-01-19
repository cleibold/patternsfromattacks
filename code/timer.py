#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 11:07:22 2021

@author: athanasiadis
"""

import numpy as np
import time

def timer(start_time):

    """
    timer
    ______________________
    
    This is a function that prints the eclapsed time of the code compliation
    
    Input:  
            - initialization timepoint 
            
    Output: 
            - time eclapsed in hours/minutes/seconds
    """
    
    stop_time = time.time()
    dt_sec = stop_time -start_time
    dt_sec = int(np.round(dt_sec))
    if dt_sec<60:  
        print('--- {} seconds ---'.format(dt_sec))
    else:  
        dt_min = int(dt_sec // 60)
        dt_sec = int(dt_sec % 60)  
        if dt_min<60:      
            print('--- {} minutes and {} seconds ---'.format(dt_min,dt_sec))      
        else:      
            dt_hour = int(dt_min // 60)
            dt_min = int(dt_min % 60)      
            print('--- {} hours and {} minutes and {} seconds ---'.format(dt_hour,dt_min,dt_sec))