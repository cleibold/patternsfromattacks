#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 17:08:46 2021

@author: athanasiadis
"""

import numpy as np
import foolbox
import torch


def linear(X_train,y_train,model,reason):

    """
    linear
    ______________________
    
    This is a function that uses Foolbox an adversarial attack python
    library to attack the  linear model trained on the subsampled dataset.
    The attack used here is FGSM.
    
    Input:  
            - Pytorch tensors holding the data and the labels.
    
            - the trained model.
            
    Output: 
            - Pytorch tensors holding the adversarial data and 
              labels from both attacks 
             
    """  
    
    #bound_min = 0
    #bound_max = 1
    bound_min = np.min(X_train.numpy())
    bound_max = np.max(X_train.numpy())
    
    # Set the trained model for the attack
    fmodel = foolbox.models.PyTorchModel(model.eval(), bounds=(bound_min,bound_max), num_classes=2)
    # Choose attack
    attack = foolbox.attacks.FGSM(fmodel)
    
    # Set Data Format
    images = X_train.numpy()
    original_labels = y_train
    # Get Adversarial Data
    adversarial_data = attack(images, original_labels,epsilons=100) 

    if reason=='Bias':
        #checks = np.copy(np.ndarray.flatten(adversarial_data))        
        #print(len([ind for ind in checks if np.isnan(ind)==True or np.isinf(ind)==True]))        
        adversarial_data = np.nan_to_num(adversarial_data)

    for n_dim in range(adversarial_data.shape[1]):
        if np.isnan(np.sum(adversarial_data[:,n_dim]))==True or np.isinf(np.sum(adversarial_data[:,n_dim]))==True:
            repeat = True
            #print('....Attack 1 Failed')
            return 0,0,0,0,repeat
    
    # Get Adversarial Labels
    adversarial_labels = fmodel.forward(adversarial_data).argmax(axis=-1)
    # Keep copies of the adversarial data for saving
    adversarial_data_sv = adversarial_data
    adversarial_labels_sv = adversarial_labels 


    # Set the trained model for the attack
    model_adv = model    
    fmodel2 = foolbox.models.PyTorchModel(model_adv.eval(), bounds=(bound_min,bound_max), num_classes=2)
    
    # Bring data to proper form for the second attack
    adversarial_data, adversarial_labels = map(torch.tensor, (adversarial_data,adversarial_labels))
    adversarial_data = adversarial_data.float()
    adversarial_labels = adversarial_labels.long()
    images = adversarial_data.numpy()
    adversarial_labels = adversarial_labels.numpy()        
    # Choose attack
    attack = foolbox.attacks.FGSM(fmodel2)
    
    # Get Adversarial Data
    adversarial_data2 = attack(images, adversarial_labels,epsilons=100)

    if reason=='Bias':
        #checks = np.copy(np.ndarray.flatten(adversarial_data2))        
        #print(len([ind for ind in checks if np.isnan(ind)==True or np.isinf(ind)==True]))        
        adversarial_data2 = np.nan_to_num(adversarial_data2)


    for n_dim in range(adversarial_data2.shape[1]):
        if np.isnan(np.sum(adversarial_data2[:,n_dim]))==True or np.isinf(np.sum(adversarial_data2[:,n_dim]))==True :
            repeat = True
            #print('....Attack 2 Failed')
            return 0,0,0,0,repeat
    
    # Get Adversarial Labels
    adversarial_labels2 = fmodel2.forward(adversarial_data2).argmax(axis=-1)
    repeat = False
    
    return adversarial_data_sv, adversarial_labels_sv, adversarial_data2, adversarial_labels2, repeat



def non_linear(X_train,y_train,model,reason):

    """
    non_linear
    ______________________
    
    This is a function that uses Foolbox an adversarial attack python
    library to attack the  non-linear model trained on the subsampled dataset.
    The attack used here is PGD.
    
    Input:  
            - Pytorch tensors holding the data and the labels.
    
            - the trained model.
            
    Output: 
            - Pytorch tensors holding the adversarial data and 
              labels from both attacks 
             
    """  
    
    #bound_min = 0
    #bound_max = 1
    bound_min = np.min(X_train.numpy())
    bound_max = np.max(X_train.numpy())
    
    # Set the trained model for the attack
    fmodel = foolbox.models.PyTorchModel(model.eval(), bounds=(bound_min,bound_max), num_classes=2)
    # Choose attack
    attack = foolbox.attacks.PGD(fmodel,distance=foolbox.distances.Linf)
    
    # Set Data Format
    images = X_train.numpy()
    original_labels = y_train
    # Get Adversarial Data
    adversarial_data = attack(images, original_labels) 

    if reason=='Bias':
        #checks = np.copy(np.ndarray.flatten(adversarial_data))        
        #print(len([ind for ind in checks if np.isnan(ind)==True or np.isinf(ind)==True]))        
        adversarial_data = np.nan_to_num(adversarial_data)

    for n_dim in range(adversarial_data.shape[1]):
        if np.isnan(np.sum(adversarial_data[:,n_dim]))==True or np.isinf(np.sum(adversarial_data[:,n_dim]))==True:
            repeat = True
            #print('....Attack 1 Failed')
            return 0,0,0,0,repeat
    
    # Get Adversarial Labels
    adversarial_labels = fmodel.forward(adversarial_data).argmax(axis=-1)
    # Keep copies of the adversarial data for saving
    adversarial_data_sv = adversarial_data
    adversarial_labels_sv = adversarial_labels 

    # Set the trained model for the attack
    model_adv = model    
    fmodel2 = foolbox.models.PyTorchModel(model_adv.eval(), bounds=(bound_min,bound_max), num_classes=2)
    
    # Bring data to proper form for the second attack
    adversarial_data, adversarial_labels = map(torch.tensor, (adversarial_data,adversarial_labels))
    adversarial_data = adversarial_data.float()
    adversarial_labels = adversarial_labels.long()
    images = adversarial_data.numpy()
    adversarial_labels = adversarial_labels.numpy()        
    # Choose attack
    attack = foolbox.attacks.PGD(fmodel2,distance=foolbox.distances.Linf)
    
    # Get Adversarial Data
    adversarial_data2 = attack(images, adversarial_labels)

    if reason=='Bias':
        #checks = np.copy(np.ndarray.flatten(adversarial_data2))        
        #print(len([ind for ind in checks if np.isnan(ind)==True or np.isinf(ind)==True]))        
        adversarial_data2 = np.nan_to_num(adversarial_data2)


    for n_dim in range(adversarial_data2.shape[1]):
        if np.isnan(np.sum(adversarial_data2[:,n_dim]))==True or np.isinf(np.sum(adversarial_data2[:,n_dim]))==True :
            repeat = True
            #print('....Attack 2 Failed')
            return 0,0,0,0,repeat
    
    # Get Adversarial Labels
    adversarial_labels2 = fmodel2.forward(adversarial_data2).argmax(axis=-1)
    repeat = False
    
    return adversarial_data_sv, adversarial_labels_sv, adversarial_data2, adversarial_labels2, repeat