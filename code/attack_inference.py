#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 21:27:23 2021

@author: athanasiadis
"""


import torch
import numpy as np
import attack_modelfunctions as md
import torch.nn as nn

def computer_local(eigenvectors,modelinfo,nsub,nvalues,model_flag):

    """
    computer_local
    ______________________
    
    This is a function that adjusts the direction of the identified eigenvectors
    which can point to any of two directions (towards one class or the other). In 
    this way all eigenvectors point towards one of the classes. 
    
    Input:  
            - the identified eigenvectors.
            - the trained model and it's parameters. 
            
    Output: 
            - the direction adjusted eigenvectors.

    """        

    
    # Find out how many useful features exist within the current subsampling 
    needed_features = len([1 for x in nvalues if x!=1])
    
    # Group the local normal vectors
    eigenvectors = np.concatenate(eigenvectors,axis=0)
    
    # Load the model of each subsampling and feed the eigenvectors to get a classification label
    model_name = modelinfo + 'Model_Sub{}.pt'.format(nsub)
    if model_flag=='linear':
        inference_model = md.LinearNet(needed_features)
    if model_flag=='non-linear':
        inference_model = md.NonLinearNet(needed_features)        
    inference_model.load_state_dict(torch.load(model_name))
    inference_model.eval()
                        
    # Turn the subsets into torch tensors in order to use them on the neural network
    cluster_test = torch.from_numpy(eigenvectors)
    cluster_test = cluster_test.float()
    cluster_test_label = inference_model.predict(cluster_test)
    cluster_test_label = cluster_test_label.numpy()                                 

    # Flip the sign in any of the eigenvectors for which the infered label belongs to class 0 
    flip_eigenvectors = []
    for neig in range(eigenvectors.shape[0]):
        eig_i = eigenvectors[neig,:]
        eig_i = np.reshape(eig_i,(1,len(eig_i)))
        if cluster_test_label[neig]==0:
            eig_i = eig_i * (-1)
        flip_eigenvectors.append(eig_i)
    eigenvectors = np.concatenate(flip_eigenvectors,axis=0)
        
    return eigenvectors

def computer_global(meaned_clusters,x_data,y_labels,dim_i,lr,epochs,modelinfo,model_flag):

    """
    computer_global
    ______________________
    
    This is a function that adjusts the direction of the averaged patterns
    which can point to any of two directions (towards one class or the other).
    To do so a new model is trained using the whole dataset.
    In this way all averaged patterns point towards one of the classes. 
    This is more of a failsafe function.
    
    Input:  
            - the averaged patterns.
            - the data, labels and the model and it's parameters. 
            
    Output: 
            - the direction adjusted averaged patterns.

    """            

    # Turn the subsets into torch tensors in order to use them on the neural network
    X_train, y_train = map(torch.tensor, (x_data, y_labels))
    X_train = X_train.float()
    y_train = y_train.long()
   
    # Initiallize the model
    model = []
    #Set the model training parameters
    epochs = epochs
    if model_flag=='linear':
        model = md.LinearNet(dim_i)
    if model_flag=='non-linear':
        model = md.NonLinearNet(dim_i)            
    #Loss Criterion 
    criterion = nn.CrossEntropyLoss()
    #Optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    # Train model        
    model.train()
    for n_epoch in range(epochs):
        #Clear previous gradients 
        optimizer.zero_grad()     
        # Predict the output of a given input
        y_pred = model.forward(X_train)         
        # Compute the losses with the criterion and add it to the list
        loss = criterion(y_pred, y_train)           
        #Compute gradients from backward propagation
        loss.backward()        
        #Adjust the weights
        optimizer.step()        
   
    # Evaluate the models accuracy on the training dataset
    model.eval()             

    # Save the model
    model_name = modelinfo +'Model_Global.pt'
    torch.save(model.state_dict(), model_name) 
    
    # Turn the subsets into torch tensors in order to use them on the neural network
    cluster_test = torch.from_numpy(meaned_clusters)
    cluster_test = cluster_test.float()
    cluster_test_label = model.predict(cluster_test)
    cluster_test_label = cluster_test_label.numpy()                                 

    # Flip the sign in any of the eigenvectors for which the infered label belongs to class 0 
    flip_meaned_clusters = []
    for neig in range(meaned_clusters.shape[0]):
        vec_i = meaned_clusters[neig,:]
        vec_i = np.reshape(vec_i,(1,len(vec_i)))
        if cluster_test_label[neig]==0:
            vec_i = vec_i * (-1)
        flip_meaned_clusters.append(vec_i)
    meaned_clusters = np.concatenate(flip_meaned_clusters,axis=0)      
    
    return meaned_clusters