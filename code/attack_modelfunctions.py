#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 11:53:52 2021

@author: athanasiadis
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import attack_foolbox as fool


class LinearNet(nn.Module):

    """
    linearnet
    ______________________
    
    This is a class composed of the linear neural network. 
    
    - input nodes  : equal to the data dimensionality.
    - output nodes : equal to 2, one per class label. 
    - softmax included in the crossentropy loss function.
    
    Input:  
            - the data dimensionality.
                                         
    """

    def __init__(self,dimensions):
        super(LinearNet,self).__init__()    
        #Input(N nodes) -> Output(2 nodes) 
        #We use 2 output nodes beacouse we give 2 probability numbers
        #Each one for each one of the possible classes. The one that is bigger wins
        self.fc1 = nn.Linear(dimensions,2)
    
    def forward(self, x):
        #Input Layer result
        x = self.fc1(x)                  
        return x
    
    # Prediction. Takes the result of output and predicts a class
    def predict(self,x):
        #We use softmax function for the output
        #pred = F.softmax(self.forward(x),dim=1)
        pred = self.forward(x)
        ans = [] 
        #Check class 1 or 0 and put it in tensor ans
        for t in pred:
            ans.append(torch.argmax(t))
        return torch.tensor(ans)

class NonLinearNet(nn.Module):

    """
    nonlinearnet
    ______________________
    
    This is a class composed of the non linear neural network 
    
    - input nodes  : equal to the data dimensionality.
    - hidden nodes : equal to the data dimensionality +1.
    - output nodes : equal to 2, one per class label. 
    - tanh(x)      : activation function between layers.
    - softmax included in the crossentropy loss function.
    
    Input:  
            - the data dimensionality.
                         
                  
    """

    def __init__(self,dimensions):
        super(NonLinearNet,self).__init__()    
        #Input(N nodes) -> Output(2 nodes) 
        #We use 2 output nodes beacouse we give 2 probability numbers
        #Each one for each one of the possible classes. The one that is bigger wins
        self.fc1 = nn.Linear(dimensions,dimensions+1)
        self.fc2 = nn.Linear(dimensions+1,2)        

    
    def forward(self, x):
        #Input Layer result
        x = self.fc1(x)                  
        x = torch.tanh(x)
        x = self.fc2(x) 
        return x
    
    # Prediction. Takes the result of output and predicts a class
    def predict(self,x):
        #We use softmax function for the output
        #pred = F.softmax(self.forward(x),dim=1)
        pred = self.forward(x)
        ans = [] 
        #Check class 1 or 0 and put it in tensor ans
        for t in pred:
            ans.append(torch.argmax(t))
        return torch.tensor(ans)


def subsampling_perms(labindices,traintarget): 

    """
    subsampling_perms
    ______________________
    
    This is a function that shuffles the data indices so that 
    different parts of the dataset are represented within each
    subsampling.
    
    Input:  
            - a list holding the indices of the specific label.
    
            - an integer equal to the amount of patterns needed.                 
            
    Output: 
            - an array holding the indices for the test set.
    
            - an array holding the indices for the train set.
              
    """  
    
    train_perms = []
    test_perms = []
    # Get the first permutation
    lperm_i = list(np.random.permutation(labindices)) 
    train_perms = lperm_i[0:traintarget]
    test_perms = lperm_i[traintarget:]  

    return train_perms, test_perms


def train_test_model(x_dt,y_lb,dimensions,lr,epochs,nsub,modelinfo,model_flag):

    """
    train_test_model
    ______________________
    
    This is a function that decodes the data using the model of choise.
    A 2-fold subsampling procedure is implemented.
    The label representation/contribution is kept within each fold.
    Once the model is trained it is also used for the adversarial attack.
    
    Input:  
            - a numpy array holding the dataset. 
            - a numpy array holding the labels.
            - an integer equal to the dataset dimensionality.
            - the learning rate and the amount of epochs for the model.
            - an integer value equal to the amount of subsamplings requested.
            - the type of model requested.
            
            Note : an easily additional input could be the option to have ballanced label contribution  within subsamplings.
            Uppon request I can implement that. 
            
    Output: 
            - a dictionary with the decoding and attack results per subsampling
              with the following entries:
                  
                  - The indices of the data used in the training set per subsampling.
                  - The indices of the data used in the testing set per subsampling.
                  - The predictions made by the trained model on the testing set.
                  - The mean CCR value per subsampling.
                  - The adversarial data computed per subsampling.
                  - The adversarial labels computed per subsampling.
                  - The model weight vector (usefull for linear ANNs).
                  - Features that are "useless" meaning that they have the same value for all.
                    patterns withing a subampling. These features are excluded.
                    from the training and the attack processes.
                  
    """

    # Get the indices per label and their contribution % in train test sets
    label0_indices = [ind for ind,x in enumerate(y_lb) if x==0]
    label1_indices = [ind for ind,x in enumerate(y_lb) if x==1]    
    # Get the size of the training and testing sets
    train_size = int(np.ceil(x_dt.shape[0]/2))
    # Get the ratio of each label in the original set
    label0_ratio = len(label0_indices)/(len(label0_indices)+len(label1_indices))
    label1_ratio = len(label1_indices)/(len(label0_indices)+len(label1_indices))
    # Get the target size for each label for the training-testing set
    label0_train_target = int(np.ceil(train_size*label0_ratio))
    label1_train_target = int(np.ceil(train_size*label1_ratio))
    
    # Set the flag for the case the attack fails
    repeat = True 
    while repeat==True:        
        
        # Generate n distinct permutations of the indices per label to be used in each subsampling
        train0_perms,test0_perms = subsampling_perms(label0_indices,label0_train_target)
        train1_perms,test1_perms = subsampling_perms(label1_indices,label1_train_target)
        # Unite the indices from each label
        ind_train = train0_perms + train1_perms
        ind_test = test0_perms + test1_perms      
        # Pick the data needed for the subsampling
        y_train = np.copy(y_lb[ind_train])
        y_test = np.copy(y_lb[ind_test])
        X_train = np.copy(x_dt[ind_train,:])
        X_test = np.copy(x_dt[ind_test,:])  

        # Check if there are features for which all datapoints have the same value
        nvalues = np.zeros((dimensions,))
        for ndim in range(dimensions):
            nvalues[ndim] = len(np.unique(x_dt[:,ndim]))

        # Create a new data matrix that only has the usefull features data
        keep_feature_inds = []
        for ndim in range(dimensions):
            if nvalues[ndim]!=1:
                keep_feature_inds.append(ndim)
        X_train = np.copy(X_train[:,keep_feature_inds])
        X_test = np.copy(X_test[:,keep_feature_inds])
        needed_features = len(keep_feature_inds)
   
        # Turn the subsets into torch tensors in order to use them on the neural network
        X_train, X_test, y_train, y_test = map(torch.tensor, (X_train, X_test, y_train, y_test))
        X_train = X_train.float()
        y_train = y_train.long()
        X_test = X_test.float()
        y_test = y_test.long()   
    
        # Initiallize the model
        model = []
        #Set the model training parameters
        epochs = epochs
        if model_flag=='linear':
            model = LinearNet(needed_features)
        if model_flag=='non-linear':
            model = NonLinearNet(needed_features)            
        #Loss Criterion 
        criterion = nn.CrossEntropyLoss()
        #Optimizer 
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        #optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    
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
             
        # Save the model
        model_name = modelinfo +'Model_Sub{}.pt'.format(nsub)
        torch.save(model.state_dict(), model_name)        

        # Get the estimated weight vector and the bias from the trained model
        weights_out = list(model.fc1.weight.data)
        bias = model.fc1.bias
        bias=list(bias)
        w_node_0 = np.reshape(weights_out[0].numpy(),(needed_features,1)) 
        w_node_1 = np.reshape(weights_out[1].numpy(),(needed_features,1))       
        w_node_0 = w_node_0/np.linalg.norm(w_node_0)
        w_node_1 = w_node_1/np.linalg.norm(w_node_1)
        weight_vector = np.subtract(w_node_0,w_node_1)
        #weight_vector = weight_vector/np.linalg.norm(weight_vector)
        weight_vector_torch = torch.from_numpy(weight_vector.T).type(torch.FloatTensor)
        weight_vector_label = model.predict(weight_vector_torch)
        weight_vector_label = weight_vector_label.numpy()
        if weight_vector_label==0:
            weight_vector = np.negative(weight_vector)

        # Evaluate the models accuracy on the training dataset
        model.eval()    
        y_pred_class_test = model.predict(X_test)
        y_pred_class_test = y_pred_class_test.numpy()
        y_test = y_test.numpy()    
        y_train = y_train.numpy()    
        ccr= np.sum(y_test==y_pred_class_test) / len(y_test)
        #ccr = np.round(ccr,2)
    
        # Set the flag for the reason of attack
        reason = 'Sub'
        # Apply 2 consecutive attacks on the dataset using Foolbox 
        if model_flag=='linear':
            adversarial_data_sv, adversarial_labels_sv, adversarial_data2, adversarial_labels2, repeat = fool.linear(X_train,y_train,model,reason)
        if model_flag=='non-linear':
            adversarial_data_sv, adversarial_labels_sv, adversarial_data2, adversarial_labels2, repeat = fool.non_linear(X_train,y_train,model,reason)
            
    # Initiallize the dictionary used to save the subsamplings results   
    subsampling_key = 'Subsampling_{}'.format(nsub)
    att_res = {}
    att_res[subsampling_key] = {}
    att_res[subsampling_key]['train_indices'] = ind_train
    att_res[subsampling_key]['test_indices'] = ind_test
    att_res[subsampling_key]['test_prediction'] = y_pred_class_test
    att_res[subsampling_key]['ccr'] = ccr
    att_res[subsampling_key]['adversarial_data'] = adversarial_data2
    att_res[subsampling_key]['adversarial_labels'] = adversarial_labels2
    att_res[subsampling_key]['useless_features'] = nvalues
    att_res[subsampling_key]['weights'] = weight_vector

    return att_res