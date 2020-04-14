# original: https://github.com/sergulaydore/Feature-Grouping-Regularizer


import pandas as pd
import numpy as np

from my_functions import create_split_indeces, load_4Ddata_for_validation, validate_model
from my_functions import load_id_for_train, load_batch, update_orig_weight
from my_functions import update_visdom_plot
#from model_try01 import Abide1DConvNet

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score

import random


k_i = 3
tot_ica_run = 50
n_timepoints = 150 # ica has been calculated on this lenght, do not change
#batch_size = 64
n_epoch = 200

def load_data(perc_data_for_train, batch_size):
#### read info about the dataset 
    k_i=3

    data_all=pd.read_csv("/data_local/deeplearning/ABIDE_ica/Model/ICA-Regularizer-SG/data_info_selected_sbj.csv", engine='python')

    # the data were not randomized when creating the ICAs therefore DO NOT 
    # randomize here --> risk to contaminate the training!

    ##### create training validation test indeces for kfold CV
    # same kfold number and randomseed used to create the ICAs, DO NOT modify
    train_split, val_split, test_split = create_split_indeces(np.array(data_all["SUB_ID"]))
    
    ######
    # mask data and ica with only not zero values
    mask_data = True 

    # ------ CV Loop ------
    ##### for testing purpose run only the 1st fold
    train_ind=train_split[k_i]
    train_ind=train_ind[0:int(len(train_ind)*perc_data_for_train)]
    val_ind=val_split[k_i]
    test_ind=test_split[k_i]

    
    # --- Prepare data for validation 
    val_data = load_4Ddata_for_validation(n_timepoints, data_all, val_ind, mask_data)
    val_loader = DataLoader(dataset= val_data, batch_size=batch_size, shuffle=True)

    # --- Load data for training

    tr_id_data= load_id_for_train(data_all, train_ind)
    train_id_loader = DataLoader(dataset=tr_id_data, batch_size=batch_size, shuffle=True)
    return train_id_loader, val_loader
  
def train_val(net, optimizer, 
            lr_SGD, 
            epoch_i, train_id_loader, val_loader,
            paths):
    
    #torch.cuda.set_device(1)

    #### model functions and optimizers
    #lr= 0.0001
    #criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(net.parameters(), lr=lr)

    #net.cuda()
    #net.train()

    # ------ Epochs Loop ------

    #ica_ind=str(random.randint(1,tot_ica_run)).zfill(2)
    ica_ind = '42'
    if epoch_i>50 & epoch_i+1%32 ==0:
        ica_ind=str(random.randint(1,tot_ica_run)).zfill(2)

    for i, (ids, labels) in enumerate(train_id_loader):

        optimizer.zero_grad()
        mask_data = True 
        data_tc, ica_matrix = load_batch(ids, n_timepoints, paths, ica_ind, mask_data)
        data_tc.requires_grad=True
        ica_matrix.requires_grad=True
        
        #output=net(data_tc.cuda(), ica_matrix= ica_matrix.cuda(), training=True)
        output=net(data_tc, ica_matrix= ica_matrix, training=True)

# when using BCEWithLogitsLoss    
        # calculate loss
        #predicted = (output.data>0.5).float()  
        #loss = criterion(output.cuda(), labels.cuda())

# when using nn.CrossEntropyLoss() 
        predicted = torch.argmax(output,1)
        #loss = criterion(output.cuda(), labels.reshape(-1).long().cuda())
        loss = criterion(output, labels.reshape(-1).long())
        loss.backward()
        optimizer.step()
        # manually update the original weight based on the updated red weight
        #update_orig_weight(net, ica_matrix.cuda(), lr=lr_SGD)      
        update_orig_weight(net, ica_matrix, lr=lr_SGD)      

    #### validation
    val_loss, val_bal_acc = validate_model(val_loader, net, criterion= nn.CrossEntropyLoss())
 
    return val_bal_acc


