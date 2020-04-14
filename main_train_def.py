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
#### REAL TIME VISUALIZATION
import visdom 

def train_val(change_ica_now,k_i):
    # real time visualization
    vis = visdom.Visdom()

    loss_window  = vis.line(X=np.zeros((1 ,2)),
                            Y=np.zeros((1,2)),
                            opts=dict(xlabel='epoch',
                                        ylabel=['train', 'val'],
                                        title='Loss',
                                        legend=['train', 'val*10-4']
                                        ))    

    accuracy_window = vis.line(X=np.zeros((1 ,2)),
                            Y=np.zeros((1,2)),
                            opts=dict(xlabel='epoch',
                                        ylabel=['train', 'val'],
                                        title='Balanced Accuracy',
                                        legend=['train', 'val']
                                        ))       
    #### define variables
    
    tot_ica_run = 50
    n_timepoints = 150 # ica has been calculated on this lenght, do not change
    batch_size = 64
    n_epoch = 800

    torch.cuda.set_device(1)

    #### read info about the dataset 
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
    val_ind=val_split[k_i]
    test_ind=test_split[k_i]


    if mask_data == True:
        from model_try01_masked import Abide1DConvNet
    else: 
        from model_try01 import Abide1DConvNet

    net=Abide1DConvNet(n_cluster=25, k_conv1_dim=7)

    #### model functions and optimizers
    lr= 0.0001
    #criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # assign correct paths for this fold
    paths =[]
    paths.append("/data_local/deeplearning/ABIDE_ica/DualRegr/Kfold_" + str(k_i+1) + "/") #timecourse data
    paths.append("/data_local/deeplearning/ABIDE_ica/Melodic_Results_v02/Kfold_"+ str(k_i+1) + "/") #ica matrix

    # --- Prepare data for validation 
    val_data = load_4Ddata_for_validation(n_timepoints, data_all, val_ind, mask_data)
    val_loader = DataLoader(dataset= val_data, batch_size=batch_size, shuffle=True)

    # --- Load data for training

    tr_id_data= load_id_for_train(data_all, train_ind)
    train_id_loader = DataLoader(dataset=tr_id_data, batch_size=batch_size, shuffle=True)

    # ------ Epochs Loop ------
    net.cuda()
    net.train()

    train_loss=[]
    train_accuracy=[]
    train_bal_acc = []

    best_acc = None # tracks best validation accuracy

    ica_ind=str(random.randint(1,tot_ica_run)).zfill(2)

    for epoch_i in range(n_epoch):
        #### collect info during epoch
        #ugly trick to have empty numpy array
        true_labels=[]
        true_labels = np.array(true_labels,dtype=float)
        predictions = []
        predictions = np.array(predictions,dtype=float)
        epoch_train_loss=0.0

        
        
        epoch_loss = 0.0
        if epoch_i>50 & epoch_i+1%change_ica_now ==0:
            ica_ind=str(random.randint(1,tot_ica_run)).zfill(2)
        

        for i, (ids, labels) in enumerate(train_id_loader):
            # randomly select an ica per batch
            #ica_ind=str(random.randint(1,tot_ica_run)).zfill(2)

            optimizer.zero_grad()
            data_tc, ica_matrix = load_batch(ids, n_timepoints, paths, ica_ind, mask_data)
            data_tc.requires_grad=True
            ica_matrix.requires_grad=True
            
            output=net(data_tc.cuda(), ica_matrix= ica_matrix.cuda(), training=True)

    # when using BCEWithLogitsLoss    
            # calculate loss
            #predicted = (output.data>0.5).float()  
            #loss = criterion(output.cuda(), labels.cuda())

    # when using nn.CrossEntropyLoss() 
            predicted = torch.argmax(output,1)
            loss = criterion(output.cuda(), labels.reshape(-1).long().cuda())
            
            loss.backward()
            optimizer.step()
            # manually update the original weight based on the updated red weight
            update_orig_weight(net, ica_matrix.cuda(), lr=0.001)
            
            epoch_loss += loss.detach()
            true_labels = np.append(true_labels, labels.numpy())
            predictions = np.append(predictions, predicted.cpu().numpy())

        cm1 = confusion_matrix(true_labels, predictions)
        total1=sum(sum(cm1))
        accuracy1=(cm1[0,0]+cm1[1,1])/total1
        bal_acc = balanced_accuracy_score(true_labels, predictions)
        epoch_train_loss = epoch_loss/len(true_labels)
        

    #### validation
        val_loss, val_bal_acc = validate_model(val_loader, net, criterion= nn.CrossEntropyLoss().cuda())
        
    # find best model and save it
        if not best_acc or best_acc < val_bal_acc:
            best_acc = val_bal_acc
            torch.save(net.state_dict(), str(k_i+1) + '_ch' + str(change_ica_now) + '_BEST_model.pt')
                    

        train_loss.append(epoch_train_loss)
        train_bal_acc.append(bal_acc)

    #### plot realtime
        update_visdom_plot(loss_window, accuracy_window, epoch_i, np.log(epoch_train_loss.cpu()), np.log(val_loss.cpu()), bal_acc, val_bal_acc)
        #update_visdom_plot(loss_window, accuracy_window, epoch_i, epoch_train_loss.cpu(), val_loss, bal_acc, val_bal_acc)
        print('Epoch n {} --- TRAINING: loss {} -- bal-accuracy {} --- VALIDATION: loss {} -- bal-accuracy {}'
            .format(epoch_i+1, epoch_train_loss, bal_acc, val_loss, val_bal_acc))

        

    print('end')
    return best_acc

#-------
import pandas as pd

df = pd.DataFrame({'kfold': [], 'changed ica every':[], 'best val bal acc': []})

#change_ica_now_list = np.array((1, 4, 8, 16, 32, 50))
change_ica_now = 32
for k_i in range(5):
    
    best_acc = train_val(change_ica_now, k_i)
    d = {'kfold': [k_i+1], 'changed ica every':[change_ica_now], 'best val bal acc': [best_acc]}
    df_update= pd.DataFrame(d)

    df = df.append(df_update)
    df.to_csv('cv_validation_best_Config.csv')

