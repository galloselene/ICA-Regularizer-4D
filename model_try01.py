import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

import numpy as np

import nibabel as nib
import os

# -------------------
#batch_size = 5
n_cluster = 25 # n of components for ica
n_timepoints = 150
tot_voxels_flat = 45*54*45 # with no mask
tot_voxels_flat = 28542 # with mask

# dimentions of the kernels

k_conv1_n = 32 #64
k_conv2_n = 16 #32
k_dim = 7
# -------------------

#in_features = n_timepoints
out_features = k_dim

# -------------------


class ICAGrouping(nn.Module):
    def __init__(self, n_cluster, in_features, out_features, k_conv1_dim, bias=True):
        # n_cluster = n of components for ica
        # in_features = original data dimentions tot_voxels_flat
        # out_features = conv2 kernel dimention

        super(ICAGrouping, self).__init__()

        self.icaconv_orig = nn.Conv1d(tot_voxels_flat, k_conv1_n, kernel_size=(k_dim,), stride=(1,))
        self.icaconv_red = nn.Conv1d(25, k_conv1_n, kernel_size=(k_dim,), stride=(1,))

# --------------------------------------------

    def forward(self, x, ica_matrix=np.empty([]), training=True):
       
        if training is True:
            clone_weight_red = self.icaconv_red.weight.clone()
            clone_weight_red.retain_grad = True

            for i, orig_wt in enumerate(self.icaconv_orig.weight.data):
                clone_weight_red[i,:,:] = torch.matmul(ica_matrix, orig_wt)


            reduced = x # this need to 
            reduced_var = autograd.Variable(reduced, requires_grad=True)
            output = self.icaconv_red(reduced_var)
            
        else:
            output = self.icaconv_orig(x)
        return output

# -----------------------------------------

class Abide1DConvNet(nn.Module):
    def __init__(self,  n_inputs=5, n_outputs=3, n_cluster=None,
                 dropout=0,
                 k_conv1_dim=7,

                 ):
        super(Abide1DConvNet, self).__init__()

        # what is out feature?
        # in the original is the dimention of the hidden layer
        # should it be the n_channels=n_ica components ?
        out_features = n_cluster

        self.ICAGrouping = ICAGrouping(
            n_cluster, n_inputs, out_features,k_conv1_dim=7)    # contains conv1
        self.batchnorm1 = nn.BatchNorm1d(k_conv1_n, affine=False)
        self.conv2 = nn.Conv1d(k_conv1_n, k_conv2_n, 7,)
        self.batchnorm2 = nn.BatchNorm1d(k_conv2_n, affine=False)
        self.avg = nn.AdaptiveAvgPool1d((1))
        #self.drop2 = nn.Dropout(p=0.0)
        self.linear1 = nn.Linear(k_conv2_n, 2)

        
    def forward(self, x, ica_matrix, training):
        x = self.ICAGrouping(x, ica_matrix, training)
        #x = self.batchnorm1(x)
        x = F.relu(self.conv2(x))
        #x = self.batchnorm2(x)
        x = self.avg(x).view(-1, k_conv2_n)
        #x = self.drop2(x)
        x =self.linear1(x)
        
        
        return x
"""
### test the backward and the weights updates
import torch.optim as optimizer

net=Abide1DConvNet(n_cluster=25, k_conv1_dim=7)
output=net(data_tc, ica_matrix)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters(),lr = 0.001, weight_decay=0.002)

# calculate loss
loss = criterion(output.cuda(1), dx.cuda(1)).cuda(1)

optimizer.zero_grad()
loss.backward()
optimizer.step()

print(a)

"""