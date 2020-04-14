import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import random
import torch.autograd as autograd

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score


def create_split_indeces(data):

    n_splits_CV=5
    kf = KFold(n_splits=n_splits_CV,    random_state=111,     shuffle=True)
    n_subjects = np.shape(data)[0]  
    X = np.zeros((n_subjects, 1))

    train_split = []
    val_split = []
    test_split = []
    for i_split, (trainval_list, test_list) in enumerate(kf.split(X)):
            
            train_split.append(trainval_list[int(len(trainval_list)*0.1):])
            val_split.append(trainval_list[0:int(len(trainval_list)*0.1)])
            test_split.append(test_list)

    return train_split, val_split, test_split



class load_id_for_train(Dataset):
    def __init__(self, 
                data_all,
                split_ind,
                ):   
        self.data_id = np.array(data_all["SUB_ID"])[split_ind]
        self.label = np.array(data_all["DX_GROUP"])[split_ind].reshape(-1,1)
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, item):
        return torch.tensor(self.data_id[item]), torch.tensor(self.label[item], dtype = float)

class load_4Ddata_for_validation(Dataset):
    def __init__(self, 
                n_timepoints,
                data_all,
                split_ind,
                mask_data
                ):   
        print('Taking my time to load 4D validation data....')
        fullpath_str = "/data_local/deeplearning/ABIDE_LC/raw/00" 
        data_id = np.array(data_all["SUB_ID"])[split_ind]
        self.label = np.array(data_all["DX_GROUP"])[split_ind].reshape(-1,1)

        data_path=[]
        for i in data_id:
            data_path.append(fullpath_str 
            #+ str(data_id.values[split_indeces[i],0]) 
            + str(i)
            + "/preproc.feat/"
            #+ "00" + str(data_id.values[split_indeces[i],0]) 
            + "00" + str(i) 
            + "_filtered_func_data_MNI4mm.nii.gz")    

        self.data=np.empty((len(data_id), 45*54*45, n_timepoints))

        for i,sbj_path in enumerate(data_path):  
                   
            sbj_data = nib.load(sbj_path)
            sbj_data = sbj_data.get_fdata()[:,:,:,0:n_timepoints]
            sbj_data = sbj_data.reshape(-1,n_timepoints)
            self.data[i,:,:]=sbj_data
        print('4D validation data: loaded')


        if mask_data == True:
            mask = np.load("/data_local/deeplearning/ABIDE_ica/Model/ICA-Regularizer-4D/params_tuning/brain_mask", allow_pickle= True)
            self.data = self.data[:, mask.reshape(-1), :]

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, item):
        return torch.tensor(self.data[item], dtype= torch.float32), torch.tensor(self.label[item], dtype= torch.float32)


def load_batch(ids, n_timepoints, paths, ica_ind, mask_data):
        # randomly select ica id
        #ica_ind=str(random.randint(1,tot_ica_run)).zfill(2) 
        
        ica_filename = paths[1] + "melodic_run"+ ica_ind + "/melodic_IC.nii.gz"
        ica_matrix = nib.load(ica_filename)
        ica_matrix = ica_matrix.get_fdata().reshape(-1,25).transpose()
        
        if mask_data==True:
            mask=np.load("/data_local/deeplearning/ABIDE_ica/Model/ICA-Regularizer-4D/params_tuning/brain_mask", allow_pickle= True)
            ica_matrix = ica_matrix[:, mask.reshape(-1)]

        data=np.empty((len(ids), 25, n_timepoints))

        for i, i_sbj in enumerate(ids):
            sbj_tc_path= paths[0] + str((i_sbj.detach().numpy())).zfill(7) + '_melodic_run' + ica_ind + '_dr1'
            sbj_tc =np.loadtxt(sbj_tc_path)
           
            data[i,:,:]=np.transpose(sbj_tc[0:n_timepoints,:]).reshape(1,-1, n_timepoints)       
        
        return torch.as_tensor(data,dtype= torch.float32), torch.as_tensor(ica_matrix, dtype= torch.float32)


def update_orig_weight(self, ica_matrix, lr): 
    p = list(self.parameters())
    grad = self.ICAGrouping.icaconv_red.weight.data.clone()
    for i,wred_grad in enumerate(grad):
        #p[0].data[i,:,:]=(torch.matmul(wred_grad.transpose_(1,0), ica_matrix)).transpose_(1,0)*lr
        p[0].data[i,:,:]+=(torch.matmul(wred_grad.transpose_(1,0), ica_matrix)).transpose_(1,0)*(-lr)

    self.ICAGrouping.icaconv_orig.bias = self.ICAGrouping.icaconv_red.bias
    self.ICAGrouping.icaconv_red.weight.grad.zero_()

    
    
def validate_model(data_loader, net, criterion):    
    net.eval()
        #ugly trick to have empty numpy array
    true_labels=[]
    true_labels = np.array(true_labels,dtype=float)
    predictions = []
    predictions = np.array(predictions,dtype=float)
    
    loss=0.0
    for i, (data, labels) in enumerate(data_loader):
        
        output=net(data, ica_matrix=np.empty([]), training=False)

# when using BCEWithLogitsLoss    
        # calculate loss
        #predicted = (output.data>0.5).float()  
        #loss += criterion(output.cuda(), labels.cuda()).detach()

# when using nn.CrossEntropyLoss() 
        predicted = torch.argmax(output,1)
        #loss += criterion(output.cuda(), labels.reshape(-1).long().cuda()).detach()
        loss += criterion(output, labels.reshape(-1).long()).detach()
        
        true_labels = np.append(true_labels, labels.numpy())
        predictions = np.append(predictions, predicted.cpu().numpy())

    cm1 = confusion_matrix(true_labels, predictions)
    total1=sum(sum(cm1))
    accuracy1=(cm1[0,0]+cm1[1,1])/total1
    bal_acc = balanced_accuracy_score(true_labels, predictions)
    
    return loss/len(true_labels), bal_acc


#### ----------   REAL TIME VISUALIZATION
import visdom 
vis = visdom.Visdom()

def update_visdom_plot(loss_window, accuracy_window, epoch_i, epoch_loss, loss_val, bal_acc, val_bal_acc):

    vis.line(
            X=np.column_stack((epoch_i,epoch_i)),
            Y=np.column_stack((epoch_loss, loss_val)),#.unsqueeze(0).cpu(),
            win=loss_window,
            update='append')

   
    vis.line(
            X=np.column_stack((epoch_i,epoch_i)),
            Y=np.column_stack((bal_acc, val_bal_acc)),#.unsqueeze(0).cpu(),
            win=accuracy_window,
            update='append')