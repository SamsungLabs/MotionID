#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tqdm
import tqdm.notebook

import numpy as np
import pandas as pd
from scipy.integrate import cumtrapz, trapz

from matplotlib import pyplot as plt

import seaborn as sns
import torch
from torch import nn

import torch
import torchvision
import torchvision.transforms as transforms

import pickle

device = 'cuda:1'

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


with open('data/data_smoltest_75/accel_dataset_smoltest.pkl', 'rb') as handle:
    accel_dataset = pickle.load(handle)
    
with open('data/data_smoltest_75/gyro_dataset_smoltest.pkl', 'rb') as handle:
    gyro_dataset = pickle.load(handle)

with open('data/data_smoltest_75/rotation_dataset_smoltest.pkl', 'rb') as handle:
    rotation_dataset = pickle.load(handle)
    
with open('data/data_smoltest_75/magnetic_dataset_smoltest.pkl', 'rb') as handle:
    magnetic_dataset = pickle.load(handle)

with open('dataset_smolltest_add_test/accel_dataset_smoltest_add_test.pkl', 'rb') as handle:
    accel_dataset_add_test = pickle.load(handle)
    
with open('dataset_smolltest_add_test/gyro_dataset_smoltest_add_test.pkl', 'rb') as handle:
    gyro_dataset_add_test = pickle.load(handle)

with open('dataset_smolltest_add_test/rotation_dataset_smoltest_add_test.pkl', 'rb') as handle:
    rotation_dataset_add_test = pickle.load(handle)
    
with open('dataset_smolltest_add_test/magnetic_dataset_smoltest_add_test.pkl', 'rb') as handle:
    magnetic_dataset_add_test = pickle.load(handle)


# In[3]:


def get_rotation_matrix_from_vector(v):
    q1 = v[0]
    q2 = v[1]
    q3 = v[2]
    if len(v) == 4:
        q0 = v[3]
    else:    
        q0 = 1 - q1**2 - q2**2 - q3**2
        q0 = np.sqrt(q0) if q0 > 0 else 0
    sq_q1 = 2 * q1**2
    sq_q2 = 2 * q2**2
    sq_q3 = 2 * q3**2
    q1_q2 = 2 * q1 * q2
    q3_q0 = 2 * q3 * q0
    q1_q3 = 2 * q1 * q3
    q1_q3 = 2 * q1 * q3
    q2_q0 = 2 * q2 * q0
    q2_q3 = 2 * q2 * q3
    q1_q0 = 2 * q1 * q0
    R = np.array([
        [1.0 - sq_q2 - sq_q3, q1_q2 - q3_q0, q1_q3 + q2_q0],
        [q1_q2 + q3_q0, 1.0 - sq_q1 - sq_q3, q2_q3 - q1_q0],
        [q1_q3 - q2_q0, q2_q3 + q1_q0, 1.0 - sq_q1 - sq_q2],
    ])
    if len(v) == 4:
        R = np.hstack((R, np.zeros((3, 1))))
        R = np.vstack((R, np.array([[0.0, 0.0, 0.0, 1.0]])))
    return R

def get_rotation_matrices(x):
    rotation_matrices = np.zeros(x.shape[:-1] + (3, 3))
    xx = x.reshape((-1, 3))
    rmt = rotation_matrices.reshape((-1, 3, 3))
    for i in range(xx.shape[0]):
        rmt[i, :, :] = get_rotation_matrix_from_vector(xx[i])
    rotation_matrices = rmt.reshape(x.shape[:-1] + (3, 3))
    return rotation_matrices


# In[4]:


def known_rot_starts(rot):
    nonzero_mask = np.any(rot != 0, axis=-1)
    pred_zero = np.concatenate((np.ones((nonzero_mask.shape[0], 1), dtype=bool), (~nonzero_mask)[:, :-1]), axis=1)
    rot_single_start_mask = np.logical_and(nonzero_mask, pred_zero)
    assertion_idx, col_idx = np.where(np.logical_and(nonzero_mask, pred_zero))
    col_idx_res = np.ones(nonzero_mask.shape[0], dtype=col_idx.dtype) * nonzero_mask.shape[0]
    for ass_id, cid in zip(assertion_idx, col_idx):
        col_idx_res[ass_id] = cid
    return col_idx_res


def rotate_known(x, rot, axis=1):
    start_indices = known_rot_starts(rot)
    rotated = np.zeros_like(x)
    for i in range(x.shape[0]):
        if start_indices[i] == rot.shape[1]:
            continue
        rmt = get_rotation_matrices(rot[i, start_indices[i]:, :])
        #print(np.einsum('dmn,dkn->dmk', rmt, rmt))
        #print(rmt[0] @ rmt[0].T)
        #assert np.allclose(np.einsum('dmn,dkn->dmk', rmt, rmt), np.stack([np.eye(3)] * rmt.shape[0], axis=0))
        xx = np.einsum('dmn,dn->dm', rmt, x[i, start_indices[i]:, :])
        rotated[i, start_indices[i]:, :] = xx
    return rotated


def diff_known(x, rot, axis=1):
    start_indices = known_rot_starts(rot)
    diffed = np.zeros_like(x)
    for i in range(x.shape[0]):
        if start_indices[i] == rot.shape[1]:
            continue
        #rmt = get_rotation_matrices(rot[i, start_indices[i]:, :])
        #print(np.einsum('dmn,dkn->dmk', rmt, rmt))
        #print(rmt[0] @ rmt[0].T)
        #assert np.allclose(np.einsum('dmn,dkn->dmk', rmt, rmt), np.stack([np.eye(3)] * rmt.shape[0], axis=0))
        #xx = np.einsum('dmn,dn->dm', rmt, x[i, start_indices[i]:, :])
        #rotated[i, start_indices[i]:, :] = xx
        xx = x[i, start_indices[i]:, :]
        diff_value = np.concatenate((
            np.zeros((1, xx.shape[-1])),
            np.diff(xx, axis=-2)), axis=-2)
        diffed[i, start_indices[i]:, :] = diff_value
    return diffed


def unrotate_known(x, rot, axis=1):
    start_indices = known_rot_starts(rot)
    rotated = np.zeros_like(x)
    for i in range(x.shape[0]):
        if start_indices[i] == rot.shape[1]:
            continue
        rmt = get_rotation_matrices(rot[i, start_indices[i]:, :])
        #print(np.einsum('dmn,dkn->dmk', rmt, rmt))
        #print(rmt[0] @ rmt[0].T)
        #assert np.allclose(np.einsum('dmn,dkn->dmk', rmt, rmt), np.stack([np.eye(3)] * rmt.shape[0], axis=0))
        xx = np.einsum('dnm,dn->dm', rmt, x[i, start_indices[i]:, :])
        rotated[i, start_indices[i]:, :] = xx
    return rotated


def integrate_known(x, rot, axis=1):
    start_indices = known_rot_starts(rot)
    integrated = np.zeros_like(x)
    for i in range(x.shape[0]):
        if start_indices[i] == rot.shape[1]:
            continue
        rmt = get_rotation_matrices(rot[i, start_indices[i]:, :])
        xx = np.einsum('dmn,dn->dm', rmt, x[i, start_indices[i]:, :])
        integrated[i, start_indices[i]:, :] = cumtrapz(xx, dx=0.02, axis=0, initial=0)
    return integrated


def degrav_known(x, rot, grav_vec, axis=1):
    start_indices = known_rot_starts(rot)
    degrav = np.zeros_like(x)
    for i in range(x.shape[0]):
        if start_indices[i] == rot.shape[1]:
            continue
        rmt = get_rotation_matrices(rot[i, start_indices[i]:, :])
        xx = np.einsum('dmn,dn->dm', rmt, x[i, start_indices[i]:, :])
        degrav[i, start_indices[i]:, :] = xx - grav_vec[None, :]
    return degrav


def get_linaccel_features(accel, rot):
    grav_const = np.array([0.0, 0.0, 9.806634201818664])
    rot_accel_known = rotate_known(accel, rot)
    rot_linaccel_known = degrav_known(accel, rot, grav_const)
    unrot_accel_known = unrotate_known(rot_accel_known, rot)
    unrot_linaccel_known = unrotate_known(rot_linaccel_known, rot)
    
    rot_linaccel_diff = diff_known(rot_linaccel_known, rot)
    unrot_linaccel_diff = diff_known(unrot_linaccel_known, rot)
    rot_linaccel_int = integrate_known(rot_linaccel_known, rot)
    unrot_linaccel_int = integrate_known(unrot_accel_known, rot)
    
    return np.concatenate((
        rot_linaccel_known,
        rot_linaccel_diff,
        rot_linaccel_int,
        unrot_linaccel_known,
        unrot_linaccel_diff,
        unrot_linaccel_int
    ), axis=-1)


# In[5]:


def make_dataset(accel_dataset, gyro_dataset, rotation_dataset, magnetic_dataset):
    dataset = {
        stage: {
            'data': np.concatenate((get_linaccel_features(accel_dataset[stage]['data'], rotation_dataset[stage]['data']),
                                    np.concatenate((
                                        integrate_known(accel_dataset[stage]['data'], rotation_dataset[stage]['data']),
                                        cumtrapz(accel_dataset[stage]['data'], dx=0.02, axis=1, initial=0.0),
                                        rotate_known(accel_dataset[stage]['data'], rotation_dataset[stage]['data']),
                                        accel_dataset[stage]['data'],
                                        np.concatenate((np.zeros((accel_dataset[stage]['data'].shape[0], 1, accel_dataset[stage]['data'].shape[2])), np.diff(accel_dataset[stage]['data'], axis=1)), axis=1)), axis=-1),
                                    np.concatenate((
                                        integrate_known(gyro_dataset[stage]['data'], rotation_dataset[stage]['data']),
                                        cumtrapz(gyro_dataset[stage]['data'], dx=0.02, axis=1, initial=0.0),
                                        rotate_known(gyro_dataset[stage]['data'], rotation_dataset[stage]['data']),
                                        gyro_dataset[stage]['data'],
                                        np.concatenate((np.zeros((gyro_dataset[stage]['data'].shape[0], 1, gyro_dataset[stage]['data'].shape[2])), np.diff(gyro_dataset[stage]['data'], axis=1)), axis=1)), axis=-1),
                                    np.concatenate((
                                        rotation_dataset[stage]['data'],
                                        np.concatenate((np.zeros((rotation_dataset[stage]['data'].shape[0], 1, rotation_dataset[stage]['data'].shape[2])), np.diff(rotation_dataset[stage]['data'], axis=1)), axis=1)), axis=-1),
                                    np.concatenate((
                                        integrate_known(magnetic_dataset[stage]['data'], magnetic_dataset[stage]['data']),
                                        rotate_known(magnetic_dataset[stage]['data'], rotation_dataset[stage]['data']),
                                        magnetic_dataset[stage]['data'],
                                        np.concatenate((np.zeros((magnetic_dataset[stage]['data'].shape[0], 1, magnetic_dataset[stage]['data'].shape[2])), np.diff(magnetic_dataset[stage]['data'], axis=1)), axis=1)), axis=-1)
                                   ), axis=-1),
            'labels': accel_dataset[stage]['labels']
        }
        for stage in accel_dataset
    }
    return dataset


# In[6]:


dataset = make_dataset(accel_dataset, gyro_dataset, rotation_dataset, magnetic_dataset)
dataset_add_test = make_dataset(accel_dataset_add_test, gyro_dataset_add_test, rotation_dataset_add_test, magnetic_dataset_add_test)


# In[7]:


train_data_tensor = torch.from_numpy(dataset['train']['data']).float()
train_labels_tensor = torch.from_numpy(dataset['train']['labels']).type(torch.LongTensor)

train_dataset = torch.utils.data.TensorDataset(train_data_tensor, train_labels_tensor)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128,
                                          shuffle=True, num_workers=0)


val_data_tensor = torch.from_numpy(dataset['val']['data']).float()
val_labels_tensor = torch.from_numpy(dataset['val']['labels']).type(torch.LongTensor)

val_dataset = torch.utils.data.TensorDataset(val_data_tensor, val_labels_tensor)

valloader = torch.utils.data.DataLoader(val_dataset, batch_size=128,
                                          shuffle=False, num_workers=0)


test_data_tensor = torch.from_numpy(dataset['test']['data']).float()
test_labels_tensor = torch.from_numpy(dataset['test']['labels']).type(torch.LongTensor)

test_dataset = torch.utils.data.TensorDataset(test_data_tensor, test_labels_tensor)

testloader = torch.utils.data.DataLoader(test_dataset, batch_size=128,
                                          shuffle=False, num_workers=0)


# In[8]:


train_data_tensor_add_test = torch.from_numpy(dataset_add_test['train']['data']).float()
train_labels_tensor_add_test = torch.from_numpy(dataset_add_test['train']['labels']).type(torch.LongTensor)

train_dataset_add_test = torch.utils.data.TensorDataset(train_data_tensor_add_test, train_labels_tensor_add_test)

trainloader_add_test = torch.utils.data.DataLoader(train_dataset_add_test, batch_size=128,
                                          shuffle=True, num_workers=0)


val_data_tensor_add_test = torch.from_numpy(dataset_add_test['val']['data']).float()
val_labels_tensor_add_test = torch.from_numpy(dataset_add_test['val']['labels']).type(torch.LongTensor)

val_dataset_add_test = torch.utils.data.TensorDataset(val_data_tensor_add_test, val_labels_tensor_add_test)

valloader_add_test = torch.utils.data.DataLoader(val_dataset_add_test, batch_size=128,
                                          shuffle=False, num_workers=0)


test_data_tensor_add_test = torch.from_numpy(dataset_add_test['test']['data']).float()
test_labels_tensor_add_test = torch.from_numpy(dataset_add_test['test']['labels']).type(torch.LongTensor)

test_dataset_add_test = torch.utils.data.TensorDataset(test_data_tensor_add_test, test_labels_tensor_add_test)

testloader_add_test = torch.utils.data.DataLoader(test_dataset_add_test, batch_size=128,
                                          shuffle=False, num_workers=0)


# In[9]:


nclasses = len(np.unique(dataset['train']['labels']))
nclasses_add_test = len(np.unique(dataset_add_test['train']['labels']))
nclasses, nclasses_add_test


# In[10]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# class SampaddingConv1D_BN(nn.Module):
#     def __init__(self,in_channels,out_channels,kernel_size):
#         super(SampaddingConv1D_BN, self).__init__()
#         self.padding = nn.ConstantPad1d((int((kernel_size-1)/2), int(kernel_size/2)), 0)
#         self.conv1d = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
#         self.bn = nn.BatchNorm1d(num_features=out_channels)
        
#     def forward(self, X):
#         X = self.padding(X)
#         X = self.conv1d(X)
#         X = self.bn(X)
#         return X
    
# class build_layer_with_layer_parameter(nn.Module):
#     def __init__(self,layer_parameters): 
#         super(build_layer_with_layer_parameter, self).__init__()
#         self.conv_list = nn.ModuleList()
        
#         for i in layer_parameters:
#             conv = SampaddingConv1D_BN(i[0],i[1],i[2])
#             self.conv_list.append(conv)
    
#     def forward(self, X):
        
#         conv_result_list = []
#         for conv in self.conv_list:
#             conv_result = conv(X)
#             conv_result_list.append(conv_result)
            
#         result = F.relu(torch.cat(tuple(conv_result_list), 1))
#         return result



def calculate_mask_index(kernel_length_now,largest_kernel_lenght):
    right_zero_mast_length = math.ceil((largest_kernel_lenght-1)/2)-math.ceil((kernel_length_now-1)/2)
    left_zero_mask_length = largest_kernel_lenght - kernel_length_now - right_zero_mast_length
    return left_zero_mask_length, left_zero_mask_length+ kernel_length_now

def creat_mask(number_of_input_channel,number_of_output_channel, kernel_length_now, largest_kernel_lenght):
    ind_left, ind_right= calculate_mask_index(kernel_length_now,largest_kernel_lenght)
    mask = np.ones((number_of_input_channel,number_of_output_channel,largest_kernel_lenght))
    mask[:,:,0:ind_left]=0
    mask[:,:,ind_right:]=0
    return mask


def creak_layer_mask(layer_parameter_list):
    largest_kernel_lenght = layer_parameter_list[-1][-1]
    mask_list = []
    init_weight_list = []
    bias_list = []
    for i in layer_parameter_list:
        conv = torch.nn.Conv1d(in_channels=i[0], out_channels=i[1], kernel_size=i[2])
        ind_l,ind_r= calculate_mask_index(i[2],largest_kernel_lenght)
        big_weight = np.zeros((i[1],i[0],largest_kernel_lenght))
        big_weight[:,:,ind_l:ind_r]= conv.weight.detach().numpy()
        
        bias_list.append(conv.bias.detach().numpy())
        init_weight_list.append(big_weight)
        
        mask = creat_mask(i[1],i[0],i[2], largest_kernel_lenght)
        mask_list.append(mask)
        
    mask = np.concatenate(mask_list, axis=0)
    init_weight = np.concatenate(init_weight_list, axis=0)
    init_bias = np.concatenate(bias_list, axis=0)
    return mask.astype(np.float32), init_weight.astype(np.float32), init_bias.astype(np.float32)

    
class build_layer_with_layer_parameter(nn.Module):
    def __init__(self,layer_parameters):
        super(build_layer_with_layer_parameter, self).__init__()

        os_mask, init_weight, init_bias= creak_layer_mask(layer_parameters)
        
        
        in_channels = os_mask.shape[1] 
        out_channels = os_mask.shape[0] 
        max_kernel_size = os_mask.shape[-1]

        self.weight_mask = nn.Parameter(torch.from_numpy(os_mask),requires_grad=False)
        
        self.padding = nn.ConstantPad1d((int((max_kernel_size-1)/2), int(max_kernel_size/2)), 0)
         
        self.conv1d = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=max_kernel_size)
        self.conv1d.weight = nn.Parameter(torch.from_numpy(init_weight),requires_grad=True)
        self.conv1d.bias =  nn.Parameter(torch.from_numpy(init_bias),requires_grad=True)

        self.bn = nn.BatchNorm1d(num_features=out_channels)
    
    def forward(self, X):
        self.conv1d.weight.data = self.conv1d.weight*self.weight_mask
        #self.conv1d.weight.data.mul_(self.weight_mask)
        result_1 = self.padding(X)
        result_2 = self.conv1d(result_1)
        result_3 = self.bn(result_2)
        result = F.relu(result_3)
        return result    
    
class OS_CNN(nn.Module):
    def __init__(self,layer_parameter_list,n_class,few_shot = True):
        super(OS_CNN, self).__init__()
        self.few_shot = few_shot
        self.layer_parameter_list = layer_parameter_list
        self.layer_list = []
        
        
        for i in range(len(layer_parameter_list)):
            layer = build_layer_with_layer_parameter(layer_parameter_list[i])
            self.layer_list.append(layer)
        
        self.net = nn.Sequential(*self.layer_list)
            
        self.averagepool = nn.AdaptiveAvgPool1d(1)
        
        out_put_channel_numebr = 0
        for final_layer_parameters in layer_parameter_list[-1]:
            out_put_channel_numebr = out_put_channel_numebr+ final_layer_parameters[1] 
            
        self.hidden = nn.Linear(out_put_channel_numebr, n_class)

    def forward(self, X):
        
        X = self.net(X)

        X = self.averagepool(X)
        X = X.squeeze_(-1)

        if not self.few_shot:
            X = self.hidden(X)
        return X

class OS_CNN_branch(nn.Module):
    def __init__(self,layer_parameter_list,n_class,few_shot = True):
        super(OS_CNN_branch, self).__init__()
        self.few_shot = few_shot
        self.layer_parameter_list = layer_parameter_list
        self.layer_list = []
        
        
        for i in range(len(layer_parameter_list)):
            layer = build_layer_with_layer_parameter(layer_parameter_list[i])
            self.layer_list.append(layer)
        
        self.net = nn.Sequential(*self.layer_list)
            
        self.averagepool = nn.AdaptiveAvgPool1d(1)
        
        out_put_channel_numebr = 0
        for final_layer_parameters in layer_parameter_list[-1]:
            out_put_channel_numebr = out_put_channel_numebr+ final_layer_parameters[1] 
            
        #self.hidden = nn.Linear(out_put_channel_numebr, n_class)

    def forward(self, X):
        
        X = self.net(X)

        X = self.averagepool(X)
        X = X.squeeze_(-1)

        #if not self.few_shot:
        #    X = self.hidden(X)
        return X


# In[12]:


def get_Prime_number_in_a_range(start, end):
    Prime_list = []
    for val in range(start, end + 1): 
        prime_or_not = True
        for n in range(2, val):
            if (val % n) == 0:
                prime_or_not = False
                break
        if prime_or_not:
            Prime_list.append(val)
    return Prime_list


def get_out_channel_number(paramenter_layer, in_channel, prime_list):
    out_channel_expect = int(paramenter_layer/(in_channel*sum(prime_list)))
    return out_channel_expect

# def generate_layer_parameter_list(start,end,paramenter_number_of_layer_list):
    
#     in_channel = 1 
#     prime_list = get_Prime_number_in_a_range(start, end)
    
#     layer_parameter_list = []
#     for paramenter_number_of_layer in paramenter_number_of_layer_list:
#         out_channel = get_out_channel_number(paramenter_number_of_layer, in_channel, prime_list)
        
#         tuples_in_layer= []
#         for prime in prime_list:
#             tuples_in_layer.append((in_channel,out_channel,prime))
#         in_channel =  len(prime_list)*out_channel
        
#         layer_parameter_list.append(tuples_in_layer)
    
#     tuples_in_layer_last = []
#     first_out_channel = len(prime_list)*get_out_channel_number(paramenter_number_of_layer_list[0], 1, prime_list)
#     tuples_in_layer_last.append((in_channel,first_out_channel,1))
#     tuples_in_layer_last.append((in_channel,first_out_channel,2))
#     layer_parameter_list.append(tuples_in_layer_last)
#     return layer_parameter_list

def generate_layer_parameter_list(start,end,paramenter_number_of_layer_list, in_channel = 1):
    prime_list = get_Prime_number_in_a_range(start, end)
    if prime_list == []:
        print('start = ',start, 'which is larger than end = ', end)
    input_in_channel = in_channel
    layer_parameter_list = []
    for paramenter_number_of_layer in paramenter_number_of_layer_list:
        out_channel = get_out_channel_number(paramenter_number_of_layer, in_channel, prime_list)
        
        tuples_in_layer= []
        for prime in prime_list:
            tuples_in_layer.append((in_channel,out_channel,prime))
        in_channel =  len(prime_list)*out_channel
        
        layer_parameter_list.append(tuples_in_layer)
    
    tuples_in_layer_last = []
    first_out_channel = len(prime_list)*get_out_channel_number(paramenter_number_of_layer_list[0], input_in_channel, prime_list)
    tuples_in_layer_last.append((in_channel,first_out_channel,start))
    tuples_in_layer_last.append((in_channel,first_out_channel,start+1))
    layer_parameter_list.append(tuples_in_layer_last)
    return layer_parameter_list


# In[13]:


train_shape = train_dataset.tensors[0][:, :51, :].shape
nclasses = train_dataset.tensors[1].max().item() + 1
print(f'train shape: {train_shape}, nclass = {nclasses}')

max_kernel_size = 89
start_kernel_size = 1

receptive_field_shape = min(int(train_shape[1]/4), max_kernel_size)

parameter_number_of_layer_list = [8*128, 5*128*256 + 2*256*128]

# generate parameter list
layer_parameter_list = generate_layer_parameter_list(start_kernel_size,
                                                     receptive_field_shape,
                                                     parameter_number_of_layer_list,
                                                     in_channel=3)


# In[14]:


class Cat(nn.Module):
    def __init__(self, dim=-1):
        self.dim = dim
        super().__init__()

    def forward(self, args):
        #print(torch.cat(args, dim=self.dim).shape)
        return torch.cat(args, dim=self.dim)


class Net(nn.Module):
    def __init__(self, n_branches=22):
        super(Net, self).__init__()
        
        train_shape = train_dataset.tensors[0][:, :51, :].shape
        nclasses = train_dataset.tensors[1].max().item() + 1
        print(f'train shape: {train_shape}, nclass = {nclasses}')

        max_kernel_size = 89
        start_kernel_size = 1

        receptive_field_shape = min(int(train_shape[1]/4), max_kernel_size)

        parameter_number_of_layer_list = [8*128, 5*128*256 + 2*256*128]

        # generate parameter list
        layer_parameter_list = generate_layer_parameter_list(start_kernel_size,
                                                             receptive_field_shape,
                                                             parameter_number_of_layer_list,
                                                             in_channel=3)
        
        
        self.n_branches = n_branches
        
        self.net_branches = nn.ModuleList(OS_CNN_branch(layer_parameter_list, nclasses, False) for _ in range(n_branches))

        self.cat = Cat(dim=-1)
    
    def forward(self, x):
        chunk_size = x.shape[1] // self.n_branches
        if x.shape[1] != chunk_size * self.n_branches:
            raise ValueError(f'branches shapes are not aligned with input shape {x.shape}')
        
        x = torch.split(x, 3, dim=1)
        branches_out = [branch(x_el) for (branch, x_el) in zip(self.net_branches, x)]
        branches_out_cat = self.cat(branches_out)
        return branches_out_cat


class ClassifierTail(nn.Module):
    def __init__(self, n_classes, n_branches=22):
        super().__init__()
        self.n_classes = n_classes
        self.n_branches = n_branches
        sqrt_features_mult = np.sqrt(self.n_branches)
        sqrt_features1 = int(np.round(256 * sqrt_features_mult))
        sqrt_features2 = int(np.round(128 * sqrt_features_mult))
        self.classifier = nn.Sequential(
            nn.Linear(132 * self.n_branches, sqrt_features1),
            nn.ReLU6(),
            nn.Dropout(p=0.2),
            nn.Linear(sqrt_features1, sqrt_features2),
            nn.ReLU6(),
            nn.Dropout(p=0.2),
            nn.Linear(sqrt_features2, sqrt_features2),
            nn.ReLU6(),
            #nn.Dropout(p=0.1),
            nn.Linear(sqrt_features2, nclasses)
        )
    
    def forward(self, x):
        return self.classifier(x)


class SiameseTail(nn.Module):
    def __init__(self, n_branches=22):
        super(SiameseTail, self).__init__()
        self.n_branches = n_branches
        sqrt_features_mult = np.sqrt(self.n_branches)
        sqrt_features1 = int(np.round(128 * sqrt_features_mult))
        sqrt_features2 = int(np.round(64 * sqrt_features_mult))
        self.nonlinear_embedding = nn.Sequential(
            nn.Linear(132 * self.n_branches, sqrt_features1),
            nn.ReLU6(),
            nn.Dropout(p=0.1),
            nn.Linear(sqrt_features1, sqrt_features2),
            nn.ReLU6(),
            nn.Dropout(p=0.1),
            nn.Linear(sqrt_features2, sqrt_features2),
        )
    
    def forward(self, x1):
        x1_embedding = self.nonlinear_embedding(x1)
        #x2_embedding = self.nonlinear_embedding(x2)
        return x1_embedding

class SimpleMLP(nn.Module):
    def __init__(self, n_features=300):
        super(SimpleMLP, self).__init__()
        self.nonlinear_embedding = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.ReLU6(),
            nn.Linear(n_features, n_features),
        )
    
    def forward(self, x1):
        x1_embedding = self.nonlinear_embedding(x1)
        #x2_embedding = self.nonlinear_embedding(x2)
        return x1_embedding

# net = Net()
# classifier_tail = ClassifierTail(n_classes=nclasses, n_branches=22)
# siamese_tail = SiameseTail(n_branches=22)
# mlp_tail = SimpleMLP(n_features=300)
# net.to(device)
# classifier_tail.to(device)
# siamese_tail.to(device)
# mlp_tail.to(device)


# In[15]:


import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, device, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


# In[16]:


from scipy.spatial.distance import pdist, squareform
from multiprocessing import Pool, cpu_count


def ncpus_from_njobs(njobs):
    return max(1, njobs) if njobs >= 0 else max(1, cpu_count() + njobs + 1)


def bootstrap_metric_thr_wrapper(args):
    ppdist_st, pdist_labels, thr, m, iters = args
    return bootstrap_metric_thr(ppdist_st, pdist_labels, thr, m=m, iters=iters)


def bootstrap_metric_thr(ppdist_st, pdist_labels, thr, m=200, iters=2000):
    if np.isscalar(thr):
        thr = np.array([thr])
    unique_labels = np.sort(np.unique(pdist_labels))
    n = len(unique_labels)
    labels2indices = {
        label: np.where(pdist_labels == label)[0]
        for label in unique_labels
    }
    labels2nonindices = {
        label: np.where(pdist_labels != label)[0]
        for label in unique_labels
    }
    fars = []
    for iter_num in np.arange(iters):#tqdm.tqdm_notebook(np.arange(iters)):
        sampled_labels = np.random.choice(unique_labels, size=n, replace=True)
        sampled_enroll_indices_per_label = np.stack([
            np.repeat(np.random.choice(labels2nonindices[label], size=n-1, replace=True), m)
            for label in sampled_labels
        ], axis=0)
        sampled_verify_indices_per_label = np.stack([
            np.tile(np.random.choice(labels2indices[label], size=m, replace=True), n-1)
            for label in sampled_labels
        ], axis=0)
        enroll_raws = sampled_enroll_indices_per_label.reshape(-1)
        verify_raws = sampled_verify_indices_per_label.reshape(-1)
        match_scores = ppdist_st[enroll_raws, verify_raws].copy()
        sample_far = np.array([
            np.mean(match_scores >= thr_el)
            for thr_el in thr
        ])
        #sns.distplot(match_scores.ravel())
        fars.append(sample_far)
    return np.stack(fars, axis=0)


def bootstrap_metric_thr_frr_wrapper(args):
    ppdist_st, pdist_labels, thr, m, iters = args
    return bootstrap_metric_thr_frr(ppdist_st, pdist_labels, thr, m=m, iters=iters)


def bootstrap_metric_thr_frr(ppdist_st, pdist_labels, thr, m=200, iters=1000):
    if np.isscalar(thr):
        thr = np.array([thr])
    unique_labels = np.sort(np.unique(pdist_labels))
    n = len(unique_labels)
    labels2indices = {
        label: np.where(pdist_labels == label)[0]
        for label in unique_labels
    }
    labels2nonindices = {
        label: np.where(pdist_labels != label)[0]
        for label in unique_labels
    }
    frrs = []
    for iter_num in np.arange(iters):#tqdm.tqdm_notebook(np.arange(iters)):
        sampled_labels = np.random.choice(unique_labels, size=n, replace=True)
        sampled_verify_indices = np.array([np.random.choice(labels2indices[label]) for label in sampled_labels])
        sampled_verify_indices_per_label = np.stack([
            np.random.choice(labels2indices[label], size=m, replace=True)
            for label in sampled_labels
        ], axis=0)
        match_scores = [
            ppdist_st[sampled_label, sampled_verify_indices].copy()
            for sampled_label, sampled_verify_indices in zip(sampled_verify_indices, sampled_verify_indices_per_label)
        ]
        #sns.distplot(np.stack(match_scores, 0).reshape(-1))
        sample_frr = np.array([
            
            np.mean(np.stack(match_scores, 0).reshape(-1) <= thr_el)
            for thr_el in thr
        ])
        frrs.append(sample_frr)
    return np.stack(frrs, axis=0)


def parallel_bootstrap_metric_thr(ppdist_st, pdist_labels, thr, m=200, iters=1000, njobs=40):
    ncpus = ncpus_from_njobs(njobs)
    
    chunk_size = iters // njobs
    last_iter_size = iters % njobs
    if last_iter_size:
        chunk_size += 1
    high_iters = chunk_size * njobs
    
    chunk_iters = [chunk_size] * njobs
    ppdist_st_list = [ppdist_st] * njobs
    pdist_labels_list = [pdist_labels] * njobs
    thr_list = [thr] * njobs
    mm = [m] * njobs
    
    pool_args = zip(ppdist_st_list, pdist_labels_list, thr_list, mm, chunk_iters)
    
    with Pool(ncpus) as pool:
        res = np.concatenate(pool.map(bootstrap_metric_thr_wrapper, pool_args), axis=0)
        print(res.shape)
        return res[:iters]


def parallel_bootstrap_metric_thr_frr(ppdist_st, pdist_labels, thr, m=200, iters=1000, njobs=40):
    ncpus = ncpus_from_njobs(njobs)
    
    chunk_size = iters // njobs
    last_iter_size = iters % njobs
    if last_iter_size:
        chunk_size += 1
    high_iters = chunk_size * njobs
    
    chunk_iters = [chunk_size] * njobs
    ppdist_st_list = [ppdist_st] * njobs
    pdist_labels_list = [pdist_labels] * njobs
    thr_list = [thr] * njobs
    mm = [m] * njobs
    
    pool_args = zip(ppdist_st_list, pdist_labels_list, thr_list, mm, chunk_iters)
    
    with Pool(ncpus) as pool:
        res = np.concatenate(pool.map(bootstrap_metric_thr_frr_wrapper, pool_args), axis=0)
        print(res.shape)
        return res[:iters]


# In[17]:


def augment_random_crop(timeseries, nseq=51):
    batch_size, n_channels, seq_length = timeseries.shape
    
    batch_indices = torch.randint(low=0, high=seq_length - nseq, size=(batch_size,))
    timeseries = torch.stack([
        timeseries[i, :, batch_index:batch_index + nseq]
        for i, batch_index in enumerate(batch_indices)
    ], dim=0)
    return timeseries


def augment_mag_noise(timeseries):
    mag_scale = 0.2
    mean, std = timeseries.mean(dim=-1, keepdim=True), timeseries.std(dim=-1, keepdim=True)
    noise = mag_scale * std * torch.randn_like(std)
    #print(timeseries.shape, noise.shape)
    return timeseries + noise
    

def augment_center_crop(timeseries, nseq=51):
    batch_size, n_channels, seq_length = timeseries.shape
    
    start = (seq_length - nseq) // 2
    
    return timeseries[:, :, start: start + nseq]


def augment_triple_crop(timeseries, nseq=51):
    batch_size, n_channels, seq_length = timeseries.shape
    
    start = (seq_length - nseq) // 2

    return torch.cat([
        timeseries[:, :, :nseq],
        timeseries[:, :, start: start + nseq],
        timeseries[:, :, -nseq:]
    ], dim=0)


# In[18]:


def augment_random_crop(timeseries, nseq=51):
    batch_size, n_channels, seq_length = timeseries.shape
    
    batch_indices = torch.randint(low=0, high=seq_length - nseq + 1, size=(batch_size,))
    timeseries = torch.stack([
        timeseries[i, :, batch_index:batch_index + nseq]
        for i, batch_index in enumerate(batch_indices)
    ], dim=0)
    return timeseries

def augment_features_drop(timeseries, nseq=51):
    batch_size, n_channels, seq_length = timeseries.shape
    max_ndrop = seq_length - nseq
    
    dropped_features = []
    
    ar = np.arange(seq_length)
    
    result = torch.zeros(batch_size, n_channels, nseq, device=timeseries.device)
    
    for i in range(batch_size):
        for j in range(n_channels):
            mask = torch.ones(seq_length, dtype=np.bool)

            size_to_drop = np.random.randint(low=0, high=max_ndrop + 1)

            mask[np.random.choice(ar, size=size_to_drop, replace=False)] = False

            filtered_series = timeseries[i, j, mask]
            
            max_shift = seq_length - size_to_drop - nseq + 1
            shift_index = np.random.randint(0, max_shift)
            result[i, j, :] = filtered_series[shift_index: shift_index + nseq]

    return result


def augment_features_drop_fast(timeseries, nseq=51):
    batch_size, n_channels, seq_length = timeseries.shape
    max_ndrop = seq_length - nseq
    
    bxc = batch_size * n_channels
    
    chooses = np.random.randint(low=0, high=seq_length - nseq + 1, size=bxc)
    chooses_inv = max_ndrop - chooses
    
    shift_max = np.random.randint(low=0, high=chooses_inv + 1, size=bxc)
    
    ar = np.arange(seq_length)
    
    mask_i = np.concatenate([
        np.ones(choose)
        for choose in chooses
    ], axis=0)
    mask_j = np.concatenate([
        np.random.choice(ar, size=choose, replace=False)
        for choose in chooses
    ], axis=0)
    mask_data = np.ones(len(mask_i), dtype=np.bool)
    first_mask = coo_matrix((mask_data, (mask_i, mask_j)), shape=(bxc, seq_length), dtype=np.bool)
    
    return first_mask


# In[19]:


finetune_paths = [
    f'/nasDATASETS/_from_nas/imu_results/smoltest_finetune_{cls}_results'
    for cls in range(nclasses_add_test)
]
for finetune_path in finetune_paths:
    if not os.path.exists(finetune_path):
        os.makedirs(finetune_path)


# In[21]:


import pickle
import heapq

results_path = '/nasDATASETS/_from_nas/imu_results/smoltest_results'

score_heap_filepath = os.path.join(results_path, 'score_heap.pkl')
with open(score_heap_filepath, 'rb') as handle:
    score_heap = pickle.load(handle)

heapq.heapify(score_heap)
checkpoint_name = max(score_heap)[1]
print(f'checkpoint name: {checkpoint_name}')


# In[22]:


checkpoint_path = os.path.join(results_path, checkpoint_name)
state = torch.load(checkpoint_path)

net = Net()
classifier_tail = ClassifierTail(n_classes=nclasses, n_branches=22)
siamese_tail = SiameseTail(n_branches=22)
mlp_tail = SimpleMLP(n_features=300)

net.load_state_dict(state['net'])
classifier_tail.load_state_dict(state['classifier_tail'])
siamese_tail.load_state_dict(state['siamese_tail'])
mlp_tail.load_state_dict(state['mlp_tail'])


# feature_extract = True

# def set_parameter_requires_grad(model, feature_extracting):
#     if feature_extracting:
#         for param in model.parameters():
#             param.requires_grad = False

# set_parameter_requires_grad(net, feature_extract)
# #set_parameter_requires_grad(classifier_tail, feature_extract)
# #set_parameter_requires_grad(siamese_tail, feature_extract)
# #set_parameter_requires_grad(mlp_tail, feature_extract)

# classifier_in_features = classifier_tail.classifier[8].in_features

# classifier_tail.classifier[8] = nn.Linear(in_features=classifier_in_features, out_features=2)

net.to(device)
classifier_tail.to(device)
siamese_tail.to(device)
mlp_tail.to(device)


# In[23]:


net.eval()
classifier_tail.eval()
siamese_tail.eval()
mlp_tail.eval()

val_embeddings = [[]]

correct = 0
total = 0

cerr = [0, 0]

with torch.no_grad():
    for data in valloader:
        inputs, labels = data
        inputs = inputs.permute((0, 2, 1))
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = augment_center_crop(inputs)

        output_embeddings = net(inputs)
        outputs = classifier_tail(output_embeddings)
        siamese_tail_embeddings = siamese_tail(output_embeddings)
        val_embeddings[-1].append(siamese_tail_embeddings.detach().cpu().numpy())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    val_embeddings[-1] = np.concatenate(val_embeddings[-1], axis=0)
    dist_matrix = squareform(pdist(val_embeddings[-1], metric='cosine'))
    #dist_matrix = squareform(pdist(val_embeddings[-1], metric='euclidean'))
    cosdist = dist_matrix / 2.0
    cos_scores = 1.0 - cosdist
    class_labels = val_labels_tensor.cpu().numpy()

    fars = bootstrap_metric_thr(cos_scores, class_labels, np.linspace(0.01, 0.99, 99), iters=100)
    frrs = bootstrap_metric_thr_frr(cos_scores, class_labels, np.linspace(0.01, 0.99, 99), iters=100)

    thr_space = np.linspace(0.01, 0.99, 99)
    frr_means = frrs.mean(0)
    far_means = fars.mean(0)

    thr_nanargmin = np.nanargmin(np.absolute((far_means - frr_means)))
    print(f'thr_nanargmin: {thr_nanargmin}, thr: {thr_space[thr_nanargmin]}')
    eer_threshold = thr_space[thr_nanargmin]
    cerr[0] = frr_means[thr_nanargmin]
    cerr[1] = far_means[thr_nanargmin]
print(cerr)


# In[24]:


net.eval()
classifier_tail.eval()
siamese_tail.eval()
mlp_tail.eval()

embeddings = []

with torch.no_grad():
    for data in trainloader_add_test:
        inputs, labels = data
        inputs = inputs.permute((0, 2, 1))
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = augment_center_crop(inputs)

        output_embeddings = net(inputs)
        outputs = classifier_tail(output_embeddings)
        siamese_tail_embeddings = siamese_tail(output_embeddings)
        embeddings.append(siamese_tail_embeddings.detach().cpu().numpy())
        #_, predicted = torch.max(outputs.data, 1)
        #total += labels.size(0)
        #correct += (predicted == labels).sum().item()
    embeddings = np.concatenate(embeddings, axis=0)
    dist_matrix = squareform(pdist(embeddings, metric='cosine'))
    #dist_matrix = squareform(pdist(embeddings, metric='euclidean'))
    cosdist = dist_matrix / 2.0
    cos_scores = 1.0 - cosdist
    class_labels = train_labels_tensor_add_test.cpu().numpy()

    thr_filler = np.linspace(thr_space[thr_nanargmin], thr_space[thr_nanargmin], 1)
    fars_filler = bootstrap_metric_thr(cos_scores, class_labels, thr_filler, iters=1000)
    frrs_filler = bootstrap_metric_thr_frr(cos_scores, class_labels, thr_filler, iters=1000)

    frr_means_filler = frrs_filler.mean(0)
    far_means_filler = fars_filler.mean(0)

    frr_estimate = frr_means_filler[0]
    fsa_estimate = far_means_filler[0]
    print(f'hold out users: frr est: {frr_estimate * 100:.4f}, fsa est: {fsa_estimate * 100:.4f}')


# In[25]:


import heapq
import pickle

class CheckpointManager():
    def __init__(self, path_to_checkpoints, n_best=-1):
        self.path_to_checkpoints = path_to_checkpoints
        self.n_best = n_best
        self.score_heap = []
        self.score_heap_filepath = os.path.join(self.path_to_checkpoints, 'score_heap.pkl')
    
    def add_checkpoint(self, state, score, checkpoint_name):
        heapq.heappush(self.score_heap, (score, checkpoint_name))
        torch.save(state, os.path.join(self.path_to_checkpoints, checkpoint_name))
        if self.n_best != -1 and len(self.score_heap) > self.n_best:
            _, worst_checkpoint_name = heapq.heappop(self.score_heap)
            os.remove(os.path.join(self.path_to_checkpoints, worst_checkpoint_name))
        with open(self.score_heap_filepath, 'wb') as handle:
            pickle.dump(list(self.score_heap), handle, protocol=pickle.HIGHEST_PROTOCOL)
        return


# In[33]:


def compute_frr10(scores, class_labels, thr_space, iters=100, njobs=10):
    fars = bootstrap_metric_thr(scores, class_labels, thr_space, iters=iters)
    frrs = bootstrap_metric_thr_frr(scores, class_labels, thr_space, iters=iters)
    
    frr_means = frrs.mean(0)
    far_means = fars.mean(0)
    
    ind_to = np.where(frr_means > 0.1)[0][0]
    return frr_means[ind_to - 1], far_means[ind_to - 1]


def finetune_model(finetune_class, checkpoint_name, nepochs):
    finetune_path = finetune_paths[finetune_class]
    print(f'finetune model {checkpoint_name} for class {finetune_class}')

    # upsample few train samples of a user
    data_to_resample = dataset_add_test['train']['data'][dataset_add_test['train']['labels'] == finetune_class]
    resampled_indices = np.random.choice(np.arange(data_to_resample.shape[0]), dataset['train']['data'].shape[0], replace=True)
    ft_data_train_resampled = data_to_resample[resampled_indices].copy()
    
    # prepare datasets for finetuning
    finetune_train_data_tensor = torch.from_numpy(np.concatenate((
        dataset['train']['data'],
        #dataset_add_test['train']['data'][dataset_add_test['train']['labels'] == finetune_class]
        ft_data_train_resampled,
    ), axis=0)).float()

    finetune_val_data_tensor = torch.from_numpy(np.concatenate((
        dataset['val']['data'],
        dataset_add_test['val']['data'][dataset_add_test['val']['labels'] == finetune_class]
    ), axis=0)).float()

    finetune_test_data_tensor = torch.from_numpy(np.concatenate((
        dataset['test']['data'],
        dataset_add_test['test']['data'][dataset_add_test['test']['labels'] == finetune_class]
    ), axis=0)).float()

    finetune_train_labels_tensor = torch.from_numpy(
        np.concatenate((
            np.zeros_like(dataset['train']['labels']),
            #dataset_add_test['train']['labels'][dataset_add_test['train']['labels'] == finetune_class]
            np.ones(ft_data_train_resampled.shape[0])
        ), axis=0)
    ).type(torch.LongTensor)

    finetune_val_labels_tensor = torch.from_numpy(
        np.concatenate((
            np.zeros_like(dataset['val']['labels']),
            np.ones_like(dataset_add_test['val']['labels'][dataset_add_test['val']['labels'] == finetune_class])
        ), axis=0)
    ).type(torch.LongTensor)

    finetune_test_labels_tensor = torch.from_numpy(
        np.concatenate((
            np.zeros_like(dataset['test']['labels']),
            np.ones_like(dataset_add_test['test']['labels'][dataset_add_test['test']['labels'] == finetune_class])
        ), axis=0)
    ).type(torch.LongTensor)
    
    finetune_train_dataset = torch.utils.data.TensorDataset(finetune_train_data_tensor, finetune_train_labels_tensor)

    finetune_trainloader = torch.utils.data.DataLoader(finetune_train_dataset, batch_size=128,
                                              shuffle=True, num_workers=0)

    finetune_val_dataset = torch.utils.data.TensorDataset(finetune_val_data_tensor, finetune_val_labels_tensor)

    finetune_valloader = torch.utils.data.DataLoader(finetune_val_dataset, batch_size=128,
                                              shuffle=False, num_workers=0)

    finetune_test_dataset = torch.utils.data.TensorDataset(finetune_test_data_tensor, finetune_test_labels_tensor)

    finetune_testloader = torch.utils.data.DataLoader(finetune_test_dataset, batch_size=128,
                                              shuffle=False, num_workers=0)
    
    # load model, freeze extractor, change classification task to binary (one vs all)
    checkpoint_path = os.path.join(results_path, checkpoint_name)
    state = torch.load(checkpoint_path)

    net = Net()
    classifier_tail = ClassifierTail(n_classes=nclasses, n_branches=22)
    siamese_tail = SiameseTail(n_branches=22)
    mlp_tail = SimpleMLP(n_features=300)

    net.load_state_dict(state['net'])
    classifier_tail.load_state_dict(state['classifier_tail'])
    siamese_tail.load_state_dict(state['siamese_tail'])
    mlp_tail.load_state_dict(state['mlp_tail'])


    feature_extract = True

    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    set_parameter_requires_grad(net, feature_extract)
    #set_parameter_requires_grad(classifier_tail, feature_extract)
    #set_parameter_requires_grad(siamese_tail, feature_extract)
    #set_parameter_requires_grad(mlp_tail, feature_extract)

    classifier_in_features = classifier_tail.classifier[8].in_features

    classifier_tail.classifier[8] = nn.Linear(in_features=classifier_in_features, out_features=2)

    net.to(device)
    classifier_tail.to(device)
    siamese_tail.to(device)
    mlp_tail.to(device)
    
    # create losses and optimizer
    from pytorch_metric_learning import miners, losses
    miner = miners.MultiSimilarityMiner()

    from pytorch_metric_learning.distances import CosineSimilarity
    from pytorch_metric_learning.reducers import ThresholdReducer
    from pytorch_metric_learning.regularizers import LpRegularizer
    from pytorch_metric_learning import losses
    #loss_func = losses.TripletMarginLoss(distance = CosineSimilarity(),
    #                                    reducer = ThresholdReducer(low=0.0, high=0.5), 
    #                                    embedding_regularizer = LpRegularizer())

    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    #contrastive_criterion = ContrastiveLoss(margin=100)
    loss_func = losses.TripletMarginLoss(margin=0.5)
    supconloss = SupConLoss(device)
    optimizer = optim.SGD(list(classifier_tail.parameters()) + list(siamese_tail.parameters()) + list(mlp_tail.parameters()), lr=0.0005, momentum=0.9)
    
    # train model
    finetuning_manager = CheckpointManager(finetune_path, n_best=50)

    best_score = 0
    best_cerr = [10, 10]

    val_embeddings = []

    for epoch in range(nepochs):  # loop over the dataset multiple times
        net.train()
        classifier_tail.train()
        siamese_tail.train()
        mlp_tail.train()
        running_loss = 0.0
        running_triplet_loss = 0.0
        running_scloss = 0.0
        running_total_loss = 0.0
        train_correct = 0
        train_total = 0

        val_embeddings.append([])

        for i, data in enumerate(finetune_trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.permute((0, 2, 1))
            inputs, labels = inputs.to(device), labels.to(device)
            inputs1 = augment_mag_noise(augment_random_crop(inputs))
            inputs2 = augment_mag_noise(augment_random_crop(inputs))

            bsize = inputs.shape[0]

            inputs_cat = torch.cat((inputs1, inputs2), dim=0)
            labels_cat = torch.cat((labels, labels), dim=0)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output_embeddings = net(inputs_cat)
            outputs = classifier_tail(output_embeddings)
            outputs1, outputs2 = outputs[:bsize, ...], outputs[bsize:, ...]
            output_embeddings1, output_embeddings2 = output_embeddings[:bsize, ...], output_embeddings[bsize:, ...]

            output_embeddings1 = net(inputs1)
            output_embeddings2 = net(inputs2)


            siamese_tail_embeddings = siamese_tail(output_embeddings)
            hard_pairs = miner(siamese_tail_embeddings, labels_cat)
            loss = criterion(outputs, labels_cat)
            alpha_c = 1.0#(min(100, epoch)) / 100
            triplet_loss = loss_func(siamese_tail_embeddings, labels_cat, hard_pairs)

            mlp_features = mlp_tail(siamese_tail_embeddings)

            scloss = supconloss(mlp_features.reshape((bsize, 2, -1)), labels)

            total_loss = 1.0 * loss + alpha_c * triplet_loss + 1.0 * scloss
            #total_loss = alpha_c * triplet_loss

            total_loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels_cat.size(0)
            train_correct += (predicted == labels_cat).sum().item()
            # print statistics
            running_loss += loss.item()
            running_triplet_loss += triplet_loss.item()
            running_scloss += scloss.item()
            running_total_loss += total_loss.item()
    #         if i % 2000 == 1999:    # print every 2000 mini-batches
    #             print('[%d, %5d] loss: %.3f' %
    #                   (epoch + 1, i + 1, running_loss / 2000))
    #             running_loss = 0.0

        print(f'epoch {epoch}: loss={running_loss / i:.4f}, t_loss={running_triplet_loss / i:.4f}, s_loss={running_scloss / i:.4f}, total_loss={running_total_loss / i:.4f}')

        net.eval()
        classifier_tail.eval()
        siamese_tail.eval()
        mlp_tail.eval()
        correct = 0
        total = 0

        cerr = [0, 0]

        with torch.no_grad():
            for data in finetune_valloader:
                inputs, labels = data
                inputs = inputs.permute((0, 2, 1))
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = augment_center_crop(inputs)

                output_embeddings = net(inputs)
                outputs = classifier_tail(output_embeddings)
                siamese_tail_embeddings = siamese_tail(output_embeddings)
                val_embeddings[-1].append(siamese_tail_embeddings.detach().cpu().numpy())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            val_embeddings[-1] = np.concatenate(val_embeddings[-1], axis=0)
            dist_matrix = squareform(pdist(val_embeddings[-1], metric='cosine'))
            cosdist = dist_matrix / 2.0
            cos_scores = 1.0 - cosdist
            class_labels = finetune_val_labels_tensor.cpu().numpy()

            fars = bootstrap_metric_thr(cos_scores, class_labels, np.linspace(0.01, 0.99, 99), iters=25)
            frrs = bootstrap_metric_thr_frr(cos_scores, class_labels, np.linspace(0.01, 0.99, 99), iters=25)

            thr_space = np.linspace(0.01, 0.99, 99)
            frr_means = frrs.mean(0)
            far_means = fars.mean(0)

            eer_threshold = thr_space[np.nanargmin(np.absolute((far_means - frr_means)))]
            cerr[0] = frr_means[np.nanargmin(np.absolute((far_means - frr_means)))]
            cerr[1] = far_means[np.nanargmin(np.absolute((far_means - frr_means)))]
        current_score = 100 * correct / total

        current_cerr = -np.mean(cerr)

        current_state = {
            'net': net.state_dict(),
            'classifier_tail': classifier_tail.state_dict(),
            'siamese_tail': siamese_tail.state_dict(),
            'mlp_tail': mlp_tail.state_dict(),
        }
        cur_checkpoint_name = f'finetune_smoltrain_model_{epoch}_cls_{finetune_class}.pth'
        finetuning_manager.add_checkpoint(current_state, current_cerr, cur_checkpoint_name)
        #torch.save(net.state_dict(), CUR_MODEL_PATH)
        #torch.save(siamese_tail.state_dict(), CUR_SIAMESE_MODEL_PATH)

        if current_score > best_score:
            best_score = current_score
        if np.mean(cerr) < np.mean(best_cerr):
            best_cerr = cerr
        #if current_score > best_score:
        #    torch.save(net.state_dict(), BEST_MODEL_PATH)
        #    torch.save(siamese_tail.state_dict(), BEST_SIAMESE_MODEL_PATH)
        #    best_score = current_score


        print(f'Accuracy on train: {100 * train_correct / train_total:.4f} %, val: {100 * correct / total:.4f} %, best val: {best_score:.4f} %, val EER: {cerr[0] * 100:.2f} {cerr[1] * 100:.2f}, best_EER: {best_cerr[0] * 100:.2f} {best_cerr[1] * 100:.2f}')

    print('Finished Training. Started testing...')
    
    net.eval()
    classifier_tail.eval()
    siamese_tail.eval()
    mlp_tail.eval()

    embeddings = []

    with torch.no_grad():
        for data in testloader_add_test:
            inputs, labels = data
            inputs = inputs.permute((0, 2, 1))
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = augment_center_crop(inputs)

            output_embeddings = net(inputs)
            outputs = classifier_tail(output_embeddings)
            siamese_tail_embeddings = siamese_tail(output_embeddings)
            embeddings.append(siamese_tail_embeddings.detach().cpu().numpy())
            #_, predicted = torch.max(outputs.data, 1)
            #total += labels.size(0)
            #correct += (predicted == labels).sum().item()
        embeddings = np.concatenate(embeddings, axis=0)
        dist_matrix = squareform(pdist(embeddings, metric='cosine'))
        #dist_matrix = squareform(pdist(embeddings, metric='euclidean'))
        cosdist = dist_matrix / 2.0
        cos_scores = 1.0 - cosdist
        #class_labels = test_labels_tensor_add_test.cpu().numpy()
        ksksks_labels = test_labels_tensor_add_test.cpu().numpy()
        ksksks_labels = (ksksks_labels != finetune_class).astype(int)

        thr_space = np.linspace(0.01, 0.99, 99)
        #thr_filler = np.linspace(thr_space, thr_space[thr_nanargmin], 1)
        fars = bootstrap_metric_thr(cos_scores, ksksks_labels, thr_space, iters=1000)
        frrs = bootstrap_metric_thr_frr(cos_scores, ksksks_labels, thr_space, iters=1000)

        frr_means = frrs.mean(0)
        far_means = fars.mean(0)

        #frr_estimate = frr_means_filler[0]
        #fsa_estimate = far_means_filler[0]
        #print(f'frr est: {frr_estimate * 100:.4f}, fsa est: {fsa_estimate * 100:.4f}')
    frr10, far_frr10 = compute_frr10(cos_scores, ksksks_labels, thr_space)
    print(f'frr10: {frr10 * 100:.4f}, fsa10: {far_frr10 * 100:.4f}')
    print('-'*80)

    return frr10, far_frr10


# In[35]:


results = []

for finetune_class in range(nclasses_add_test):
    results.append(finetune_model(finetune_class, checkpoint_name, 50))


# In[37]:


results


# In[44]:


[el[1] * 100 for el in results]


# In[41]:


sns.histplot([el[1] * 100 for el in results], bins=10)


# In[43]:


print(f'mean FSA @ 10FRR: {np.mean([el[1] for el in results]) * 100}')


# In[ ]:





# In[45]:


pwd


# # THE END

# In[229]:





# In[201]:


classifier_tail = ClassifierTail(n_classes=2, n_branches=22)
siamese_tail = SiameseTail(n_branches=22)
mlp_tail = SimpleMLP(n_features=300)
classifier_tail.to(device)
siamese_tail.to(device)
mlp_tail.to(device)


# In[231]:





# In[232]:


finetuning_manager = CheckpointManager(finetune_path, n_best=50)

best_score = 0
best_cerr = [10, 10]

val_embeddings = []

for epoch in range(2000):  # loop over the dataset multiple times
    net.train()
    classifier_tail.train()
    siamese_tail.train()
    mlp_tail.train()
    running_loss = 0.0
    running_triplet_loss = 0.0
    running_scloss = 0.0
    running_total_loss = 0.0
    train_correct = 0
    train_total = 0
            
    val_embeddings.append([])
    
    for i, data in enumerate(finetune_trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.permute((0, 2, 1))
        inputs, labels = inputs.to(device), labels.to(device)
        inputs1 = augment_mag_noise(augment_random_crop(inputs))
        inputs2 = augment_mag_noise(augment_random_crop(inputs))
        
        bsize = inputs.shape[0]
        
        inputs_cat = torch.cat((inputs1, inputs2), dim=0)
        labels_cat = torch.cat((labels, labels), dim=0)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output_embeddings = net(inputs_cat)
        outputs = classifier_tail(output_embeddings)
        outputs1, outputs2 = outputs[:bsize, ...], outputs[bsize:, ...]
        output_embeddings1, output_embeddings2 = output_embeddings[:bsize, ...], output_embeddings[bsize:, ...]
        
        output_embeddings1 = net(inputs1)
        output_embeddings2 = net(inputs2)
        

        siamese_tail_embeddings = siamese_tail(output_embeddings)
        hard_pairs = miner(siamese_tail_embeddings, labels_cat)
        loss = criterion(outputs, labels_cat)
        alpha_c = 1.0#(min(100, epoch)) / 100
        triplet_loss = loss_func(siamese_tail_embeddings, labels_cat, hard_pairs)
        
        mlp_features = mlp_tail(siamese_tail_embeddings)
        
        scloss = supconloss(mlp_features.reshape((bsize, 2, -1)), labels)
        
        total_loss = 1.0 * loss + alpha_c * triplet_loss + 1.0 * scloss
        #total_loss = alpha_c * triplet_loss
        
        total_loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        train_total += labels_cat.size(0)
        train_correct += (predicted == labels_cat).sum().item()
        # print statistics
        running_loss += loss.item()
        running_triplet_loss += triplet_loss.item()
        running_scloss += scloss.item()
        running_total_loss += total_loss.item()
#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0
    
    print(f'epoch {epoch}: loss={running_loss / i:.4f}, t_loss={running_triplet_loss / i:.4f}, s_loss={running_scloss / i:.4f}, total_loss={running_total_loss / i:.4f}')
    
    net.eval()
    classifier_tail.eval()
    siamese_tail.eval()
    mlp_tail.eval()
    correct = 0
    total = 0
    
    cerr = [0, 0]
    
    with torch.no_grad():
        for data in finetune_valloader:
            inputs, labels = data
            inputs = inputs.permute((0, 2, 1))
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = augment_center_crop(inputs)

            output_embeddings = net(inputs)
            outputs = classifier_tail(output_embeddings)
            siamese_tail_embeddings = siamese_tail(output_embeddings)
            val_embeddings[-1].append(siamese_tail_embeddings.detach().cpu().numpy())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        val_embeddings[-1] = np.concatenate(val_embeddings[-1], axis=0)
        dist_matrix = squareform(pdist(val_embeddings[-1], metric='cosine'))
        cosdist = dist_matrix / 2.0
        cos_scores = 1.0 - cosdist
        class_labels = finetune_val_labels_tensor.cpu().numpy()
        
        fars = bootstrap_metric_thr(cos_scores, class_labels, np.linspace(0.01, 0.99, 99), iters=25)
        frrs = bootstrap_metric_thr_frr(cos_scores, class_labels, np.linspace(0.01, 0.99, 99), iters=25)
                
        thr_space = np.linspace(0.01, 0.99, 99)
        frr_means = frrs.mean(0)
        far_means = fars.mean(0)
        
        eer_threshold = thr_space[np.nanargmin(np.absolute((far_means - frr_means)))]
        cerr[0] = frr_means[np.nanargmin(np.absolute((far_means - frr_means)))]
        cerr[1] = far_means[np.nanargmin(np.absolute((far_means - frr_means)))]
    current_score = 100 * correct / total
    
    current_cerr = -np.mean(cerr)
    
    current_state = {
        'net': net.state_dict(),
        'classifier_tail': classifier_tail.state_dict(),
        'siamese_tail': siamese_tail.state_dict(),
        'mlp_tail': mlp_tail.state_dict(),
    }
    cur_checkpoint_name = f'finetune_smoltrain_model_{epoch}_cls_{finetune_class}.pth'
    finetuning_manager.add_checkpoint(current_state, current_cerr, cur_checkpoint_name)
    #torch.save(net.state_dict(), CUR_MODEL_PATH)
    #torch.save(siamese_tail.state_dict(), CUR_SIAMESE_MODEL_PATH)
    
    if current_score > best_score:
        best_score = current_score
    if np.mean(cerr) < np.mean(best_cerr):
        best_cerr = cerr
    #if current_score > best_score:
    #    torch.save(net.state_dict(), BEST_MODEL_PATH)
    #    torch.save(siamese_tail.state_dict(), BEST_SIAMESE_MODEL_PATH)
    #    best_score = current_score
    

    print(f'Accuracy on train: {100 * train_correct / train_total:.4f} %, val: {100 * correct / total:.4f} %, best val: {best_score:.4f} %, val EER: {cerr[0] * 100:.2f} {cerr[1] * 100:.2f}, best_EER: {best_cerr[0] * 100:.2f} {best_cerr[1] * 100:.2f}')

print('Finished Training')


# In[ ]:





# In[ ]:





# In[ ]:





# In[241]:





# In[242]:





# In[243]:





# In[244]:


print(frr10, far_frr10)


# In[ ]:





# In[ ]:





# In[ ]:





# In[177]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[138]:


train_data_tensor = torch.from_numpy(dataset['train']['data']).float()
train_labels_tensor = torch.from_numpy(dataset['train']['labels']).type(torch.LongTensor)

train_dataset = torch.utils.data.TensorDataset(train_data_tensor, train_labels_tensor)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128,
                                          shuffle=True, num_workers=0)


val_data_tensor = torch.from_numpy(dataset['val']['data']).float()
val_labels_tensor = torch.from_numpy(dataset['val']['labels']).type(torch.LongTensor)

val_dataset = torch.utils.data.TensorDataset(val_data_tensor, val_labels_tensor)

valloader = torch.utils.data.DataLoader(val_dataset, batch_size=128,
                                          shuffle=False, num_workers=0)


test_data_tensor = torch.from_numpy(dataset['test']['data']).float()
test_labels_tensor = torch.from_numpy(dataset['test']['labels']).type(torch.LongTensor)

test_dataset = torch.utils.data.TensorDataset(test_data_tensor, test_labels_tensor)

testloader = torch.utils.data.DataLoader(test_dataset, batch_size=128,
                                          shuffle=False, num_workers=0)


# In[140]:


train_data_tensor_add_test = torch.from_numpy(dataset_add_test['train']['data']).float()
train_labels_tensor_add_test = torch.from_numpy(dataset_add_test['train']['labels']).type(torch.LongTensor)

train_dataset_add_test = torch.utils.data.TensorDataset(train_data_tensor_add_test, train_labels_tensor_add_test)

trainloader_add_test = torch.utils.data.DataLoader(train_dataset_add_test, batch_size=128,
                                          shuffle=True, num_workers=0)


val_data_tensor_add_test = torch.from_numpy(dataset_add_test['val']['data']).float()
val_labels_tensor_add_test = torch.from_numpy(dataset_add_test['val']['labels']).type(torch.LongTensor)

val_dataset_add_test = torch.utils.data.TensorDataset(val_data_tensor_add_test, val_labels_tensor_add_test)

valloader_add_test = torch.utils.data.DataLoader(val_dataset_add_test, batch_size=128,
                                          shuffle=False, num_workers=0)


test_data_tensor_add_test = torch.from_numpy(dataset_add_test['test']['data']).float()
test_labels_tensor_add_test = torch.from_numpy(dataset_add_test['test']['labels']).type(torch.LongTensor)

test_dataset_add_test = torch.utils.data.TensorDataset(test_data_tensor_add_test, test_labels_tensor_add_test)

testloader_add_test = torch.utils.data.DataLoader(test_dataset_add_test, batch_size=128,
                                          shuffle=False, num_workers=0)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[60]:





# In[ ]:





# In[ ]:





# In[ ]:


best_score = 0

val_embeddings = []

for epoch in range(2000):  # loop over the dataset multiple times
    net.train()
    siamese_tail.train()
    mlp_tail.train()
    running_loss = 0.0
    running_triplet_loss = 0.0
    running_scloss = 0.0
    running_total_loss = 0.0
    train_correct = 0
    train_total = 0
            
    val_embeddings.append([])
    
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.permute((0, 2, 1))
        inputs, labels = inputs.to(device), labels.to(device)
        inputs1 = augment_mag_noise(augment_random_crop(inputs))
        inputs2 = augment_mag_noise(augment_random_crop(inputs))
        
        bsize = inputs.shape[0]
        
        inputs_cat = torch.cat((inputs1, inputs2), dim=0)
        labels_cat = torch.cat((labels, labels), dim=0)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs, output_embeddings = net(inputs_cat)
        outputs1, outputs2 = outputs[:bsize, ...], outputs[bsize:, ...]
        output_embeddings1, output_embeddings2 = output_embeddings[:bsize, ...], output_embeddings[bsize:, ...]
        
        outputs1, output_embeddings1 = net(inputs1)
        outputs2, output_embeddings2 = net(inputs2)
        

        siamese_tail_embeddings = siamese_tail(output_embeddings)
        hard_pairs = miner(siamese_tail_embeddings, labels_cat)
        
        loss = criterion(outputs, labels_cat)
        alpha_c = 1.0#(min(100, epoch)) / 100
        triplet_loss = loss_func(siamese_tail_embeddings, labels_cat, hard_pairs)
        
        mlp_features = mlp_tail(siamese_tail_embeddings)
        
        scloss = supconloss(mlp_features.reshape((bsize, 2, -1)), labels)
        
        total_loss = loss + alpha_c * triplet_loss + scloss
        #total_loss = alpha_c * triplet_loss
        
        total_loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        train_total += labels_cat.size(0)
        train_correct += (predicted == labels_cat).sum().item()
        # print statistics
        running_loss += loss.item()
        running_triplet_loss += triplet_loss.item()
        running_scloss += scloss.item()
        running_total_loss += total_loss.item()
#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0
    
    print(f'epoch {epoch}: loss={running_loss / i:.4f}, t_loss={running_triplet_loss / i:.4f}, s_loss={running_scloss / i:.4f}, total_loss={running_total_loss / i:.4f}')
    
    
    net.eval()
    siamese_tail.eval()
    mlp_tail.eval()
    correct = 0
    total = 0
    
    cerr = [0, 0]
    
    with torch.no_grad():
        for data in valloader:
            inputs, labels = data
            inputs = inputs.permute((0, 2, 1))
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = augment_center_crop(inputs)

            outputs, output_embeddings = net(inputs)
            siamese_tail_embeddings = siamese_tail(output_embeddings)
            val_embeddings[-1].append(siamese_tail_embeddings.detach().cpu().numpy())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        val_embeddings[-1] = np.concatenate(val_embeddings[-1], axis=0)
        dist_matrix = squareform(pdist(val_embeddings[-1], metric='cosine'))
        cosdist = dist_matrix / 2.0
        cos_scores = 1.0 - cosdist
        class_labels = val_labels_tensor.cpu().numpy()
        
        fars = bootstrap_metric_thr(cos_scores, class_labels, np.linspace(0.01, 0.99, 99), iters=25)
        frrs = bootstrap_metric_thr_frr(cos_scores, class_labels, np.linspace(0.01, 0.99, 99), iters=25)
        
        thr_space = np.linspace(0.01, 0.99, 99)
        frr_means = frrs.mean(0)
        far_means = fars.mean(0)
        
        eer_threshold = thr_space[np.nanargmin(np.absolute((far_means - frr_means)))]
        cerr[0] = frr_means[np.nanargmin(np.absolute((far_means - frr_means)))]
        cerr[1] = far_means[np.nanargmin(np.absolute((far_means - frr_means)))]
    current_score = 100 * correct / total
    
    current_state = {
        'net': net.state_dict(),
        'siamese_tail': siamese_tail.state_dict(),
        'mlp_tail': mlp_tail.state_dict(),
    }
    cur_checkpoint_name = f'smoltrain_model_{epoch}.pth'
    training_manager.add_checkpoint(current_state, current_score, cur_checkpoint_name)
    #torch.save(net.state_dict(), CUR_MODEL_PATH)
    #torch.save(siamese_tail.state_dict(), CUR_SIAMESE_MODEL_PATH)
    
    if current_score > best_score:
        best_score = current_score
    #if current_score > best_score:
    #    torch.save(net.state_dict(), BEST_MODEL_PATH)
    #    torch.save(siamese_tail.state_dict(), BEST_SIAMESE_MODEL_PATH)
    #    best_score = current_score
    

    print(f'Accuracy on train: {100 * train_correct / train_total:.4f} %, val: {100 * correct / total:.4f} %, best val: {best_score:.4f} %, val EER: {cerr[0] * 100:.2f} {cerr[1] * 100:.2f}')

print('Finished Training')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


from pytorch_metric_learning import miners, losses
miner = miners.MultiSimilarityMiner()

from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.regularizers import LpRegularizer
from pytorch_metric_learning import losses
#loss_func = losses.TripletMarginLoss(distance = CosineSimilarity(),
#                                    reducer = ThresholdReducer(low=0.0, high=0.5), 
#                                    embedding_regularizer = LpRegularizer())

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
#contrastive_criterion = ContrastiveLoss(margin=100)
loss_func = losses.TripletMarginLoss()
supconloss = SupConLoss(device)
optimizer = optim.SGD(list(net.parameters()) + list(siamese_tail.parameters()) + list(mlp_tail.parameters()), lr=0.01, momentum=0.9)

