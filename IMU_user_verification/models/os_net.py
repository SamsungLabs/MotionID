import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


from .layers import Cat


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
        self.out_embedding_size = out_put_channel_numebr

    def forward(self, X):
        
        X = self.net(X)

        X = self.averagepool(X)
        X = X.squeeze_(-1)

        return X


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


class ClassifierTail(nn.Module):
    def __init__(self, n_classes, n_branches, branch_embedding_size):
        super().__init__()
        self.n_classes = n_classes
        self.n_branches = n_branches
        self.branch_embedding_size = branch_embedding_size
        sqrt_features_mult = np.sqrt(self.n_branches)
        sqrt_features1 = int(np.round(256 * sqrt_features_mult))
        sqrt_features2 = int(np.round(128 * sqrt_features_mult))
        self.classifier = nn.Sequential(
            nn.Linear(self.branch_embedding_size * self.n_branches, sqrt_features1),
            nn.ReLU6(),
            nn.Dropout(p=0.2),
            nn.Linear(sqrt_features1, sqrt_features2),
            nn.ReLU6(),
            nn.Dropout(p=0.2),
            nn.Linear(sqrt_features2, sqrt_features2),
            nn.ReLU6(),
            nn.Linear(sqrt_features2, self.n_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)


class SiameseTail(nn.Module):
    def __init__(self, n_branches, branch_embedding_size, out_embedding_size):
        super(SiameseTail, self).__init__()
        self.n_branches = n_branches
        self.branch_embedding_size = branch_embedding_size
        self.out_embedding_size = out_embedding_size
        sqrt_features_mult = np.sqrt(self.n_branches)
        sqrt_features1 = int(np.round(128 * sqrt_features_mult))
        sqrt_features2 = int(np.round(64 * sqrt_features_mult))
        self.nonlinear_embedding = nn.Sequential(
            nn.Linear(self.branch_embedding_size * self.n_branches, sqrt_features1),
            nn.ReLU6(),
            nn.Dropout(p=0.1),
            nn.Linear(sqrt_features1, sqrt_features2),
            nn.ReLU6(),
            nn.Dropout(p=0.1),
            nn.Linear(sqrt_features2, self.out_embedding_size),
        )
    
    def forward(self, x1):
        x1_embedding = self.nonlinear_embedding(x1)
        return x1_embedding

class SimpleMLP(nn.Module):
    def __init__(self, out_embedding_size=300):
        super(SimpleMLP, self).__init__()
        self.nonlinear_embedding = nn.Sequential(
            nn.Linear(out_embedding_size, out_embedding_size),
            nn.ReLU6(),
            nn.Linear(out_embedding_size, out_embedding_size),
        )
    
    def forward(self, x1):
        x1_embedding = self.nonlinear_embedding(x1)
        return x1_embedding


class Net(nn.Module):
    def __init__(self, n_branches, n_classes, out_embedding_size=300, crop_size=51):
        super(Net, self).__init__()
        self.crop_size = crop_size
        
        max_kernel_size = 89
        start_kernel_size = 1

        receptive_field_shape = min(int(self.crop_size / 4), max_kernel_size)

        parameter_number_of_layer_list = [8*128, 5*128*256 + 2*256*128]

        # generate parameter list
        layer_parameter_list = generate_layer_parameter_list(start_kernel_size,
                                                             receptive_field_shape,
                                                             parameter_number_of_layer_list,
                                                             in_channel=3)
        
        
        self.n_classes = n_classes
        self.n_branches = n_branches
        self.out_embedding_size = out_embedding_size
        
        self.net_branches = nn.ModuleList(OS_CNN_branch(layer_parameter_list, self.n_classes, False) for _ in range(n_branches))
        self.branch_embedding_size = self.net_branches[0].out_embedding_size
        self.cat = Cat(dim=-1)
        self.classifier_tail = ClassifierTail(self.n_classes, self.n_branches, self.branch_embedding_size)
        self.siamese_tail = SiameseTail(self.n_branches, self.branch_embedding_size, self.out_embedding_size)
        self.mlp_tail = SimpleMLP(self.out_embedding_size)
    
    def forward(self, x):
        chunk_size = x.shape[1] // self.n_branches
        if x.shape[1] != chunk_size * self.n_branches:
            raise ValueError(f'branches shapes are not aligned with input shape {x.shape}')
        
        x = torch.split(x, 3, dim=1)
        branches_out = [branch(x_el) for (branch, x_el) in zip(self.net_branches, x)]
        branches_out_cat = self.cat(branches_out)

        clf_outputs = self.classifier_tail(branches_out_cat)
        siamese_tail_embeddings = self.siamese_tail(branches_out_cat)
        mlp_features = self.mlp_tail(siamese_tail_embeddings)

        return clf_outputs, siamese_tail_embeddings, mlp_features
