import torch
import scipy
import numpy as np

from scipy.sparse import coo_matrix


def augment_random_crop(timeseries, nseq=51):
    batch_size, n_channels, seq_length = timeseries.shape
    
    batch_indices = torch.randint(low=0, high=seq_length - nseq + 1, size=(batch_size,))
    timeseries = torch.stack([
        timeseries[i, :, batch_index:batch_index + nseq]
        for i, batch_index in enumerate(batch_indices)
    ], dim=0)
    return timeseries


def augment_mag_noise(timeseries):
    mag_scale = 0.2
    mean, std = timeseries.mean(dim=-1, keepdim=True), timeseries.std(dim=-1, keepdim=True)
    noise = mag_scale * std * torch.randn_like(std)
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
