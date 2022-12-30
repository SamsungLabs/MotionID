import os
import numpy as np
import pickle
import torch


def load_imu_data(path_to_imu_data, data_template_name=None):
    if data_template_name is None:
        filelist = os.listdir(path_to_imu_data)
        if len(filelist) != 4:
            raise ValueError('make sure path contains only 4 files')
        try:
            accel_filename = [el for el in filelist if el.startswith('accel_')][0]
            gyro_filename = [el for el in filelist if el.startswith('accel_')][0]
            rotation_filename = [el for el in filelist if el.startswith('accel_')][0]
            magnetic_filename = [el for el in filelist if el.startswith('accel_')][0]
        except:
            raise ValueError('make sure that path contains only 4 files that starts with correct names')
    else:
        accel_filename = f'accel_{data_template_name}.pkl'
        gyro_filename = f'gyro_{data_template_name}.pkl'
        rotation_filename = f'rotation_{data_template_name}.pkl'
        magnetic_filename = f'magnetic_{data_template_name}.pkl'

    with open(os.path.join(path_to_imu_data, accel_filename), 'rb') as handle:
        accel_dataset = pickle.load(handle)
        
    with open(os.path.join(path_to_imu_data, gyro_filename), 'rb') as handle:
        gyro_dataset = pickle.load(handle)

    with open(os.path.join(path_to_imu_data, rotation_filename), 'rb') as handle:
        rotation_dataset = pickle.load(handle)
        
    with open(os.path.join(path_to_imu_data, magnetic_filename), 'rb') as handle:
        magnetic_dataset = pickle.load(handle)

    return accel_dataset, gyro_dataset, rotation_dataset, magnetic_dataset


def to_tensor_dataloader(data, labels, batch_size=128, shuffle=False, num_workers=0):
    data_tensor = torch.from_numpy(data).float()
    labels_tensor = torch.from_numpy(labels).type(torch.LongTensor)
    dataset = torch.utils.data.TensorDataset(data_tensor, labels_tensor)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return loader
