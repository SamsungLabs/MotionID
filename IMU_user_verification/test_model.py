import os
import torch
import pickle
import heapq
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
import argparse

import scipy
from scipy.spatial.distance import pdist, squareform


import torch.optim as optim

from utils import augment_random_crop, augment_center_crop, augment_mag_noise
from utils import load_imu_data, to_tensor_dataloader
from utils import bootstrap_metric_thr_far, bootstrap_metric_thr_frr, est_frr_q
from utils import generate_dataset_features
from utils import CheckpointManager
from models import model_by_name, optimizer_by_name, lr_scheduler_by_name, SupConLoss

from pytorch_metric_learning import miners, losses
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.regularizers import LpRegularizer
from pytorch_metric_learning import losses


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)


def val_step(net, val_loader, config, iters=1000, m=90, q=0.1, p=0.8):
    device = config['device']
    batch_size = config['batch_size']

    net.eval()
    correct = 0
    total = 0
    
    val_embeddings = []
    val_labels = []
    
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            inputs = inputs.permute((0, 2, 1))
            
            inputs, labels = inputs.to(device), labels.to(device)
            val_labels.append(labels.data.cpu().numpy())
            #inputs = augment_center_crop(inputs)
            clf_outputs, siamese_tail_embeddings, mlp_features = net(inputs)
            val_embeddings.append(siamese_tail_embeddings.detach().cpu().numpy())
            _, predicted = torch.max(clf_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        val_embeddings = np.concatenate(val_embeddings, axis=0)
        dist_matrix = squareform(pdist(val_embeddings, metric='cosine'))
        cosdist = dist_matrix / 2.0
        cos_scores = 1.0 - cosdist
        val_labels = np.concatenate(val_labels)
        
        far_frr10, frr10 = est_frr_q(cos_scores, val_labels, iters=iters, m=m, q=q, p=p)
        

    current_score = 100 * correct / total
    
    current_far_frr10 = (far_frr10, frr10)

    return current_score, current_far_frr10


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_template.yaml', help='Path to the config file.')
    opts = parser.parse_args()

    config = get_config(opts.config)

    path_to_imu_data = config['path_to_imu_data']
    train_template_name = config['train_template_name'] if 'train_template_name' in config else None
    batch_size = config['batch_size']

    accel_dataset, gyro_dataset, rotation_dataset, magnetic_dataset = load_imu_data(
        path_to_imu_data,
        train_template_name
    )

    dataset = generate_dataset_features(accel_dataset, gyro_dataset, rotation_dataset, magnetic_dataset)

    train_loader = to_tensor_dataloader(
        dataset['train']['data'],
        dataset['train']['labels'],
        batch_size,
    )
    val_loader = to_tensor_dataloader(
        dataset['val']['data'],
        dataset['val']['labels'],
        batch_size,
    )
    test_loader = to_tensor_dataloader(
        dataset['test']['data'],
        dataset['test']['labels'],
        batch_size,
    )

    device = config['device']
    results_path = config['results_path']
    experiment_name = config['experiment_name']
    model_name = config.get('model_name', 'os_net')
    model_params = config.get('model', {})
    optimizer_name = config.get('optimizer_name', 'sgd')
    optimizer_params = config.get('optimizer', {})
    lr_scheduler_name = config.get('lr_scheduler_name', 'none')
    lr_scheduler_params = config.get('lr_scheduler', {})
    epochs = config['epochs']
    batch_size = config['batch_size']

    ce_weight = config.get('ce_weight', 1.0)
    triplet_weight = config.get('triplet_weight', 1.0)
    contrastive_weight = config.get('contrastive_weight', 1.0)

    experiment_path = os.path.join(results_path, experiment_name)

    score_heap_filepath = os.path.join(experiment_path, 'score_heap.pkl')
    with open(score_heap_filepath, 'rb') as handle:
        score_heap = pickle.load(handle)

    heapq.heapify(score_heap)
    checkpoint_name = heapq.nlargest(5, score_heap)[0][1]

    test_log_path = os.path.join(experiment_path, 'test_log.txt')
    with open(test_log_path, 'w') as test_log_writer:
        message = f'checkpoint name: {checkpoint_name}'
        print(message)
        test_log_writer.write(message + '\n')

        checkpoint_path = os.path.join(experiment_path, checkpoint_name)

        Net = model_by_name(model_name)
        net = Net(**model_params)

        state = torch.load(checkpoint_path)
        net.load_state_dict(state['net'])
        net.to(device)

        train_score, train_far_frr10 = val_step(net, train_loader, config, iters=2000, m=90, q=0.1, p=0.8)
        val_score, val_far_frr10 = val_step(net, val_loader, config, iters=2000, m=90, q=0.1, p=0.8)
        test_score, test_far_frr10 = val_step(net, test_loader, config, iters=2000, m=90, q=0.1, p=0.8)
        train_message = (f'train acc: {train_score:.4f} %, '
            f'train far@frr: {train_far_frr10[0] * 100:.4f}%@{train_far_frr10[1] * 100:.2f}%'
        )
        val_message = (f'val acc: {val_score:.4f} %, '
            f'val far@frr: {val_far_frr10[0] * 100:.4f}%@{val_far_frr10[1] * 100:.2f}%'
        )
        test_message = (f'test acc: {test_score:.4f} %, '
            f'test far@frr: {test_far_frr10[0] * 100:.4f}%@{test_far_frr10[1] * 100:.2f}%'
        )
        print(train_message)
        print(val_message)
        print(test_message)
        test_log_writer.write(train_message + '\n')
        test_log_writer.write(val_message + '\n')
        test_log_writer.write(test_message + '\n')