import os
import torch
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


def train(train_loader, val_loader, config):
    device = config['device']
    results_path = config['results_path']
    experiment_name = config['experiment_name']
    n_best_checkpoints = config.get('n_best_checkpoint', 50)
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
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)


    training_manager = CheckpointManager(
        path_to_checkpoints=experiment_path,
        n_best=n_best_checkpoints
    )

    best_score = 0
    best_far_frr10 = (100.0, 100.0)

    Net = model_by_name(model_name)

    net = Net(**model_params)
    net.to(device)

    miner = miners.MultiSimilarityMiner()

    criterion = nn.CrossEntropyLoss()
    loss_func = losses.TripletMarginLoss()
    supconloss = SupConLoss(device)
    optimizer = optimizer_by_name(optimizer_name)(list(net.parameters()), **optimizer_params)
    lr_scheduler = lr_scheduler_by_name(lr_scheduler_name)(optimizer, **lr_scheduler_params)

    log_path = os.path.join(experiment_path, 'log.txt')

    with open(log_path, 'w') as log_writer:
        for epoch in range(epochs):  # loop over the dataset multiple times
            net.train()

            running_loss = 0.0
            running_triplet_loss = 0.0
            running_scloss = 0.0
            running_total_loss = 0.0
            train_correct = 0
            train_total = 0
                    
            for i, data in enumerate(train_loader, 0):
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

                clf_outputs, siamese_tail_embeddings, mlp_features = net(inputs_cat)
                
                hard_pairs = miner(siamese_tail_embeddings, labels_cat)
                
                loss = criterion(clf_outputs, labels_cat)
                triplet_loss = loss_func(siamese_tail_embeddings, labels_cat, hard_pairs)
                            
                scloss = supconloss(mlp_features.reshape((bsize, 2, -1)), labels)
                
                total_loss = ce_weight * loss + triplet_weight * triplet_loss + contrastive_weight * scloss
                
                total_loss.backward()
                optimizer.step()
                lr_scheduler.step()

                _, predicted = torch.max(clf_outputs.data, 1)
                train_total += labels_cat.size(0)
                train_correct += (predicted == labels_cat).sum().item()
                # print statistics
                running_loss += loss.item()
                running_triplet_loss += triplet_loss.item()
                running_scloss += scloss.item()
                running_total_loss += total_loss.item()
            
            message = f'epoch {epoch}: loss={running_loss / i:.4f}, t_loss={running_triplet_loss / i:.4f}, s_loss={running_scloss / i:.4f}, total_loss={running_total_loss / i:.4f}'
            print(message)
            log_writer.write(message + '\n')

            current_score, current_far_frr10 = val_step(net, val_loader, config, iters=1000, m=90, q=0.1, p=0.8)
            
            current_state = {
                'net': net.state_dict(),
                'optim': optimizer.state_dict(),
                'epoch': epoch,
            }
            cur_checkpoint_name = f'trained_model_{epoch}.pth'
            training_manager.add_checkpoint(current_state, (-current_far_frr10[0], -current_far_frr10[1]), cur_checkpoint_name)
            
            if current_score > best_score:
                best_score = current_score

            if current_far_frr10 < best_far_frr10:
                best_far_frr10 = current_far_frr10
            
            message = (
                f'Accuracy on train: {100 * train_correct / train_total:.4f} %, '
                f'val: {current_score:.4f} %, best val: {best_score:.4f} %, '
                f'val far@frr: {current_far_frr10[0] * 100:.4f}%@{current_far_frr10[1] * 100:.2f}%, '
                f'best far@frr: {best_far_frr10[0]*100:.4f}%@{best_far_frr10[1]*100:.2f}%'
            )
            print(message)
            log_writer.write(message + '\n')
    return    


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
        shuffle=True,
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

    train(train_loader, val_loader, config)