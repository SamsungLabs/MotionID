import os
import re
import copy
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
from utils import binary_bootstrap_metric_thr_far, binary_bootstrap_metric_thr_frr, binary_est_frr_q
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
        
        far_frr10, frr10 = binary_est_frr_q(cos_scores, val_labels, iters=iters, m=m, q=q, p=p)
        

    current_score = 100 * correct / total
    
    current_far_frr10 = (far_frr10, frr10)

    return current_score, current_far_frr10


def finetune(train_loader, val_loader, finetune_id, config):
    device = config['device']
    results_path = config['results_path']
    experiment_name = config['experiment_name']
    n_best_checkpoints = config.get('n_best_checkpoint', 10)
    model_name = config.get('model_name', 'os_net')
    model_params = config.get('model', {})
    optimizer_name = config.get('optimizer_name', 'sgd')
    optimizer_params = config.get('optimizer', {})
    lr_scheduler_name = config.get('lr_scheduler_name', 'none')
    lr_scheduler_params = config.get('lr_scheduler', {})
    lr_scheduler_params = copy.deepcopy(lr_scheduler_params)
    #if 'lr' in lr_scheduler_params:
    #    lr_scheduler_params['lr'] = lr_scheduler_params['lr'] * 5e-2
    import torch.optim as optim
    
    ft_lr = 5e-4
    epochs = 100#config['epochs']

    batch_size = config['batch_size']

    ce_weight = config.get('ce_weight', 1.0)
    triplet_weight = config.get('triplet_weight', 1.0)
    contrastive_weight = config.get('contrastive_weight', 1.0)

    experiment_path = os.path.join(results_path, experiment_name)

    #with open(os.path.join(experiment_path, 'reval_log.txt'), 'r') as reader_stream:
    with open(os.path.join(experiment_path, 'test_log.txt'), 'r') as reader_stream:
        checkpoint_name = re.findall(r'[\S]+\.pth', reader_stream.readline())[0]

    message = f'starting from checkpoint {checkpoint_name}'
    print(message)

    training_manager = CheckpointManager(
        path_to_checkpoints=experiment_path,
        score_heap_filename=f'score_heap_finetune_{finetune_id}.pkl',
        #n_best=n_best_checkpoints,\
        n_best=10
    )

    best_score = 0
    best_far_frr10 = (100.0, 100.0)

    Net = model_by_name(model_name)

    net = Net(**model_params)
    net.load_state_dict(torch.load(os.path.join(experiment_path, checkpoint_name))['net'])

    feature_extract = True

    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    set_parameter_requires_grad(net.net_branches, feature_extract)
    net.classifier_tail.classifier[-1] = nn.Linear(net.classifier_tail.classifier[-1].in_features, 2)
    net.to(device)

    miner = miners.MultiSimilarityMiner()

    criterion = nn.CrossEntropyLoss()
    loss_func = losses.TripletMarginLoss(margin=0.5)
    supconloss = SupConLoss(device)
#    optimizer = optimizer_by_name(optimizer_name)(list(net.parameters()), **optimizer_params)
#    lr_scheduler = lr_scheduler_by_name(lr_scheduler_name)(optimizer, **lr_scheduler_params)
    optimizer = optim.SGD(list(net.parameters()), lr=ft_lr, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * epochs, eta_min=0.1 * ft_lr)

    log_path = os.path.join(experiment_path, f'finetune_log_{finetune_id}.txt')

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

            current_score, current_far_frr10 = val_step(
                net,
                val_loader,
                config,
                iters=2000,
                m=90,
                q=0.1,
                p=0.8
            )
            
            current_state = {
                'net': net.state_dict(),
                'optim': optimizer.state_dict(),
                'epoch': epoch,
            }
            cur_checkpoint_name = f'finetuned_model_{finetune_id}_{epoch}.pth'
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
    path_to_holdout_imu_data = config['path_to_holdout_imu_data']
    train_template_name = config['train_template_name'] if 'train_template_name' in config else None
    holdval_template_name = config['holdval_template_name'] if 'holdval_template_name' in config else None
    holdout_template_name = config['holdout_template_name'] if 'holdout_template_name' in config else None

    batch_size = config['batch_size']

    train_datadicts = load_imu_data(
        path_to_imu_data,
        train_template_name,
    )

    holdval_datadicts = load_imu_data(
        path_to_imu_data,
        holdval_template_name,
    )

    holdout_datadicts = load_imu_data(
        path_to_holdout_imu_data,
        holdout_template_name,
    )


    def merge_stages(staged_dataset):
        merged = {}
        merged['data'] = np.concatenate([
            staged_dataset[stage]['data']
            for stage in ['train', 'val', 'test']
            ], axis=0)
        merged['labels'] = np.concatenate([
            staged_dataset[stage]['labels']
            for stage in ['train', 'val', 'test']
            ], axis=0)
        return merged



    train_dataset = generate_dataset_features(*train_datadicts)
    holdval_dataset = generate_dataset_features(*holdval_datadicts)
    holdout_dataset = generate_dataset_features(*holdout_datadicts)

    merged_train = merge_stages(train_dataset)
    merged_holdval = merge_stages(holdval_dataset)
    merged_holdout = merge_stages(holdout_dataset)

    merged_train['labels'] = np.zeros_like(merged_train['labels'])
    merged_holdval['labels'] = np.zeros_like(merged_holdval['labels'])

    n_holdout = len(np.unique(holdout_dataset['train']['labels']))
    message = f'there are {n_holdout} users to finetune to..'
    print(message)

    for finetune_id in range(0, n_holdout):
        message = f'finetune user #{finetune_id}:'
        print(message)

        # train loader
        bool_mask = (holdout_dataset['train']['labels'] == finetune_id)
        choice_len = len(merged_train['data'])
        random_choice = np.random.choice(np.arange(np.sum(bool_mask)), size=choice_len, replace=True)
        holdout_train_masked = {
            'data': holdout_dataset['train']['data'][bool_mask].copy(),
            'labels': np.ones_like(holdout_dataset['train']['labels'][bool_mask].copy()),
        }
        choiced_train_masked = {
            'data': holdout_train_masked['data'][random_choice].copy(),
            'labels': holdout_train_masked['labels'][random_choice].copy(),
        }
        print(f"train distribution: {len(merged_train['data'])} vs {np.sum(bool_mask)}|{choice_len}|{len(choiced_train_masked['data'])}")
        train_loader = to_tensor_dataloader(
            np.concatenate((
                merged_train['data'],
                #holdout_dataset['train']['data'][bool_mask],
                choiced_train_masked['data'],
            ), axis=0),
            np.concatenate((
                merged_train['labels'],
                #np.ones_like(holdout_dataset['train']['labels'][bool_mask]),
                choiced_train_masked['labels'],
            ), axis=0),
            batch_size,
            shuffle=True,
        )

        # val loader
        bool_mask = (holdout_dataset['val']['labels'] == finetune_id)
        print(f"val distribution: {len(merged_holdval['data'])} vs {np.sum(bool_mask)}")
        val_loader = to_tensor_dataloader(
            np.concatenate((
                merged_holdval['data'],
                holdout_dataset['val']['data'][bool_mask],
            ), axis=0),
            np.concatenate((
                merged_holdval['labels'],
                np.ones_like(holdout_dataset['val']['labels'][bool_mask])
            ), axis=0),
            batch_size,
            shuffle=False,
        )

        # val loader
        bool_mask = (holdout_dataset['test']['labels'] == finetune_id)
        bool_umask = (merged_holdout['labels'] != finetune_id)
        #bool_umasks = {
        #    'train': (holdout_dataset['train']['labels'] != finetune_id),
        #    'val': (holdout_dataset['val']['labels'] != finetune_id),
        #    'test': (holdout_dataset['test']['labels'] != finetune_id),
        #}
        print(f'test distribution: {np.sum(bool_umask)} vs {np.sum(bool_mask)}')

        test_loader = to_tensor_dataloader(
            np.concatenate((
                merged_holdout['data'][bool_umask],
                holdout_dataset['test']['data'][bool_mask],
            ), axis=0),
            np.concatenate((
                np.zeros_like(merged_holdout['labels'][bool_umask]),
                np.ones_like(holdout_dataset['test']['labels'][bool_mask])
            ), axis=0),
            batch_size,
            shuffle=False,
        )

        device = config['device']
        results_path = config['results_path']
        experiment_name = config['experiment_name']
        model_name = config.get('model_name', 'os_net')
        model_params = config.get('model', {})
        # optimizer_name = config.get('optimizer_name', 'sgd')
        # optimizer_params = config.get('optimizer', {})
        # lr_scheduler_name = config.get('lr_scheduler_name', 'none')
        # lr_scheduler_params = config.get('lr_scheduler', {})
        # if 'lr' in lr_scheduler_params:
        #     lr = lr * 1e-2
        # epochs = config['epochs']
        # batch_size = config['batch_size']

        # ce_weight = config.get('ce_weight', 1.0)
        # triplet_weight = config.get('triplet_weight', 1.0)
        # contrastive_weight = config.get('contrastive_weight', 1.0)

        experiment_path = os.path.join(results_path, experiment_name)

        # with open(os.path.join(experiment_path, 'reval_log.txt'), 'r') as reader_stream:
        #     checkpoint_name = re.findall(r'[\S]+\.pth', reader_stream.readline())

        # message = f'starting from checkpoint {checkpoint_name}'
        # print(message)

        finetune(train_loader, val_loader, finetune_id, config)

        score_heap_filepath = os.path.join(experiment_path, f'score_heap_finetune_{finetune_id}.pkl')
        with open(score_heap_filepath, 'rb') as handle:
            score_heap = pickle.load(handle)       
        heapq.heapify(score_heap)
        checkpoint_name = heapq.nlargest(5, score_heap)[0][1]

        test_log_path = os.path.join(experiment_path, f'finetune_test_log_{finetune_id}.txt')
        with open(test_log_path, 'w') as test_log_writer:
            message = f'checkpoint name: {checkpoint_name}'
            print(message)
            test_log_writer.write(message + '\n')

            checkpoint_path = os.path.join(experiment_path, checkpoint_name)

            Net = model_by_name(model_name)
            model_params = copy.deepcopy(model_params)
            model_params['n_classes'] = 2
            net = Net(**model_params)

            state = torch.load(checkpoint_path)
            net.load_state_dict(state['net'])
            net.to(device)

            #train_score, train_far_frr10 = val_step(net, train_loader, config, iters=5000, m=90, p=0.8)
            val_score, val_far_frr10 = val_step(net, val_loader, config, iters=5000, m=36, p=0.8)
            test_score, test_far_frr10 = val_step(net, test_loader, config, iters=5000, m=90, p=0.8)
            #train_message = (f'train acc: {train_score:.4f} %, '
            #    f'train far@frr: {train_far_frr10[0] * 100:.4f}%@{train_far_frr10[1] * 100:.2f}%'
            #)
            val_message = (f'val acc: {val_score:.4f} %, '
                f'val far@frr: {val_far_frr10[0] * 100:.4f}%@{val_far_frr10[1] * 100:.2f}%'
            )
            test_message = (f'test acc: {test_score:.4f} %, '
                f'test far@frr: {test_far_frr10[0] * 100:.4f}%@{test_far_frr10[1] * 100:.2f}%'
            )
            #print(train_message)
            print(val_message)
            print(test_message)
            #test_log_writer.write(train_message + '\n')
            test_log_writer.write(val_message + '\n')
            test_log_writer.write(test_message + '\n')
