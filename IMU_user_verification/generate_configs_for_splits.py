import os
import yaml
import argparse


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)


def generate_config(nuser, nsplit, device):
    config_template = f"""device: {device}

# logger options
#results_path: /nasDATASETS/_from_nas/imu_results/prod_results
results_path: /nasDATASETS/IMU_specific_motion/results
experiment_name: train_osnet_on_split_{nuser}_{nsplit}

# optimization options
epochs: 2000
batch_size: 128
optimizer_name: sgd
#  weight_decay: 0.0001
optimizer:
  lr: 0.01
  momentum: 0.9
lr_scheduler_name: none
lr_scheduler:
  step_size: 2000
  gamma: 1.0
ce_weight: 1.0
triplet_weight: 1.0
contrast_weight: 1.0

# model options
#branch_embedding_size: 64
model_name: os_net
model:
  n_branches: 22
  out_embedding_size: 300
  n_classes: {nuser}
  crop_size: 51

# data options
path_to_imu_data: /nasDATASETS/IMU_specific_motion/tensorized_datasets/split_{nuser}_{nsplit}
train_template_name: train
holdval_template_name: holdval

path_to_holdout_imu_data: /nasDATASETS/IMU_specific_motion/tensorized_datasets/holdout_users
holdout_template_name: holdout_test

"""
    return config_template


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default='configs/dataset_preparation_config.yaml',
        help='Path to the config file.'
    )
    opts = parser.parse_args()

    config = get_config(opts.config)
    path_to_tensorized_datasets = config['path_to_tensorized_datasets']
    resplit = config['resplit']
    nusers = resplit['nusers']
    nsplits = resplit['nsplits']

    for i, nuser in enumerate(nusers):
        for nsplit in range(nsplits):
            gpu_num = (nsplit + nsplits * i) % 8
            device = f'cuda:{gpu_num}'
            config_content = generate_config(nuser, nsplit, device)
            config_name = f'config_train_split_{nuser}_{nsplit}.yaml'
            print(os.path.join('configs', config_name))
            with open(os.path.join('configs', config_name), 'w') as writestream:
                writestream.write(config_content)





