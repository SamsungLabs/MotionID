import os
import yaml
import argparse


from utils import get_dfs_from_data, get_group_labeling, get_holdout_group_labeling
from utils import split_dfs_by_users, get_stamp_dataset_for_split
from utils import create_tensorized_datasets, write_tensorized_dataset


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/dataset_preparation_config.yaml', help='Path to the config file.')
    opts = parser.parse_args()

    config = get_config(opts.config)

    path_to_users = config['path_to_users']
    path_to_holdout_users = config['path_to_holdout_users']
    path_to_tensorized_datasets = config['path_to_tensorized_datasets']

    nseq_stat = config['nseq_stat']

    resplit_dict = config['resplit']
    resplit_nusers = resplit_dict['nusers']
    resplit_nsplits = resplit_dict['nsplits']
    resplit_init_seeds = resplit_dict['init_seeds']

    holdout_users_dfs = get_dfs_from_data(path_to_holdout_users)
    get_holdout_group_labeling(holdout_users_dfs)

    for nusers in resplit_nusers:
        for (resplit_idx, init_seed) in zip(range(resplit_nsplits), resplit_init_seeds):
            main_users_dfs = get_dfs_from_data(path_to_users)
            get_group_labeling(main_users_dfs)

            data_dir_template = f'split_{nusers}_{resplit_idx}'
            train_fname_template = 'train'
            holdval_fname_template = 'holdval'

            train_dfs, holdval_dfs = split_dfs_by_users(
                main_users_dfs,
                nusers,
                seed=init_seed + nusers)

            train_stamp_dataset, train_stamp_labels = get_stamp_dataset_for_split(train_dfs)
            holdval_stamp_dataset, holdval_stamp_labels = get_stamp_dataset_for_split(holdval_dfs)

            train_ds = create_tensorized_datasets(
                train_dfs,
                train_stamp_dataset,
                train_stamp_labels,
                nseq_stat,
            )
            holdval_ds = create_tensorized_datasets(
                holdval_dfs,
                holdval_stamp_dataset,
                holdval_stamp_labels,
                nseq_stat
            )
            dataset_path = os.path.join(path_to_tensorized_datasets, data_dir_template)
            write_tensorized_dataset(train_ds, dataset_path, train_fname_template)
            write_tensorized_dataset(holdval_ds, dataset_path, holdval_fname_template)

    holdout_dir_template = 'holdout_users'
    holdout_stamp_dataset, holdout_stamp_labels = get_stamp_dataset_for_split(holdout_users_dfs)
    holdout_ds = create_tensorized_datasets(
        holdout_users_dfs,
        holdout_stamp_dataset,
        holdout_stamp_labels,
        nseq_stat,
    )
    holdout_dataset_path = os.path.join(path_to_tensorized_datasets, holdout_dir_template)
    write_tensorized_dataset(holdout_ds, holdout_dataset_path, 'holdout_test')























