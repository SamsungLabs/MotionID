import os
import re
import numpy as np
import pandas as pd
import argparse


def extract_fields(path):
    with open(path, 'r') as reader_stream:
        text = ''.join(reader_stream.readlines())
    
    digit_matcher = r'[-+]?(?:\d*\.\d+|\d+)'
    return re.findall(digit_matcher, text)


def get_model_epoch(results_path, log_name, experiment_name):
    log_filepath = os.path.join(results_path, experiment_name, log_name)
    return extract_fields(log_filepath)[0]


def extract_finetune_stats(results_path):
    finetune_csv_colnames = [
        'user_id',
        'ft_epoch',
        'val_acc',
        'val_far',
        'val_frr',
        'test_acc',
        'test_far',
        'test_frr',
    ]
    test_log_name = 'test_log.txt'
    experiment_names = os.listdir(results_path)
    zips = []
    for experiment_name in experiment_names:
        model_epoch = get_model_epoch(results_path, test_log_name, experiment_name)
        results = []
        for user_id in range(0, 11):
            log_filepath = os.path.join(results_path, experiment_name, f'finetune_test_log_{user_id}.txt')
            results.append(extract_fields(log_filepath))
        experiment_df = pd.DataFrame(data=results, columns=finetune_csv_colnames)
        experiment_df['epoch_from'] = model_epoch
        zips.append((experiment_name, experiment_df))
    return zips


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', type=str, help='Path to the results folder')
    parser.add_argument('--output', type=str, default='finetune_test_results.xlsx', help='Path to metrics file')
    opts = parser.parse_args()

    results_path = opts.results_path
    output_path = opts.output

    experiment_names = os.listdir(results_path)

    zips = extract_finetune_stats(results_path)

    with pd.ExcelWriter(output_path) as writer:
        for experiment_name, experiment_results in zips:
            experiment_results.to_excel(writer, sheet_name=experiment_name)
