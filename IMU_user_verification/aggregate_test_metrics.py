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


def extract_test_stats(results_path, log_name):
    csv_colnames = [
        'experiment_name',
        'epoch',
        'train_acc',
        'train_far',
        'train_frr',
        'val_acc',
        'val_far',
        'val_frr',
        'test_acc',
        'test_far',
        'test_frr'
    ]
    experiment_names = os.listdir(results_path)
    results = []
    for experiment_name in experiment_names:
        log_filepath = os.path.join(results_path, experiment_name, log_name)
        results.append([experiment_name] + extract_fields(log_filepath))
    return pd.DataFrame(data=results, columns=csv_colnames)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', type=str, help='Path to the results folder')
    parser.add_argument('--output', type=str, default='test_results.xlsx', help='Path to metrics file')
    opts = parser.parse_args()

    results_path = opts.results_path
    output_path = opts.output

    experiment_names = os.listdir(results_path)

    test_log_name = 'test_log.txt'

    test_results = extract_test_stats(results_path, test_log_name)

    with pd.ExcelWriter(output_path) as writer:
        test_results.to_excel(writer, sheet_name='test results')