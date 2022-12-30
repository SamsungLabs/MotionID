import os
import argparse
import numpy as np
import pandas as pd
import struct
import datetime
import tqdm
from multiprocessing import Pool, cpu_count


def streaming_extract(path_to_text_file, path_to_out_file, verbose=True, njobs=-1):
    """Determination of extraction methods for each of the sensors
    Args:
        path_to_text_file: path to the bin file
        path_to_out_file: save path for converted file
    Returns:
    """
    filename_to_header = {
        'accel.txt': ['timestamp', 'X', 'Y', 'Z'],
        'gravity.txt': ['timestamp', 'X', 'Y', 'Z'],
        'gyro.txt': ['timestamp', 'X', 'Y', 'Z'],
        'Light.txt': ['timestamp', 'level'],
        'linAccel.txt': ['timestamp', 'X', 'Y', 'Z'],
        'MagneticField.txt': ['timestamp', 'X', 'Y', 'Z'],
        'Pressure.txt': ['timestamp', 'pressure'],
        'Proximity.txt': ['timestamp', 'distance_cm'],
        'Rotation.txt': ['timestamp', 'X', 'Y', 'Z'],
        'screen.txt': ['timestamp', 'event']
    }
    
    filename_to_processfunc = {
        'accel.txt': parallel_bin_extract,
        'gravity.txt': parallel_bin_extract,
        'gyro.txt': parallel_bin_extract,
        'Light.txt': streaming_text_extract,
        'linAccel.txt': parallel_bin_extract,
        'MagneticField.txt': parallel_bin_extract,
        'Pressure.txt': streaming_text_extract,
        'Proximity.txt': streaming_text_extract,
        'Rotation.txt': parallel_bin_extract,
        'screen.txt': streaming_screen_text_extract,
    }
    filename = os.path.split(path_to_text_file)[-1]
    filename_to_processfunc[filename](path_to_text_file, path_to_out_file, filename_to_header[filename], verbose, njobs)
    return


def streaming_text_extract(path_to_text_file, path_to_out_file, header, verbose=True, njobs=-1):
    """Estimation of extraction time for the Pressure, Proximity, Light sensors
    Args:
        path_to_text_file: path to the bin file
        path_to_out_file: save path for the converted file
    Returns:
    """
    if verbose:
        print(f'stream processing text file {path_to_text_file}', flush=True)
    with open(path_to_text_file, 'r') as handle_from:
        with open(path_to_out_file, 'w') as handle_to:
            handle_to.write(f'{" ".join(header)}\n')
            for line in handle_from:
                if not line.startswith('STOP'):
                    handle_to.write(line)
    return


def streaming_screen_text_extract(path_to_text_file, path_to_out_file, header, verbose=True, njobs=-1):
    """Estimation of extraction time for the Screen
    Args:
        path_to_text_file: save path to the bin file
        path_to_out_file: save path for the converted file
    Returns:
    """
    if verbose:
        print(f'stream processing screen text file {path_to_text_file}', flush=True)
    accel_path = path_to_out_file.replace('screen.txt', 'accel.txt')
    with open(accel_path, 'r') as handle_accel:
        for line_number, line in enumerate(handle_accel):
            if line_number == 1:
                anchor_timestamp = np.int64(line.split(' ')[0])

    time_hour_delta = None
    previous_date_time = None
    current_date_time = None

    with open(path_to_text_file, 'r') as handle_from:
        with open(path_to_out_file, 'w') as handle_to:
            handle_to.write(f'{" ".join(header)}\n')
            for line in handle_from:
                if not line.startswith('STOP'):
                    line_split = line.split(' ')
                    if len(line_split) == 3:
                        date, time, event = line_split
                        # old format with date time and 
                        if time_hour_delta is None:
                            time_delta_range = np.arange(-23, 24)
                            delta_timestamp_guesses = np.array([np.int64((pd.to_datetime(date + ' ' + time) + pd.to_timedelta(i, unit='h')).value) // 10**6 for i in time_delta_range])
                            time_hour_delta = time_delta_range[np.argmin(np.abs(delta_timestamp_guesses - anchor_timestamp))]
                            print(f'time_hour_delta for file {path_to_text_file} is {time_hour_delta}h')
                        previous_date_time, current_date_time = current_date_time, pd.to_datetime(date + ' ' + time)
                        if previous_date_time is not None and current_date_time < previous_date_time:
                            # time travel engaged
                            travel_distance = (previous_date_time - current_date_time).round('h').seconds // 3600
                            time_hour_delta += travel_distance
                            print(f'time_hour_delta for file {path_to_text_file} is {time_hour_delta}h (shifted for {travel_distance} by time travel)')
                        timestamp = np.int64((pd.to_datetime(date + ' ' + time) + pd.to_timedelta(time_hour_delta, unit='h')).value) // 10**6
                        handle_to.write(' '.join([str(timestamp), event]))
                    else:
                        handle_to.write(line)
    return



def chunk_bin_extract(arr_chunk):
    """Extract data from bin file for specific sensor
    Args:
        path_to_bin_file: path to the bin file
        path_to_out_file: save path for the converted file
        chunk_size: "batch" size of data to be processed simultaneously
    Returns:
    """
    step = 8+4+4+4
    sum = 0

    output = []
    # unpack
    for i in range(0, len(arr_chunk), 8+4+4+4):
        time = struct.unpack('>q', bytearray(arr_chunk[i:i+8]))[0]
        date = datetime.datetime.fromtimestamp(time / 1e3)
        f = struct.unpack('<fff', bytearray(arr_chunk[i + 8:i + 8 + 3*4]))
        f = np.asarray(f).astype('float32')
        sum += np.sum(f)
        output.append(f'{time} {f[0]:9.10f} {f[1]:9.10f} {f[2]:9.10f}\n')
    return output, sum


def parallel_bin_extract(path_to_bin_file, path_to_out_file, header, verbose=True, njobs=-1, chunk_size=(8+4+4+4)*1024):
    """Extract data from bin file for the Accelerometer, Gravity, Gyroscope, Linear Accelerometer, Magnetic
    and Rotation sensors
    Args:
        path_to_bin_file: path to the bin file
        path_to_out_file: save path for the converted file
        chunk_size: "batch" size of data to be processed simultaneously
    Returns:
    """
    arr = np.fromfile(path_to_bin_file, dtype='int8')
    if verbose:
        print(f'parallel chunk processing bin file {path_to_bin_file} of length {len(arr)}', flush=True)
    step = 8+4+4+4
    assert ((len(arr) % step) == 0)
    nsteps = len(arr) // step
    
    bin_checksum = 0

    ncpus = ncpus_from_njobs(njobs)
    
    start_indices = np.arange(0, len(arr), chunk_size)
    end_indices = start_indices[1:]
    end_indices = np.concatenate((end_indices, [len(arr)]))
    arr_sliced = tuple(arr[start_index: end_index] for (start_index, end_index) in zip(start_indices, end_indices))
    with open(path_to_out_file, 'w') as handle_to:
        handle_to.write(f'{" ".join(header)}\n')
        parallel_args = arr_sliced
        with Pool(ncpus) as pool:
            with tqdm.tqdm(total=len(arr_sliced)) as pbar:
                for (chunk_output, chunk_checksum) in pool.imap(chunk_bin_extract, parallel_args):
                    pbar.update()
                    for chunk_line in chunk_output:
                        handle_to.write(chunk_line)
                    bin_checksum += chunk_checksum
    return


def makedirs_if(filepath):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    return


def convert_root_filepath(filepath, root, new_root):
    """Creation of path for saving converted data
    Args:
        filepath: path to bin (raw) dataset + specific IMU sensor
        root: path to bin (raw) dataset
        new_root: path to text (converted) dataset
    Returns:
        number of CPUs
    """
    return os.path.join(new_root, os.path.relpath(filepath, root))


def ncpus_from_njobs(njobs):
    """Evaluation of max number of available CPUs depending of number of jobs
    Args:
    Returns:
        number of CPUs
    """
    return max(1, njobs) if njobs >= 0 else max(1, cpu_count() + njobs + 1)


def extract_dataset(path_to_sensor_data, path_to_extracted_dataset, verbose=True, njobs=-1):
    """Extract dataset from bin (raw) and convert to the text format
    Args:
        path_to_sensor_data: path to raw dataset
        path_to_extracted_dataset: path to converted dataset
    Returns:
    """
    dataset_from = os.path.abspath(path_to_sensor_data)
    dataset_to = os.path.abspath(path_to_extracted_dataset)
    
    all_filepaths_from = []
    all_filepaths_to = []
    all_verbose = []
    
    users = os.listdir(dataset_from)
    # 1st level -- user
    for user in users:
        path_user_from = os.path.join(dataset_from, user)
        user_sensors = os.listdir(path_user_from)
        #2nd level -- user sensors
        for sensor in user_sensors:
            path_user_sensor_from = os.path.join(path_user_from, sensor)
            user_sensor_collections = os.listdir(path_user_sensor_from)
            #3rd level -- user sensor collections
            for collection in user_sensor_collections:
                path_user_sensor_collection_from = os.path.join(path_user_sensor_from, collection)
                path_user_sensor_collection_to = convert_root_filepath(path_user_sensor_collection_from, dataset_from, dataset_to)
                makedirs_if(path_user_sensor_collection_to)
                filenames_from = os.listdir(path_user_sensor_collection_from)
                if 'screen.txt' in filenames_from:
                    filenames_from.remove('screen.txt')
                    filenames_from.append('screen.txt')
                #4th level -- measurement files
                all_filepaths_from += [os.path.join(path_user_sensor_collection_from, filename) for filename in filenames_from]
                all_filepaths_to += [os.path.join(path_user_sensor_collection_to, filename) for filename in filenames_from]
                all_verbose += [verbose] * len(filenames_from)
    if verbose:
        print(f'there are {len(all_filepaths_from)} files to extract.', flush=True)
        
    for filepath_from, filepath_to in tqdm.tqdm(zip(all_filepaths_from, all_filepaths_to), total=len(all_filepaths_from)):
        streaming_extract(filepath_from, filepath_to, verbose, njobs)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dataset extractor from binfiles to csv-like txt files.')
    parser.add_argument('--extract_from',
                        help='binary dataset to extract from')
    parser.add_argument('--extract_to', help='path to extracted dataset')
    parser.add_argument('--njobs', type=int, default=8, help='number of workers')
    parser.add_argument('--verbose', action='store_true', help='verbosity flag')
    
    args = parser.parse_args()
    
    print(f'extracting data from dataset {args.extract_from} to dataset {args.extract_to} using {ncpus_from_njobs(args.njobs)} workers...')
    extract_dataset(args.extract_from, args.extract_to, args.verbose, args.njobs)
