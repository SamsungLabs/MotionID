"""In case of availability of several separate measurements from one device for one user,
it is worth to concatenate data to the one Data Frame for obtaining better performance
"""

import pandas as pd
import numpy as np
import os


PATH = "path_to_converted_data"


def get_path_to_user_device(user, device):
    """Get path to user device
    Args:
        user: user name/id
        device: device's serial number
    Returns:
        path to the certain device of selected user
    """
    return f"{PATH}/{user}/{device}"


def get_user_devices(user):
    """Get list of all devices for specific user
    Args:
        user: user name/id
    Returns:
        list of devices
    """
    return os.listdir(f"{PATH}/{user}/")


def get_users():
    """Get list of users
    Args:
    Returns:
        list of users
    """
    return os.listdir(f"{PATH}/")


def load_data(path: str):
    """Load all files in folder path to df_set.
    Args:
        path: path to files (without last /)
    Returns:
         map (key, df), where key = file name, df = file as pd.DataFrame
    """
    df_set = {}
    for file in os.listdir(path):
        df_set[file] = pd.read_csv(f"{path}/{file}", delimiter=" ")
        print(f"Success for {path}/{file}")
    return df_set


def union_dfs(dfs1, dfs2):
    """Concatenation of Data Frames if keys are not the same
    Args:
        dfs1: first Data Frame
        dfs2: second Data Frame
    Returns:
         concatenated Data Frame
    """
    keys = set(list(dfs2.keys()) + list(dfs1.keys()))
    new_dfs = {}
    for key in keys:
        if key not in dfs1.keys():
            new_dfs[key] = dfs2[key]
        elif key not in dfs2.keys():
            new_dfs[key] = dfs1[key]
        else:
            new_dfs[key] = pd.concat((dfs1[key], dfs2[key]), ignore_index=True)
    return new_dfs


for user in get_users():
    for device in get_user_devices(user):
        path = get_path_to_user_device(user, device)
        dirs = os.listdir(path)
        if len(dirs) == 1:
            continue

        files = []
        for cur_dir in dirs:
            files += os.listdir(f"{path}/{cur_dir}")
        files = set(files)

        path_to_dir_union = f"{path}/{user}_20000_union"
        print(f"Path to union dir = {path_to_dir_union}")
        os.mkdir(path_to_dir_union)

        print(f"Files = {files}")
        for file in files:
            df = None
            for cur_dir in dirs:
                if file in os.listdir(f"{path}/{cur_dir}"):
                    print(f"Start reading {path}/{cur_dir}/{file}")
                    new_df = pd.read_csv(f"{path}/{cur_dir}/{file}", delimiter=" ")
                    print(f"End reading {path}/{cur_dir}/{file}")
                    if df is None:
                        df = new_df
                    else:
                        df = pd.concat((df, new_df), ignore_index=True)
            df.to_csv(f"{path_to_dir_union}/{file}", sep=" ", index=False)
            print(f"Success {path_to_dir_union}/{file}")