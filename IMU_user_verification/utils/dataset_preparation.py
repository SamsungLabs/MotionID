import os
import pickle
import sklearn
import numpy as np
import pandas as pd


def get_dfs(path):
    """Loads data to data frame
    Args:
        path: path to the data of user
    Returns:
        dict where keys are sensors names
    """
    dfs = {}
    for key in ['accel', 'gravity', 'gyro', 'Light', 'linAccel', 'MagneticField', 'Rotation', 'screen']:
        dfs[key] = pd.read_csv(os.path.join(path, f'{key}.txt'), sep=' ')
    return dfs


def get_dfs_from_data(path_to_data):
	user_names = os.listdir(path_to_data)
	data_paths = {
		username: os.path.join(path_to_data, f'{username}', 's20', f'{username}_20000')
        for username in user_names
	}
	dfs_dict = {username: get_dfs(data_path) for username, data_path in data_paths.items()}
	return dfs_dict


def timestamp_to_delta_s(timestamp, ref_timestamp):
    timestamp_delta_s = 1.0 * (timestamp - ref_timestamp) / 10**3
    return timestamp_delta_s


def get_group_labeling(dfs_dict):
    """Processing: splitting of collected data to the six clusters (for each user)
    Args:
        dfs_dict: dict with data for all users
    Returns:
        processed dict
    """
    user_names = list(dfs_dict.keys())
    for username in dfs_dict:
        screen_df = dfs_dict[username]['screen']
        user_present_df = screen_df[screen_df.event == 'android.intent.action.USER_PRESENT'].copy()

        timedelta_data = timestamp_to_delta_s(user_present_df.timestamp, user_present_df.timestamp.iloc[0])

        from sklearn.cluster import DBSCAN

        eps_map = {uname: 31 for uname in user_names}
        eps_map['069'] = 35
        eps_map['088'] = 40
        eps_map['041'] = 60
        eps_map['034'] = 68
        eps_map['090'] = 180
        eps_map['005'] = 12
        eps_map['044'] = 20
        eps_map['032'] = 12
        eps_map['089'] = 5
        eps_map['058'] = 15
        eps_map['074'] = 12
        eps_map['037'] = 12
        eps_map['029'] = 20
        eps_map['020'] = 20
        eps_map['010'] = 8
        eps_map['021'] = 13
        eps_map['028'] = 9
        eps_map['026'] = 17
        eps_map['009'] = 9
        eps_map['070'] = 15
        eps_map['023'] = 15
        eps_map['046'] = 10
        eps_map['056'] = 15
        eps_map['063'] = 13
        eps_map['062'] = 20
        eps_map['078'] = 6
        eps_map['064'] = 80
        eps_map['084'] = 10
        eps_map['036'] = 15
        eps_map['051'] = 20
        eps_map['016'] = 60

        clustering = DBSCAN(eps=eps_map[username], min_samples=1).fit(timedelta_data.values.reshape((-1, 1)))
        #print(clustering.labels_)
        n_clusters = max(clustering.labels_) + 1
        #print([len(clustering.labels_[clustering.labels_ == cluster_id]) for cluster_id in range(n_clusters)])

        if username == '090':
            clustering.labels_[clustering.labels_ == 5] = 4
            clustering.labels_[np.logical_or(clustering.labels_ == 6, clustering.labels_ == 7)] = 5
        if username == '054':
            clustering.labels_[np.logical_and(clustering.labels_ >= 1, clustering.labels_ <= 3)] = 1
            clustering.labels_[clustering.labels_ > 3] -= 2
            clustering.labels_[clustering.labels_ > 5] = 5
        if username == '077':
            clustering.labels_[clustering.labels_ >= 1] += 1
            where_zero = np.where(clustering.labels_ == 0)[0]
            clustering.labels_[where_zero[50:]] = 1
        if username == '005':
            clustering.labels_[from_to(clustering.labels_, 0, 4)] = 0
            clustering.labels_[from_to(clustering.labels_, 4, 6)] = 1
            clustering.labels_[from_to(clustering.labels_, 6, 8)] = 2
            clustering.labels_[from_to(clustering.labels_, 8, 9)] = 3
            clustering.labels_[from_to(clustering.labels_, 9, 11)] = 4
            clustering.labels_[from_to(clustering.labels_, 11, 14)] = 5
        if username == '044':
            clustering.labels_[from_to(clustering.labels_, 0, 2)] = 0
            clustering.labels_[from_to(clustering.labels_, 2, 4)] = 1
            clustering.labels_[from_to(clustering.labels_, 4, 6)] = 2
            clustering.labels_[from_to(clustering.labels_, 6, 7)] = 3
            clustering.labels_[from_to(clustering.labels_, 7, 8)] = 4
            clustering.labels_[from_to(clustering.labels_, 8, 9)] = 5
            clustering.labels_[from_to(clustering.labels_, 9, 10)] = 6
        if username == '045':
            clustering.labels_[from_to(clustering.labels_, 4, 6)] = 4
            clustering.labels_[from_to(clustering.labels_, 6, 7)] = 5
        if username == '059':
            clustering.labels_ = np.repeat(np.arange(6), 50)
        if username == '040':
            clustering.labels_ = np.concatenate((np.repeat(np.arange(6), 50), [5]))
        if username == '032':
            clustering.labels_[from_to(clustering.labels_, 0, 3)] = 0
            clustering.labels_[from_to(clustering.labels_, 3, 5)] = 1
            clustering.labels_[from_to(clustering.labels_, 5, 6)] = 2
            clustering.labels_[from_to(clustering.labels_, 6, 7)] = 3
            clustering.labels_[from_to(clustering.labels_, 7, 8)] = 4
            clustering.labels_[from_to(clustering.labels_, 8, 11)] = 5
        if username == '089':
            clustering.labels_[from_to(clustering.labels_, 0, 6)] = 0
            clustering.labels_[from_to(clustering.labels_, 6, 11)] = 1
            clustering.labels_[from_to(clustering.labels_, 11, 12)] = 2
            clustering.labels_[from_to(clustering.labels_, 12, 13)] = 3
            clustering.labels_[from_to(clustering.labels_, 13, 14)] = 4
            clustering.labels_[from_to(clustering.labels_, 14, 17)] = 5
            clustering.labels_[from_to(clustering.labels_, 17, 18)] = 6
        if username == '058':
            clustering.labels_[from_to(clustering.labels_, 0, 8)] = 0
            clustering.labels_[from_to(clustering.labels_, 8, 9)] = 1
            clustering.labels_[from_to(clustering.labels_, 9, 12)] = 2
            clustering.labels_[from_to(clustering.labels_, 12, 14)] = 3
            clustering.labels_[from_to(clustering.labels_, 14, 15)] = 4
            clustering.labels_[from_to(clustering.labels_, 15, 16)] = 5
            clustering.labels_[from_to(clustering.labels_, 16, 17)] = 6
        if username == '020':
            clustering.labels_[from_to(clustering.labels_, 0, 1)] = 0
            clustering.labels_[from_to(clustering.labels_, 1, 5)] = 1
            clustering.labels_[from_to(clustering.labels_, 5, 10)] = 2
            clustering.labels_[from_to(clustering.labels_, 10, 11)] = 3
            clustering.labels_[213:264] = 4
            clustering.labels_[264:] = 5
            clustering.labels_[-1] = 6
        if username == '021':
            clustering.labels_[from_to(clustering.labels_, 3, 5)] = 3
            clustering.labels_[from_to(clustering.labels_, 5, 6)] = 4
            clustering.labels_[from_to(clustering.labels_, 6, 7)] = 5
            clustering.labels_[from_to(clustering.labels_, 7, 8)] = 6
        if username == '028':
            clustering.labels_[from_to(clustering.labels_, 0, 8)] -= 1
            np.clip(clustering.labels_, a_min=0, a_max=10)
        if username == '009':
            clustering.labels_[from_to(clustering.labels_, 0, 3)] = 0
            clustering.labels_[from_to(clustering.labels_, 3, 4)] = 1
            clustering.labels_[from_to(clustering.labels_, 4, 5)] = 2
            clustering.labels_[from_to(clustering.labels_, 5, 6)] = 3
            clustering.labels_[220:274] = 4
            clustering.labels_[274:] = 5
            clustering.labels_[-1] = 6
        if username == '002':
            clustering.labels_[from_to(clustering.labels_, 3, 5)] = 3
            clustering.labels_[from_to(clustering.labels_, 5, 6)] = 4
            clustering.labels_[from_to(clustering.labels_, 6, 7)] = 5
        if username == '070':
            clustering.labels_[from_to(clustering.labels_, 0, 3)] = 0
            clustering.labels_[from_to(clustering.labels_, 3, 4)] = 1
            clustering.labels_[from_to(clustering.labels_, 4, 5)] = 2
            clustering.labels_[from_to(clustering.labels_, 5, 6)] = 3
            clustering.labels_[198:240] = 4
            clustering.labels_[240:] = 5
            clustering.labels_[-1] = 6
        if username == '046':
            clustering.labels_[from_to(clustering.labels_, 0, 5)] = 0
            clustering.labels_[from_to(clustering.labels_, 5, 6)] = 1
            clustering.labels_[from_to(clustering.labels_, 6, 7)] = 2
            clustering.labels_[150:201] = 3
            clustering.labels_[201:251] = 4
            clustering.labels_[from_to(clustering.labels_, 9, 11)] = 5
            clustering.labels_[-1] = 6
        if username == '056':
            clustering.labels_[from_to(clustering.labels_, 5, 8)] = 5
            clustering.labels_[-1] = 6
        if username == '062':
            clustering.labels_[from_to(clustering.labels_, 1, 9)] = 2
            clustering.labels_[from_to(clustering.labels_, 9, 18)] = 3
            clustering.labels_[from_to(clustering.labels_, 18, 19)] = 4
            clustering.labels_[from_to(clustering.labels_, 19, 30)] = 5
            clustering.labels_[0:52] = 0
            clustering.labels_[52:103] = 1
            clustering.labels_[-1] = 6
        if username == '078':
            clustering.labels_[from_to(clustering.labels_, 0, 6)] = 0
            clustering.labels_[from_to(clustering.labels_, 6, 9)] = 1
            clustering.labels_[from_to(clustering.labels_, 9, 10)] = 2
            clustering.labels_[from_to(clustering.labels_, 10, 12)] = 3
            clustering.labels_[from_to(clustering.labels_, 12, 13)] = 4
            clustering.labels_[from_to(clustering.labels_, 13, 14)] = 5
            clustering.labels_[-1] = 6
        if username == '084':
            clustering.labels_[clustering.labels_ == 6] = 5
            clustering.labels_[-1] = 6
        if username == '083':
            clustering.labels_[-1] = 6
        if username == '025':
            clustering.labels_[-1] = 6
        if username == '036':
            clustering.labels_[clustering.labels_ == 1] = 0
            clustering.labels_[clustering.labels_ == 2] = 1
            clustering.labels_[clustering.labels_ == 3] = 2
            clustering.labels_[clustering.labels_ == 4] = 3
            clustering.labels_[np.logical_or(clustering.labels_ == 5, clustering.labels_ == 6)] = 4
            clustering.labels_[np.logical_or(clustering.labels_ == 7, clustering.labels_ == 8)] = 5
            clustering.labels_[-1] = 6
        if username == '087':
            clustering.labels_[-1] = 6
        if username == '015':
            clustering.labels_[-1] = 6
        if username == '051':
            clustering.labels_[clustering.labels_ == 4] = 3
            clustering.labels_[clustering.labels_ == 6] = 4
            clustering.labels_[clustering.labels_ == 7] = 6

        #print(username)
        #print(clustering.labels_)

        fictive_n_clusters = max(clustering.labels_) + 1
        actual_n_clusters = 6
        cdiff = 0
        actual_cluster_id = 0
        true_grouping = np.ones_like(clustering.labels_) * -1
        endlock_condition = True
        for current_cluster_id in range(fictive_n_clusters):
            fictive_group_size = len(clustering.labels_[clustering.labels_ == current_cluster_id])
            if fictive_group_size > 41:
                # actual cluster
                true_grouping[clustering.labels_ == current_cluster_id] = actual_cluster_id
                actual_cluster_id += 1
                if actual_cluster_id >= actual_n_clusters:
                    if current_cluster_id < fictive_n_clusters - 1:
                        endlock_condition = False
                    break
        #print(actual_cluster_id, actual_n_clusters)
        assert actual_cluster_id == actual_n_clusters
        if len(true_grouping[true_grouping == (actual_n_clusters - 1)]) > 50 and endlock_condition:
            true_grouping[np.where(true_grouping == (actual_n_clusters - 1))[0][-1]] = -1

        user_present_df['grouping'] = true_grouping
        dfs_dict[username]['user_present'] = user_present_df.copy()
    return


def get_holdout_group_labeling(dfs_dict):
    """Processing: splitting of collected data to the six clusters (for each user)
    Args:
        dfs_dict: dict with data for all users
    Returns:
        processed dict
    """
    user_names = list(dfs_dict.keys())
    for username in dfs_dict:
        screen_df = dfs_dict[username]['screen']
        user_present_df = screen_df[screen_df.event == 'android.intent.action.USER_PRESENT'].copy()

        timedelta_data = timestamp_to_delta_s(user_present_df.timestamp, user_present_df.timestamp.iloc[0])

        from sklearn.cluster import DBSCAN

        eps_map = {uname: 31 for uname in user_names}
        eps_map['091'] = 10
        eps_map['092'] = 10
        eps_map['093'] = 10
        eps_map['099'] = 20
        eps_map['101'] = 25

        clustering = DBSCAN(eps=eps_map[username], min_samples=1).fit(timedelta_data.values.reshape((-1, 1)))
        #print(clustering.labels_)

        n_clusters = max(clustering.labels_) + 1
        #print([len(clustering.labels_[clustering.labels_ == cluster_id]) for cluster_id in range(n_clusters)])

        if username == '091':
            clustering.labels_[clustering.labels_ == 4] = 3
            clustering.labels_[clustering.labels_ == 5] = 4
            clustering.labels_[np.logical_or(clustering.labels_ == 6, clustering.labels_ == 7)] = 5
            clustering.labels_[-1] = 6
        if username == '092':
            clustering.labels_[clustering.labels_ == 1] = 0
            clustering.labels_[clustering.labels_ == 2] = 1
            clustering.labels_[clustering.labels_ == 3] = 2
            clustering.labels_[clustering.labels_ == 4] = 3
            clustering.labels_[clustering.labels_ == 5] = 4
            clustering.labels_[np.logical_or(clustering.labels_ == 6, clustering.labels_ == 7)] = 5
            clustering.labels_[-1] = 6
        if username == '093':
            clustering.labels_[clustering.labels_ == 6] = 5
            clustering.labels_[-1] = 6
        if username == '094':
            clustering.labels_[clustering.labels_ == 1] = 0
            clustering.labels_[clustering.labels_ == 2] = 1
            clustering.labels_[clustering.labels_ == 3] = 2
            clustering.labels_[clustering.labels_ == 4] = 3
            clustering.labels_[clustering.labels_ == 5] = 4
            clustering.labels_[clustering.labels_ == 6] = 5
            clustering.labels_[-1] = 6
        if username == '099':
            clustering.labels_[clustering.labels_ == 1] = 0
            clustering.labels_[clustering.labels_ == 2] = 1
            clustering.labels_[clustering.labels_ == 3] = 2
            clustering.labels_[clustering.labels_ == 4] = 3
            clustering.labels_[clustering.labels_ == 5] = 4
            clustering.labels_[np.logical_or(clustering.labels_ == 6, clustering.labels_ == 7)] = 5
            clustering.labels_[-1] = 6
        if username == '101':
            clustering.labels_[-1] = 6
        if username == '096':
            clustering.labels_[-1] = 6
        if username == '095':
            clustering.labels_[-1] = 6
        if username == '097':
            clustering.labels_[-1] = 6
        if username == '100':
            clustering.labels_[-1] = 6
        if username == '098':
            clustering.labels_[-1] = 6

        #print(username)
        #print(clustering.labels_)

        fictive_n_clusters = max(clustering.labels_) + 1
        actual_n_clusters = 6
        cdiff = 0
        actual_cluster_id = 0
        true_grouping = np.ones_like(clustering.labels_) * -1
        endlock_condition = True
        for current_cluster_id in range(fictive_n_clusters):
            fictive_group_size = len(clustering.labels_[clustering.labels_ == current_cluster_id])
            if fictive_group_size > 41:
                # actual cluster
                true_grouping[clustering.labels_ == current_cluster_id] = actual_cluster_id
                actual_cluster_id += 1
                if actual_cluster_id >= actual_n_clusters:
                    if current_cluster_id < fictive_n_clusters - 1:
                        endlock_condition = False
                    break
        #print(actual_cluster_id, actual_n_clusters)
        assert actual_cluster_id == actual_n_clusters
        if len(true_grouping[true_grouping == (actual_n_clusters - 1)]) > 50 and endlock_condition:
            true_grouping[np.where(true_grouping == (actual_n_clusters - 1))[0][-1]] = -1

        user_present_df['grouping'] = true_grouping
        dfs_dict[username]['user_present'] = user_present_df.copy()
    return


def from_to(x, a, b):
    return np.logical_and(x >= a, x < b)


def split_dfs_by_users(dfs_dict, group_size, seed=None):
	if seed is not None:
		np.random.seed(seed)
	user_names = np.array([username for username in dfs_dict])
	choice = np.random.choice(np.arange(len(user_names)), size=group_size, replace=False)
	first_group_user_names = user_names[choice].copy()
	second_group_user_names = [
		user_name
		for user_name in user_names
		if user_name not in first_group_user_names
	]
	first_group_dfs_dict = {
		user_name: value
		for user_name, value in dfs_dict.items()
		if user_name in first_group_user_names
	}
	second_group_dfs_dict = {
		user_name: value
		for user_name, value in dfs_dict.items()
		if user_name in second_group_user_names
	}
	return first_group_dfs_dict, second_group_dfs_dict


def get_group_split(user_present_group, train_val_ratio=0.85, test_size=15):
    """Data splitting by certain ratio
    Args:
        train_val_ratio: length(train) / length(train + val) = 0.85
        test_size: number of attempts (for test) for each of the six places of dataset collection. In total for each
        user number of test attempts = 15*6 = 90
    Returns:
        Splitted data frame by indices
    """
    group_index = user_present_group.index
    selected_test_indices = np.random.choice(group_index, size=test_size, replace=False)
    selected_test_indices = group_index[group_index.isin(selected_test_indices)]
    selected_train_indices = group_index[~group_index.isin(selected_test_indices)]

    train_size = int(len(selected_train_indices) * train_val_ratio)
    val_size = len(selected_train_indices) - train_size

    selected_val_indices = np.random.choice(selected_train_indices, size=val_size, replace=False)
    selected_val_indices = selected_train_indices[selected_train_indices.isin(selected_val_indices)]
    selected_train_indices = selected_train_indices[~selected_train_indices.isin(selected_val_indices)]

    #print(selected_train_indices)
    #print(selected_val_indices)
    #print(selected_test_indices)
    #print('-' * 80)
    split_column = pd.Series(data=['train'] * len(group_index), index=group_index, name='split_type')
    split_column.loc[selected_val_indices] = 'val'
    split_column.loc[selected_test_indices] = 'test'
    #print(split_column)
    return split_column


def get_user_split(user_present):
    """Data splitting on train, validation and test
    Args:
        user_present:
    Returns:
        dict with keys: train, val, test
    """
    user_present[user_present.grouping != -1].copy()
    data_user_present = user_present[user_present.grouping != -1].copy()
    res = data_user_present[['timestamp', 'grouping']].groupby(by='grouping').transform(get_group_split)
    data_user_present['split_type'] = res['timestamp'].copy()
    return {
        'train': data_user_present[data_user_present['split_type'] == 'train'],
        'val': data_user_present[data_user_present['split_type'] == 'val'],
        'test': data_user_present[data_user_present['split_type'] == 'test'],
    }


def get_stamp_dataset_for_split(dfs_dict):
    stamp_dataset = {
        'train': [],
        'val': [],
        'test': [],
    }
    stamp_labels = {
        'train': [],
        'val': [],
        'test': []
    }

    for user_id, username in enumerate(sorted(dfs_dict.keys())):
        user_present = dfs_dict[username]['user_present']
        user_split = get_user_split(user_present)
        for stage in user_split:
            user_split[stage]['username'] = username
            user_split[stage]['user_id'] = user_id
        stamp_dataset['train'].append(user_split['train'])
        stamp_dataset['val'].append(user_split['val'])
        stamp_dataset['test'].append(user_split['test'])
        stamp_labels['train'].append(np.ones(len(user_split['train'])) * user_id)
        stamp_labels['val'].append(np.ones(len(user_split['val'])) * user_id)
        stamp_labels['test'].append(np.ones(len(user_split['test'])) * user_id)

    for stage in stamp_dataset:
        stamp_dataset[stage] = pd.concat(stamp_dataset[stage], ignore_index=True)

    for stage in stamp_labels:
        stamp_labels[stage] = np.concatenate(stamp_labels[stage])
    return stamp_dataset, stamp_labels


def accel_est_getseq(accel_est_df, timestamp, time_back_s=1, time_forward_s=0):
    """Get data's segment which started at 'unlock_event-time_back_s' and finished at 'unlock_event+time_forward_s'
    Args:
        accel_est_df: data frame
    Returns:
        cropped segments
    """
    timestamp_back = timestamp - 1000 * time_back_s
    timestamp_forward = timestamp + 1000 * time_forward_s

    return accel_est_df[(accel_est_df.timestamp >= timestamp_back) & (accel_est_df.timestamp <= timestamp_forward)].copy()



def getseqlen(df, timestamp, time_back_s=1, time_forward_s=0):
    """Get length of data's segment which started at 'unlock_event-time_back_s' and finished at
    'unlock_event+time_forward_s'
    Args:
        df: data frame
    Returns:
        length of cropped segment
    """
    timestamp_back = timestamp - 1000 * time_back_s
    timestamp_forward = timestamp + 1000 * time_forward_s

    return len(df[(df.timestamp >= timestamp_back) & (df.timestamp <= timestamp_forward)])


def combine_grav_linaccel(df_grav, df_linaccel):
    """Combine data from gravity and linear accelerometer
    Args:
    Returns:
        combined data frame
    """
    df_accel_est = df_grav.copy()
    df_accel_est.X += df_linaccel.X
    df_accel_est.Y += df_linaccel.Y
    df_accel_est.Z += df_linaccel.Z
    return df_accel_est


def build_tensorized_data(dfs_dict, stamp_dataset, stamp_labels, data_key, nseq_stat):
    nseq = nseq_stat['nseq']
    time_back_s = (nseq_stat['nseq_back'] - 1) * (1.0 / nseq_stat['freq'])
    time_forward_s = nseq_stat['nseq_forward'] * (1.0 / nseq_stat['freq'])
    tensorized_data = {
        stage: np.zeros((len(stamp_dataset[stage]), nseq, 3))
        for stage in stamp_dataset
    }

    for stage in stamp_dataset:
        for row_id, row in stamp_dataset[stage].iterrows():
            timestamp, _, _, _, username, user_id = row

            data = accel_est_getseq(dfs_dict[username][data_key], timestamp, time_back_s, time_forward_s)
            data = data[-nseq:].copy()
            nlen = len(data)
            tensorized_data[stage][row_id, nseq - nlen:, :] = data[['X', 'Y', 'Z']].values

    tensorized_dict = {
        stage: {
            'data': tensorized_data[stage],
            'labels': stamp_labels[stage],
        }
        for stage in stamp_dataset
    }

    return tensorized_dict


def create_tensorized_datasets(dfs_dict, stamp_dataset, stamp_labels, nseq_stat):
    for username in dfs_dict:
        dfs_dict[username]['accel_est'] = combine_grav_linaccel(dfs_dict[username]['gravity'], dfs_dict[username]['linAccel'])

    accel_data = build_tensorized_data(dfs_dict, stamp_dataset, stamp_labels, 'accel_est', nseq_stat)
    gyro_data = build_tensorized_data(dfs_dict, stamp_dataset, stamp_labels, 'gyro', nseq_stat)
    #gravity_data = build_tensorized_data(dfs_dict, stamp_dataset, 'gravity', nseq_stat)
    rotation_data = build_tensorized_data(dfs_dict, stamp_dataset, stamp_labels, 'Rotation', nseq_stat)
    magnetic_data = build_tensorized_data(dfs_dict, stamp_dataset, stamp_labels, 'MagneticField', nseq_stat)

    return {
        'accel': accel_data,
        'gyro': gyro_data,
        #'gravity': gravity_data,
        'rotation': rotation_data,
        'magnetic': magnetic_data
    }


def write_tensorized_dataset(dataset, dataset_path, data_template_name):
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    for key, data in dataset.items():
        filename = f'{key}_{data_template_name}.pkl'
        print(f'write data {key} with template {data_template_name} into {dataset_path}')
        with open(os.path.join(dataset_path, filename), 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return