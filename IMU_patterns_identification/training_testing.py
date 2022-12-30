import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from  matplotlib.pyplot import plot

from tslearn.utils import to_time_series
from tslearn.utils import to_time_series_dataset
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score


PATH = "path_to_converted_data"
DETECT_FIXED_EPS = 1.1 #constant to determine whether the device is moving or not
T = pd.Timedelta(3, unit="sec") #length of dataframe till the unlock event
COUNT = 100 #min number of sensors meausurements to fix the length of dataframes


def get_path_to_user_device(user, device):
    """Get path to user device
    Args:
        user: user name/id
        device: device's serial number
    Returns:
        path to the certain device of selected user
    """
    return os.path.join(PATH, user, device, user + '_20000')


def get_user_devices(user):
    """Get list of all devices for specific user
    Args:
        user: user name/id
    Returns:
        list of devices
    """
    return os.listdir(os.path.join(PATH, user))


def get_users():
    """Get list of users
    Args:
    Returns:
        list of users
    """
    return os.listdir(PATH)


def load_data(path: str, file: str):
    """load data from txt file to dataframe
    Args:
        path: path to the file
        file: file name
    Returns:
        data frame
    """
    df = pd.read_csv(f"{path}/{file}", delimiter=" ")
    df.timestamp = pd.to_datetime(df.timestamp * 1000000)
    return df


def detect_moving(df_lin_accel):
    """Make mask with determined periods of time when device was moving. The basement is the data from linear
    accelerometer
    Args:
        df_lin_accel: dataframe of linear accelerometer
    Returns:
        mask
    """
    mask_fixed = (df_lin_accel.X**2 + df_lin_accel.Y**2 + df_lin_accel.Z**2 > DETECT_FIXED_EPS).astype(int)
    # mask_fixed[i] = True <=> df_lin_accel.timestamp[i] - moving
    mask_fixed_av = mask_fixed.copy()
    for i in range(1, 3):
        mask_fixed_av += mask_fixed.shift(i, fill_value=0)[:df_lin_accel.shape[0]]
        mask_fixed_av += mask_fixed.shift(-i, fill_value=0)[:df_lin_accel.shape[0]]
    mask = mask_fixed_av >= 2
    # mask[i] = True <=> moving
    return mask


def get_moving_off_segments(df_lin_accel, df_screen):
    """ Find segments, while phone was moving and turned off
    Args:
        df_lin_accel: data frame of linear accelerometer
        df_screen: data frame of screen
    Returns:
        l_times: left borders of segments above
        r_times: right borders of segments above
        [l_times[i], r_times[i]] - i-th segment
    """
    mask = detect_moving(df_lin_accel)
    l_times = [] #left borders of segments
    r_times = [] #right borders of segments

    # indices corresponding to OFF, and previous value != OFF <=> left borders of segments, when the phone is turned off
    off_left_ids = pd.Series(df_screen[
        (df_screen.event == "android.intent.action.SCREEN_OFF") &
        (df_screen.shift(-1).event != "android.intent.action.SCREEN_OFF")
    ].index)

    T = pd.Timedelta(3, unit="sec")

    #indices: it was stationary -> became mobile <=> left borders of mobility segments
    active_left = pd.Series(df_lin_accel.timestamp[mask[
        (mask.shift(1) == False) & (mask == True)
    ].index].values)

    #indices: it was mobile -> became stationary <=> right borders of mobility segments
    active_right = pd.Series(df_lin_accel.timestamp[mask[
        (mask.shift(1) == True) & (mask == False)
    ].index].values)

    for off_left_id in off_left_ids:
        off_left = df_screen.timestamp[off_left_id]
        off_right = df_screen.timestamp[off_left_id + 1] - T

        l_first = active_left.searchsorted(off_left)
        l_last = active_left.searchsorted(off_right)

        for l in range(l_first, l_last):
            l_time = active_left[l]
            r = active_right.searchsorted(l_time)
            r_time = active_right[r]
            """
            change if -> while to split segment into many segments
            """
            if l_time + T <= r_time:
                l_times.append(l_time)
                r_times.append(l_time + T)
                l_time += T
    return l_times, r_times


def cut_sample_data(df_screen, df_lin_accel, path, l_times, r_times):
    """ Data cutting based on determined segments. Splitting of data on true and false classes
    Args:
        df_screen: data frame from screen
        df_lin_accel: data frame from linear accelerometer
        l_times: left borders of segments, while phone was moving and turned off
        r_times: right borders of segments, while phone was moving and turned off
    Returns:
        true_unblocking[i]: list of features (X, Y, Z) for each sensor for i-th segment leading to unblocking
        false_unblocking[i]: list of features (X, Y, Z) for each sensor for i-th segment that doesn't lead to unblocking
    """
    path = get_path_to_user_device(user, device)
    # false_unblocking[i] = list of features (each feature - several columns of sensor) for [l_times[i], r_times[i]]
    false_unblocking = [[] for _ in range(len(l_times))]

    on_times = df_screen[df_screen.event == "android.intent.action.USER_PRESENT"].timestamp
    # true_unblocking[i] = list of features (each feature - several columns of sensor) для [on_times[i] - T, on_times[i]]
    true_unblocking = [[] for _ in range(len(on_times))]
    for file in os.listdir(path):
        if file == "screen.txt" or file == "Proximity.txt" or file == "Pressure.txt" or file == "Light.txt":
            continue
        if file != "linAccel.txt":
            df = load_data(path, file)
        else:
            df = df_lin_accel

        # data frames leading to unlocking
        true_left_ids = df.timestamp.searchsorted(on_times - T)
        true_right_ids = df.timestamp.searchsorted(on_times)
        for i, tm in enumerate(on_times):
            left = true_left_ids[i]
            right = true_right_ids[i]
            true_unblocking[i].append(df[left:right].drop("timestamp", axis=1).values)

        # rows that do not lead to unlocking
        false_left_ids = df.timestamp.searchsorted(l_times)
        false_right_ids = df.timestamp.searchsorted(r_times)

        for i in range(len(l_times)):
            left = false_left_ids[i]
            right = false_right_ids[i]
            false_unblocking[i].append(df[left:right].drop("timestamp", axis=1).values)
    return true_unblocking, false_unblocking


def merge_sample_data(true_unblocking, false_unblocking):
    """ Merging of all prepared samples
    Args:
        true_unblocking[i]: list of features (X, Y, Z) for each sensor for i-th segment leading to unblocking
        false_unblocking[i]: list of features (X, Y, Z) for each sensor for i-th segment that doesn't lead to
        unlocking
    Returns:
        data_true_unblocking[i]: list of features (merged) for i-th segment leading to unblocking
        data_false_unblocking[i]: list of features (merged) for i-th segment that doesn't lead to unblocking
    """
    data_false_unblocking = []

    for i in range(len(false_unblocking)):
        ar = None
        for d in false_unblocking[i]:
            if len(d) < COUNT:
                ar = None
                break
            elif ar is None:
                ar = d[:COUNT]
            else:
                ar = np.concatenate((ar, d[:COUNT]), axis=1)
        if ar is not None:
            data_false_unblocking.append(to_time_series(ar))
    data_false_unblocking = to_time_series_dataset(data_false_unblocking)

    data_true_unblocking = []

    for i in range(len(true_unblocking)):
        ar = None
        for d in true_unblocking[i]:
            if len(d) < COUNT:
                ar = None
                break
            elif ar is None:
                ar = d[:COUNT]
            else:
                ar = np.concatenate((ar, d[:COUNT]), axis=1)
        if ar is not None:
            data_true_unblocking.append(to_time_series(ar))
    data_true_unblocking = to_time_series_dataset(data_true_unblocking)
    return data_true_unblocking, data_false_unblocking


def save_sample_data(path, data):
    """Saves handled data
    Args:
        path: path to the data
        data: data frame
    """
    np.savetxt(path, data.reshape(data.shape[0] * data.shape[1], data.shape[2]))


def load_sample_data(path):
    """Loads handled data
    Args:
        path: path to the data
    Returns:
        data
    """
    data = np.loadtxt(path)
    return data.reshape((data.shape[0] // COUNT, COUNT, data.shape[1]))


def get_sample_data(user, device):
    """Handles dataset
    Args:
        user: user name
        device: device id
    Returns:
        true and false classes
    """
    path = get_path_to_user_device(user, device)
    df_screen = load_data(path, "screen.txt")
    df_lin_accel = load_data(path, "linAccel.txt")

    l_times, r_times = get_moving_off_segments(df_lin_accel, df_screen)
    true_unblocking, false_unblocking = cut_sample_data(df_screen, df_lin_accel, path, l_times, r_times)
    data_true_unblocking, data_false_unblocking = merge_sample_data(true_unblocking, false_unblocking)

    save_sample_data(os.path.join(PATH, user, device, "true_unblocking.txt"), data_true_unblocking)
    save_sample_data(os.path.join(PATH, user, device, "false_unblocking.txt"), data_false_unblocking)
    return data_true_unblocking, data_false_unblocking


user = "user_id"
device = get_user_devices(user)[0] #id of device (from 0 to 5)
data_true_unblocking, data_false_unblocking = get_sample_data(user, device) #to make txt inputs for NN
#data_true_unblocking = load_sample_data(os.path.join(PATH, user, device,"true_unblocking.txt")) #to load txt NN's inputs
#data_false_unblocking = load_sample_data(os.path.join(PATH, user, device, "false_unblocking.txt"))


X = np.concatenate((data_false_unblocking, data_true_unblocking), axis=0)
y = np.zeros((X.shape[0]))
y[data_false_unblocking.shape[0]:] = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=47)

X_torch_test = torch.tensor(X_test).float().transpose(1, 2)
y_torch_test = torch.tensor(y_test).long()


class SensorDataset(Dataset):
    """Creation of dataset

    Longer class information....
    Longer class information....

    Attributes:
        X: data
        y: labels
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X).transpose(1, 2).float()
        self.y = torch.tensor(y).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {"time series": self.X[idx], "label": self.y[idx]}


training_dataset = SensorDataset(X_train, y_train)
test_dataset = SensorDataset(X_test, y_test)

train_dataloader = DataLoader(training_dataset, batch_size=100, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False)


class Net(nn.Module):
    """NN architecture
        """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(18, 9, 11)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(9, 4, 10)
        self.linear1 = nn.Linear(4 * 18, 150)
        self.linear2 = nn.Linear(150, 30)
        self.linear3 = nn.Linear(30, 2)

    def forward(self, X):
        X = self.pool(F.relu(self.conv1(X)))
        X = self.pool(F.relu(self.conv2(X)))
        X = X.view(-1, 4 * 18)
        X = F.relu(self.linear1(X))
        X = F.relu(self.linear2(X))
        X = self.linear3(X)
        return X


def process(train_dataloader, test_dataloader, lr=0.001, momentum=0.9, weight_decay=0.1, n_epochs=200,
            debug=False, plot=False, print_results=False):
    if print_results:
        print(f"process(lr={lr}, momentum={momentum}, weight_decay={weight_decay}, n_epochs={n_epochs})")
    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    train_losses = np.zeros(n_epochs)
    test_losses = np.zeros(n_epochs)

    train_accuracies = np.zeros(n_epochs)
    test_accuracies = np.zeros(n_epochs)

    train_rocaucs = np.zeros(n_epochs)
    test_rocaucs = np.zeros(n_epochs)

    for epoch in range(n_epochs):
        train_loss = 0.0
        train_accuracy = 0
        train_rocauc = 0
        for i, data in enumerate(train_dataloader):
            inputs, labels = data["time series"], data["label"]

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, pred = torch.max(outputs, 1)

            train_loss += loss.item()
            train_accuracy += (pred == labels).float().mean()
            train_rocauc += roc_auc_score(labels.numpy(), torch.softmax(outputs, dim=1).detach().numpy()[:, 1])

        test_loss = 0.0
        test_accuracy = 0
        test_rocauc = 0
        for i, data in enumerate(test_dataloader):
            inputs, labels = data["time series"], data["label"]
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            _, pred = torch.max(outputs, 1)

            test_loss += loss.item()
            test_accuracy += (pred == labels).float().mean()
            test_rocauc += roc_auc_score(labels.numpy(), torch.softmax(outputs, dim=1).detach().numpy()[:, 1])

        train_losses[epoch] = train_loss / len(train_dataloader)
        test_losses[epoch] = test_loss / len(test_dataloader)
        train_accuracies[epoch] = train_accuracy / len(train_dataloader)
        test_accuracies[epoch] = test_accuracy / len(test_dataloader)
        train_rocaucs[epoch] = train_rocauc / len(train_dataloader)
        test_rocaucs[epoch] = test_rocauc / len(test_dataloader)

        if (epoch + 1) % 10 == 0 and debug:
            print(f"{epoch + 1} training loss -> {train_loss / len(train_dataloader)}")
            print(f"{epoch + 1} test loss -> {test_loss / len(test_dataloader)}")
            print(f"{epoch + 1} training accuracy -> {train_accuracy / len(train_dataloader)}")
            print(f"{epoch + 1} test accuracy -> {test_accuracy / len(test_dataloader)}")
            print(f"{epoch + 1} training roc-auc -> {train_rocauc / len(train_dataloader)}")
            print(f"{epoch + 1} test roc-auc -> {test_rocauc / len(test_dataloader)}")
            print()

    if plot:
        plt.rcParams["figure.figsize"] = (20, 10)
        fig, axes = plt.subplots(3, 2)
        x = np.arange(n_epochs)

        axes[0, 0].plot(x, train_losses)
        axes[0, 0].set_title("train cross-entropy loss")

        axes[0, 1].plot(x, test_losses)
        axes[0, 1].set_title("test cross-entropy loss")

        axes[1, 0].plot(x, train_accuracies)
        axes[1, 0].set_title("train accuracy")

        axes[1, 1].plot(x, test_accuracies)
        axes[1, 1].set_title("test accuracy")

        axes[2, 0].plot(x, train_rocaucs)
        axes[2, 0].set_title("train roc-auc")

        axes[2, 1].plot(x, test_rocaucs)
        axes[2, 1].set_title("test roc-auc")

    if print_results:
        print(f"Cross-entropy metric = {test_losses.min()} on epoch {test_losses.argmin()}")
        print(f"Accuracy metric = {test_accuracies.max()} on epoch {test_accuracies.argmax()}")
        print(f"Roc-auc metric = {test_rocaucs.max()} on epoch {test_rocaucs.argmax()}")
        print(f"Accuracy on best roc-auc epoch ({test_rocaucs.argmax()}) = {test_accuracies[test_rocaucs.argmax()]}")
        print()


for weight_decay in np.linspace(0.01, 0.1, 5):
    process(train_dataloader, test_dataloader, weight_decay=weight_decay, plot=True, print_results=True, n_epochs=1000)


print('check')





