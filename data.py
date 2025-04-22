import pandas as pd
import numpy as np
import random
import os
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from sklearn.model_selection import train_test_split
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


set_seed(seed=42)


def normal(x, min_val=1):
    '''
    Normalize a vector

    Parameters:
    x = numerical vector
    min_val = integer; choice of [0,1], setting whether normalization in range[0,1] or [-1,1]

    Return:
    x_norm = normalized vector
    '''
    x_min = np.min(x)
    x_max = np.max(x)
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_min == 0 and x_max == 0:
        return x
    if min_val == -1:
        x_norm = 2 * ((x - x_min) / (x_max - x_min)) - 1
    if min_val == 0:
        x_norm = ((x - x_min) / (x_max - x_min))
    if min_val == 1:
        x_norm = (x - x_mean) / x_std
    return x_norm


def get_cali_housing_data(norm_coords=True, norm_x=True, norm_y=False, norm_min_val=1):

    '''


    norm_x = logical; should features be normalized
    norm_y = logical;
    norm_min_val = integer; choice of [0,1], setting whether normalization in range[0,1] or [-1,1]
    Return:
    coords = spatial coordinates (lon/lat)
    x = features at location
    y = outcome variable
    '''

    cali_housing_ds = pd.read_csv('./fetch_california_housing.csv')
    cali_housing_ds = np.array(cali_housing_ds)
    id = cali_housing_ds[:, 0]
    coords = cali_housing_ds[:, 1:3]
    y = cali_housing_ds[:, 9]
    x = cali_housing_ds[:, 3:9]
    if norm_coords:
        for i in range(coords.shape[1]):
            coords[:, i] = normal(coords[:, i], norm_min_val)
    if norm_x:
        for i in range(x.shape[1]):
            x[:, i] = normal(x[:, i], norm_min_val)
    if norm_y:
        y = normal(y, norm_min_val)
    return torch.tensor(id).to(device).float(), torch.tensor(coords).to(device).float(), torch.tensor(x).to(device).float(), torch.tensor(y).to(device).float()


def get_GData_data(norm_coords=True, norm_x=True, norm_y=False, norm_min_val=1):
    GData = pd.read_csv(r'./GData_utm.csv')
    GData = np.array(GData)
    id = GData[:, 0]
    coords = GData[:, 1:3]
    y = GData[:, 10]
    x = GData[:, 3:10]
    if norm_coords:
        for i in range(coords.shape[1]):
            coords[:, i] = normal(coords[:, i], norm_min_val)
    if norm_x:
        for i in range(x.shape[1]):
            x[:, i] = normal(x[:, i], norm_min_val)
    if norm_y:
        y = normal(y, norm_min_val)
    return torch.tensor(id).to(device).float(), torch.tensor(coords).to(device).float(), torch.tensor(x).to(device).float(), torch.tensor(y).to(device).float()




def get_generation_data(norm_coords=False, norm_x=False, norm_y=False, norm_min_val=1):
    generate = pd.read_csv(r'./generate3.csv', encoding='gbk')
    generate = np.array(generate)
    id = generate[:, 0]
    coords = generate[:, 1:3]
    y = generate[:, 6]
    x = generate[:, 3:6]
    if norm_coords:
        for i in range(coords.shape[1]):
            coords[:, i] = normal(coords[:, i], norm_min_val)
    if norm_x:
        for i in range(x.shape[1]):
            x[:, i] = normal(x[:, i], norm_min_val)
    if norm_y:
        y = normal(y, norm_min_val)
    return torch.tensor(id).to(device).float(), torch.tensor(coords).to(device).float(), torch.tensor(x).to(device).float(), torch.tensor(y).to(device).float()


def get_near_surface_data(norm_coords=True, norm_x=False, norm_y=False, norm_min_val=1):
    data_frame = pd.read_csv(r'./near_surface.csv', encoding='gbk')
    data_frame.columns = ['ID', 'North', "East", 'Ele.', 'H0', 'V0', 'H1', 'V1']
    data_frame['H1'].fillna(0, inplace=True)  # 填充 H1 中的 NaN 值为0
    H0 = data_frame['H0'].to_numpy()
    H1 = data_frame['H1'].to_numpy()
    data_frame['H1'] = H0 + H1  # 计算 Hθ 和 H1
    data_frame = np.array(data_frame)
    id = data_frame[:, 0]
    coords = data_frame[:, 1:3]
    y = data_frame[:, 7]
    x = data_frame[:, 3:7]
    if norm_coords:
        for i in range(coords.shape[1]):
            coords[:, i] = normal(coords[:, i], norm_min_val)
    if norm_x:
        for i in range(x.shape[1]):
            x[:, i] = normal(x[:, i], norm_min_val)
    if norm_y:
        y = normal(y, norm_min_val)
    return torch.tensor(id).to(device).float(), torch.tensor(coords).to(device).float(), torch.tensor(x).to(device).float(), torch.tensor(y).to(device).float()


def get_generation_data_test(norm_coords=False, norm_x=False, norm_y=False, norm_min_val=1):
    generate = pd.read_csv(r'./generate1.csv', encoding='gbk') #gegeration test dataset
    generate = np.array(generate)
    # cali_housing_ds = cali_housing_ds[:2000, :]
    id = generate[:, 0]
    coords = generate[:, 1:3]
    y = generate[:, 6]
    x = generate[:, 3:6]
    if norm_coords:
        for i in range(coords.shape[1]):
            coords[:, i] = normal(coords[:, i], norm_min_val)
    if norm_x:
        for i in range(x.shape[1]):
            x[:, i] = normal(x[:, i], norm_min_val)
    if norm_y == True:
        y = normal(y, norm_min_val)
    return torch.tensor(id).to(device).float(), torch.tensor(coords).to(device).float(), torch.tensor(x).to(device).float(), torch.tensor(y).to(device).float()


def loader(id, coords, attributes, targets, train_size, batch_size, seed):
    train_id, test_id, train_coords, test_coords, train_attr, test_attr, train_targets, test_targets = train_test_split(
        id, coords, attributes, targets, test_size=1 - train_size, random_state=seed)
    # if generation data
    test_id, test_coords, test_attr, test_targets = get_generation_data_test(norm_coords=False, norm_x=False, norm_y=False, norm_min_val=1)

    # load data
    train_data = torch.utils.data.TensorDataset(torch.tensor(train_id), torch.tensor(train_coords), torch.tensor(train_attr), torch.tensor(train_targets))

    test_data = torch.utils.data.TensorDataset(torch.tensor(test_id), torch.tensor(test_coords), torch.tensor(test_attr), torch.tensor(test_targets))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    val_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    return train_loader, val_loader, test_loader
