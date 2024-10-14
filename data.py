import pandas as pd
import numpy as np
import random
import os
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
    # def get_cali_housing_data(norm_x=True, norm_y=TRUE, norm_min_val=0, spat_int=False):
    '''
    Download and process the California Housing Dataset

    Parameters:
    norm_x = logical; should features be normalized
    norm_y = logical; should outcome be normalized
    norm_min_val = integer; choice of [0,1], setting whether normalization in range[0,1] or [-1,1]

    Return:
    coords = spatial coordinates (lon/lat)
    x = features at location
    y = outcome variable
    '''

    cali_housing_ds = pd.read_csv('./fetch_california_housing.csv')
    cali_housing_ds = np.array(cali_housing_ds)
    # cali_housing_ds = cali_housing_ds[:2000, :]
    id = cali_housing_ds[:, 0]
    coords = cali_housing_ds[:, 1:3]
    y = cali_housing_ds[:, 9]
    x = cali_housing_ds[:, 3:9]

    if norm_coords == True:
        for i in range(coords.shape[1]):
            coords[:, i] = normal(coords[:, i], norm_min_val)
    if norm_x == True:
        for i in range(x.shape[1]):
            x[:, i] = normal(x[:, i], norm_min_val)
    if norm_y == True:
        y = normal(y, norm_min_val)
    return torch.tensor(id).to(device).float(), torch.tensor(coords).to(device).float(), torch.tensor(x).to(device).float(), torch.tensor(y).to(device).float()

def get_GData_data(norm_coords=True, norm_x=True, norm_y=False, norm_min_val=1):
    GData = pd.read_csv(r'./GData_utm.csv')
    GData = np.array(GData)
    id = GData[:, 0]
    coords = GData[:, 1:3]
    y = GData[:, 10]
    x = GData[:, 3:10]
    if norm_coords == True:
        for i in range(coords.shape[1]):
            coords[:, i] = normal(coords[:, i], norm_min_val)
    if norm_x == True:
        for i in range(x.shape[1]):
            x[:, i] = normal(x[:, i], norm_min_val)
    if norm_y == True:
        y = normal(y, norm_min_val)
    return torch.tensor(id).to(device).float(), torch.tensor(coords).to(device).float(), torch.tensor(x).to(device).float(), torch.tensor(y).to(device).float()

def get_rick_data(norm_coords=True, norm_x=True, norm_y=False, norm_min_val=1):
    rick = pd.read_csv(r'./rick35.csv')
    rick = np.array(rick)
    # cali_housing_ds = cali_housing_ds[:2000, :]
    id = rick[:, 0]
    coords = rick[:, 1:3]
    y = rick[:, 24]
    x = rick[:, 3:24]
    if norm_coords == True:
        for i in range(coords.shape[1]):
            coords[:, i] = normal(coords[:, i], norm_min_val)
    if norm_x == True:
        for i in range(x.shape[1]):
            x[:, i] = normal(x[:, i], norm_min_val)
    if norm_y == True:
        y = normal(y, norm_min_val)
    return torch.tensor(id).to(device).float(), torch.tensor(coords).to(device).float(), torch.tensor(x).to(device).float(), torch.tensor(y).to(device).float()

def get_metla_data(norm_coords=True, norm_x=True, norm_y=False, norm_min_val=1):
    meta = pd.read_csv(r'./meta_20120522-17.csv',encoding='gbk')
    meta = np.array(meta)
    # cali_housing_ds = cali_housing_ds[:2000, :]
    id = meta[:, 0]
    coords = meta[:, 1:3]
    y = meta[:, 14]
    x = meta[:, 3:14]
    if norm_coords == True:
        for i in range(coords.shape[1]):
            coords[:, i] = normal(coords[:, i], norm_min_val)
    if norm_x == True:
        for i in range(x.shape[1]):
            x[:, i] = normal(x[:, i], norm_min_val)
    if norm_y == True:
        y = normal(y, norm_min_val)
    return torch.tensor(id).to(device).float(), torch.tensor(coords).to(device).float(), torch.tensor(x).to(device).float(), torch.tensor(y).to(device).float()

def get_generation_data(norm_coords=False, norm_x=False, norm_y=False, norm_min_val=1):
    generate = pd.read_csv(r'./generate3.csv',encoding='gbk')
    generate = np.array(generate)
    # cali_housing_ds = cali_housing_ds[:2000, :]
    id = generate[:, 0]
    coords = generate[:, 1:3]
    y = generate[:, 6]
    x = generate[:, 3:6]
    if norm_coords == True:
        for i in range(coords.shape[1]):
            coords[:, i] = normal(coords[:, i], norm_min_val)
    if norm_x == True:
        for i in range(x.shape[1]):
            x[:, i] = normal(x[:, i], norm_min_val)
    if norm_y == True:
        y = normal(y, norm_min_val)
    return torch.tensor(id).to(device).float(), torch.tensor(coords).to(device).float(), torch.tensor(x).to(device).float(), torch.tensor(y).to(device).float()


def get_Tokyomortality_data(norm_coords=True, norm_x=False, norm_y=False, norm_min_val=1):
    generate = pd.read_csv(r'./Tokyomortality.csv',encoding='gbk')
    generate = np.array(generate)
    # cali_housing_ds = cali_housing_ds[:2000, :]
    id = generate[:, 0]
    coords = generate[:, 1:3]
    y = generate[:, 8]
    x = generate[:, 3:8]
    if norm_coords == True:
        for i in range(coords.shape[1]):
            coords[:, i] = normal(coords[:, i], norm_min_val)
    if norm_x == True:
        for i in range(x.shape[1]):
            x[:, i] = normal(x[:, i], norm_min_val)
    if norm_y == True:
        y = normal(y, norm_min_val)
    return torch.tensor(id).to(device).float(), torch.tensor(coords).to(device).float(), torch.tensor(x).to(device).float(), torch.tensor(y).to(device).float()



def loader(id, coords, attributes, targets, train_size, batch_size, seed):

    train_id, test_id, train_coords, test_coords, train_attr, test_attr, train_targets, test_targets = train_test_split(
        id, coords, attributes, targets, test_size=1 - train_size, random_state=seed)
    


    # 创建数据加载器
    train_data = torch.utils.data.TensorDataset(torch.tensor(train_id), torch.tensor(train_coords), torch.tensor(train_attr), torch.tensor(train_targets))
    # val_data = torch.utils.data.TensorDataset(torch.tensor(val_id), torch.tensor(val_coords), torch.tensor(val_attr), torch.tensor(val_targets))
    test_data = torch.utils.data.TensorDataset(torch.tensor(test_id), torch.tensor(test_coords), torch.tensor(test_attr), torch.tensor(test_targets))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)


    return train_loader, val_loader, test_loader
    # return train_loader, test_loader
