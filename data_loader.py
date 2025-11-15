import os
import numpy as np
import random
import glob

from torch.utils import data
from utils import *
from tqdm import tqdm
from collections import Counter


class MaskDataset(data.Dataset):
    def __init__(self, paths, type):
        super(MaskDataset, self).__init__()
        self.image_paths = paths
        self.type = type

    def __getitem__(self, index):
        data, label = self.image_paths[index, :-1], self.image_paths[index, -1]
        if self.type == 'train':
            """if train dataset, then apply data enhancement"""
            if random.random() < 0.5:
                data = np.flip(data, axis=1).copy()
                label = label[::-1].copy()
        else:
            pass
        return data, label

    def __len__(self):
        return self.image_paths.shape[0]
    
def load_data(batch_size=64, split_rate=0.8, 
              test_mode=False, 
              is_standardization=False,
              province=None,
              fixed_split_indices=None):  # ADD: 新增固定划分参数
    print(f'Batch size: {batch_size}, split rate: {split_rate}')
    cols2keep=['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'VV', 'VH', 'label']
    if test_mode:
        cols2keep.insert(-1, 'lat_lon')
    # samples for train & samples for valid
    samples4train, samples4valid = get_all_files_in_samples(".\\samples", split_rate=split_rate, province=province, fixed_split_indices=fixed_split_indices)  # MODIFY: 传递固定划分参数
    print('Processing samples files...')
    samples4train, samples4valid = [proc_bands_value(file, cols2keep=cols2keep) for file in tqdm(samples4train)], \
                                   [proc_bands_value(file, cols2keep=cols2keep) for file in tqdm(samples4valid)]
    array_list_train, array_list_valid = [ts.values.transpose(1, 0) for ts in samples4train if ts is not None and ts.shape == (60, len(cols2keep))], \
                                         [ts.values.transpose(1, 0) for ts in samples4valid if ts is not None and ts.shape == (60, len(cols2keep))]
    train, test = np.stack(array_list_train, axis=0), \
                  np.stack(array_list_valid, axis=0)
                  
    # samples distribution adjustment``
    mask_train, mask_valid = (train[:, -1, :] != 4).all(axis=1), (test[:, -1, :] != 4).all(axis=1)
    train, test = train[mask_train],  test[mask_valid] # remove background type
    train[:, -1, :][train[:, -1, :] == 5], test[:, -1, :][test[:, -1, :] == 5] = 4, 4   # herbicide -> index 4
    train[:, -1, :][train[:, -1, :] == 6], test[:, -1, :][test[:, -1, :] == 6] = 5, 5   # other vegetation -> index 5
    train[:, -1, :], test[:, -1, :] = train[:, -1, :] - 1, test[:, -1, :] - 1 # index minus 1
    
    # Load data
    print(f'train shape: {train.shape}, test shape: {test.shape}')
    if is_standardization:
        print('Data standardization.')
        train[:, :-1, :], _ = standarlization(train[:, :-1, :])
        test[:, :-1, :], _ = standarlization(test[:, :-1, :])
        # standardization opt + sar together
        # train[:, :-1, :], _ = standardization(train[:, :-1, :], optical_idx=range(12), sar_idx=list())
        # test[:, :-1, :], _ = standardization(test[:, :-1, :], optical_idx=range(12), sar_idx=list())
        
    # static lcc
    if __name__ == '__main__':
        lc = {
            0: 'SA',
            1: 'TF',
            2: 'OW',
            3: 'HL',
            4: 'OVe'
        }
        all = np.concatenate([train, test], axis=0)
        _, _ , changetypes = FilteringSeries(all[:, -1, :].reshape(all.shape[0], -1), method='Majority', window_size=5)
        lcc_counter = Counter([tuple(i.astype(int)) for i in changetypes])
        total = sum(lcc_counter.values())
        for key, value in lcc_counter.most_common():
            key_str = '→'.join([lc[i] for i in key])
            percent = (value / total) * 100
            print(f'{key_str} : {value} ({percent:.2f}%)')
            
    return train, test, train[:, np.r_[0:10, -1], :], test[:, np.r_[0:10, -1], :]

def make_dataloader(dataset, type, is_shuffle, batch_size=64):
    ds = MaskDataset(paths=dataset, type=type)
    return data.DataLoader(dataset=ds, batch_size=batch_size, shuffle=is_shuffle)
def random_permutation(tralid_ds, n_split=5, split_rate=0.8, batch_size=64):
    n_samples, test_size = tralid_ds.shape[0], 1 - split_rate
    n_test = int(n_samples * test_size)
    print(n_samples, n_test)
    for _ in range(n_split):
        # permutation indices
        indices = np.random.permutation(n_samples)
        train_indices, valid_indices = indices[n_test:], indices[:n_test]
        train_dl, valid_dl = make_dataloader(tralid_ds[train_indices], type='train', is_shuffle=True, batch_size=batch_size), \
                             make_dataloader(tralid_ds[valid_indices], type='test', is_shuffle=False, batch_size=batch_size)   
        yield train_dl, valid_dl

if __name__ == '__main__':
    # Batch × Channel × Length
    model_idx = 1037
    provinces = ['SD', 'JS', 'SH', 'ZJ', 'FJ', 'GDGX']
    if input(f'model idx is {model_idx}, continue? (y/n)\t') != 'y':
        exit('Aborted.')
    is_standardization = input(f'Standardization ? (y/n)\t') == 'y'
    print(f'is_standardization: {is_standardization}')
    
    # MODIFY: 先获取全国所有样本文件列表，用于后续固定划分
    all_samples, _ = get_all_files_in_samples(".\\samples", split_rate=1.0, province=None)
    n_samples = len(all_samples)
    n_test = int(n_samples * 0.1)  # 对应split_rate=0.9
    # 使用固定随机种子确保划分一致性
    np.random.seed(42)
    indices = np.random.permutation(n_samples)
    national_train_indices, national_test_indices = indices[n_test:], indices[:n_test]
    
    # 全国数据使用固定划分
    tralid, test, trailid_opt_only, test_opt_only = load_data(
        batch_size=64, 
        split_rate=0.9, 
        is_standardization=True,
        fixed_split_indices=(national_train_indices, national_test_indices)
    )
    
    print(f'train shape: {tralid.shape}, test shape: {test.shape}')
    
    save_dir = os.path.join('./models/model_data/dataset', str(model_idx))
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, 'tralid.npy'), tralid)
    np.save(os.path.join(save_dir, 'test.npy'), test)
    np.save(os.path.join(save_dir, 'tralid_opt_only.npy'), trailid_opt_only)
    np.save(os.path.join(save_dir, 'test_opt_only.npy'), test_opt_only)
    
    for province in provinces:
        # 分省数据使用相同的划分方式
        train, test, train_opt, test_opt = load_data(
            batch_size=64,
            split_rate=0.8,
            is_standardization=is_standardization,
            province=province,
            fixed_split_indices=(national_train_indices, national_test_indices)
        )
        print(f'province: {province}, train shape: {train.shape}, test shape: {test.shape}')
        np.save(os.path.join(save_dir, f'{province}_tralid.npy'), train)
        np.save(os.path.join(save_dir, f'{province}_test.npy'), test)
        np.save(os.path.join(save_dir, f'{province}_tralid_opt_only.npy'), train_opt)
        np.save(os.path.join(save_dir, f'{province}_test_opt_only.npy'), test_opt)