import os
import re
import torch
import random
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from matplotlib.colors import Normalize

# from typing import Tuple, Union, Sequence
from collections import Counter
# from tqdm import tqdm
from osgeo import gdal, ogr, osr
from scipy.special import softmax

class EarlyStopping:
    '''
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    '''

    def __init__(self, patience=8, min_delta=0):
        '''
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        '''
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f'INFO: Early stopping counter {self.counter} of {self.patience}')
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

class DualGammaNorm(Normalize):
    def __init__(self, vmin=None, vmax=None, threshold=0.5, gamma_low=0.5, gamma_high=2.0):
        self.threshold = threshold
        self.gamma_low = gamma_low
        self.gamma_high = gamma_high
        super().__init__(vmin, vmax)
    
    def __call__(self, value, clip=None):
        value = np.ma.masked_array(value, mask=np.isnan(value))
        result = np.zeros_like(value, dtype=np.float64)
        
        low_mask = value <= self.threshold
        result[low_mask] = (value[low_mask]/self.threshold) ** self.gamma_low * self.threshold

        high_mask = value > self.threshold
        result[high_mask] = self.threshold + (1-self.threshold) * ((value[high_mask]-self.threshold)/(1-self.threshold)) ** self.gamma_high
        
        return np.ma.masked_array(result, mask=value.mask)

def device_on():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Uisng {device} device')
    return device

def vec2mask(ref_image, vector_path, output_mask_path):
    # open reference image
    geotrans, proj = ref_image.GetGeoTransform(), ref_image.GetProjection()
    cols, row = ref_image.RasterXSize, ref_image.RasterYSize
    # create mask image
    driver = gdal.GetDriverByName('GTiff')
    mask_ds = driver.Create(output_mask_path, cols, row, 1, gdal.GDT_Byte, options=['COMPRESS=LZW'])
    mask_ds.SetGeoTransform(geotrans), mask_ds.SetProjection(proj)
    
    mask_band = mask_ds.GetRasterBand(1)
    mask_band.Fill(0)
    # open vec file
    vec_src = ogr.Open(vector_path)
    layer = vec_src.GetLayer()
    
    gdal.RasterizeLayer(mask_ds, [1], layer, burn_values=[1])
    mask_ds, vec_src, layer = None, None, None
    print(f'Mask image saved to: {output_mask_path}')  

def MajorityFilter(preds, kernel_size=3):
    """
    Sequential majority filter with symmetric padding and temporal dependency.
    Each time step is updated using a sliding window that includes already updated values.
    Padding is applied symmetrically to ensure all positions are processed.
    
    Args:
        preds (ndarray): shape (samples, time_steps)
        kernel_size (int): must be odd
    
    Returns:
        filtered (ndarray): same shape as preds
    """
    assert kernel_size % 2 == 1, "Kernel size must be odd"

    half_k = kernel_size // 2
    n_samples, n_steps = preds.shape

    filtered = np.zeros_like(preds)

    for i in range(n_samples):
        # Symmetric padding along time dimension
        padded = np.pad(preds[i], pad_width=(half_k, half_k), mode='symmetric')

        # Convert padded array to list for mutable in-place updates
        padded = padded.tolist()

        # Apply in-place majority filtering with temporal update
        for t in range(half_k, len(padded) - half_k):
            # Window is centered at t: [t - half_k, ..., t + half_k]
            window = padded[t - half_k : t + half_k + 1]
            
            # Count frequency of values in the window
            counter = Counter(window)

            # Mode value (with tie-breaking by smallest value)
            mode_val = max(counter.items(), key=lambda x: (x[1], -x[0]))[0]

            # Update current position with mode
            padded[t] = mode_val

        # Extract the central part (original length)
        filtered[i] = padded[half_k : -half_k]

    return filtered

def mode_filter(arr, nodata=99):
    """
    对二维分类图像做 3x3 众数滤波（等价于 GEE 中 radius=1, kernelType='square'）。
    参数:
      arr : 2D array-like (height x width) - 可以是 np.ndarray 或 np.ma.MaskedArray
      nodata : 用于填充掩膜或边界的值（默认 99，与 GEE 中 unmask(99) 等价）
    返回:
      result : 2D np.ndarray，shape 与 arr 相同，经过 3x3 众数滤波的结果。
    备注:
      - 若邻域内出现平局（多个值出现次数相同），函数返回数值**最小**的那个类别。
      - 该实现假定类别总数不太大（np.unique 的数量可控）。
    """
    arr = np.asarray(arr)
    # 如果是 MaskedArray，先用 nodata 填充
    if np.ma.isMaskedArray(arr):
        arr = arr.filled(nodata)
    # pad 边界，等价于 GEE 先 unmask 再对边界外像素使用 nodata
    padded = np.pad(arr, pad_width=1, mode='constant', constant_values=nodata)
    # sliding window -> shape (H, W, 3, 3)
    windows = sliding_window_view(padded, (3, 3))
    H, W = windows.shape[:2]
    # 展平每个 3x3 窗口为长度 9 向量，形状 (H, W, 9)
    flat = windows.reshape(H, W, 9)
    # 找到窗口内所有出现的唯一值（包含 nodata）
    uniques = np.unique(flat)
    # 对每个唯一值统计出现次数（向量化）
    counts = np.empty((uniques.size, H, W), dtype=np.int32)
    for i, u in enumerate(uniques):
        counts[i, :, :] = np.sum(flat == u, axis=-1)
    # 取出现次数的 argmax -> 得到每个像素对应的唯一值索引
    idx = counts.argmax(axis=0)  # shape (H, W)
    result = uniques[idx]        # 把索引映射回类别值
    return result

def proc_bands_value(csv_file, bands_col='bands_value', 
                     cols2keep=['B2', 'B3', 'B4','B5', 'B6', 'B7','B8', 'B8A', 'B11', 'B12',
                                 'VV', 'VH', 'label']):
    df = pd.read_csv(csv_file)
    try:
        # deal with bands values {'B2=114, B3=513, ... '}
        df[bands_col] = df[bands_col].str.strip('{}')
        df_dicts = df[bands_col].str.split(', ').apply(
            lambda x: {k: float(v) if v.strip() != 'null' else np.nan for k, v in (item.split('=') for item in x)}
        )
        df_expanded = pd.DataFrame(df_dicts.tolist())
        # Interpolate NaN values
        df_expanded = df_expanded.interpolate(method='linear', limit_direction='both')
        if df_expanded.isnull().any().any():
            raise ValueError(f'NaN values found in {csv_file}')
        # Date and location
        df['date'] = pd.to_datetime(df['system:time_start']).dt.to_period('M')
        df['lat_lon'] = df.apply(lambda row: [row['latitude'], row['longitude']], axis=1)
        df_final = pd.concat([df.drop(columns=[bands_col]), df_expanded], axis=1)
        
        # ================== make sure 3 consecutive label at least ==================
        def adjust_continuous_labels(labels):
            adjusted = labels.copy()
            n = len(adjusted)
            i = 0
            while i < n:
                current_label = adjusted[i]
                j = i
                while j < n and adjusted[j] == current_label:
                    j += 1
                segment_length = j - i
                if segment_length < 3 and i > 0:
                    adjusted[i-1] = current_label
                    foo = True
                i = j
            return adjusted
        df_final['label'] = adjust_continuous_labels(df_final['label'].values)
        # ================================================================
        return df_final[cols2keep]
    except Exception as e:
        print(f'Error processing DataFrame: {e}, ', csv_file)
        return None

# 修改后的get_all_files_in_samples函数
def get_all_files_in_samples(dir, split_rate=0.8, show_tonum=False, province=None, fixed_split_indices=None):
    '''
        Get all samples file path in root_dir.
        Return:
            samples4train: Train samples file path.
            samples4valid: Valid samples file path.
    '''
    samples4train, samples4valid = list(), list()
    total_samples_num = 0
    
    # 按文件夹组织文件路径和索引映射
    folder_files = []  # 存储每个文件夹的文件列表
    file_to_global_idx = {}  # 文件路径到全局索引的映射
    global_idx_to_folder_info = []  # 全局索引到(文件夹索引, 文件夹内索引)的映射
    
    # 第一遍遍历：收集文件并建立映射关系
    current_global_idx = 0
    for dirpath, _, filenames in os.walk(dir):
        if province is not None and province not in dirpath:    continue
        
        csv_files = [f for f in filenames if f.endswith('.csv')]
        total_samples_num += len(csv_files)

        if dirpath != dir:
            dir_name = os.path.basename(dirpath)
            print(f'{dir_name}: {len(csv_files)}')
        
        # 为当前文件夹的文件创建完整路径
        folder_file_paths = [os.path.join(dirpath, filename) for filename in csv_files]
        folder_files.append((dirpath, folder_file_paths))
        
        # 建立映射关系
        for folder_idx, file_path in enumerate(folder_file_paths):
            file_to_global_idx[file_path] = current_global_idx
            global_idx_to_folder_info.append((len(folder_files) - 1, folder_idx))
            current_global_idx += 1
    
    # 如果提供了固定划分索引
    if fixed_split_indices is not None:
        train_indices, test_indices = fixed_split_indices
        
        # 将全局索引转换为对应的文件路径
        for idx in train_indices:
            if 0 <= idx < len(global_idx_to_folder_info):
                folder_idx, file_idx = global_idx_to_folder_info[idx]
                dirpath, folder_file_paths = folder_files[folder_idx]
                if file_idx < len(folder_file_paths):
                    samples4train.append(folder_file_paths[file_idx])
        
        for idx in test_indices:
            if 0 <= idx < len(global_idx_to_folder_info):
                folder_idx, file_idx = global_idx_to_folder_info[idx]
                dirpath, folder_file_paths = folder_files[folder_idx]
                if file_idx < len(folder_file_paths):
                    samples4valid.append(folder_file_paths[file_idx])
    else:
        # 按文件夹单独进行8:2划分
        for dirpath, folder_file_paths in folder_files:
            # 对当前文件夹内的文件进行随机打乱
            random.shuffle(folder_file_paths)
            # 计算划分索引
            split_index = int(len(folder_file_paths) * split_rate)
            # 添加到训练集和验证集
            samples4train.extend(folder_file_paths[:split_index])
            if split_rate != 1:
                samples4valid.extend(folder_file_paths[split_index:])
    
    if show_tonum: print('Total Samples:', total_samples_num)
    return samples4train, samples4valid

def standarlization(data, optical_idx=range(0, 10), sar_idx=range(10, 12), eps=1e-6):
    '''
        Standardize the data along the time dimension for optical and SAR bands.
    '''
    # return data
    B, C, T = data.shape
    out = data.astype(np.float32, copy=True)
    opt_idx, sar_idx = np.array(list(optical_idx), dtype=int), np.array(list(sar_idx), dtype=int)

    # optical: compute per-sample mean / std over (channels,time)
    if opt_idx.size > 0:
        # shape (B,)
        mu_opt = out[:, opt_idx, :].reshape(B, -1).mean(axis=1, keepdims=True)  # (B,1)
        sd_opt = out[:, opt_idx, :].reshape(B, -1).std(axis=1, keepdims=True) + eps
        # reshape to (B,1,1) for broadcasting to (B, n_opt, T)
        mu_opt, sd_opt = mu_opt.reshape(B, 1, 1), sd_opt.reshape(B, 1, 1)
        out[:, opt_idx, :] = (out[:, opt_idx, :] - mu_opt) / sd_opt

    # sar: same
    if sar_idx.size > 0:
        mu_sar = out[:, sar_idx, :].reshape(B, -1).mean(axis=1, keepdims=True)
        sd_sar = out[:, sar_idx, :].reshape(B, -1).std(axis=1, keepdims=True) + eps

        mu_sar, sd_sar = mu_sar.reshape(B, 1, 1), sd_sar.reshape(B, 1, 1)
        out[:, sar_idx, :] = (out[:, sar_idx, :] - mu_sar) / sd_sar

    return out, {
        'mu_opt': mu_opt,
        'sd_opt': sd_opt,
        # 'mu_sar': mu_sar,
        # 'sd_sar': sd_sar,
    }

def extract_time_series_data(lat_lon, img_path):
    lat, lon = lat_lon[1], lat_lon[0]
    data = gdal.Open(img_path)
    # get geo transform
    transform = data.GetGeoTransform()
    x, y = int((lon - transform[0]) / transform[1]), int((lat - transform[3]) / transform[5])
    pixel_values = data.ReadAsArray(x, y, 1, 1)
    del data
    return pixel_values.flatten()

def extract_change_event_from_pixel(lcc, cd):
    def dtc_flooding(lcc):
        if len(lcc) >= 3:
            for i in range(len(lcc) - 2):
                if np.array_equal(lcc[i:i+3], np.array([0, 1, 2])):
                    return True, i + 1
        return False, -1

    def dtc_recurring(lcc):
        if len(lcc) >= 3:
            for i in range(len(lcc) - 2):
                if np.array_equal(lcc[i:i+3], np.array([0, 1, 0])):
                    return True, i + 1
            return False, -1
        else:
            return False, -1
        
    def dtc_herbicide(lcc):
        idx = np.argmax(lcc == 3)
        if lcc[idx] == 3:
            return idx
        else:
            return -1
        
    def dtc_mangrove(lcc):
        idx = np.argmax(lcc == 4)
        if lcc[idx] == 4:
            return idx
        else:
            return -1
    
    def dtc_mowing_1st(lcc):
        if len(lcc) < 2:
            return 99
        consecutive_01 = (lcc[:-1] == 0) & (lcc[1:] == 1)
        idx = np.argmax(consecutive_01)
        return idx if consecutive_01[idx] else 99
    
    def dtc_flooding_fast(lcc):
        for i in range(len(lcc) - 1):
            if np.array_equal(lcc[i:i+2], np.array([0, 2])):
                return True, i
        return False, -1
    
    mowing_1st, mowing_2nd, recurring, no_change, flooding_fast = 99, 99, 99, 99, 99
    # invasion
    # invasion = cd[0] if lcc[0] == 1 and lcc[1] == 0 else 99
    invasion_idx = next((i for i in range(len(lcc)-1) if lcc[i] == 1 and lcc[i+1] == 0 and 0 not in lcc[:i]), None)
    invasion = cd[invasion_idx] if invasion_idx is not None else 99
    # flooding
    is_flooding, flooding_cd = dtc_flooding(lcc)
    flooding = cd[flooding_cd] if is_flooding else 99
    # recurring
    is_recurring, recurring_cd = dtc_recurring(lcc)
    recurring = cd[recurring_cd] if is_recurring else 99
    # herbicide
    is_herbicide, herbicide = dtc_herbicide(lcc), 99
    if is_herbicide != -1 and is_herbicide != 0:
        herbicide = cd[is_herbicide - 1]
    # mangrove
    is_mangrove, mangrove = dtc_mangrove(lcc), 99
    if is_mangrove != -1 and is_mangrove != 0:
        mangrove = cd[is_mangrove - 1]

    # flooding_fast
    is_flooding_fast, flooding_fast_cd = dtc_flooding_fast(lcc)
    flooding_fast = cd[flooding_fast_cd] if is_flooding_fast else 99
    
    # mowing_1st
    idx_mowing_1st = dtc_mowing_1st(lcc)
    if idx_mowing_1st != 99:
        # mowing_2nd
        mowing_1st = cd[idx_mowing_1st]
        if len(lcc) >= 4:
            mowing_2nd = dtc_mowing_1st(lcc[idx_mowing_1st + 1:])
            if mowing_2nd != 99:
                mowing_2nd += idx_mowing_1st + 1
                mowing_2nd = cd[mowing_2nd]
            
    return [invasion, mowing_1st, mowing_2nd, flooding, herbicide, recurring, flooding_fast, mangrove ,no_change] # 《no change》 must be last

def generate_event_map(model_preds, valid_area, events, max_lc_change=5, is_static=False):
    lccmap = list()
    static_info = None
    height, width = model_preds.shape[0], model_preds.shape[1]
    for ts in model_preds[valid_area]:
        change_points = np.where(ts[:-1]!= ts[1:])[0] + 1
        # no change
        if change_points.shape[0] == 0:
            lccmap.append(np.concatenate([np.repeat(99, len(events) - 1), np.array([ts[0]])]))  # No change type is last one 
        # too many lc changes
        elif len(change_points) > (max_lc_change - 1):
            cd = change_points[: max_lc_change - 1] # as [2, 12, 19, 30]
            change_points = np.concatenate([[0], change_points])
            lc = ts[change_points][: max_lc_change] # as [1, 0, 1, 0, 1]
            event = extract_change_event_from_pixel(lc, cd)
            lccmap.append(np.array(event))
        # lc changes < max_lc_change
        else:
            lc = ts[np.concatenate([[0], change_points])]
            event = extract_change_event_from_pixel(lc, change_points)
            lccmap.append(np.array(event))
    lccmap = np.stack(lccmap)
    lcc = np.full((height, width, len(events)), 99)
    lcc[valid_area] = lccmap
    return lcc, static_info

def extract_accuracy_from_log(file_path):
    pth = file_path.split('\\')[-1].split('.')[0]
    try:
        with open(file_path, 'r') as file:
            log_content = file.read()

        epoch_info = dict()
        epoch_pattern = re.compile(r'Epoch (\d+), Train loss: ([\d.]+)')
        metric_pattern = re.compile(r'(\w+): ([\d.]+)')
        confusion_matrix_pattern = re.compile(r'Confusion Matrix\s*\n((?:\s*\[.*?\]\s*\n?)+)', re.DOTALL)
        change_type_acc_pattern = re.compile(r'Change Type Accuracy Matrix\s*\n((?:\s*\[.*?\]\s*\n?)+)', re.DOTALL)
        last_saved_pattern = re.compile(r'last saved epoch: (\d+)')

        last_saved_epoch = 0
        # Epoch 100, Train loss: 0.002951946808025241
        for epoch_match in epoch_pattern.finditer(log_content):
            epoch_num = int(epoch_match.group(1))
            train_loss = float(epoch_match.group(2)[:-1])
            metrics = {'pth': pth,
                       'train_loss': train_loss}
            start_index = epoch_match.end()
            next_epoch_start = log_content.find('Epoch', start_index)
            if next_epoch_start == -1:
                next_epoch_start = len(log_content)

            metric_text = log_content[start_index:next_epoch_start]
            
            # metric ———— mIoU: 0.9537; OA: 0.9822; AA: 0.9681; F1: 0.9761; Kappa: 0.9693;
            for metric_match in metric_pattern.finditer(metric_text):
                metric_name = metric_match.group(1)
                metric_value = float(metric_match.group(2))
                metrics[metric_name] = metric_value
            
            # extract confusion matrix
            cm_match, ct_acc_match = confusion_matrix_pattern.search(metric_text), \
                                     change_type_acc_pattern.search(metric_text)
            for cm_, cm_name in zip([cm_match, ct_acc_match], ['confusion_matrix', 'change_type_acc']):
                if cm_:
                    cm_block = cm_.group(1)
                    rows = list()
                    for line in cm_block.strip().splitlines():
                        if not line.strip():
                            continue
                        nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', line)
                        if nums:
                            row = [float(x) for x in nums]
                            rows.append(row)
                    if rows:
                        import numpy as _np
                        cm = _np.array(rows, dtype=float)
                        if _np.all(cm == _np.floor(cm)):
                            cm = cm.astype(int)
                        metrics[cm_name] = cm
            
            last_saved_match = last_saved_pattern.search(metric_text)
            if last_saved_match:
                last_saved_epoch = int(last_saved_match.group(1))
            epoch_info[epoch_num] = metrics
        return epoch_info, last_saved_epoch, pth

    except FileNotFoundError:
        print('file not found')
    except Exception as e:
        raise Exception(f'Error: {e}')

def delete_files_in_folder(folder_path):
    """ delete all files in folder_path(list type) """
    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
        print(f"deleted all files in {folder_path}")
    except Exception as e:
        print(f"{e}")
def mosaic_post(region):
    """ default path: .\\TimeSeriesImages\\SA_blocks_clipped&mosaic\\{**region**}_cls"""
    if type(region) is list:
        province, file_name = region, 'cls_all'
    elif type(region) is str:
        province, file_name = [region], region
    folder_path = [f".\\TimeSeriesImages\\SA_blocks_clipped&mosaic\\{p}_cls" for p in province]
    
    classification_all = list()
    for folder in folder_path:  
        files = os.listdir(folder)
        classification = [os.path.join(folder, file) for file in files]
        classification_all.append(classification)
    # all cls block for mosaic
    classification_all = [item for sublist in classification_all for item in sublist]
    
    # Mosaic
    try:
        vrt_options = gdal.BuildVRTOptions(VRTNodata=99)
        vrt_file = gdal.BuildVRT("temp.vrt", classification_all, options=vrt_options)
        translate_options = gdal.TranslateOptions(format='GTiff', creationOptions=['COMPRESS=LZW', 'BIGTIFF=YES'])
        gdal.Translate(f'.\\TimeSeriesImages\\classification\\{file_name}.tif', vrt_file, options=translate_options)
        vrt_file = None
        if os.path.exists("temp.vrt"):
            os.remove("temp.vrt")
    except Exception as e:
        print(e)
def DetectChangepoints(data):
    changepoints, changetypes = [], []
    for series in data:
        id = np.where((series[1:] - series[:-1]) != 0)[0]
        changepoints.append(id)
        changetypes.append(np.append(series[id], series[-1]))
    return changepoints, changetypes
def FilteringSeries(data, method='NoFilter', window_size=3):
    '''Temporal consistency modification'''
    if method == 'NoFilter':
        changepoints, changetypes = DetectChangepoints(data)
        return data, changepoints, changetypes
    elif method == 'Majority':
        res = MajorityFilter(data, kernel_size=window_size)
        changepoints, changetypes = DetectChangepoints(res)
        return res, changepoints, changetypes