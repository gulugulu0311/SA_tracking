import os
import sys
import time
import re
sys.path.append('.')
import math
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import *
from torch import nn, Tensor
from typing import Optional

plt.rcParams['font.family'] = ['Arial']
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.unicode_minus'] = False

# ================= Configuration & Model Classes (保持不变) =================
class Configs():
    def __init__(self, is_opt_only=False):
        self.classes = 5
        self.input_channels = 12 if not is_opt_only else 10
        self.model_hidden = [64, 128, 256, 512, 1024]
        self.Transformer_hparams = {
            'd_model': 256, 'nhead': 8, 'num_layers': 6, 'dim_feedforward': 2048
        }

class TransformerEncoderLayerWithAttn(nn.TransformerEncoderLayer):
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x, _ = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=True, is_causal=is_causal)
        return self.dropout1(x)

class DoubleConv1d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool1d(2, stride=2, ceil_mode=True)
        self.conv = DoubleConv1d(in_ch, out_ch)
    def forward(self, x): return self.conv(self.pool(x))
    
class UpBlock(nn.Module):
    def __init__(self, cat_ch, mid_ch, out_ch, kernel_size=4):
        super().__init__()
        self.conv = DoubleConv1d(cat_ch, mid_ch)
        self.up = nn.ConvTranspose1d(mid_ch, out_ch, kernel_size=kernel_size, stride=2, padding=1, bias=False)
    def forward(self, x, skip):
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return self.up(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

class TSSCD_FCN(nn.Module):
    def __init__(self, in_channels, out_channels, config):
        super(TSSCD_FCN, self).__init__()
        c1, c2, c3, c4, c5 = config
        self.layer1 = nn.Sequential(DoubleConv1d(in_channels, c1), nn.MaxPool1d(2, stride=2, ceil_mode=True))
        self.layer2 = nn.Sequential(DoubleConv1d(c1, c2), nn.MaxPool1d(2, stride=2, ceil_mode=True))
        self.layer3 = nn.Sequential(DoubleConv1d(c2, c3), nn.MaxPool1d(2, stride=2, ceil_mode=True))
        self.layer4 = nn.Sequential(DoubleConv1d(c3, c4), nn.MaxPool1d(2, stride=2, ceil_mode=True))
        self.score_1 = nn.Conv1d(c4, out_channels, 1)
        self.score_2 = nn.Conv1d(c3, out_channels, 1)
        self.score_3 = nn.Conv1d(c2, out_channels, 1)
        self.upsampling_2x = nn.ConvTranspose1d(out_channels, out_channels, 4, 2, 1, bias=False)
        self.upsampling_4x = nn.ConvTranspose1d(out_channels, out_channels, 3, 2, 1, bias=False)
        self.upsampling_8x = nn.ConvTranspose1d(out_channels, out_channels, 6, 4, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.layer1(x)
        self.s1 = self.layer2(h)
        self.s2 = self.layer3(self.s1)
        self.s3 = self.layer4(self.s2)
        s3 = self.score_1(self.s3)
        s3 = self.upsampling_2x(s3)
        s2 = self.score_2(self.s2)
        s2 += s3
        s2 = self.upsampling_4x(s2)
        s1 = self.score_3(self.s1)
        score = s1 + s2
        score = self.upsampling_8x(score)
        return score

class TSSCD_Unet(nn.Module):
    def __init__(self, in_channels, out_channels, cfg):
        super().__init__()
        c1, c2, c3, c4, c5 = cfg
        self.enc0 = DoubleConv1d(in_channels, c1)
        self.enc1 = DownBlock(c1, c2)
        self.enc2 = DownBlock(c2, c3)
        self.enc3 = DownBlock(c3, c4)
        self.bot = nn.Sequential(DownBlock(c4, c5), nn.ConvTranspose1d(c5, c4, 4, 2, 1, bias=False))
        self.dec3 = UpBlock(cat_ch=c4 + c4, mid_ch=c4, out_ch=c3, kernel_size=3)
        self.dec2 = UpBlock(cat_ch=c3 + c3, mid_ch=c3, out_ch=c2)
        self.dec1 = UpBlock(cat_ch=c2 + c2, mid_ch=c2, out_ch=c1)
        self.head = nn.Conv1d(c1, out_channels, kernel_size=1)

    def forward(self, x):
        s0 = self.enc0(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        x = self.bot(s3)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        x = self.head(x)
        return x

class TSSCD_TransEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, Transformer_hparams):
        super(TSSCD_TransEncoder, self).__init__()
        d_model, nhead, num_layers, dim_feedforward = Transformer_hparams.values()
        self.embedding = nn.Linear(in_channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = TransformerEncoderLayerWithAttn(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))
        self.decoder = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, out_channels))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.decoder(x)
        return x.permute(0, 2, 1)

def generate_model_instances(is_opt_only=False, model_idx='1038'):
    configs = Configs(is_opt_only=is_opt_only)
    model_names = ['TSSCD_TransEncoder', 'TSSCD_Unet', 'TSSCD_FCN']
    if is_opt_only: model_idx = str(model_idx) + '_opt_only'
    return list(zip(model_names, [
        TSSCD_TransEncoder(configs.input_channels, configs.classes, configs.Transformer_hparams),
        TSSCD_Unet(configs.input_channels, configs.classes, configs.model_hidden),
        TSSCD_FCN(configs.input_channels, configs.classes, configs.model_hidden)
    ]))

# ================= Visualization Functions =================
# --- 文件：TSSCD.py ---

def plot_confusion_matrix(cm, classes, save_path, normalize=True, 
                          pa_mat=None, ua_mat=None, title=None):
    """
    绘制混淆矩阵或精度矩阵。
    Change Type Detection 模式下：
    - 左上角显示 UA
    - 右下角显示 PA
    - 中间显示 /
    """
    cm = cm.T.copy()
    font_size = 24
    plt.figure(figsize=(10, 8))
    
    # === 1. 颜色映射逻辑 (保持不变) ===
    vmin, vmax = 0, 1
    dual_gamma_norm = DualGammaNorm(
        vmin=vmin,
        vmax=vmax,
        threshold=0.5, 
        gamma_low=0.3,
        gamma_high=2
    )
    if pa_mat is not None and ua_mat is not None:
        cm_display = cm.T.copy() # F1 矩阵
        cmap = plt.cm.Purples
        plt.imshow(cm_display, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    elif normalize:
        # cm_display = cm.astype('float') / (cm.sum(axis=0)[:, np.newaxis] + 1e-6)
        cm_display = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-6)
        cmap = dual_gamma_norm
        plt.imshow(cm_display.T.copy(), interpolation='nearest', cmap=plt.cm.Reds, norm=dual_gamma_norm)
    else:
        cm_display = cm
        cmap = dual_gamma_norm
        vmin, vmax = None, None
        plt.imshow(cm_display, interpolation='nearest', cmap=plt.cm.Reds, norm=dual_gamma_norm)
    
    if title:
        plt.title(title, fontsize=font_size, pad=20)
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=font_size)
    plt.yticks(tick_marks, classes, fontsize=font_size)
    
    # === 2. 文本显示逻辑 (核心修改) ===
    thresh = (cm_display.max() + cm_display.min()) / 2. if vmin is None else 0.5
    
    # 调整特殊模式下的字体大小，防止挤在一起
    corner_font_size = font_size  # 变小一点
    slash_font_size = font_size  # 斜杠大小
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val_color = cm_display[i, j]
            text_color = "white" if val_color > thresh else "black"
            
            # --- 分支 A: Change Type Detection 模式 (UA/PA 分角显示) ---
            if pa_mat is not None and ua_mat is not None:
                f1_val = cm[j, i]
                
                if f1_val == 0:
                    # 如果没有数据，只在中间画个短横线
                    plt.text(j, i, "—", 
                             horizontalalignment="center", verticalalignment="center",
                             fontsize=font_size, color=text_color, fontweight='bold')
                else:
                    ua_str = f"{ua_mat[i, j] * 100:.2f}%"
                    pa_str = f"{pa_mat[i, j] * 100:.2f}%"
                    
                    # 1. 左上角显示 UA
                    # x 减小, y 减小 (origin is top-left)
                    plt.text(j - 0.4, i - 0.4, ua_str,
                             horizontalalignment="left", 
                             verticalalignment="top",
                             fontsize=corner_font_size, 
                             color=text_color, 
                             fontweight='bold')
                    
                    # 2. 中间显示斜杠 (更细一点或颜色淡一点也可以，这里保持一致)
                    plt.text(j, i, "/",
                             horizontalalignment="center", 
                             verticalalignment="center",
                             fontsize=slash_font_size, 
                             color=text_color, 
                             fontweight='bold')

                    # 3. 右下角显示 PA
                    # x 增加, y 增加
                    plt.text(j + 0.4, i + 0.4, pa_str,
                             horizontalalignment="right", 
                             verticalalignment="bottom",
                             fontsize=corner_font_size, 
                             color=text_color, 
                             fontweight='bold')

            # --- 分支 B: 普通混淆矩阵模式 (居中显示) ---
            else:
                if normalize:
                    value_text = f"{val_color*100:.2f}%"
                else:
                    value_text = f"{int(cm[i, j])}"
                
                plt.text(j, i, value_text,
                         horizontalalignment="center",
                         verticalalignment="center",
                         fontsize=font_size,
                         color=text_color,
                         fontweight='bold')
    
    # 设置轴标签
    if pa_mat is not None:
        # 为了配合左上/右下，这里可以稍微提示一下
        # 比如 X轴标签加上 (UA: Top-Left)，Y轴标签加上 (PA: Bottom-Right)
        # 或者保持原样，靠图例或文字说明
        x_label, y_label = 'To', 'From'
    elif normalize: 
        x_label, y_label = 'Actual label', 'Predicted label'
        
    plt.ylabel(y_label, fontsize=font_size, fontweight='bold')
    plt.xlabel(x_label, fontsize=font_size, fontweight='bold')
    
    plt.tight_layout()
    print(save_path)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
def process_regional_data(log_dir, model_idx, model_name, metircs_rename, classes):
    """
    处理区域数据并生成可视化。
    model_idx: e.g. '1038'
    model_name: e.g. 'TSSCD_Unet'
    """
    permutation_folders = [str(i) for i in range(1, 6) if os.path.isdir(os.path.join(log_dir, str(i)))]
    if not permutation_folders:
        return None
    
    all_province_data = {}
    
    for perm_folder in permutation_folders:
        perm_path = os.path.join(log_dir, perm_folder)
        province_files = [f for f in os.listdir(perm_path) if os.path.isfile(os.path.join(perm_path, f))]
        
        for province_file in province_files:
            province_name = province_file.split('_')[0]
            if province_name not in all_province_data:
                all_province_data[province_name] = []
            
            file_path = os.path.join(perm_path, province_file)
            info = extract_accuracy_from_log(file_path)
            # info = (epoch_info, last_epoch, pth_name)
            
            # 获取最后一次保存的 epoch 数据
            if info[0] and info[1] in info[0]:
                data = info[0][info[1]]
                data['pth_idx'] = perm_folder # 记录是第几个 fold (1, 2, 3...)
                # 记录是否为 opt_only，用于文件名生成
                data['is_opt'] = '_opt' in info[2]
                all_province_data[province_name].append(data)

    province_dfs = dict()
    
    for province_name, acc_list in all_province_data.items():
        if not acc_list: continue
        
        metric_values = dict()
        
        # 遍历该省份所有 fold 的数据
        for acc_dict in acc_list:
            pth_idx = acc_dict['pth_idx']
            is_opt = acc_dict.get('is_opt', False)
            
            # === 处理可视化 (Confusion Matrix & Change Type Matrix) ===
            
            # 1. 准备 Change Type Vis 需要的三个矩阵
            if 'ct_f1' in acc_dict and 'ct_pa' in acc_dict and 'ct_ua' in acc_dict:
                vis_save_dir = os.path.join('models', 'model_data', 'log', 'change_type_acc', 
                                          model_idx, model_name, pth_idx)
                if not os.path.exists(vis_save_dir): os.makedirs(vis_save_dir)
                
                # 文件名: SD.png 或 SD_opt.png
                fname = f"{province_name}_opt.png" if is_opt else f"{province_name}.png"
                
                plot_confusion_matrix(
                    cm=acc_dict['ct_f1'], 
                    classes=classes,
                    save_path=os.path.join(vis_save_dir, fname),
                    normalize=False,
                    pa_mat=acc_dict['ct_pa'],
                    ua_mat=acc_dict['ct_ua'],
                    title=None
                )
            
            # 2. 处理普通指标汇总
            if 'ct_f1' in acc_dict:
                # 计算 Macro-F1
                f1_mat = acc_dict['ct_f1']
                valid_elements = f1_mat[f1_mat > 0]
                macro_f1 = np.mean(valid_elements) if len(valid_elements) > 0 else 0.0
                acc_dict['Macro_Event_F1'] = macro_f1

            for metric, value in acc_dict.items():
                if metric in ['pth', 'pth_idx', 'train_loss', 'F1', 'mIoU', 'is_opt']: continue
                
                # 绘制普通混淆矩阵 (仍然保存在 log/confusion_matrix 下，保持原样)
                if metric == 'confusion_matrix':
                    cm_save_path = os.path.join(f'models\\model_data\\log\\{metric}\\', model_idx, model_name, pth_idx)
                    if not os.path.exists(cm_save_path): os.makedirs(cm_save_path)
                    fname = f"{province_name}_opt.png" if is_opt else f"{province_name}.png"
                    plot_confusion_matrix(value, classes, os.path.join(cm_save_path, fname), normalize=True)
                    continue
                
                if metric in ['ct_f1', 'ct_pa', 'ct_ua']: continue # 跳过矩阵数据
                
                if metric not in metric_values: metric_values[metric] = []
                metric_values[metric].append(value)
        
        # 生成 DataFrame
        if metric_values:
            _columns = list(metric_values.keys())
            _columns.insert(0, 'model')
            df = pd.DataFrame(columns=_columns).astype(object)
            _row = {'model': province_name}
            for metric, values in metric_values.items():
                _row[metric] = f'{np.mean(values)*100:.1f}±{np.std(values)*100:.2f}'
            df.loc[0] = _row
            df.rename(columns=metircs_rename, inplace=True)
            province_dfs[province_name] = df
    
    return province_dfs

# ================= Main Execution =================
if __name__ == '__main__':
    batch_size, seq_len, device = 64, 60, torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classes = ['SA', 'BF', 'WB', 'HL', 'OV']
    target_dir = 'models\\model_data\\log'
    
    model_idxs = ['1038'] 
    print(f'Current model indices: {model_idxs}')
          
    models_name = ['TSSCD_TransEncoder', 'TSSCD_Unet', 'TSSCD_FCN']
    
    metircs_rename = {
       'temporal_CdAccuracy': 'T-SMA',
       'spatial_LccAccuracy': 'S-SMA',
       'spatial_PA': 'S-PA', 'spatial_UA': 'S-UA', 'spatial_F1': 'S-F1',
       'temporal_PA': 'T-PA', 'temporal_UA': 'T-UA', 'temporal_F1': 'T-F1',
       'Macro_Event_F1': 'Macro-F1'
    }
    
    for model_idx in model_idxs:
        df_4_csv, all_province_dfs = None, dict()
        
        for is_opt_only in [False, True]:
            _model_idx = model_idx if not is_opt_only else model_idx + '_opt_only'
            
            for model_name in models_name:
                log_dir = os.path.join(target_dir, model_name, _model_idx)
                if not os.path.exists(log_dir): continue
                    
                log_files = [os.path.join(log_dir, i) for i in os.listdir(log_dir) if os.path.isfile(os.path.join(log_dir, i))]
                infos = [extract_accuracy_from_log(log_file) for log_file in log_files]
                model_saved_accuracy = [info[0][info[1]] for info in infos if info is not None and info[1] in info[0]]
                
                metric_values = dict()
                
                # 遍历全国数据（用于计算均值和绘制全国图）
                for acc_dict in model_saved_accuracy:
                    pth_idx = acc_dict['pth'].split('_')[1] # Fold ID (1, 2, 3...)
                    
                    # 1. 准备 Change Type Vis 需要的矩阵
                    if 'ct_f1' in acc_dict and 'ct_pa' in acc_dict and 'ct_ua' in acc_dict:
                        vis_save_dir = os.path.join('models', 'model_data', 'log', 'change_type_acc', 
                                                  model_idx, model_name, pth_idx)
                        if not os.path.exists(vis_save_dir): os.makedirs(vis_save_dir)
                        
                        # 文件名: 1.png 或 1_opt_only.png
                        fname = f"{pth_idx}_opt_only.png" if is_opt_only else f"{pth_idx}.png"
                        
                        plot_confusion_matrix(
                            cm=acc_dict['ct_f1'], 
                            classes=classes,
                            save_path=os.path.join(vis_save_dir, fname),
                            normalize=False,
                            pa_mat=acc_dict['ct_pa'],
                            ua_mat=acc_dict['ct_ua'],
                            title=None
                        )

                    # 2. 计算 Macro-F1
                    if 'ct_f1' in acc_dict:
                        f1_mat = acc_dict['ct_f1']
                        valid_elements = f1_mat[f1_mat > 0]
                        acc_dict['Macro_Event_F1'] = np.mean(valid_elements) if len(valid_elements) > 0 else 0.0
                    
                    # 3. 收集其他指标
                    for metric, value in acc_dict.items():
                        if metric in ['pth', 'train_loss', 'F1', 'mIoU']: continue
                        
                        # 普通混淆矩阵单独绘制
                        if metric == 'confusion_matrix':
                            cm_save_path = os.path.join(f'models\\model_data\\log\\{metric}\\', model_idx, model_name, pth_idx)
                            if not os.path.exists(cm_save_path): os.makedirs(cm_save_path)
                            fname = f"{pth_idx}_opt_only.png" if is_opt_only else f"{pth_idx}.png"
                            plot_confusion_matrix(value, classes, os.path.join(cm_save_path, fname), normalize=True)
                            continue
                        
                        if metric in ['ct_f1', 'ct_pa', 'ct_ua']: continue # 矩阵不放入 metric_values
                        
                        if metric not in metric_values: metric_values[metric] = []
                        metric_values[metric].append(value)
                
                # 生成全国数据 Excel 行
                if df_4_csv is None:
                    _columns = list(metric_values.keys())
                    _columns.insert(0, 'model')
                    df_4_csv = pd.DataFrame(columns=_columns).astype(object)

                _model_name_disp = model_name[5:] if not is_opt_only else model_name[5:] + ' (opt only)'
                _row_idx = models_name.index(model_name) * 2 + (1 if is_opt_only else 0)
                
                _row = {'model': _model_name_disp}
                for metric, values in metric_values.items():
                    _row[metric] = f'{np.mean(values)*100:.1f}±{np.std(values)*100:.2f}'
                
                df_4_csv.loc[_row_idx] = _row
               
                # 处理分省数据（并在函数内部绘制分省的 CT 图）
                # 注意：这里需要传入 classes 列表以供绘图
                province_dfs = process_regional_data(log_dir, model_idx, model_name, metircs_rename, classes)
                if province_dfs:
                    model_key = f"{model_name}_{'opt_only' if is_opt_only else 'full'}"
                    all_province_dfs[model_key] = province_dfs
        
        # 数据表后处理与保存
        df_4_csv.rename(columns=metircs_rename, inplace=True)
        df_4_csv = df_4_csv.sort_index()
        
        combined_province_df = pd.DataFrame()
        for model_key, province_dfs in all_province_dfs.items():
            model_full_name = model_key.split('_')[1]
            is_opt = 'opt_only' in model_key
            for province_name, df in province_dfs.items():
                temp_df = df.copy()
                temp_df['model'] = f"{model_full_name}_{province_name}" + ('_opt_only' if is_opt else '')
                temp_df['original_model'] = model_full_name
                temp_df['province'] = province_name
                temp_df['is_opt_only'] = is_opt
                combined_province_df = pd.concat([combined_province_df, temp_df], ignore_index=True)
        
        if not combined_province_df.empty:
            model_order = {'TransEncoder': 0, 'Unet': 1, 'FCN': 2}
            province_order = {'SD': 0, 'JS': 1, 'SH': 2, 'ZJ': 3, 'FJ': 4, 'GDGX': 5}
            combined_province_df['model_order'] = combined_province_df['original_model'].map(model_order)
            combined_province_df['province_order'] = combined_province_df['province'].map(province_order).fillna(999)
            
            combined_province_df = combined_province_df.sort_values(
                by=['model_order', 'province_order', 'is_opt_only'],
                ascending=[True, True, True]
            )
            combined_province_df = combined_province_df.drop(['original_model', 'is_opt_only', 'model_order', 'province_order'], axis=1)
            cols = combined_province_df.columns.tolist()
            if 'province' in cols:
                cols.insert(1, cols.pop(cols.index('province')))
                combined_province_df = combined_province_df[cols]
            if 'mIoU' in combined_province_df.columns:
                combined_province_df = combined_province_df.drop('mIoU', axis=1)
        
        excel_path = os.path.join(target_dir, f'model accs.xlsx')
        mode = 'a' if os.path.exists(excel_path) else 'w'
        with pd.ExcelWriter(excel_path, engine='openpyxl', mode=mode, if_sheet_exists='replace') as writer:
            df_4_csv.to_excel(writer, index=False, sheet_name=f'{model_idx}')
            if not combined_province_df.empty:
                sheet_name = f"{model_idx}_province_data"
                try:
                    combined_province_df.to_excel(writer, index=False, sheet_name=sheet_name)
                except Exception as e:
                    print(f"Failed to write sheet {sheet_name}: {e}")
                    
        print(f"Results saved to {excel_path}")