import os
import sys
import time
sys.path.append('.')
import math
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import *
from torch import nn, Tensor
from typing import Optional
# from openpyxl import load_workbook
plt.rcParams['font.family'] = ['Arial']
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.unicode_minus'] = False
class Configs():
    def __init__(self, is_opt_only=False):
        # Classes: 1 2 3 5 6 (indices: 0 1 2 4 5)
        self.classes = 5
        self.input_channels = 12 if not is_opt_only else 10
        self.model_hidden = [64, 128, 256, 512, 1024]

        self.Transformer_hparams = {
            'd_model': 256,
            'nhead': 8,
            'num_layers': 6,
            'dim_feedforward': 2048
        }

def count_parameters(model: torch.nn.Module, trainable_only=True):
    """Calculate number of parameters in the model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

class TransformerEncoderLayerWithAttn(nn.TransformerEncoderLayer):
    """Transformer encoder layer with attention weights."""
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x, _ = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True, is_causal=is_causal)
        return self.dropout1(x)
    
# ================ Network Components ====================
class DoubleConv1d(nn.Module):
    """Two consecutive (Conv → BN → ReLU) blocks."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.net(x)

class DownBlock(nn.Module):
    """Downsampling block: MaxPool with 2x reduction + DoubleConv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool1d(2, stride=2, ceil_mode=True)
        self.conv = DoubleConv1d(in_ch, out_ch)
    
    def forward(self, x):
        return self.conv(self.pool(x))
    
class UpBlock(nn.Module):
    """Upsampling block: ConvTranspose with 2x upsampling + concatenation + DoubleConv."""
    def __init__(self, cat_ch, mid_ch, out_ch, kernel_size=4):
        super().__init__()
        self.conv = DoubleConv1d(cat_ch, mid_ch)
        self.up = nn.ConvTranspose1d(mid_ch, out_ch,
                                    kernel_size=kernel_size, stride=2, padding=1,
                                    bias=False)

    def forward(self, x, skip):
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return self.up(x)

class PositionalEncoding(nn.Module):
    """Add positional encoding to input embeddings."""
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [B, T, C]
        x = x + self.pe[:, :x.size(1), :]
        return x

# ================ Main Network Architectures ====================
class TSSCD_FCN(nn.Module):
    """Fully Convolutional Network for Time Series Change Detection."""
    def __init__(self, in_channels, out_channels, config):
        super(TSSCD_FCN, self).__init__()
        self.out_channels = out_channels
        self.config = config
        # Channel configuration: 64, 128, 256, 512, 1024
        c1, c2, c3, c4, c5 = config
        
        # Downsampling path (60 → 30)
        self.layer1 = nn.Sequential(
            DoubleConv1d(in_channels, c1),
            nn.MaxPool1d(2, stride=2, ceil_mode=True)  # 1/2 reduction, length = 30
        )
        
        # Downsampling path (30 → 15)
        self.layer2 = nn.Sequential(
            DoubleConv1d(c1, c2),
            nn.MaxPool1d(2, stride=2, ceil_mode=True)  # 1/4 reduction, length = 15
        )

        # Downsampling path (15 → 8)
        self.layer3 = nn.Sequential(
            DoubleConv1d(c2, c3),
            nn.MaxPool1d(2, stride=2, ceil_mode=True)  # 1/8 reduction, length = 8
        )

        # Downsampling path (8 → 4)
        self.layer4 = nn.Sequential(
            DoubleConv1d(c3, c4),
            nn.MaxPool1d(2, stride=2, ceil_mode=True)  # 1/16 reduction, length = 4
        )
        
        # Score prediction layers
        self.score_1 = nn.Conv1d(c4, out_channels, 1)
        self.score_2 = nn.Conv1d(c3, out_channels, 1)
        self.score_3 = nn.Conv1d(c2, out_channels, 1)

        # Upsampling layers
        self.upsampling_2x = nn.ConvTranspose1d(out_channels, out_channels, 4, 2, 1, bias=False)
        self.upsampling_4x = nn.ConvTranspose1d(out_channels, out_channels, 3, 2, 1, bias=False)    # 8 → 15
        self.upsampling_8x = nn.ConvTranspose1d(out_channels, out_channels, 6, 4, 1, bias=False)    # 15 → 60

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.layer1(x)  # Length: 30
        self.s1 = self.layer2(h)  # Length: 15
        self.s2 = self.layer3(self.s1)  # Length: 8
        self.s3 = self.layer4(self.s2)  # Length: 4
        
        s3 = self.score_1(self.s3)  # Score from deepest layer
        s3 = self.upsampling_2x(s3)  # Length: 4 → 8
        s2 = self.score_2(self.s2)  # Score from middle layer
        
        s2 += s3  # Skip connection
        s2 = self.upsampling_4x(s2)  # Length: 8 → 15
        s1 = self.score_3(self.s1)  # Score from shallow layer
        
        score = s1 + s2  # Final skip connection
        score = self.upsampling_8x(score)  # Length: 15 → 60
        return score

class TSSCD_Unet(nn.Module):
    """U-Net architecture for Time Series Change Detection."""
    def __init__(self, in_channels, out_channels, cfg):
        """Initialize with channel configuration (c1, c2, c3, c4, c5)."""
        super().__init__()
        c1, c2, c3, c4, c5 = cfg

        # Encoder
        self.enc0 = DoubleConv1d(in_channels, c1)            # Length remains
        self.enc1 = DownBlock(c1, c2)                        # Length → Length/2
        self.enc2 = DownBlock(c2, c3)                        # Length/2 → Length/4
        self.enc3 = DownBlock(c3, c4)                        # Length/4 → Length/8

        # Bottleneck (maintains Length/8, then upscales 2x)
        self.bot = nn.Sequential(
            DownBlock(c4, c5),                               # Length/8 → Length/16
            nn.ConvTranspose1d(c5, c4, 4, 2, 1, bias=False)  # Length/16 → Length/8
        )

        # Decoder
        self.dec3 = UpBlock(cat_ch=c4 + c4, mid_ch=c4, out_ch=c3, kernel_size=3)  # 8 → 16
        self.dec2 = UpBlock(cat_ch=c3 + c3, mid_ch=c3, out_ch=c2)  # 16 → 30
        self.dec1 = UpBlock(cat_ch=c2 + c2, mid_ch=c2, out_ch=c1)  # 30 → 60

        # Output head
        self.head = nn.Conv1d(c1, out_channels, kernel_size=1)

    def forward(self, x):
        # -------- Encoder --------
        s0 = self.enc0(x)   # Original length
        s1 = self.enc1(s0)  # Length/2
        s2 = self.enc2(s1)  # Length/4
        s3 = self.enc3(s2)  # Length/8

        # -------- Bottleneck --------
        x = self.bot(s3)    # Length/8

        # -------- Decoder --------
        x = self.dec3(x, s3)  # Length/4
        x = self.dec2(x, s2)  # Length/2
        x = self.dec1(x, s1)  # Original length

        x = self.head(x)

        return x

class TSSCD_TransEncoder(nn.Module):
    """Transformer-based encoder for Time Series Change Detection."""
    def __init__(self, in_channels, out_channels, Transformer_hparams):
        super(TSSCD_TransEncoder, self).__init__()

        d_model = Transformer_hparams['d_model']
        nhead = Transformer_hparams['nhead']
        num_layers = Transformer_hparams['num_layers']
        dim_feedforward = Transformer_hparams['dim_feedforward']

        self.embedding = nn.Linear(in_channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder with attention
        encoder_layer = TransformerEncoderLayerWithAttn(d_model=d_model, nhead=nhead, 
                                                       batch_first=True, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, 
                                                       norm=nn.LayerNorm(d_model))

        # Output decoder
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_channels)
        )

    def forward(self, x):
        # x: [B, C, T] -> [B, T, C]
        x = x.permute(0, 2, 1)
        x = self.embedding(x)           # [B, T, d_model]
        x = self.pos_encoder(x)         # Add positional encoding
        x = self.transformer_encoder(x) # [B, T, d_model]
        x = self.decoder(x)             # [B, T, out_channels]
        return x.permute(0, 2, 1)       # [B, out_channels, T]

class ConvTransEncoder(nn.Module):
    """Convolutional-Transformer hybrid encoder for change detection."""
    def __init__(self, in_channels, out_channels, d_model):
        super(ConvTransEncoder, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, 64, 3, padding=1), nn.GELU(),
            nn.Conv1d(64, d_model, 3, padding=1), nn.GELU(),
        )

        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, 
                                                 batch_first=True, dim_feedforward=256)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6, 
                                                       norm=nn.LayerNorm(d_model))

        self.decoder = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1), nn.GELU(),
            nn.Conv1d(d_model, d_model, 3, padding=1), nn.GELU(),
            nn.Conv1d(d_model, out_channels, 1)
        )

    def forward(self, x):
        # x: [B, C, T]
        x = self.conv_block(x)            # [B, conv_channels, T]
        x = x.permute(0, 2, 1)            # [B, T, conv_channels]
        
        x = self.pos_encoder(x)           # [B, T, d_model]
        x = self.transformer_encoder(x)   # [B, T, d_model]
        
        x = x.permute(0, 2, 1)            # [B, d_model, T]
        x = self.decoder(x)               # [B, out_channels, T]
        return x

def generate_model_instances(is_opt_only=False, model_idx='1036'):
    """Generate model instances with specified configurations."""
    configs = Configs(is_opt_only=is_opt_only)
    model_names = ['TSSCD_TransEncoder', 'TSSCD_Unet', 'TSSCD_FCN']
    
    if is_opt_only: 
        model_idx = str(model_idx) + '_opt_only'
    
    # Create model directories if they don't exist
    for name in model_names:
        model_dir = os.path.join(f'models\\model_data\\{name}', str(model_idx))
        log_dir = os.path.join(f'models\\model_data\\log\\{name}', str(model_idx))

        while not (os.path.exists(model_dir) and os.path.exists(log_dir)):
            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)
            time.sleep(1)
    
    # Return list of (model_name, model_instance) pairs
    return list(zip(
            model_names,
            [
                TSSCD_TransEncoder(configs.input_channels, configs.classes, configs.Transformer_hparams),
                TSSCD_Unet(configs.input_channels, configs.classes, configs.model_hidden),
                TSSCD_FCN(configs.input_channels, configs.classes, configs.model_hidden)
            ]
        ))
def plot_confusion_matrix(cm, classes, save_path, normalize=True):
    font_size = 20
    cm_copy = cm.T.copy()
    # Apply normalization if requested
    if normalize:
        col_sums = cm_copy.sum(axis=0, keepdims=True)
        # Avoid division by zero
        col_sums[col_sums == 0] = 1
        # Normalize each column to sum
        cm_copy = cm_copy / col_sums
    # Set up the plot with appropriate figure size
    plt.figure(figsize=(8, 8))
    
    # Create a custom normalization instance using DualGammaNorm
    # For percentage data, we can set threshold at 50% (middle of 0-100 range)
    dual_gamma_norm = DualGammaNorm(
        vmin=0,
        vmax=1,
        threshold=0.5, 
        gamma_low=0.3,
        gamma_high=2
    )
    
    # Display the confusion matrix with DualGammaNorm for color mapping
    if normalize:
        # for confusion matrix
        plt.imshow(cm_copy, interpolation='nearest', cmap=plt.cm.Reds, norm=dual_gamma_norm)
    else:
        # for change_type_acc
        plt.imshow(cm_copy, interpolation='nearest', cmap=plt.cm.Purples)
    
    # Set class labels with proper rotation and alignment
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=30, fontsize=font_size)
    plt.yticks(tick_marks, classes, fontsize=font_size)
    
    # Format for displaying numbers - use .2f for percentages to match example
    # fmt = '.2f' if normalize else 'd'
    thresh = cm_copy.max() / 2.
    
    # Add numerical labels on the heatmap
    for i in range(cm_copy.shape[0]):
        for j in range(cm_copy.shape[1]):
            value_text = format(cm_copy[i, j] * 100, '.2f')
            value_text += '%'
            plt.text(j, i, value_text,
                     horizontalalignment="center",
                     verticalalignment="center",
                     fontsize=font_size,
                     color="white" if cm_copy[i, j] > thresh else "black")
    
    # Set axis labels with larger font
    if normalize: 
        x_label, y_label = 'Actual label', 'Predicted class'
    else:
        x_label, y_label = 'To', 'From'
    plt.ylabel(y_label, fontsize=font_size, fontweight='bold')
    plt.xlabel(x_label, fontsize=font_size, fontweight='bold')
    
    # Adjust layout to prevent label clipping
    plt.tight_layout()
    
    # Save with high resolution
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def process_regional_data(log_dir, metircs_rename):
    """Process regional dataset accuracy statistics."""
    # Check for permutation folders (1-5)
    permutation_folders = [str(i) for i in range(1, 6) if os.path.isdir(os.path.join(log_dir, str(i)))]
    if not permutation_folders:
        print(f"No regional data folders found at: {log_dir}")
        return None
    
    # Dictionary to store all province data
    all_province_data = {}
    
    # Process each permutation folder
    for perm_folder in permutation_folders:
        perm_path = os.path.join(log_dir, perm_folder)
        # Get all province files in the folder
        province_files = [f for f in os.listdir(perm_path) if os.path.isfile(os.path.join(perm_path, f))]
        
        for province_file in province_files:
            # Extract province name (assuming format: "provinceName_xxx.txt")
            province_name = province_file.split('_')[0]
            
            if province_name not in all_province_data:
                all_province_data[province_name] = []
            
            # Extract accuracy information
            file_path = os.path.join(perm_path, province_file)
            info = extract_accuracy_from_log(file_path)
            info[0][info[1]]['pth_idx'] = perm_folder
            if info is not None:
                all_province_data[province_name].append(info[0][info[1]])
    # Create DataFrames for each province
    province_dfs = dict()
    
    for province_name, acc_list in all_province_data.items():
        if not acc_list:
            continue
        
        # Calculate mean and std for each metric
        metric_values = dict()
        for acc_dict in acc_list:
            for metric, value in acc_dict.items():
                pth_idx = acc_dict['pth_idx']
                sn = province_name if not 'opt' in acc_dict['pth'] else province_name + '_opt'
                # Exclude unwanted metrics
                if metric in ['pth', 'pth_idx', 'train_loss', 'F1', 'mIoU']:  # Added 'mIoU' to skip list
                    continue
                if metric in ['confusion_matrix', 'change_type_acc']:
                    cm_save_path = os.path.join(f'models\\model_data\\log\\{metric}\\', model_idx, model_name, pth_idx)
                    if not os.path.exists(cm_save_path):    os.makedirs(cm_save_path)
                    is_norm = True if metric == 'confusion_matrix' else False
                    plot_confusion_matrix(value, classes, os.path.join(cm_save_path, f'{sn}.png'), normalize=is_norm)
                    continue
                if metric not in metric_values:
                    metric_values[metric] = []
                metric_values[metric].append(value)
        
        if metric_values:
            # Create DataFrame
            _columns = list(metric_values.keys())
            _columns.insert(0, 'model')
            df = pd.DataFrame(columns=_columns).astype(object)
            
            # Create row for this province
            _row = {'model': province_name}
            for metric, values in metric_values.items():
                _row[metric] = f'{np.mean(values)*100:.1f}±{np.std(values)*100:.2f}'
            
            df.loc[0] = _row
            df.rename(columns=metircs_rename, inplace=True)
            province_dfs[province_name] = df
    
    return province_dfs


if __name__ == '__main__':
    batch_size, seq_len, device = 64, 60, device_on()
    classes = ['S. alterniflora', 'Bare flats', 'Water body', 'Herbicide\n-treated\nS. alterniflora\nlitter', 'Other\nvegetation']
    target_dir = 'models\\model_data\\log'
    
    # Get or specify model indices
    model_idxs = os.listdir('models\\model_data\\dataset')
    model_idxs = ['1037']  # Override with specific models if needed
    print(f'Current model: {model_idxs}')
          
    models_name = ['TSSCD_TransEncoder', 'TSSCD_Unet', 'TSSCD_FCN']
    metircs_rename = {
       'temporal_CdAccuracy': 'T-SMA',
       'spatial_LccAccuracy': 'S-SMA',
       'spatial_PA': 'S-PA', 'spatial_UA': 'S-UA', 'spatial_F1': 'S-F1',
       'temporal_PA': 'T-PA', 'temporal_UA': 'T-UA', 'temporal_F1': 'T-F1',
    }
    
    for model_idx in model_idxs:
        df_4_csv, all_province_dfs = None, dict()
        # Process both standard and opt-only models
        for is_opt_only in [False, True]:
            _model_idx = model_idx if not is_opt_only else model_idx + '_opt_only'
            # Process each model architecture
            for model_name in models_name:
                # Get metrics from logs
                log_dir = os.path.join(target_dir, model_name, _model_idx)
                log_files = [os.path.join(log_dir, i) for i in os.listdir(log_dir) 
                            if os.path.isfile(os.path.join(log_dir, i))]
                
                infos = [extract_accuracy_from_log(log_file) for log_file in log_files]
                model_saved_accuracy = [info[0][info[1]] for info in infos if info is not None]
                
                # Calculate mean & std for metrics
                metric_values, metric_values_for_main_model = dict(), dict()
                for acc_dict in model_saved_accuracy:
                    for metric, value in acc_dict.items():
                        # Skip unwanted metrics
                        if metric in ['pth', 'train_loss', 'F1']:
                            continue
                        if metric in ['confusion_matrix', 'change_type_acc']:
                            pth_idx = acc_dict['pth'].split('_')[1]
                            cm_save_path = os.path.join(f'models\\model_data\\log\\{metric}\\', model_idx, model_name, pth_idx)
                            if not os.path.exists(cm_save_path):    os.makedirs(cm_save_path)
                            is_norm = True if metric == 'confusion_matrix' else False
                            plot_confusion_matrix(value, classes, os.path.join(cm_save_path, f'{pth_idx if not is_opt_only else pth_idx + '_opt_only'}.png'), normalize=is_norm)
                            continue
                        if metric not in metric_values:
                            metric_values[metric], metric_values_for_main_model[metric] = list(), None
                        
                        # Store main model accuracy separately
                        if acc_dict['pth'] == _model_idx:
                            metric_values_for_main_model[metric] = value
                        else:
                            metric_values[metric].append(value)
                
                # Initialize DataFrame for CSV output
                if df_4_csv is None:
                    _columns = list(metric_values.keys())
                    _columns.insert(0, 'model')
                    df_4_csv = pd.DataFrame(columns=_columns).astype(object)

                # Format model name
                _model_name = model_name[5:] if not is_opt_only else model_name[5:] + ' (opt only)'
                _row_idx = models_name.index(model_name) * 2
                if is_opt_only:
                    _row_idx += 1
                
                # Add metrics to DataFrame
                _row = {'model': _model_name}
                for metric, values in metric_values.items():
                    _row[metric] = f'{np.mean(values)*100:.1f}±{np.std(values)*100:.2f}'
                df_4_csv.loc[_row_idx] = _row
               
                # Process regional data
                province_dfs = process_regional_data(log_dir, metircs_rename)
                if province_dfs:
                    # Store regional data by model
                    model_key = f"{model_name}_{'opt_only' if is_opt_only else 'full'}"
                    all_province_dfs[model_key] = province_dfs
        
        # Rename metrics and sort
        df_4_csv.rename(columns=metircs_rename, inplace=True)
        df_4_csv = df_4_csv.sort_index()
        
        # Process combined regional data
        combined_province_df = pd.DataFrame()
        
        # Process each model's regional data
        for model_key, province_dfs in all_province_dfs.items():
            # Extract full model name
            model_full_name = model_key.split('_')[1]
            # Determine if opt-only model
            is_opt_only = 'opt_only' in model_key
            
            # Add model info to each province's data
            for province_name, df in province_dfs.items():
                temp_df = df.copy()
                # Set full model name with type and province
                if is_opt_only:
                    temp_df['model'] = f"{model_full_name}_{province_name}_opt_only"
                else:
                    temp_df['model'] = f"{model_full_name}_{province_name}"
                
                # Add columns for sorting
                temp_df['original_model'] = model_full_name
                temp_df['province'] = province_name
                temp_df['is_opt_only'] = is_opt_only
                
                # Merge into combined DataFrame
                combined_province_df = pd.concat([combined_province_df, temp_df], ignore_index=True)
        
        # Optimize sorting: by original model name in specified order, then province in custom order, then opt-only flag
        if not combined_province_df.empty:
            # Define custom order for models: Trans first, then Unet, then FCN
            model_order = {'TransEncoder': 0, 'Unet': 1, 'FCN': 2}
            # Define custom order for provinces as requested: SD JS SH ZJ FJ GDGX
            province_order = {'SD': 0, 'JS': 1, 'SH': 2, 'ZJ': 3, 'FJ': 4, 'GDGX': 5}
            
            # Create custom sort keys
            combined_province_df['model_order'] = combined_province_df['original_model'].map(model_order)
            # For provinces not in the custom order, assign a high value to place them at the end
            combined_province_df['province_order'] = combined_province_df['province'].map(province_order).fillna(999)
            
            # Sort by custom model order, then custom province order, then opt-only flag
            combined_province_df = combined_province_df.sort_values(
                by=['model_order', 'province_order', 'is_opt_only'],
                ascending=[True, True, True]  # Non-opt models first, followed by opt-only
            )
            
            # Remove temporary sorting columns
            combined_province_df = combined_province_df.drop(['original_model', 'is_opt_only', 'model_order', 'province_order'], axis=1)
            
            # Reorder columns to show province first
            cols = combined_province_df.columns.tolist()
            if 'province' in cols:
                cols.insert(1, cols.pop(cols.index('province')))
                combined_province_df = combined_province_df[cols]
                
            # Exclude mIoU columns if they exist
            if 'mIoU' in combined_province_df.columns:
                combined_province_df = combined_province_df.drop('mIoU', axis=1)
        
        # Write results to Excel
        excel_path = os.path.join(target_dir, f'model accs.xlsx')
        if os.path.exists(excel_path):
            with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                # Write national data
                df_4_csv.to_excel(writer, index=False, sheet_name=f'{model_idx}')
                # Write combined regional data
                if not combined_province_df.empty:
                    sheet_name = f"{model_idx}_province_data"
                    try:
                        combined_province_df.to_excel(writer, index=False, sheet_name=sheet_name)
                    except Exception as e:
                        print(f"Failed to write to sheet {sheet_name}: {e}")
        else:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Write national data
                df_4_csv.to_excel(writer, index=False, sheet_name=f'{model_idx}')
                # Write combined regional data
                if not combined_province_df.empty:
                    sheet_name = f"{model_idx}_province_data"
                    try:
                        combined_province_df.to_excel(writer, index=False, sheet_name=sheet_name)
                    except Exception as e:
                        print(f"Failed to write to sheet {sheet_name}: {e}")