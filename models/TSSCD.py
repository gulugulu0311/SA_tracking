import os
import sys
import time
sys.path.append('.')
import math
import torch
import torch.nn.functional as F

from utils import *
from torch import nn, Tensor
from typing import Optional
# from openpyxl import load_workbook

class Configs():
    def __init__(self, is_opt_only=False):
        # 1 2 3 5 6 (0 1 2 4 5)
        self.classes = 5
        self.input_channels = 12 if not is_opt_only else 10

        # self.model_hidden = [128, 256, 512, 1024, 4096]
        self.model_hidden = [64, 128, 256, 512, 1024]

        self.Transformer_hparams = {
            'd_model': 256,
            'nhead': 8,
            'num_layers': 6,
            'dim_feedforward': 2048
        }

def count_parameters(model: torch.nn.Module, trainable_only=True):
    """
    compute the number of parameters in the model
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

class TransformerEncoderLayerWithAttn(nn.TransformerEncoderLayer):
    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x, _ = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True, is_causal=is_causal)
        return self.dropout1(x)
    
# ================   Components ====================
class DoubleConv1d(nn.Module):
    " (Conv → BN → RelU) × 2"
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
    """MaxPool ↓2  + DoubleConv"""
    def __init__(self, in_ch, out_ch,):
        super().__init__()
        self.pool = nn.MaxPool1d(2, stride=2, ceil_mode=True)
        self.conv = DoubleConv1d(in_ch, out_ch)
    def forward(self, x):
        return self.conv(self.pool(x))
    
class UpBlock(nn.Module):
    """ConvTransp ↑2  + concat + DoubleConv"""
    def __init__(self, cat_ch, mid_ch, out_ch, kernel_size=4):
        super().__init__()
        self.conv = DoubleConv1d(cat_ch, mid_ch)
        self.up   = nn.ConvTranspose1d(mid_ch, out_ch,
                                       kernel_size=kernel_size, stride=2, padding=1,
                                       bias=False)

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
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [B, T, C]
        x = x + self.pe[:, :x.size(1), :]
        return x

# ================ Main Network ====================
class TSSCD_FCN(nn.Module):
    def __init__(self, in_channels, out_channels, config):
        super(TSSCD_FCN, self).__init__()
        self.out_channels = out_channels
        self.config = config
        # 64, 128, 256, 512, 1024
        c1, c2, c3, c4, c5 = config
        
        # 60 → 30
        self.layer1 = nn.Sequential(
            DoubleConv1d(in_channels, c1),
            nn.MaxPool1d(2, stride=2, ceil_mode=True)  # Downsampling 1/2, Temporal Length = 30
        )
        
        # 30 → 15
        self.layer2 = nn.Sequential(
            DoubleConv1d(c1, c2),
            nn.MaxPool1d(2, stride=2, ceil_mode=True)  # Downsampling 1/4, Temporal Length = 15
        )

        # 15 → 8
        self.layer3 = nn.Sequential(
            DoubleConv1d(c2, c3),
            # nn.Conv1d(c3, c3, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2, ceil_mode=True)  # Downsampling 1/8, Temporal Length = 8
        )

        # 8 → 4
        self.layer4 = nn.Sequential(
            DoubleConv1d(c3, c4),
            # nn.Conv1d(c4, c4, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2, ceil_mode=True)  # Downsampling 1/16, Temporal Length = 4
        )
        # 4 → 4
        # self.layer5 = DoubleConv1d(c4, c5)  # 4 → 4
        
        # self.score_1 = nn.Conv1d(c5, out_channels, 1)
        self.score_1 = nn.Conv1d(c4, out_channels, 1)
        self.score_2 = nn.Conv1d(c3, out_channels, 1)
        self.score_3 = nn.Conv1d(c2, out_channels, 1)

        # L_out = (L_in - 1) × stride - 2 × padding + dilation × (kernel_size - 1) + output_padding + 1
        self.upsampling_2x = nn.ConvTranspose1d(out_channels, out_channels, 4, 2, 1, bias=False)
        self.upsampling_4x = nn.ConvTranspose1d(out_channels, out_channels, 3, 2, 1, bias=False)    #  8 → 15
        self.upsampling_8x = nn.ConvTranspose1d(out_channels, out_channels, 6, 4, 1, bias=False)    # 15 → 60

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.layer1(x) # length: 30
        self.s1 = self.layer2(h) # s1: length: 15
        self.s2 = self.layer3(self.s1)  # s2: length: 8
        self.s3 = self.layer4(self.s2)  # s3: length: 4
        # self.s4 = self.layer5(self.s3)  # s4: length: 4
        
        # s4 = self.score_1(self.s4)
        s3 = self.score_1(self.s3)
        # s4 = self.upsampling_2x(s4) # s3: length: 4 → 8
        s3 = self.upsampling_2x(s3) # s3: length: 4 → 8
        s2 = self.score_2(self.s2)
        
        # s2 += s4 # length: 8
        s2 += s3 # length: 8
        s2 = self.upsampling_4x(s2) # s2: length: 8 → 15
        s1 = self.score_3(self.s1) # s1: length: 15
        
        score = s1 + s2
        score = self.upsampling_8x(score) # s1: length: 15 → 60
        return score

class TSSCD_Unet(nn.Module):
    def __init__(self, in_channels, out_channels, cfg):
        """
        config: (c1, c2, c3, c4, c5)
        """
        super().__init__()
        c1, c2, c3, c4, c5 = cfg

        # Encoder
        self.enc0 = DoubleConv1d(in_channels, c1)            # L  → L
        self.enc1 = DownBlock(c1,  c2)                       # L  → L/2
        self.enc2 = DownBlock(c2,  c3)                       # L/2→ L/4
        self.enc3 = DownBlock(c3,  c4)                       # L/4→ L/8

        # Bottleneck (保持 L/8 不变，再↑2)
        self.bot  = nn.Sequential(
            DownBlock(c4, c5),                                  # L/8 → L/16
            nn.ConvTranspose1d(c5, c4, 4, 2, 1, bias=False)     # L/16→ L/8
        )

        # Decoder
        self.dec3 = UpBlock(cat_ch=c4 + c4, mid_ch=c4, out_ch=c3, kernel_size=3)   # 8     → 16
        self.dec2 = UpBlock(cat_ch=c3 + c3, mid_ch=c3, out_ch=c2)   # 16    → 30
        self.dec1 = UpBlock(cat_ch=c2 + c2, mid_ch=c2, out_ch=c1)   # 30    → 60

        # Head
        self.head = nn.Conv1d(c1, out_channels, kernel_size=1)

    def forward(self, x):
        # -------- Encoder --------
        s0 = self.enc0(x)   # L
        s1 = self.enc1(s0)  # L/2
        s2 = self.enc2(s1)  # L/4
        s3 = self.enc3(s2)  # L/8

        # -------- Bottleneck --------
        x  = self.bot(s3)   # L/8

        # -------- Decoder --------
        x  = self.dec3(x, s3)   # L/4
        x  = self.dec2(x, s2)   # L/2
        x  = self.dec1(x, s1)   # L

        x  = self.head(x)

        return x
class TSSCD_TransEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, Transformer_hparams):
        super(TSSCD_TransEncoder, self).__init__()

        d_model = Transformer_hparams['d_model']
        nhead = Transformer_hparams['nhead']
        num_layers = Transformer_hparams['num_layers']
        dim_feedforward = Transformer_hparams['dim_feedforward']

        self.embedding = nn.Linear(in_channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=dim_feedforward)
        encoder_layer = TransformerEncoderLayerWithAttn(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_channels)

        )

    def forward(self, x):
        # x: [B, C, T] -> [B, T, C]
        x = x.permute(0, 2, 1)
        x = self.embedding(x)           # [B, T, d_model]
        x = self.pos_encoder(x)         # [B, T, d_model] with positional encoding
        x = self.transformer_encoder(x) # [B, T, d_model]
        x = self.decoder(x)             # [B, out_channels, T]
        return x.permute(0, 2, 1)  # [B, T, out_channels]
class ConvTransEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, d_model):
        super(ConvTransEncoder, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, 64, 3, padding=1), nn.GELU(),
            nn.Conv1d(64, d_model, 3, padding=1), nn.GELU(),
        )

        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True, dim_feedforward=256)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6, norm=nn.LayerNorm(d_model))

        self.decoder = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1),nn.GELU(),
            nn.Conv1d(d_model, d_model, 3, padding=1),nn.GELU(),
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
        return x        # [B, out_channels, T]

def generate_model_instances(is_opt_only=False, model_idx=1027, ):
    configs = Configs(is_opt_only=is_opt_only)
    model_names = ['TSSCD_TransEncoder', 'TSSCD_Unet', 'TSSCD_FCN']
    if is_opt_only: 
        model_idx = str(model_idx) + '_opt_only'
    for name in model_names:
        model_dir = os.path.join(f'models\\model_data\\{name}', model_idx)
        log_dir = os.path.join(f'models\\model_data\\log\\{name}', model_idx)

        while not (os.path.exists(model_dir) and os.path.exists(log_dir)):
            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)
            time.sleep(1)
    return list(zip(
            model_names,
            [
                TSSCD_TransEncoder(configs.input_channels, configs.classes, configs.Transformer_hparams),
                TSSCD_Unet(configs.input_channels, configs.classes, configs.model_hidden),
                TSSCD_FCN(configs.input_channels, configs.classes, configs.model_hidden)
            ]
        ))

if __name__ == '__main__':
    batch_size, seq_len, device = 64, 60, device_on()
    target_dir = 'models\\model_data\\log'
    
    model_idxs = os.listdir('models\\model_data\\dataset')
    print(f'current model: {model_idxs}')
          
    models_name = ['TSSCD_TransEncoder', 'TSSCD_Unet', 'TSSCD_FCN']
    metircs_rename = {
       'temporal_CdAccuracy': 'T-SMA',
       'spatial_LccAccuracy': 'S-SMA',
       'spatial_PA': 'S-PA', 'spatial_UA': 'S-UA', 'spatial_F1': 'S-F1',
       'temporal_PA': 'T-PA', 'temporal_UA': 'T-UA', 'temporal_F1': 'T-F1',
    }
    
    for model_idx in model_idxs:
        df_4_csv = None
        # for opt_only and opt + sar
        for is_opt_only in [False, True]:
            _model_idx = model_idx if not is_opt_only else model_idx + '_opt_only'
            # iterate for each model
            for model_name in models_name:
                # metrics
                log_dir = os.path.join(target_dir, model_name, _model_idx)
                log_files = [os.path.join(log_dir, i) for i in os.listdir(log_dir)]
                
                infos = [extract_accuracy_from_log(log_file) for log_file in log_files]
                model_saved_accuracy = [info[0][info[1]] for info in infos if info is not None]
                # calc for mean & std
                metric_values, metric_values_for_main_model = dict(), dict()
                for acc_dict in model_saved_accuracy:
                    for metric, value in acc_dict.items():
                        # metrics exclued
                        if metric in ['confusion_matrix', 'pth', 'train_loss', 'F1']: continue
                        if metric not in metric_values: # init ...
                            metric_values[metric], metric_values_for_main_model[metric] = list(), None
                        # append acc separately for main model and others
                        if acc_dict['pth'] == _model_idx : metric_values_for_main_model[metric] = value
                        else: metric_values[metric].append(value)
                        
                # init data frame for csv
                if df_4_csv is None:
                    _columns = list(metric_values.keys())
                    _columns.insert(0, 'model')
                    df_4_csv = pd.DataFrame(columns=_columns).astype(object)

                _model_name = model_name[5:] if not is_opt_only else model_name[5:] + ' (opt only)'
                _row, _row_idx = {'model': _model_name}, models_name.index(model_name)
                _row_idx = (2 * _row_idx) if not is_opt_only else (2 * _row_idx + 1)
                
                # write acc in pandas
                for metric, values in metric_values.items():
                    _row[metric] = f'{np.mean(values)*100:.1f}±{np.std(values)*100:.2f}'
                df_4_csv.loc[_row_idx] = _row
                
        df_4_csv.rename(columns=metircs_rename, inplace=True)
        df_4_csv = df_4_csv.sort_index()
        
        # write acc in excel
        excel_path = os.path.join(target_dir, f'model accs.xlsx')
        if os.path.exists(excel_path):
            with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                df_4_csv.to_excel(writer, index=False, sheet_name=f'{model_idx}')
                
        else:
            df_4_csv.to_excel(excel_path, index=False, sheet_name=f'{model_idx}')