import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import scale
from matplotlib.scale import ScaleBase
from matplotlib.transforms import Transform

plt.rcParams['font.family'] = ['Arial']
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.unicode_minus'] = False

class CustomTransform(Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True
    
    def __init__(self, data_points):
        Transform.__init__(self)
        self.data_points = data_points
        self.display_points = np.linspace(0, 1, len(data_points))
    
    def transform_non_affine(self, a):
        return np.interp(a, self.data_points, self.display_points)
    
    def inverted(self):
        return InvertedCustomTransform(self.data_points)

class InvertedCustomTransform(Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True
    
    def __init__(self, data_points):
        Transform.__init__(self)
        self.data_points = data_points
        self.display_points = np.linspace(0, 1, len(data_points))
    
    def transform_non_affine(self, a):
        return np.interp(a, self.display_points, self.data_points)
    
    def inverted(self):
        return CustomTransform(self.data_points)

class CustomScale(ScaleBase):
    name = 'custom'
    
    def __init__(self, axis, **kwargs):
        super().__init__(axis)
        self.data_points = kwargs.get('data_points', [0.0, 0.5, 1, 3, 5])
        self.data_points = sorted(self.data_points)

        self.minor_points = kwargs.get('minor_points', [])
        self.minor_points = sorted(self.minor_points)
    
    def get_transform(self):
        return CustomTransform(self.data_points)
    
    def set_default_locators_and_formatters(self, axis):
        from matplotlib.ticker import FixedLocator, FixedFormatter
        axis.set_major_locator(FixedLocator(self.data_points))
        axis.set_major_formatter(FixedFormatter([str(x) for x in self.data_points]))
        if self.minor_points:
            axis.set_minor_locator(
                FixedLocator(self.minor_points)
            )

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return self.data_points[0], self.data_points[-1]
    
    import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
import numpy as np

# 注册自定义刻度
scale.register_scale(CustomScale)

# ================= 配置区域 =================
STYLE_CONFIG = {
    'Invasion': '#54B345',
    'Chemical control': '#8E8BFE',
    'Mowing 1st': '#C82423',
    'Waterlogging': '#547BB4',
    'Waterlogging 2': '#547BB4',
    'Mowing 2nd': '#BB9727',
    'Recurring': '#6F6F6F',
    'Bio Sub': "#FF00F7",
}

SCALES_CONFIG = {
    'SD': ([0.0, 0.25, 0.5, 2.5, 4.5], 0.5),
    'JS': ([0.0, 0.5, 1.0, 3.0, 5.0], 1.0),
    'ZJ': ([0.0, 0.5, 1.0, 3.0, 5.0], 1.0),
    'SH': ([0.0, 0.25, 0.5, 5.0, 9.5], 0.5),
    'FJ': ([0.0, 0.1, 0.2, 0.3, 2.5, 4.7], 0.3),
    'GX': ([0, 0.02, 0.04, 0.06, 0.08, 1], -66)
}

active_provinces = ['SD', 'JS', 'SH', 'ZJ', 'FJ', 'GX']

# ================= 辅助函数 ================= 

def get_minor_ticks(points, break_point):
    """生成非线性轴的次级刻度"""
    minor_points = []
    for a, b in zip(points[:-1], points[1:]):
        if a >= break_point:
            delta = (b - a) / 5
            minor_points.extend([a + i * delta for i in range(5)])
        else:
            delta = (b - a) / 3
            minor_points.extend([a + i * delta for i in range(3)])
    return minor_points

def draw_axis_break(ax, data_points, break_val):
    """绘制断轴的视觉效果"""
    try:
        break_idx = data_points.index(break_val)
        display_points = np.linspace(0, 1, len(data_points))
        rel_pos = display_points[break_idx]
    except ValueError:
        return

    y_center = rel_pos
    y_span = 0.02 
    
    d = .01 
    # 斜线绘制在极高的 zorder 层级，确保盖住 Grid 和 Bar
    kwargs = dict(transform=ax.transAxes, color='black', clip_on=False, zorder=100, linewidth=1)
    
    # 绘制斜线
    ax.plot((-d, +d), (y_center - y_span + 0.05, y_center + y_span + 0.05), **kwargs) 
    ax.plot((-d, +d), (y_center - y_span, y_center + y_span), 
            transform=ax.transAxes, color='white', linewidth=5, alpha=1, clip_on=False, zorder=99)
    ax.plot((-d, +d), (y_center - y_span - 0.05, y_center + y_span - 0.05), **kwargs)

# ================= 主绘图逻辑 =================

n_plots = len(active_provinces)
fig, axs = plt.subplots(nrows=n_plots, ncols=1, 
                        figsize=(5, 1.5 * n_plots), 
                        sharex=True, 
                        dpi=300)

if n_plots == 1: axs = [axs]

legend_handles = {}

for ax, province in zip(axs, active_provinces):
    df = pd.read_csv('data.csv', parse_dates=['date'], encoding='utf-8-sig')
    df = df[df['province'] == province].copy()
    
    # 数据清洗
    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = df[num_cols].where(df[num_cols] >= 5, 0)
    
    ax.set_axisbelow(True)
    
    # 2. 设置 Y 轴
    points, break_val = SCALES_CONFIG[province]
    ax.set_yscale('custom', data_points=points, minor_points=get_minor_ticks(points, break_val))
    
    draw_axis_break(ax, points, break_val)

    # 3. 绘制右轴 (折线图)
    ax2 = ax.twinx()
    for spine in ax2.spines.values(): spine.set_visible(False)
    
    rate_data = df['removal_rate']
        
    line_plot, = ax2.plot(df['date'], rate_data,
             label='Removal rate',
             marker='D', markersize=1, 
             # 设置较高的 zorder 确保折线在柱子上方
             linestyle='-', linewidth=1, color='black', alpha=0.6, zorder=20)
    ax2.set_ylim(-10, 110)
    ax2.set_ylabel('Control rate (%)', fontweight='bold', fontsize=8)
    ax2.tick_params(axis='y', labelsize=8)
    
    if 'Rate' not in legend_handles:
        legend_handles['Rate'] = line_plot

    # 4. 绘制左轴 (堆叠柱状图)
    bar_bottom = np.zeros(len(df))
    
    for col, color in STYLE_CONFIG.items():
        if col not in df.columns: continue
        
        values = df[col].values / 1000.0
        
        bars = ax.bar(df['date'], values, bottom=bar_bottom,
               color=color, label=col, width=25,
               edgecolor='none', alpha=0.9, zorder=10)
        
        bar_bottom += values
        
        if col not in legend_handles:
            legend_handles[col] = bars[0]

    # 5. 轴样式设置
    ax.set_xlim(pd.Timestamp('2019-12-15'), pd.Timestamp('2024-12-15'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_ylabel(f'Area (× 10³ ha)', fontweight='bold', fontsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    
    ax.text(0.01, 0.9, province, transform=ax.transAxes, 
            fontsize=12, fontweight='bold', va='top')

    ax.tick_params(axis='x', which='both', labelbottom=True, labelsize=9)
    
    # 设置主刻度为年
    ax.xaxis.set_major_locator(mdates.YearLocator())
    # 设置主刻度格式仅显示年份
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # 设置次刻度为月
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

# 7. 全局图例
keys = list(STYLE_CONFIG.keys()) + ['Rate']
final_handles = [legend_handles[k] for k in keys if k in legend_handles]
final_labels = [k for k in keys if k in legend_handles]


plt.subplots_adjust(top=0.92, hspace=0.3)
plt.savefig('fig 9.jpg', dpi=300)
# plt.show()