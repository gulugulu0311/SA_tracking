import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = ['Arial']
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.unicode_minus'] = False

pure_mowing_recur = {   # 纯刈割面积占比
    'SD': 33.47,
    'JS': 8.01,
    'SH': 11.66,
    'ZJ': 13,
    'FJ': 3.83,
}
rer_stat = {'SD': [np.float64(0.3128), np.float64(3064.2188)], 
            'JS': [np.float64(0.0509), np.float64(1344.746)], 
            'SH': [np.float64(0.0225), np.float64(409.945)], 
            'ZJ': [np.float64(0.0951), np.float64(2256.7725)], 
            'FJ': [np.float64(0.0362), np.float64(339.8864)]}

for key, value in rer_stat.items():
    print(f"{key}: Recurrence rate = {value[0]*100:.2f}%, Total recurrence area = {value[1]:.2f} ha")

mowing_info = {'SD': [np.float64(6090.5742), np.float64(0.6217765487944557)], # 面积 ha； 占比 %
               'JS': [np.float64(15433.54978), np.float64(0.5850275317967684)], 
               'SH': [np.float64(3105.4234), np.float64(0.1706850626336556)], 
               'ZJ': [np.float64(15097.9221), np.float64(0.6367478878116298)], 
               'FJ': [np.float64(8545.7663), np.float64(0.911484839595933)]}

recur_info =  {'SD': [np.float64(3064.2188), np.float64(0.3128209800342124)], 
               'JS': [np.float64(1344.746), np.float64(0.0509742375194576)], 
               'SH': [np.float64(409.9452), np.float64(0.022532039314950254)], 
               'ZJ': [np.float64(2256.7725), np.float64(0.09517833733202075)], 
               'FJ': [np.float64(339.8864), np.float64(0.036252021165713265)]}

print('recur account for pure mowing:')
for province in ['SD', 'JS', 'SH', 'ZJ', 'FJ']:
    mowing_area = mowing_info[province][0]
    recur_area = recur_info[province][0]
    ratio = (recur_area / (mowing_area + recur_area)) * 100 if mowing_area > 0 else 0
    print(f"{province}: Recurrence area = {recur_area:.2f} ha, Mowing area = {mowing_area:.2f} ha, Ratio = {ratio:.2f}%")
    

# recur_after_mow数据 （中位数 最小值 最大值）
recur_after_mow = {
    'SD': [6, 6, 7],
    'JS': [5.199824181, 3, 7],
    'SH': [5, 2, 8],
    'ZJ': [8, 4, 9],
    'FJ': [12, 6, 15],
}

stats = []
provinces = list(recur_after_mow.keys())
for i, province in enumerate(provinces):
    median, q1, q3 = recur_after_mow[province]
    iqr = q3 - q1
    stats.append({
        'med': median, 'q1': q1, 'q3': q3,
        'whislo': max(0, q1 - 1.5 * iqr),
        'whishi': q3 + 1.5 * iqr,
        'fliers': [], 'label': province
    })

line_data = [rer_stat[p][0] * 100 for p in provinces] # 折线图
line_data_2 = [pure_mowing_recur[p] for p in provinces]
bar_data = [rer_stat[p][1] / 1000 for p in provinces]  # 柱状图
print('line_data_2', line_data_2, line_data)

_fig, _ax_ = plt.subplots(figsize=(6, 3), dpi=300)

# === 柱状图 (右轴 Ax2) ===
ax2 = _ax_.twinx()
bar_color = "#F5B623"
bars = ax2.bar(range(len(stats)), bar_data, color=bar_color, alpha=0.7, label='Bar (Index 1)')
ax2.set_ylabel('Recurrence area (×10³ ha)', color=bar_color, fontweight='bold')
ax2.tick_params(axis='y', colors=bar_color)
ax2.set_ylim(0, 5)
# === 箱型图 (主轴 _ax_) ===
_ax_.set_zorder(10)
_ax_.patch.set_visible(False)

boxplot_dict = _ax_.bxp(stats, positions=range(len(stats)), showfliers=True, patch_artist=True, zorder=10)

# 箱型图样式
box_color = "#1B69DE"
for box in boxplot_dict['boxes']:
    box.set_facecolor(box_color)
    box.set_edgecolor('black')
    box.set_linewidth(2)
for cap in boxplot_dict['caps']:
    cap.set_linewidth(2)
for whisker in boxplot_dict['whiskers']:
    whisker.set_linewidth(2)
for median in boxplot_dict['medians']:
    median.set_linewidth(2)
    
for i in range(len(provinces)):
    boxplot_dict['whiskers'][i*2].set_visible(False)
    boxplot_dict['caps'][i*2].set_visible(False)

_ax_.set_xticks(range(len(stats)))
_ax_.set_xticklabels(provinces)
_ax_.set_ylabel('Months to recurrence', color=box_color, fontweight='bold')
_ax_.tick_params(axis='y', colors=box_color)

# 折线图
ax3 = _ax_.twinx()
ax3.spines["right"].set_position(("axes", 1.15))
ax3.spines["right"].set_visible(True)
ax3.set_zorder(99) # 最上层
line_color = "#100404"
line_plot, = ax3.plot(range(len(stats)), line_data, color=line_color, marker='o', 
                     linewidth=2, markersize=6, label='Line (Index 0)', zorder=99)

line_plot_2, = ax3.plot(range(len(stats)), line_data_2, color=line_color, marker='^', 
                     linewidth=2, markersize=6, label='Line (Index 99)', zorder=99)

ax3.set_ylabel('Recurrence rate (%)', color=line_color, fontweight='bold')
ax3.tick_params(axis='y', colors=line_color)
ax3.spines["right"].set_color(line_color)
ax3.set_ylim(0, 60)

# --- 图例 ---
legends_handles = [boxplot_dict['boxes'][0], bars, line_plot, line_plot_2]
legends_labels = ['Months to recurrence', 'Recurrence area', 'Recurrence rate', 'Recurrence rate (mowing area)']

# 图例左上角
_ax_.legend(legends_handles, legends_labels, loc='upper left', frameon=True, framealpha=0.9)

_ax_.grid(True, alpha=0.3, axis='y', linestyle='--')
plt.tight_layout()
plt.savefig('output.jpg', dpi=300)
plt.show()