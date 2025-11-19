import matplotlib.pyplot as plt
import numpy as np

# 数据
epochs = np.arange(1, 16)
BadNets = [0.00, 20.00, 20.00, 20.00, 20.00, 10.00, 40.00, 40.00, 40.00, 40.00, 20.00, 40.00, 40.00, 40.00, 40.00]
Blended = [0.00, 65.00, 65.00, 65.00, 75.00, 75.00, 80.00, 90.00, 90.00, 90.00, 90.00, 90.00, 90.00, 90.00, 90.00]
ISSBA = [0.00, 45.00, 45.00, 45.00, 50.00, 50.00, 70.00, 80.00, 80.00, 80.00, 70.00, 70.00, 80.00, 80.00, 80.00]
GLA = [0.00, 80.00, 90.00, 85.00, 90.00, 90.00, 90.00, 85.00, 90.00, 95.00, 95.00, 95.00, 95.00, 95.00, 95.00]

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.linewidth'] = 1.0 # 稍微调细轴线

# 优化颜色方案，使用更专业的调色板 (例如 Matplotlib 的 tab10 或自定义颜色)
# 这里我自定义了一组相对柔和且区分度高的颜色
colors = ['#CC79A7', '#0072B2', '#D55E00', '#009E73'] # 紫红，蓝色，橙色，绿

# 绘图
plt.figure(figsize=(9, 5.5), dpi=300) # 调整图表大小以获得更好的视觉平衡

plt.plot(epochs, BadNets, marker='o', linestyle='-', color=colors[0], linewidth=1.8, markersize=6, label='BadNets', markeredgecolor='black', markerfacecolor=colors[0], markeredgewidth=0.5)
plt.plot(epochs, Blended, marker='s', linestyle='-', color=colors[1], linewidth=1.8, markersize=6, label='Blended', markeredgecolor='black', markerfacecolor=colors[1], markeredgewidth=0.5)
plt.plot(epochs, ISSBA, marker='^', linestyle='-', color=colors[2], linewidth=1.8, markersize=6, label='ISSBA', markeredgecolor='black', markerfacecolor=colors[2], markeredgewidth=0.5)
plt.plot(epochs, GLA, marker='D', linestyle='-', color=colors[3], linewidth=1.8, markersize=6, label='GLA', markeredgecolor='black', markerfacecolor=colors[3], markeredgewidth=0.5)

# 标签与标题
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('ASR (%)', fontsize=12)
# 如果需要标题，可以取消注释并调整
# plt.title('Attack Success Rate Across Training Epochs', fontsize=14, pad=10)

# 网格与范围
# plt.grid(True, linestyle=':', linewidth=0.5, alpha=0.6, color='gray') # 更细更淡的网格线
plt.xlim(1, 15)
plt.ylim(0, 100)
plt.xticks(np.arange(1, 16, 4)) # 调整x轴刻度，避免过于密集
plt.yticks(np.arange(0, 101, 20)) # 调整y轴刻度，增加细致度

# 图例
# bbox_to_anchor 允许在图表外部放置图例
# ncol 可以设置图例的列数
# plt.legend(fontsize=11, frameon=False, loc='lower right', bbox_to_anchor=(0.98, 0.05), borderaxespad=0.)
# 图例优化：移动到顶部，横向排列
plt.legend(fontsize=11, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(colors),columnspacing=5.0, borderaxespad=0., handlelength=3.0)
# 调整刻度标签字体大小
plt.tick_params(axis='both', which='major', labelsize=10)

# 布局
plt.tight_layout()

# 保存高分辨率图片
plt.savefig('ASR_training_curve_Base_optimized.png', dpi=600, bbox_inches='tight')

plt.show()