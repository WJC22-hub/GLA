import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import make_interp_spline

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# --- 1. 准备数据 ---

# 定义所有数据系列（使用真实数据点）
data_series = {
    'GLA': {
        'y_level': 40, 'color': '#d62728', 'marker': 'o',
        'marker_x': [2.5, 5, 10],
        'marker_z': [75.00, 85.00, 100.00]
    },
    'Blended': {
        'y_level': 30, 'color': '#1f77b4', 'marker': 's',
        'marker_x': [2.5, 5, 10],
        'marker_z': [60.00, 90.00, 95.00]
    },
    'ISSBA': {
        'y_level': 20, 'color': '#2ca02c', 'marker': 'd',
        'marker_x': [2.5, 5, 10],
        'marker_z': [35.00, 50.00, 75.00]
    },
    'BadNets': {
        'y_level': 10, 'color': '#ff7f0e', 'marker': '^',
        'marker_x': [2.5, 5, 10],
        'marker_z': [20.00, 25.00, 40.00]
    }
}

# --- 2. 开始绘图 ---

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# 遍历每个系列
for series_name, d in data_series.items():
    y = d['y_level']
    color = d['color']
    
    # 使用样条插值生成平滑曲线
    marker_x = np.array(d['marker_x'])
    marker_z = np.array(d['marker_z'])
    
    # 创建B样条插值（三次样条）
    # k=2 表示二次样条，更平滑
    spline = make_interp_spline(marker_x, marker_z, k=2)
    
    # 生成平滑的曲线点（更多点使曲线更平滑）
    x_smooth = np.linspace(marker_x.min(), marker_x.max(), 600)
    z_smooth = spline(x_smooth)
    
    # 确保曲线在合理范围内
    z_smooth = np.clip(z_smooth, 0, 105)

    # 1. 绘制平滑曲线
    ax.plot(x_smooth, np.full_like(x_smooth, y), z_smooth, 
            color=color, linewidth=2, alpha=0.9, zorder=10)

    # 2. 绘制填充区域
    verts = [(x_smooth[0], y, 0)] + \
            list(zip(x_smooth, np.full_like(x_smooth, y), z_smooth)) + \
            [(x_smooth[-1], y, 0)]
    poly = Poly3DCollection([verts], facecolors=color, alpha=0.3, 
                           edgecolors=color, linewidths=0.5)
    ax.add_collection3d(poly)

    # 3. 绘制数据标记点
    ax.scatter(d['marker_x'], np.full_like(d['marker_x'], y), d['marker_z'], 
               marker=d['marker'], color=color, s=120, edgecolors='black', 
               linewidths=1.5, depthshade=False, zorder=15)
    
    # 4. 在左侧添加系列标签
    label_offset = 6 if series_name == 'GLA' else 4
    ax.text(1.5, y, marker_z[0] + label_offset, series_name, 
            color=color, fontsize=11, fontweight='bold',
            ha='right', va='center')

# --- 3. 调整坐标轴和视图 ---

# 设置X轴 (Poison Rate) - 使用斜体
ax.set_xlabel('Poison Rate(%)', fontsize=12, labelpad=12, 
              fontstyle='italic', fontweight='bold')
ax.set_xlim(2, 10.5)
ax.set_xticks([2.5, 5,7.5,10])
ax.set_xticklabels(['2.5', '5','7.5', '10'], fontsize=12)

# 设置Z轴 (ASR) - 使用斜体
ax.set_zlabel('ASR (%)', fontsize=12, labelpad=12, 
              fontstyle='italic', fontweight='bold')
ax.set_zlim(0, 100)
ax.set_zticks(np.arange(0, 101, 20))
# ax.set_zticklabels([f'{z}' for z in np.arange(0, 101, 20)], fontsize=12)

# 隐藏Y轴
ax.set_yticks([])
ax.set_ylim(5, 45)

# 在右侧添加ASR标注
ax.text(11.2, 38, 110, "ASR(%)", color='black', fontsize=12, 
        ha='left', fontstyle='italic', fontweight='bold')
# for z in [0, 20, 40, 60, 80, 100]:
#     ax.text(11.2, 40, z, f"{z}", color='gray', fontsize=11, 
#             ha='left', va='center')

# 设置3D视图角度
# ax.view_init(elev=22, azim=-58)
ax.view_init(elev=10, azim=-50)

# 美化设置
# ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
# ax.xaxis.pane.set_edgecolor('lightgray')
# ax.yaxis.pane.set_edgecolor('lightgray')
# ax.zaxis.pane.set_edgecolor('lightgray')

# # 添加轻微的网格线
# ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.6, color='gray')
ax.grid(False)
# # 标题
# ax.set_title('Attack Success Rate vs Poison Rate', 
#              fontsize=17, fontweight='bold', pad=30)

# 调整布局
# plt.tight_layout()

# 保存图像
plt.savefig('improved_3d_ridge_plot.png', dpi=600, bbox_inches='tight', 
            facecolor='white')
print("✓ 可视化已保存为 'improved_3d_ridge_plot.png'")
print(f"✓ 数据系列: {list(data_series.keys())}")
print("✓ 使用平滑样条曲线插值")
print("✓ 所有坐标轴标签已使用斜体")

plt.show()