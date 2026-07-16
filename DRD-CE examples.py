# ============================================================
# 三国预测效果图（PPT答辩高级版）
# ============================================================

import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'PingFang SC', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

print("=" * 60)
print("生成三国预测效果图（高级版）")
print("=" * 60)

# ============================================================
# 1. 三国预测数据
# ============================================================
data = {
    '美国': {
        'sport': '女子体操',
        'current': 2,
        'predicted': 4,
        'improvement': 2,
        'reason': '历史上有Béla Károlyi成功先例，反弹空间大',
        'color': '#6B9F8A',  # 高级莫兰迪绿
        'light_color': '#D4E8DE',
        'years': [2016, 2020, 2024, 2028],
        'history': [1, 3, 2, 4],
        'coach_year': 1984
    },
    '日本': {
        'sport': '男子佩剑',
        'current': 1,
        'predicted': 3,
        'improvement': 2,
        'reason': '2020年换帅已获金牌突破，继续深化',
        'color': '#C49A8A',  # 高级莫兰迪粉
        'light_color': '#F0DFD8',
        'years': [2016, 2020, 2024, 2028],
        'history': [0, 0, 1, 3],
        'coach_year': 2020
    },
    '德国': {
        'sport': '女子曲棍球',
        'current': 0,
        'predicted': 2,
        'improvement': 2,
        'reason': '传统强项，近三届无牌，急需重振',
        'color': '#7B9BB5',  # 高级莫兰迪蓝
        'light_color': '#D5E3ED',
        'years': [2016, 2020, 2024, 2028],
        'history': [1, 0, 0, 2],
        'coach_year': 2016
    }
}

countries = ['美国', '日本', '德国']

# ============================================================
# 图1：柱状对比图（浅色柔和版）
# ============================================================
print("\n生成图1：柱状对比图...")

fig, axes = plt.subplots(1, 3, figsize=(9, 4.5))
fig.patch.set_facecolor('#F8F6F4')

for idx, (country, ax) in enumerate(zip(countries, axes)):
    d = data[country]
    current = d['current']
    predicted = d['predicted']
    color = d['color']
    light_color = d['light_color']

    # 柔和的柱子
    bars = ax.bar(['当前', '预测'], [current, predicted],
                  color=[light_color, color],
                  width=0.5, edgecolor='white', linewidth=2)

    # 数值标签
    for bar, val in zip(bars, [current, predicted]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.25,
                f'{int(val)}枚', ha='center', va='bottom',
                fontsize=16, fontweight='normal', color='#2C3E50')

    # 提升箭头（更精致）
    if predicted > current:
        ax.annotate('', xy=(0.5, predicted + 0.6), xytext=(0.5, current + 0.6),
                    arrowprops=dict(arrowstyle='->', color='#E67E22', lw=2.5))
        ax.text(0.5, (current + predicted) / 2 + 0.5,
                f'+{d["improvement"]}枚', ha='center', va='center',
                fontsize=14, fontweight='normal', color='#E67E22')

    # 原因说明（更清爽的框）
    ax.text(0.5, -0.28, d['reason'],
            transform=ax.transAxes, fontsize=9, ha='center',
            linespacing=1.3,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='#E8E0D8', linewidth=1))

    ax.set_ylim(0, max(current, predicted) + 1.8)
    ax.set_title(f'{country}\n{d["sport"]}', fontsize=15, fontweight='normal', pad=15)
    ax.set_ylabel('奖牌数', fontsize=11)
    ax.set_facecolor('#FCFAF8')
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('三国名帅引进推荐方案与预测', fontsize=18, fontweight='normal', y=0.96, color='#2C3E50')
plt.tight_layout()
plt.savefig('三国预测柱状图_高级版.png', dpi=200, bbox_inches='tight', facecolor='#F8F6F4')
plt.show()
print("  ✅ 已保存: 三国预测柱状图_高级版.png")

# ============================================================
# 图2：趋势图（优雅线条版）
# ============================================================
print("\n生成图2：趋势图...")

fig, axes = plt.subplots(1, 3, figsize=(9, 4.5))
fig.patch.set_facecolor('#F8F6F4')

for idx, (country, ax) in enumerate(zip(countries, axes)):
    d = data[country]
    years = d['years']
    history = d['history']
    color = d['color']
    light_color = d['light_color']

    # 绘制历史趋势（半透明区域 + 实线）
    ax.fill_between(years[:-1], 0, history[:-1], alpha=0.15, color=color)
    ax.plot(years[:-1], history[:-1], 'o-', color=color,
            linewidth=2.5, markersize=10, markerfacecolor='white',
            markeredgewidth=2.5, label='历史')

    # 预测点（大一点，醒目但柔和）
    ax.plot(years[-1:], history[-1:], 's-', color=color,
            linewidth=2.5, markersize=14, markerfacecolor=color,
            markeredgewidth=2, label='预测')

    # 数值标签（柔和）
    for y, val in zip(years, history):
        offset = 15 if val > 0 else -20
        va = 'bottom' if val > 0 else 'top'
        ax.annotate(f'{val}枚', xy=(y, val),
                    xytext=(0, offset), textcoords="offset points",
                    ha='center', va=va, fontsize=11, color='#4A4A4A')

    # 换帅标记（更精致）
    ax.axvline(x=d['coach_year'], color='#E67E22', linestyle='--',
               linewidth=1.5, alpha=0.6)
    ax.text(d['coach_year'], ax.get_ylim()[1] * 0.88, '换帅',
            ha='center', fontsize=9, color='#E67E22',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    # 提升标注
    improvement_text = f'↑ +{d["improvement"]}枚'
    ax.annotate(improvement_text, xy=(2028, history[-1]),
                xytext=(2026, history[-1] + 0.8),
                ha='center', fontsize=13, fontweight='normal',
                color='#E67E22',
                arrowprops=dict(arrowstyle='->', color='#E67E22', lw=1.5, alpha=0.7))

    ax.set_xlabel('年份', fontsize=11)
    ax.set_ylabel('奖牌数', fontsize=11)
    ax.set_title(f'{country} · {d["sport"]}', fontsize=15, fontweight='normal')
    ax.legend(loc='upper left', fontsize=10, frameon=True, facecolor='white', edgecolor='none')
    ax.grid(True, alpha=0.15, linestyle='--')
    ax.set_xticks(years)
    ax.set_ylim(0, max(history) + 1.8)
    ax.set_facecolor('#FCFAF8')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('三国名帅引进：历史趋势与2028预测', fontsize=18, fontweight='normal', y=0.96, color='#2C3E50')
plt.tight_layout()
plt.savefig('三国预测趋势图_高级版.png', dpi=200, bbox_inches='tight', facecolor='#F8F6F4')
plt.show()
print("  ✅ 已保存: 三国预测趋势图_高级版.png")

# ============================================================
# 图3：汇总对比（横向条形图，更清爽）
# ============================================================
print("\n生成图3：汇总对比...")

fig, ax = plt.subplots(figsize=(10, 5.5))
fig.patch.set_facecolor('#F8F6F4')

y_pos = np.arange(len(countries))
bar_height = 0.3

# 绘制横向条形图
for i, country in enumerate(countries):
    d = data[country]
    current = d['current']
    predicted = d['predicted']

    # 当前值（浅色）
    ax.barh(i - bar_height / 2, current, height=bar_height,
            color=d['light_color'], edgecolor='white', linewidth=1.5,
            label='当前' if i == 0 else '')
    # 预测值（主色）
    ax.barh(i + bar_height / 2, predicted, height=bar_height,
            color=d['color'], edgecolor='white', linewidth=1.5,
            label='预测' if i == 0 else '')

    # 数值标注
    ax.text(current + 0.15, i - bar_height / 2, f'{int(current)}枚',
            va='center', fontsize=9, color='#7F8C8D')
    ax.text(predicted + 0.15, i + bar_height / 2, f'{int(predicted)}枚',
            va='center', fontsize=9, fontweight='normal', color='#2C3E50')

    # 提升箭头
    if predicted > current:
        mid_x = (current + predicted) / 2
        ax.annotate(f'↑+{d["improvement"]}枚', xy=(mid_x, i),
                    ha='center', va='center', fontsize=12,
                    fontweight='normal', color='#E67E22')

# 设置y轴标签
ax.set_yticks(y_pos)
ax.set_yticklabels([f'{c}\n({data[c]["sport"]})' for c in countries], fontsize=12)
ax.set_xlabel('奖牌数', fontsize=12)
ax.set_title('三国名帅引进：当前 vs 2028预测', fontsize=16, fontweight='normal', pad=15)
ax.legend(loc='lower right', fontsize=11, frameon=True, facecolor='white', edgecolor='none')
ax.set_xlim(0, 5)
ax.grid(axis='x', alpha=0.15, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_facecolor('#FCFAF8')

plt.tight_layout()
plt.savefig('三国预测汇总对比_高级版.png', dpi=200, bbox_inches='tight', facecolor='#F8F6F4')
plt.show()
print("  ✅ 已保存: 三国预测汇总对比_高级版.png")

# ============================================================
# 图4：雷达图（展示多维度提升）
# ============================================================
print("\n生成图4：多维度雷达图...")

fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(projection='polar'))
fig.patch.set_facecolor('#F8F6F4')

categories = ['当前水平', '提升空间', '历史底蕴', '成功概率', '实施难度']
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# 三国数据（5分制）
scores = {
    '美国': [3, 4, 5, 4, 3],
    '日本': [2, 5, 2, 5, 4],
    '德国': [1, 5, 4, 3, 4]
}

colors = ['#6B9F8A', '#C49A8A', '#7B9BB5']
light_colors = ['#D4E8DE', '#F0DFD8', '#D5E3ED']

for i, (country, color) in enumerate(zip(countries, colors)):
    values = scores[country]
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2.5, color=color, label=country)
    ax.fill(angles, values, alpha=0.1, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 5.5)
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_yticklabels(['1', '2', '3', '4', '5'], fontsize=9)
ax.grid(True, alpha=0.2)
ax.legend(loc='upper right', fontsize=11, frameon=True, facecolor='white', edgecolor='none')

plt.title('三国名帅引进：多维度评估', fontsize=16, fontweight='normal', pad=20, color='#2C3E50')
plt.tight_layout()
plt.savefig('三国预测雷达图_高级版.png', dpi=200, bbox_inches='tight', facecolor='#F8F6F4')
plt.show()
print("  ✅ 已保存: 三国预测雷达图_高级版.png")

# ============================================================
# 打印总结
# ============================================================
print("\n" + "=" * 60)
print("✅ 三国预测效果图生成完成！")
print("=" * 60)

for country in countries:
    d = data[country]
    print(f"  {country} | {d['sport']}: {d['current']}枚 → {d['predicted']}枚 (+{d['improvement']}枚)")
