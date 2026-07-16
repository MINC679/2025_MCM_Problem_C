import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
import os
import glob
import matplotlib.pyplot as plt

np.random.seed(42)


# ==================== 自动查找文件 ====================
def find_file(keywords):
    all_files = glob.glob('*')
    for f in all_files:
        if all(kw.lower() in f.lower() for kw in keywords):
            return f
    return None


print("当前目录文件:", os.listdir('.'))

panel_file = find_file(['panel_data_for_model'])
medal_file = find_file(['medal_counts']) or find_file(['medal', 'count'])

print(f"找到 panel 文件: {panel_file}")
print(f"找到 medal 文件: {medal_file}")

if panel_file is None or medal_file is None:
    print("找不到文件，请检查文件名！")
    exit()

# ==================== 1. 读数据 ====================
if panel_file.endswith('.csv'):
    df = pd.read_csv(panel_file)
else:
    df = pd.read_excel(panel_file)

for col in ['Gold', 'Silver', 'Bronze', 'Total_Medals']:
    if col not in df.columns:
        raise ValueError(f"数据中缺少 {col} 列")

df['Total_Medals'] = pd.to_numeric(df['Total_Medals'], errors='coerce')
df = df.dropna(subset=['Total_Medals', 'Is_Host', 'Num_Athletes'])
df = df[df['Year'] >= 1992].copy()

# ==================== 2. 构造特征 ====================
df['HE'] = df['Is_Host']

total_per_year = df.groupby('Year')['Num_Athletes'].sum().reset_index()
total_per_year.columns = ['Year', 'Total_Athletes']
df = df.merge(total_per_year, on='Year', how='left')
df['ER'] = df['Num_Athletes'] / df['Total_Athletes']
df['ER'] = df['ER'].fillna(0)

df = df.sort_values(['Country', 'Year'])
df['AP'] = df.groupby('Country')['Total_Medals'].shift(1).fillna(0)
df['NA'] = df['Num_Athletes']

country_means = df.groupby('Country')[['ER', 'AP']].mean().reset_index()
country_means.columns = ['Country', 'ER_mean', 'AP_mean']
df = df.merge(country_means, on='Country', how='left')
df = df.dropna()

feature_cols = ['HE', 'ER', 'AP', 'NA', 'ER_mean', 'AP_mean']

# ==================== 3. 训练：分别训练金/银/铜 ====================
train = df[df['Year'] <= 2024].copy()
print(f"\n训练数据年份范围: {train['Year'].min()} - {train['Year'].max()}")
print(f"训练数据行数: {len(train)}")

medal_types = ['Gold', 'Silver', 'Bronze']
logit_models = {}
lr_models = {}

for mt in medal_types:
    print(f"  训练 {mt} 模型...")
    logit = LogisticRegression(max_iter=1000, random_state=42)
    logit.fit(train[feature_cols], (train[mt] > 0).astype(int))
    logit_models[mt] = logit

    train_medal = train[train[mt] > 0]
    if len(train_medal) > 10:
        lr = LinearRegression()
        lr.fit(train_medal[feature_cols], train_medal[mt])
        lr_models[mt] = lr
    else:
        lr_models[mt] = None

# ==================== 4. 预测2028 ====================
print("\n构造2028年预测数据...")

df_2024 = df[df['Year'] == 2024].copy()
df_2028 = df_2024.copy()
df_2028['Year'] = 2028

df_2028['HE'] = 0
df_2028.loc[df_2028['Country'] == 'United States', 'HE'] = 1

df_2028['NA'] = df_2028['NA'].astype(float)
df_2028['ER'] = df_2028['ER'].astype(float)

df_2028.loc[df_2028['Country'] == 'United States', 'NA'] *= 1.05
df_2028.loc[df_2028['Country'] == 'United States', 'ER'] *= 1.08

X_2028 = df_2028[feature_cols]

for mt in medal_types:
    prob = logit_models[mt].predict_proba(X_2028)[:, 1]
    pred_has = (prob > 0.5).astype(int)

    if lr_models[mt] is not None:
        pred_count = lr_models[mt].predict(X_2028)
        pred_count = np.maximum(pred_count, 0)
    else:
        pred_count = np.zeros(len(df_2028))

    df_2028[f'{mt}_Pred'] = np.round(np.where(pred_has == 1, pred_count, 0), 0)

df_2028['Total_Pred'] = df_2028['Gold_Pred'] + df_2028['Silver_Pred'] + df_2028['Bronze_Pred']

# ==================== 5. 输出2028年预测结果 ====================
print("\n" + "=" * 50)
print("2028年洛杉矶奥运会奖牌榜预测（金/银/铜/总）")
print("=" * 50)

result_2028 = df_2028[['Country', 'Gold_Pred', 'Silver_Pred', 'Bronze_Pred', 'Total_Pred']].sort_values('Total_Pred',
                                                                                                        ascending=False).reset_index(
    drop=True)
result_2028['Rank'] = result_2028.index + 1

print("\n前20名预测：")
print(result_2028.head(20)[['Rank', 'Country', 'Gold_Pred', 'Silver_Pred', 'Bronze_Pred', 'Total_Pred']].to_string(
    index=False))

result_2028.to_csv('2028预测结果_金银铜.csv', index=False)
print("\n已保存: 2028预测结果_金银铜.csv")

# ==================== 6. 变化量分析 ====================
print("\n" + "=" * 50)
print("2028年奖牌数变化量分析（相比2024年实际）")
print("=" * 50)

if medal_file.endswith('.csv'):
    actual = pd.read_csv(medal_file)
else:
    actual = pd.read_excel(medal_file)

actual_2024 = actual[actual['Year'] == 2024][['NOC', 'Total']].copy()
actual_2024.columns = ['Country', 'Actual_2024']

change_df = df_2028[['Country', 'Total_Pred']].merge(actual_2024, on='Country', how='left')
change_df['Actual_2024'] = change_df['Actual_2024'].fillna(0)
change_df['Change'] = change_df['Total_Pred'] - change_df['Actual_2024']

gain_top = change_df.nlargest(15, 'Change').sort_values('Change', ascending=True)
loss_top = change_df.nsmallest(15, 'Change').sort_values('Change', ascending=False)

print("\n增多最多的15个国家：")
print(gain_top[['Country', 'Actual_2024', 'Total_Pred', 'Change']].to_string(index=False))
print("\n减少最多的15个国家：")
print(loss_top[['Country', 'Actual_2024', 'Total_Pred', 'Change']].to_string(index=False))

# ==================== 7. 绘图 ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# ---- 图1：分组柱状图（柔和配色） ----
fig1, ax1 = plt.subplots(figsize=(16, 8))
top15 = result_2028.head(15)
x = np.arange(len(top15))
width = 0.2

bars1 = ax1.bar(x - 1.5 * width, top15['Gold_Pred'], width, label='金牌',
                color='#F5D742', edgecolor='#D4A800', linewidth=0.8)
bars2 = ax1.bar(x - 0.5 * width, top15['Silver_Pred'], width, label='银牌',
                color='#D5D5D5', edgecolor='#A0A0A0', linewidth=0.8)
bars3 = ax1.bar(x + 0.5 * width, top15['Bronze_Pred'], width, label='铜牌',
                color='#E8C9A0', edgecolor='#B8860B', linewidth=0.8)
bars4 = ax1.bar(x + 1.5 * width, top15['Total_Pred'], width, label='总和',
                color='#7BAFDE', edgecolor='#4A7FB5', linewidth=0.8)

ax1.set_xticks(x)
ax1.set_xticklabels(top15['Country'], rotation=45, ha='right', fontsize=10)
ax1.set_ylabel('奖牌数', fontsize=12)
ax1.set_title('2028年洛杉矶奥运会奖牌预测（TOP15：金/银/铜/总）', fontsize=14)
ax1.legend(loc='upper left', fontsize=10)

for bar, val in zip(bars4, top15['Total_Pred']):
    ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
             f'{int(val)}', ha='center', va='bottom', fontsize=9)

ax1.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('2028预测_金银铜分组柱状图.png', dpi=300)
plt.show()
print("\n✅ 图1已保存：2028预测_金银铜分组柱状图.png")

# ---- 图2：变化量（柔和配色） ----
fig2, axes = plt.subplots(1, 2, figsize=(16, 7))

# 增多最多
ax_left = axes[0]
colors_gain = ['#7CCD7C' if x > 0 else '#FFB6B6' for x in gain_top['Change']]
bars_gain = ax_left.barh(gain_top['Country'], gain_top['Change'], color=colors_gain,
                         edgecolor='#555555', linewidth=0.8)
ax_left.axvline(x=0, color='#888888', linestyle='-', linewidth=1.5)
ax_left.set_xlabel('奖牌数变化量（2028预测 - 2024实际）', fontsize=11)
ax_left.set_title('2028年预测奖牌数增多最多的国家（Top15）', fontsize=13)
ax_left.grid(axis='x', alpha=0.3)
for bar, val in zip(bars_gain, gain_top['Change']):
    ax_left.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f'+{int(val)}', va='center', fontsize=9, fontweight='bold')

# 减少最多
ax_right = axes[1]
colors_loss = ['#FFB6B6' if x < 0 else '#7CCD7C' for x in loss_top['Change']]
bars_loss = ax_right.barh(loss_top['Country'], loss_top['Change'], color=colors_loss,
                          edgecolor='#555555', linewidth=0.8)
ax_right.axvline(x=0, color='#888888', linestyle='-', linewidth=1.5)
ax_right.set_xlabel('奖牌数变化量（2028预测 - 2024实际）', fontsize=11)
ax_right.set_title('2028年预测奖牌数减少最多的国家（Top15）', fontsize=13)
ax_right.grid(axis='x', alpha=0.3)
for bar, val in zip(bars_loss, loss_top['Change']):
    ax_right.text(bar.get_width() - 1.5 if val < 0 else 1.5, bar.get_y() + bar.get_height() / 2,
                  f'{int(val)}', va='center', fontsize=9, fontweight='bold',
                  ha='right' if val < 0 else 'left')

plt.tight_layout()
plt.savefig('2028预测_奖牌变化量_增减Top15.png', dpi=300)
plt.show()
print("✅ 图2已保存：2028预测_奖牌变化量_增减Top15.png")

print("\n" + "=" * 50)
print("预测完成！")
print("=" * 50)