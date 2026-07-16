import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression  # 导入线性回归和逻辑回归模型
from sklearn.metrics import mean_absolute_error  # 导入平均绝对误差（算预测准不准）

np.random.seed(42)

# ==================== 1. 读数据 ====================
df = pd.read_csv('panel_data_for_model.csv')
df['Total_Medals'] = pd.to_numeric(df['Total_Medals'], errors='coerce')
df = df.dropna(subset=['Total_Medals', 'Is_Host', 'Num_Athletes'])
df = df[df['Year'] >= 1992].copy()

# ==================== 2. 构造特征 ====================
df['HE'] = df['Is_Host']

total_per_year = df.groupby('Year')['Num_Athletes'].sum().reset_index()
total_per_year.columns = ['Year', 'Total_Athletes']
df = df.merge(total_per_year, on='Year', how='left')
df['ER'] = df['Num_Athletes'] / df['Total_Athletes']  # 该国参赛人数 ÷ 当年总人数 = 参赛率
df['ER'] = df['ER'].fillna(0)

df = df.sort_values(['Country', 'Year'])
df['AP'] = df.groupby('Country')['Total_Medals'].shift(1).fillna(0)
df['NA'] = df['Num_Athletes']

country_means = df.groupby('Country')[['ER', 'AP']].mean().reset_index()  # 对每个国家，算历史平均参赛率和历史平均表现
country_means.columns = ['Country', 'ER_mean', 'AP_mean']
df = df.merge(country_means, on='Country', how='left')
df = df.dropna()

feature_cols = ['HE', 'ER', 'AP', 'NA', 'ER_mean', 'AP_mean']  # 模型的"输入"

# ==================== 3. 训练（用2020前） ====================
train = df[df['Year'] < 2020]  # 2020年之前的数据作为训练集
logit = LogisticRegression(max_iter=1000, random_state=42)  # 创建逻辑回归模型
logit.fit(train[feature_cols], (train['Total_Medals'] > 0).astype(int))  # 用训练数据教模型"什么特征下能得牌"（

train_medal = train[train['Total_Medals'] > 0]  # 只挑有奖牌的国家出来
lr = LinearRegression()  # 创建线性回归模型
lr.fit(train_medal[feature_cols], train_medal['Total_Medals'])  # 用有奖牌的数据教模型"能得几块牌"

# ==================== 4. 预测2024 ====================
test = df[df['Year'] == 2024].copy()
X = test[feature_cols]

prob = logit.predict_proba(X)[:, 1]  # 用逻辑回归算得牌概率
pred_has = (prob > 0.5).astype(int)  # 概率>50%就算能得牌（1），否则不得牌（0）
pred_count = lr.predict(X)  # 用线性回归预测具体得牌数
pred_count = np.maximum(pred_count, 0)  # 如果预测是负数，强行变成0（奖牌不能为负）
test['Pred_Total'] = np.round(np.where(pred_has == 1, pred_count, 0), 0)  # 如果能得牌，就用回归值；不能得牌就是0

# ==================== 5. 读真实2024数据 ====================
actual = pd.read_csv('summerOly_medal_counts.csv')
actual_2024 = actual[actual['Year'] == 2024][['NOC', 'Total']].copy()
actual_2024.columns = ['Country', 'Actual']

compare = test[['Country', 'Total_Medals', 'Pred_Total']].merge(actual_2024, on='Country', how='left')
compare['Actual'] = compare['Actual'].fillna(0)

# ==================== 6. 计算准确率 ====================
mae = mean_absolute_error(compare['Actual'], compare['Pred_Total'])
compare['Within_3'] = np.abs(compare['Actual'] - compare['Pred_Total']) <= 3  # 判断"误差是否在3块以内"（True/False）
compare['Within_1'] = np.abs(compare['Actual'] - compare['Pred_Total']) <= 1

acc_3 = compare['Within_3'].mean()  # 算比例：有多少国家误差在3块内
acc_1 = compare['Within_1'].mean()

print("="*40)
print("2024年预测准确率")
print("="*40)
print(f"平均绝对误差 (MAE)：{mae:.2f} 块奖牌")
print(f"误差≤3块的比例：{acc_3:.1%} ({compare['Within_3'].sum()}/{len(compare)})")
print(f"误差≤1块的比例：{acc_1:.1%} ({compare['Within_1'].sum()}/{len(compare)})")

# ==================== 7. 主要国家对比表 ====================
top_countries = ['United States', 'China', 'Japan', 'France', 'Australia', 'Great Britain', 'Italy', 'Germany', 'Netherlands', 'South Korea']
top_compare = compare[compare['Country'].isin(top_countries)].sort_values('Actual', ascending=False)

print("\n主要国家预测对比：")
print(top_compare[['Country', 'Actual', 'Pred_Total']].to_string(index=False))

# 一致性（±3块内）
consistent = top_compare['Within_3'].sum()
print(f"\n主要国家中预测在±3块内的：{consistent}/{len(top_compare)}")

# ==================== 8. 画图 ====================
import matplotlib.pyplot as plt
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左：散点图
ax1 = axes[0]
ax1.scatter(compare['Actual'], compare['Pred_Total'], alpha=0.6, color='#2196F3')
max_val = max(compare['Actual'].max(), compare['Pred_Total'].max()) + 5
ax1.plot([0, max_val], [0, max_val], '--', color='red', linewidth=2, label='完美预测')
ax1.fill_between([0, max_val], [0, max_val-3], [0, max_val+3], alpha=0.15, color='green', label='±3块区间')
ax1.set_xlabel('实际奖牌数'); ax1.set_ylabel('预测奖牌数')
ax1.set_title('2024年预测 vs 实际'); ax1.legend(); ax1.grid(alpha=0.3)

# ------ 右图：柱状图 ------
ax2 = axes[1]
top10 = compare.sort_values('Actual', ascending=False).head(10)
x = np.arange(len(top10))

# 画柱子
bars1 = ax2.bar(x - 0.2, top10['Actual'], 0.4, label='实际', color='#4A7FB5')
bars2 = ax2.bar(x + 0.2, top10['Pred_Total'], 0.4, label='预测', color='#D4837A')

# ★★★ 在柱子上加数字 ★★★
for bar in bars1:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='normal')

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='normal')

# 设置x轴标签
ax2.set_xticks(x)
ax2.set_xticklabels(top10['Country'], rotation=45, ha='right', fontsize=9)
ax2.set_ylabel('奖牌数')
ax2.set_title('TOP10国家对比')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('2024预测验证图.png', dpi=300)
plt.show()
print("\n✅ 图已保存：2024预测验证图.png")