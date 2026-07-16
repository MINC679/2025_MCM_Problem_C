import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取运动员数据
athletes = pd.read_csv('summerOly_athletes.csv')
print("=== 数据加载成功 ===")
print(f"共 {len(athletes)} 行数据")

# 2. 只保留有奖牌的数据（去掉 No medal）
medal_data = athletes[athletes['Medal'] != 'No medal'].copy()  # 复制一份，避免影响原文件
print(f"有奖牌的数据：{len(medal_data)} 行")

# 3. 给金银铜打分（Gold=3, Silver=2, Bronze=1）
medal_score = {'Gold': 3, 'Silver': 2, 'Bronze': 1}
medal_data['Score'] = medal_data['Medal'].map(medal_score)

# 4. 计算每个国家在每项运动上的总分
sport_score = medal_data.groupby(['NOC', 'Sport'])['Score'].sum().reset_index()
sport_score.columns = ['Country', 'Sport', 'Sport_Total_Score']
print(f"\n共有 {len(sport_score)} 条 国家-项目 记录")

# 5. 计算每个国家的总得分（所有项目加总）
country_score = medal_data.groupby('NOC')['Score'].sum().reset_index()  # 把分组结果变成一张普通的表
country_score.columns = ['Country', 'Country_Total_Score']
print("\n各国总得分（前10）：")
print(country_score.head(10))

# 6. 合并，计算 SCS = 该项目得分 / 该国总得分
scs_data = sport_score.merge(country_score, on='Country')  # 把两张表按"国家"合并成一张大表
scs_data['SCS'] = scs_data['Sport_Total_Score'] / scs_data['Country_Total_Score']

# 按国家、SCS降序排序
scs_data = scs_data.sort_values(['Country', 'SCS'], ascending=[True, False])
print("\nSCS计算完成！共", len(scs_data), "条记录")
print("前20行预览：")
print(scs_data.head(20))

# 7. 找出每个国家最依赖的项目（Top 3）
top3_scs = scs_data.groupby('Country').head(3).reset_index(drop=True)

# 保存为CSV
top3_scs.to_csv('每个国家最依赖的3个项目.csv', index=False)
print("\n✅ 已保存到：每个国家最依赖的3个项目.csv")

# 8. 国家分类（论文图11）
# 计算每个国家的最大SCS
max_scs = scs_data.groupby('Country')['SCS'].max().reset_index()
max_scs.columns = ['Country', 'Max_SCS']

def classify_country(scs_value):
    if scs_value == 0 or pd.isna(scs_value):
        return '体育新兴国'
    elif scs_value <= 0.25:
        return '体育强国'
    elif scs_value <= 0.75:
        return '体育中等国'
    else:
        return '体育薄弱国'

max_scs['类别'] = max_scs['Max_SCS'].apply(classify_country)
category_counts = max_scs['类别'].value_counts()  # 计算每类各有多少个国家
print("\n" + "="*50)
print("各类别国家数量：")
print(category_counts)

# 9. 画图（论文图11 饼图）
colors = ['#8DA0B3', '#B8B5A8', '#C4A882', '#A8B5A0']
explode = (0.03, 0.03, 0.03, 0.03)

plt.figure(figsize=(10, 8))
plt.pie(
    category_counts.values,  # 数据（各类国家数量）
    labels=category_counts.index,  # 标签（各类名称）
    autopct='%1.1f%%',  # 显示百分比，保留1位小数
    colors=colors[:len(category_counts)],
    explode=explode[:len(category_counts)],
    startangle=90,  # 从12点方向开始画
    textprops={'fontsize': 14, 'fontweight': 'normal'}
)
plt.title('各国体育实力分类', fontsize=18, fontweight='normal')
plt.tight_layout()  # 自动调整布局
plt.savefig('国家分类图_饼图.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ 饼图已保存为：国家分类图_饼图.png")

# 10. 画柱状图（论文图11 另一种展示）
plt.figure(figsize=(10, 6))
bars = plt.bar(
    category_counts.index,
    category_counts.values,
    color=colors[:len(category_counts)],
    edgecolor='white',
    linewidth=2
)
plt.title('各国体育实力分类', fontsize=18, fontweight='normal')
plt.xlabel('国家类别', fontsize=14)
plt.ylabel('国家数量', fontsize=14)

# 在柱子上标数字
for bar, count in zip(bars, category_counts.values):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 1,
        str(count),
        ha='center',
        va='bottom',
        fontsize=14,
        fontweight='normal'
    )
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('国家分类图_柱状图.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ 柱状图已保存为：国家分类图_柱状图.png")
print("✅ SCS分析全部完成！")

# 11. 输出典型案例（用于PPT）
print("\n" + "="*50)
print("【典型案例 - 用于PPT】")
print("="*50)

# 体育强国（SCS <= 0.25）
strong = max_scs[max_scs['类别'] == '体育强国'].head(5)
print("\n🏆 体育强国（多项目均衡发展）：")
print(strong['Country'].tolist())

# 体育薄弱国（SCS > 0.75）
weak = max_scs[max_scs['类别'] == '体育薄弱国'].head(5)
print("\n🎯 体育薄弱国（高度依赖少数项目）：")
for country in weak['Country'].tolist():
    sport = scs_data[scs_data['Country'] == country].iloc[0]['Sport']
    scs_val = scs_data[scs_data['Country'] == country].iloc[0]['SCS']
    print(f"   {country}: {sport} ({scs_val:.1%})")

# 体育新兴国（没有奖牌）
emerging = max_scs[max_scs['类别'] == '体育新兴国'].head(5)
print("\n🌱 体育新兴国（尚未获得奖牌）：")
print(emerging['Country'].tolist())

print("\n" + "="*50)



# #简化版
# import pandas as pd
# import matplotlib.pyplot as plt
#
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
# plt.rcParams['axes.unicode_minus'] = False
#
# # 读数据
# athletes = pd.read_csv('summerOly_athletes.csv')
#
# # 筛选有奖牌 + 打分
# medal_data = athletes[athletes['Medal'] != 'No medal'].copy()
# medal_data['Score'] = medal_data['Medal'].map({'Gold': 3, 'Silver': 2, 'Bronze': 1})
#
# # 按国家+项目求和
# sport_score = medal_data.groupby(['NOC', 'Sport'], as_index=False)['Score'].sum()
# sport_score.columns = ['Country', 'Sport', 'Sport_Total_Score']
#
# # 按国家求和
# country_score = medal_data.groupby('NOC', as_index=False)['Score'].sum()
# country_score.columns = ['Country', 'Country_Total_Score']
#
# # 计算SCS
# scs = sport_score.merge(country_score, on='Country')
# scs['SCS'] = scs['Sport_Total_Score'] / scs['Country_Total_Score']
#
# # 每个国家取最大SCS
# max_scs = scs.groupby('Country')['SCS'].max().reset_index()
# max_scs.columns = ['Country', 'Max_SCS']
#
# # 分类
# def classify(x):
#     if x == 0: return '体育新兴国'
#     return '体育强国' if x <= 0.25 else '体育中等国' if x <= 0.75 else '体育薄弱国'
#
# max_scs['类别'] = max_scs['Max_SCS'].apply(classify)
# counts = max_scs['类别'].value_counts()
#
# # 画图
# colors = ['#8DA0B3', '#B8B5A8', '#C4A882']
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
#
# # 饼图
# ax1.pie(counts.values, labels=counts.index, autopct='%1.1f%%', colors=colors[:len(counts)], startangle=90)
# ax1.set_title('各国体育实力分类（饼图）')
#
# # 柱状图
# bars = ax2.bar(counts.index, counts.values, color=colors[:len(counts)], edgecolor='white')
# for bar, v in zip(bars, counts.values):
#     ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, str(v), ha='center', va='bottom')
# ax2.set_xlabel('类别'); ax2.set_ylabel('国家数量'); ax2.set_title('各国体育实力分类（柱状图）')
# ax2.grid(axis='y', alpha=0.3)
#
# plt.tight_layout()
# plt.savefig('SCS分类图.png', dpi=300)
# plt.show()
# print("✅ 图已保存：SCS分类图.png")
