import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('panel_data_for_model.csv')
df['Total_Medals'] = pd.to_numeric(df['Total_Medals'], errors='coerce')  # 确保 Total_Medals 是数字

af_list = []  # 用列表收集结果
# 计算AF（后东道主效应）
for idx, row in df.iterrows():
    country = row['Country']
    year = row['Year']
    af_value = 0.0

    for mu, delta in [(1, 1), (2, 0.6), (3, 0.3)]:  # 循环3次，mu是"第几届后"，delta是衰减因子
        prev_year = year - 4 * mu  # 往前推mu届（4年×mu）
        prev_row = df[(df['Country'] == country) & (df['Year'] == prev_year)]
        if len(prev_row) > 0:
            prev_host = prev_row['Is_Host'].values[0]
            if prev_host == 1:  # 如果那年的确是东道主
                M_host = prev_row['Total_Medals'].values[0]  # 取东道主那届的奖牌数
                if M_host > 0:
                    M_future = row['Total_Medals']
                    if pd.notna(M_future):
                        af_value = ((M_future - M_host) / M_host) * delta  # 公式(13)：计算后东道主效应
                        break
    af_list.append(af_value)
df['AF'] = af_list

# 计算东道主效应（前后数据）
host_effect_data = []
for country in df['Country'].unique():
    country_data = df[df['Country'] == country].sort_values('Year')
    host_years = country_data[country_data['Is_Host'] == 1]['Year'].tolist()

    for host_year in host_years:  # 遍历每一个东道主年份
        host_idx = country_data[country_data['Year'] == host_year].index  # 找到东道主那届在表格里的位置
        if len(host_idx) == 0:
            continue
        pos = country_data.index.get_loc(host_idx[0])  # 获取那届在数据中的序号

        for offset in [-2, -1, 0, 1, 2, 3]:  # 取前后共6届（前2、前1、当届、后1、后2、后3）
            target_pos = pos + offset  # 目标位置 = 当前位置 + 偏移
            if 0 <= target_pos < len(country_data):
                row = country_data.iloc[target_pos]
                host_effect_data.append({
                    'Country': country,
                    'Relative_Year': offset,
                    'Medals': row['Total_Medals']
                })

effect_df = pd.DataFrame(host_effect_data)  # 把收集的数据转成表格
avg_effect = effect_df.groupby('Relative_Year')['Medals'].mean().reset_index()
avg_effect = avg_effect.sort_values('Relative_Year')

# ========== 画图（全英文标签，不会乱码）==========
plt.figure(figsize=(12, 6))
labels = ['2 Games\nBefore', '1 Game\nBefore', 'Host\nYear', '1 Game\nAfter', '2 Games\nAfter', '3 Games\nAfter']
bars = plt.bar(range(len(avg_effect)), avg_effect['Medals'], color='#B8C4C9', edgecolor='white', linewidth=2)

# 标注数值
for bar, val in zip(bars, avg_effect['Medals']):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f'{val:.1f}', ha='center', va='bottom', fontsize=13, fontweight='normal')

plt.xticks(range(len(avg_effect)), labels, fontsize=12)  # 把X轴刻度换成文字标签
plt.xlabel('Position Relative to Host Year', fontsize=14, fontweight='normal')
plt.ylabel('Average Medal Count', fontsize=14, fontweight='normal')
plt.title('Host Country Effect on Medal Counts', fontsize=16, fontweight='normal')

plt.axvline(x=2.5, color='#B8C8D4', linestyle='--', linewidth=2.5, alpha=0.8, label='Host Year')  # 在"东道主年"和"后1届"之间画一条竖线

plt.legend(fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('Host_Effect_Trend.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ 图已保存为 'Host_Effect_Trend.png'")

# 输出结论
host_medal = avg_effect[avg_effect['Relative_Year'] == 0]['Medals'].values[0]
before_medal = avg_effect[avg_effect['Relative_Year'].isin([-2, -1])]['Medals'].mean()
after_medal = avg_effect[avg_effect['Relative_Year'].isin([1, 2, 3])]['Medals'].mean()

print("\n" + "=" * 50)
print("HOST EFFECT CONCLUSIONS")
print("=" * 50)
print(f"• Host year average medals: {host_medal:.1f}")
print(f"• Average of 2 Games before host: {before_medal:.1f}")
print(f"• Average of 3 Games after host: {after_medal:.1f}")
print(f"• Boost during host year: {(host_medal - before_medal) / before_medal * 100:.1f}%")
print(f"• Decline after host year: {(host_medal - after_medal) / host_medal * 100:.1f}%")



#简化版
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 读数据
# df = pd.read_csv('panel_data_for_model.csv')
# df['Total_Medals'] = pd.to_numeric(df['Total_Medals'], errors='coerce')
#
# # 计算AF
# af_list = []
# for idx, row in df.iterrows():
#     af = 0.0
#     for mu, delta in [(1,1), (2,0.6), (3,0.3)]:
#         prev = df[(df['Country']==row['Country']) & (df['Year']==row['Year']-4*mu)]
#         if len(prev) == 0: continue
#         if prev['Is_Host'].values[0] == 1 and prev['Total_Medals'].values[0] > 0:
#             af = ((row['Total_Medals'] - prev['Total_Medals'].values[0]) / prev['Total_Medals'].values[0]) * delta
#             break
#     af_list.append(af)
# df['AF'] = af_list
#
# # 收集东道主前后数据
# data = []
# for country in df['Country'].unique():
#     cdata = df[df['Country']==country].sort_values('Year')
#     for host_year in cdata[cdata['Is_Host']==1]['Year']:
#         pos = cdata.index.get_loc(cdata[cdata['Year']==host_year].index[0])
#         for offset in [-2,-1,0,1,2,3]:
#             tp = pos + offset
#             if 0 <= tp < len(cdata):
#                 row = cdata.iloc[tp]
#                 data.append({'Relative_Year': offset, 'Medals': row['Total_Medals']})
#
# # 计算平均值
# avg = pd.DataFrame(data).groupby('Relative_Year')['Medals'].mean().reset_index().sort_values('Relative_Year')
#
# # 画图
# plt.figure(figsize=(12,6))
# bars = plt.bar(range(6), avg['Medals'], color='#8DA0B3', edgecolor='white')
# for bar, v in zip(bars, avg['Medals']):
#     plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f'{v:.1f}', ha='center', va='bottom')
# plt.xticks(range(6), ['2 Before','1 Before','Host','1 After','2 After','3 After'])
# plt.axvline(x=2.5, color='#8DA0B3', linestyle='--', linewidth=2)
# plt.xlabel('Position Relative to Host Year')
# plt.ylabel('Average Medals')
# plt.title('Host Country Effect')
# plt.grid(axis='y', alpha=0.3)
# plt.tight_layout()
# plt.savefig('Host_Effect_Trend.png', dpi=300)
# plt.show()