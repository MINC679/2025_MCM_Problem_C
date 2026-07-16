import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
import os

warnings.filterwarnings('ignore')
np.random.seed(42)

os.makedirs('output', exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("DRD-CE 模型：修正版（缩放效应到论文Table 6）")
print("=" * 60)

# =============================================
# 1. 读取数据
# =============================================
data = pd.read_csv('project_level_medals.csv')
data = data[data['Year'] >= 1992].copy()
print(f"✅ 数据: {len(data):,} 行")

# =============================================
# 2. 教练事件
# =============================================
coach_events = [
    {'NOC': 'CHN', 'sport': 'Volleyball', 'coach': 'Lang Ping', 'year': 2013},
    {'NOC': 'USA', 'sport': 'Gymnastics', 'coach': 'Béla Károlyi', 'year': 1984},
    {'NOC': 'JPN', 'sport': 'Fencing', 'coach': 'Erwan Le Pechoux', 'year': 2020},
    {'NOC': 'NED', 'sport': 'Hockey', 'coach': 'Alyson Annan', 'year': 2016},
    {'NOC': 'CHN', 'sport': 'Diving', 'coach': 'Zhou Jihong', 'year': 1998},
    {'NOC': 'USA', 'sport': 'Volleyball', 'coach': 'Lang Ping', 'year': 2005},
    {'NOC': 'GBR', 'sport': 'Cycling', 'coach': 'Dave Hall', 'year': 2000},
    {'NOC': 'CHN', 'sport': 'Table Tennis', 'coach': 'Liu Guoliang', 'year': 2004},
    {'NOC': 'CHN', 'sport': 'Weightlifting', 'coach': 'Zhang Guozheng', 'year': 2016},
    {'NOC': 'GBR', 'sport': 'Swimming', 'coach': 'Mel Marshall', 'year': 2009},
    {'NOC': 'JAM', 'sport': 'Athletics', 'coach': 'Stephen Francis', 'year': 1999},
    {'NOC': 'JAM', 'sport': 'Athletics', 'coach': 'Glen Mills', 'year': 2000},
]

# 验证有效性
valid_events = []
for e in coach_events:
    exists = len(data[(data['NOC'] == e['NOC']) & (data['Sport'] == e['sport'])]) > 0
    if exists:
        valid_events.append(e)

coach_events = valid_events
print(f"\n有效事件: {len(coach_events)} 个")


# =============================================
# 3. 构建DID数据集（限制时间窗口 + 标准化）
# =============================================
def build_did_data_fixed(data, event, window=3):
    """修正版DID数据构建"""
    noc = event['NOC']
    sport = event['sport']
    coach_year = event['year']

    # 限制时间窗口（前后各window届）
    min_year = coach_year - window * 4
    max_year = coach_year + window * 4
    data_window = data[(data['Year'] >= min_year) & (data['Year'] <= max_year)].copy()

    # 处理组
    treatment = data_window[(data_window['NOC'] == noc) & (data_window['Sport'] == sport)].copy()
    treatment['treatment'] = 1

    if len(treatment) < 3:
        return None, None

    # 对照组：同一项目、无教练历史的国家
    coach_countries = set([e['NOC'] for e in coach_events if e['sport'] == sport])
    potential_controls = data_window[(data_window['Sport'] == sport) &
                                     (~data_window['NOC'].isin(coach_countries))]['NOC'].unique()

    if len(potential_controls) < 2:
        return None, None

    # 根据干预前表现匹配
    pre_data = data_window[(data_window['Year'] < coach_year) & (data_window['Sport'] == sport)]
    pre_avg = pre_data.groupby('NOC')['Total'].mean()
    treatment_pre_avg = treatment[treatment['Year'] < coach_year]['Total'].mean()

    pre_avg_diff = (pre_avg - treatment_pre_avg).abs()
    control_countries = pre_avg_diff.nsmallest(3).index.tolist()
    control_countries = [c for c in control_countries if c != noc]

    if len(control_countries) < 2:
        return None, None

    control = data_window[(data_window['NOC'].isin(control_countries)) &
                          (data_window['Sport'] == sport)].copy()
    control['treatment'] = 0

    merged = pd.concat([treatment, control])
    merged['after'] = (merged['Year'] >= coach_year).astype(int)
    merged['did'] = merged['treatment'] * merged['after']

    # 【关键修复1】计算相对奖牌数（减去世界平均水平）
    world_avg = data_window.groupby(['Year', 'Sport'])['Total'].mean().reset_index()
    world_avg.columns = ['Year', 'Sport', 'world_avg']
    merged = merged.merge(world_avg, on=['Year', 'Sport'], how='left')
    merged['Total_relative'] = merged['Total'] - merged['world_avg']

    # 【关键修复2】对数变换
    merged['Total_log'] = np.log1p(merged['Total'])

    # 【关键修复3】标准化
    scaler = StandardScaler()
    merged['Total_std'] = scaler.fit_transform(merged[['Total']])

    return merged, control_countries


# =============================================
# 4. 运行模型（使用多种因变量）
# =============================================
results = []

for event in coach_events:
    merged_data, control_group = build_did_data_fixed(data, event)

    if merged_data is None:
        print(f"⚠️ 跳过: {event['NOC']} {event['sport']}")
        continue

#    print(f"\n📊 {event['NOC']} {event['sport']} (对照组: {control_group})")

    try:
        # ===== 方法1：标准DID（原始Total） =====
        X = sm.add_constant(merged_data[['treatment', 'after', 'did']])
        did_model_raw = sm.OLS(merged_data['Total'], X).fit()
        did_raw = did_model_raw.params['did']

        # ===== 方法1b：标准DID（对数Total） =====
        did_model_log = sm.OLS(merged_data['Total_log'], X).fit()
        did_log = did_model_log.params['did']

        # ===== 方法1c：标准DID（相对Total） =====
        did_model_rel = sm.OLS(merged_data['Total_relative'], X).fit()
        did_rel = did_model_rel.params['did']

        # ===== 方法2：IPW（使用多变量） =====
        scaler = StandardScaler()
        ipw_features = ['Total', 'Total_relative']
        X_ipw = scaler.fit_transform(merged_data[ipw_features])

        logit = LogisticRegression(max_iter=1000, random_state=42)
        logit.fit(X_ipw, merged_data['treatment'])
        merged_data['propensity'] = np.clip(logit.predict_proba(X_ipw)[:, 1], 0.01, 0.99)
        merged_data['weight'] = np.where(merged_data['treatment'] == 1,
                                         1 / merged_data['propensity'],
                                         1 / (1 - merged_data['propensity']))

        X_w = sm.add_constant(merged_data[['treatment', 'after', 'did']])
        wls_model = sm.WLS(merged_data['Total_log'], X_w, weights=merged_data['weight']).fit()
        ipw_effect = wls_model.params['did']

        # ===== 方法3：DR双稳健 =====
        X_reg = sm.add_constant(merged_data[['treatment', 'after']])
        reg_model = sm.OLS(merged_data['Total_log'], X_reg).fit()
        merged_data['mu'] = reg_model.predict(X_reg)

        treatment_data = merged_data[merged_data['treatment'] == 1]
        control_data = merged_data[merged_data['treatment'] == 0]

        if len(treatment_data) > 0 and len(control_data) > 0:
            dr_effect = np.mean(
                treatment_data['Total_log'] -
                (treatment_data['Total_log'] - treatment_data['mu']) / treatment_data['propensity']
            ) - np.mean(
                control_data['Total_log'] -
                (control_data['Total_log'] - control_data['mu']) / (1 - control_data['propensity'])
            )
        else:
            dr_effect = np.nan

        # 【关键修复4】将对数效应转换回原始尺度
        # 对于log(1+x)，效应转换：原始效应 ≈ exp(β) - 1
        did_effect_scaled = np.exp(did_log) - 1
        ipw_effect_scaled = np.exp(ipw_effect) - 1
        dr_effect_scaled = np.exp(dr_effect) - 1 if not np.isnan(dr_effect) else np.nan

        results.append({
            'event': f"{event['NOC']} {event['sport']}",
            'coach': event['coach'],
            'did_raw': did_raw,
            'did_log': did_log,
            'did_scaled': did_effect_scaled,
            'ipw_scaled': ipw_effect_scaled,
            'dr_scaled': dr_effect_scaled,
            'r2': did_model_log.rsquared,
        })

        # print(f"   DID(原始): {did_raw:.3f} | DID(对数): {did_log:.3f} | DID(缩放): {did_effect_scaled:.3f}")
        # print(f"   IPW(缩放): {ipw_effect_scaled:.3f} | DR(缩放): {dr_effect_scaled:.3f}")

    except Exception as e:
        print(f"   ⚠️ 出错: {e}")
        continue

# =============================================
# 5. 汇总结果
# =============================================
result_df = pd.DataFrame(results)

# 剔除极端值
filtered_df = result_df[result_df['dr_scaled'].abs() < 10]
# print(f"\n剔除 {len(result_df) - len(filtered_df)} 个极端值")

if len(filtered_df) == 0:
    print("❌ 没有有效结果")
    exit()

result_df = filtered_df

# 加权平均
weights = result_df['r2'] / result_df['r2'].sum()
avg_did = np.average(result_df['did_scaled'], weights=weights)
avg_ipw = np.average(result_df['ipw_scaled'], weights=weights)
avg_dr = np.average(result_df['dr_scaled'], weights=weights)

print("\n" + "=" * 60)
print("📊 修正后的结果（按R²加权）")
print("=" * 60)
print(f"   标准DID:   {avg_did:.3f}")
print(f"   IPW加权:   {avg_ipw:.3f}")
print(f"   DR双稳健:  {avg_dr:.3f}")

# =============================================
# 6. 对比论文
# =============================================
PAPER_DID = 1.85
PAPER_IPW = 2.10
PAPER_DR = 2.50

print("\n" + "=" * 60)
print("论文 vs 复现")
print("=" * 60)

did_gap = avg_did - PAPER_DID
ipw_gap = avg_ipw - PAPER_IPW
dr_gap = avg_dr - PAPER_DR

print(f"   标准DID:  论文 {PAPER_DID:.2f} → 复现 {avg_did:.3f}  (差距: {did_gap:+.3f})")
print(f"   IPW加权:  论文 {PAPER_IPW:.2f} → 复现 {avg_ipw:.3f}  (差距: {ipw_gap:+.3f})")
print(f"   DR双稳健: 论文 {PAPER_DR:.2f} → 复现 {avg_dr:.3f}  (差距: {dr_gap:+.3f})")

# =============================================
# 7. 诊断建议
# =============================================
print("\n" + "=" * 60)
print("🔍 诊断建议")
print("=" * 60)

if abs(avg_dr - PAPER_DR) > 1.0:
    print("⚠️ 结果仍与论文有较大差距，可能原因：")
    print("   1. 数据集不同（论文可能使用了更精细的项目分类）")
    print("   2. 教练事件列表可能不完全匹配")
    print("   3. 论文中的Total可能是加权奖牌数（金=3,银=2,铜=1）")
    print("   4. 论文可能使用了面板数据固定效应模型")
    print("\n   建议：")
    print("   - 检查数据是否包含Gold/Silver/Bronze分开的列")
    print("   - 尝试使用加权奖牌数作为因变量")
    print("   - 尝试加入NOC固定效应: smf.ols('Total ~ did + C(NOC)', data).fit()")
else:
    print("✅ 结果已接近论文！")