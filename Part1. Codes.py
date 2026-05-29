import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
 
# ════════════════════════════════════════════════════════════════
# 1. 读取数据 & 基本探索
# ════════════════════════════════════════════════════════════════
df = pd.read_csv('polymer_electrolyte.csv')
 
print("=" * 55)
print("  STEP 1 — 基本信息")
print("=" * 55)
print(f"数据维度   : {df.shape[0]:,} 行  ×  {df.shape[1]} 列")
print(f"\n列名与类型:\n{df.dtypes}")
print(f"\n缺失值统计:\n{df.isnull().sum()}")
print(f"\n数值列统计摘要:\n{df.describe().round(4)}")
 
 
# ════════════════════════════════════════════════════════════════
# 2. 预处理：对数变换（电导率 & 扩散系数跨越多个数量级）
# ════════════════════════════════════════════════════════════════
df['log_CONDUCTIVITY']     = np.log10(df['CONDUCTIVITY'])
df['log_Li_Diffusivity']   = np.log10(df['Li Diffusivity'])
df['log_TFSI_Diffusivity'] = np.log10(df['TFSI Diffusivity'])
df['log_Poly_Diffusivity'] = np.log10(df['Poly Diffusivity'])
 
print("\n" + "=" * 55)
print("  STEP 2 — 关键统计发现")
print("=" * 55)
 
cond = df['CONDUCTIVITY']
tn   = df['Transference Number']
log_cond = df['log_CONDUCTIVITY']
 
print(f"\n【电导率】")
print(f"  范围    : {cond.min():.2e}  ~  {cond.max():.2e}  S/cm")
print(f"  中位数  : {cond.median():.2e}  S/cm")
print(f"  最高/最低比值 : {cond.max()/cond.min():.0f} 倍  → 必须取对数分析")
 
print(f"\n【迁移数 Transference Number】")
print(f"  正值比例（Li⁺ 主导）: {(tn > 0).mean()*100:.1f}%")
print(f"  范围 : {tn.min():.3f}  ~  {tn.max():.3f}")
print(f"  均值 : {tn.mean():.4f}")
 
print(f"\n【与 log(电导率) 的 Pearson 相关系数】")
features_corr = {
    'log(Li Diffusivity) '  : df['log_Li_Diffusivity'],
    'log(TFSI Diffusivity)'  : df['log_TFSI_Diffusivity'],
    'log(Poly Diffusivity)'  : df['log_Poly_Diffusivity'],
    'Molality               ': df['Molality'],
    'Monomer MW             ': df['Monomer Molecular Weight'],
    'Degree of Polym.       ': df['Degree of Polymerization'],
    'Density                ': df['Density'],
}
for name, col in features_corr.items():
    r = np.corrcoef(col, log_cond)[0, 1]
    bar = '█' * int(abs(r) * 20)
    print(f"  {name} : r = {r:+.3f}  {bar}")
 
print(f"\n【聚合度 (DP) 样本分布】")
for dp, cnt in df['Degree of Polymerization'].value_counts().sort_index().items():
    print(f"  DP = {int(dp):2d} : {cnt:4d} 样本  ({cnt/len(df)*100:.1f}%)")
 
 
# ════════════════════════════════════════════════════════════════
# 3. 图1 — 总览 Dashboard (3×3)
# ════════════════════════════════════════════════════════════════
COLORS = ['#2E86AB','#A23B72','#F18F01','#C73E1D',
          '#3B1F2B','#44BBA4','#E94F37','#393E41','#8B8BAE']
 
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#F8F9FA')
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
 
# ── 子图1：电导率分布（对数坐标）
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(log_cond, bins=40, color=COLORS[0], edgecolor='white', linewidth=0.5)
ax1.axvline(log_cond.mean(), color='red', linestyle='--', linewidth=1.5,
            label=f'mean = {log_cond.mean():.2f}')
ax1.set_xlabel('log₁₀(Conductivity)');  ax1.set_ylabel('Count')
ax1.set_title('Conductivity Distribution', fontweight='bold')
ax1.legend(fontsize=9);  ax1.set_facecolor('#FFFFFF')
 
# ── 子图2：迁移数分布
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(df['Transference Number'], bins=40, color=COLORS[1],
         edgecolor='white', linewidth=0.5)
ax2.axvline(0, color='gray', linestyle=':', linewidth=1)
ax2.set_xlabel('Transference Number');  ax2.set_ylabel('Count')
ax2.set_title('Transference Number Distribution', fontweight='bold')
ax2.set_facecolor('#FFFFFF')
 
# ── 子图3：聚合度分布（条形图）
ax3 = fig.add_subplot(gs[0, 2])
dp_counts = df['Degree of Polymerization'].value_counts().sort_index()
ax3.bar(dp_counts.index.astype(str), dp_counts.values,
        color=COLORS[2], edgecolor='white')
ax3.set_xlabel('Degree of Polymerization');  ax3.set_ylabel('Count')
ax3.set_title('Degree of Polymerization', fontweight='bold')
ax3.tick_params(axis='x', rotation=45, labelsize=7)
ax3.set_facecolor('#FFFFFF')
 
# ── 子图4：三种扩散系数对比（箱线图）
ax4 = fig.add_subplot(gs[1, 0])
diff_data = [df['log_TFSI_Diffusivity'],
             df['log_Li_Diffusivity'],
             df['log_Poly_Diffusivity']]
bp = ax4.boxplot(diff_data, patch_artist=True, labels=['TFSI','Li','Polymer'])
for patch, c in zip(bp['boxes'], [COLORS[0], COLORS[3], COLORS[4]]):
    patch.set_facecolor(c);  patch.set_alpha(0.7)
ax4.set_ylabel('log₁₀(Diffusivity)')
ax4.set_title('Diffusivity Comparison', fontweight='bold')
ax4.set_facecolor('#FFFFFF')
 
# ── 子图5：电导率 vs Li扩散系数（散点图，按迁移数着色）
ax5 = fig.add_subplot(gs[1, 1])
sc5 = ax5.scatter(df['log_Li_Diffusivity'], log_cond,
                  c=df['Transference Number'],
                  cmap='RdYlGn', alpha=0.4, s=8)
plt.colorbar(sc5, ax=ax5, label='Transference Number', shrink=0.8)
ax5.set_xlabel('log₁₀(Li Diffusivity)');  ax5.set_ylabel('log₁₀(Conductivity)')
ax5.set_title('Conductivity vs Li Diffusivity\n(colored by Transference Number)',
              fontweight='bold')
ax5.set_facecolor('#FFFFFF')
 
# ── 子图6：密度 vs 电导率（按聚合度着色）
ax6 = fig.add_subplot(gs[1, 2])
dp_vals  = sorted(df['Degree of Polymerization'].unique())
cmap6    = plt.cm.get_cmap('viridis', len(dp_vals))
for i, dp in enumerate(dp_vals):
    sub = df[df['Degree of Polymerization'] == dp]
    ax6.scatter(sub['Density'], np.log10(sub['CONDUCTIVITY']),
                color=cmap6(i), alpha=0.4, s=8, label=f'DP={int(dp)}')
ax6.set_xlabel('Density (g/cm³)');  ax6.set_ylabel('log₁₀(Conductivity)')
ax6.set_title('Density vs Conductivity\n(by Degree of Polymerization)',
              fontweight='bold')
ax6.legend(fontsize=7, markerscale=2, ncol=2)
ax6.set_facecolor('#FFFFFF')
 
# ── 子图7：相关系数热图
ax7 = fig.add_subplot(gs[2, :2])
corr_df = df[['Molality', 'Monomer Molecular Weight', 'Degree of Polymerization',
              'Density', 'log_CONDUCTIVITY', 'log_TFSI_Diffusivity',
              'log_Li_Diffusivity', 'log_Poly_Diffusivity',
              'Transference Number']].copy()
corr_df.columns = ['Molality','MW','DP','Density',
                   'log(Cond)','log(TFSI_D)','log(Li_D)','log(Poly_D)','T_num']
mask = np.triu(np.ones(corr_df.shape[1], dtype=bool))
sns.heatmap(corr_df.corr(), ax=ax7, mask=mask,
            annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, linewidths=0.5,
            annot_kws={'size': 8})
ax7.set_title('Correlation Matrix (log-transformed)', fontweight='bold')
ax7.tick_params(axis='x', rotation=30, labelsize=8)
ax7.tick_params(axis='y', rotation=0, labelsize=8)
 
# ── 子图8：单体分子量分布
ax8 = fig.add_subplot(gs[2, 2])
ax8.hist(df['Monomer Molecular Weight'], bins=35,
         color=COLORS[5], edgecolor='white', linewidth=0.5)
ax8.set_xlabel('Monomer MW (g/mol)');  ax8.set_ylabel('Count')
ax8.set_title('Monomer Molecular Weight', fontweight='bold')
ax8.set_facecolor('#FFFFFF')
 
fig.suptitle(
    'Polymer Electrolyte Dataset — Exploratory Data Analysis\n'
    '(n = 6,270 molecular dynamics simulations)',
    fontsize=14, fontweight='bold', y=0.98)
plt.savefig('polymer_eda_dashboard.png', dpi=150,
            bbox_inches='tight', facecolor='#F8F9FA')
plt.close()
print("\n图1 保存 → polymer_eda_dashboard.png")
 
 
# ════════════════════════════════════════════════════════════════
# 4. 图2 — 深度分析（2×2）
# ════════════════════════════════════════════════════════════════
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
fig2.patch.set_facecolor('#F8F9FA')
fig2.suptitle('Polymer Electrolyte — Deep Dive Analysis',
              fontsize=14, fontweight='bold')
 
# ── 高 vs 低电导率样本特征均值对比
ax = axes[0, 0]
high = df[df['CONDUCTIVITY'] >= df['CONDUCTIVITY'].quantile(0.75)]
low  = df[df['CONDUCTIVITY'] <  df['CONDUCTIVITY'].quantile(0.25)]
feats = ['Molality','Monomer Molecular Weight',
         'Degree of Polymerization','Density']
feat_labels = ['Molality','Monomer MW','Deg. Polym.','Density']
means_all  = [df[f].mean()   for f in feats]
high_norm  = [high[f].mean() / m for f, m in zip(feats, means_all)]
low_norm   = [low[f].mean()  / m for f, m in zip(feats, means_all)]
x = np.arange(len(feats));  w = 0.35
ax.bar(x - w/2, high_norm, w, label='Top 25% conductivity',
       color='#2E86AB', alpha=0.85)
ax.bar(x + w/2, low_norm,  w, label='Bot 25% conductivity',
       color='#C73E1D', alpha=0.85)
ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, label='Dataset mean')
ax.set_xticks(x);  ax.set_xticklabels(feat_labels, fontsize=9)
ax.set_ylabel('Normalized value (ratio to mean)', fontsize=9)
ax.set_title('High vs Low Conductivity — Feature Comparison',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=8);  ax.set_facecolor('#FFFFFF')
 
# ── 迁移数 vs TFSI扩散系数（按电导率四分位着色）
ax = axes[0, 1]
cond_bins = pd.qcut(df['CONDUCTIVITY'], q=4,
                    labels=['Q1\n(lowest)','Q2','Q3','Q4\n(highest)'])
for i, (label, group) in enumerate(df.groupby(cond_bins)):
    ax.scatter(group['log_TFSI_Diffusivity'],
               group['Transference Number'],
               alpha=0.3, s=8,
               color=plt.cm.RdYlGn(i / 3), label=label)
ax.axhline(0, color='gray', linestyle=':', linewidth=1)
ax.set_xlabel('log₁₀(TFSI Diffusivity)', fontsize=10)
ax.set_ylabel('Transference Number', fontsize=10)
ax.set_title('Transference Number vs TFSI Diffusivity\n'
             '(by conductivity quartile)', fontsize=10, fontweight='bold')
ax.legend(title='Conductivity', fontsize=8, title_fontsize=8)
ax.set_facecolor('#FFFFFF')
 
# ── 聚合度 vs 电导率
ax = axes[1, 0]
dp_order  = sorted(df['Degree of Polymerization'].unique())
data_viol = [df[df['Degree of Polymerization']==dp]['log_CONDUCTIVITY'].values
             for dp in dp_order]
vp = ax.violinplot(data_viol, positions=range(len(dp_order)), showmedians=True)
for i, body in enumerate(vp['bodies']):
    body.set_facecolor(plt.cm.viridis(i / len(dp_order)));  body.set_alpha(0.7)
ax.set_xticks(range(len(dp_order)))
ax.set_xticklabels([f'DP={int(d)}' for d in dp_order], fontsize=7, rotation=40)
ax.set_ylabel('log₁₀(Conductivity)', fontsize=10)
ax.set_title('Conductivity by Degree of Polymerization',
             fontsize=10, fontweight='bold')
ax.set_facecolor('#FFFFFF')
 
# ── Top 20 最高电导率样本
ax = axes[1, 1]
top20 = df.nlargest(20, 'CONDUCTIVITY')[
    ['Trajectory ID','CONDUCTIVITY','Transference Number']
].reset_index(drop=True)
bar_colors = ['#2E86AB' if t > 0 else '#C73E1D'
              for t in top20['Transference Number']]
ax.barh(range(20), np.log10(top20['CONDUCTIVITY']),
        color=bar_colors, alpha=0.85)
ax.set_yticks(range(20))
ax.set_yticklabels(
    [f"ID {int(r['Trajectory ID'])}  (T={r['Transference Number']:.2f})"
     for _, r in top20.iterrows()], fontsize=7)
ax.set_xlabel('log₁₀(Conductivity)', fontsize=10)
ax.set_title('Top 20 Highest Conductivity Samples\n'
             '(blue = T > 0,  red = T ≤ 0)',
             fontsize=10, fontweight='bold')
ax.legend(handles=[mpatches.Patch(color='#2E86AB', label='T > 0'),
                   mpatches.Patch(color='#C73E1D', label='T ≤ 0')],
          fontsize=8)
ax.set_facecolor('#FFFFFF')
 
plt.tight_layout()
plt.savefig('polymer_deep_analysis.png', dpi=150,
            bbox_inches='tight', facecolor='#F8F9FA')
plt.close()
print("图2 保存 → polymer_deep_analysis.png")
 
 
# ════════════════════════════════════════════════════════════════
# 5. 图3 — 结构-性能关系（1×3）
# ════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))
fig3.patch.set_facecolor('#F8F9FA')
fig3.suptitle('Structure–Property Relationships in Polymer Electrolytes',
              fontsize=13, fontweight='bold')
main_dps = [14, 17, 19, 26, 31]
cmap3    = plt.cm.viridis
 
# ── Li vs TFSI 扩散系数
ax = axes3[0]
sc = ax.scatter(df['log_TFSI_Diffusivity'], df['log_Li_Diffusivity'],
                c=df['Transference Number'], cmap='RdYlGn',
                alpha=0.5, s=10, vmin=-0.5, vmax=0.8)
lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1])]
ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.5, label='Li = TFSI')
plt.colorbar(sc, ax=ax, label='Transference Number', shrink=0.8)
ax.set_xlabel('log₁₀(TFSI Diffusivity)');  ax.set_ylabel('log₁₀(Li Diffusivity)')
ax.set_title('Li vs TFSI Diffusivity\n(above line = Li faster)',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=8);  ax.set_facecolor('#FFFFFF')
 
# ── 单体分子量 vs 电导率（按主要DP着色）
ax = axes3[1]
for dp in main_dps:
    sub = df[df['Degree of Polymerization'] == dp]
    idx = dp_vals.index(dp) / len(dp_vals)
    ax.scatter(sub['Monomer Molecular Weight'], sub['log_CONDUCTIVITY'],
               color=cmap3(idx), alpha=0.4, s=12,
               label=f'DP={int(dp)} (n={len(sub)})')
ax.set_xlabel('Monomer Molecular Weight (g/mol)')
ax.set_ylabel('log₁₀(Conductivity)')
ax.set_title('Monomer MW vs Conductivity\n(major DP groups)',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=8, markerscale=2);  ax.set_facecolor('#FFFFFF')
 
# ── Molality vs 电导率（按主要DP着色）
ax = axes3[2]
for dp in main_dps:
    sub = df[df['Degree of Polymerization'] == dp]
    idx = dp_vals.index(dp) / len(dp_vals)
    ax.scatter(sub['Molality'], sub['log_CONDUCTIVITY'],
               color=cmap3(idx), alpha=0.4, s=12,
               label=f'DP={int(dp)}')
ax.set_xlabel('Molality (mol/kg)');  ax.set_ylabel('log₁₀(Conductivity)')
ax.set_title('Molality vs Conductivity\n(major DP groups)',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=8, markerscale=2);  ax.set_facecolor('#FFFFFF')
 
plt.tight_layout()
plt.savefig('polymer_structure_property.png', dpi=150,
            bbox_inches='tight', facecolor='#F8F9FA')
plt.close()
print("图3 保存 → polymer_structure_property.png")
 
 
# ════════════════════════════════════════════════════════════════
# 6. 保存处理后的数据
# ════════════════════════════════════════════════════════════════
df.to_csv('polymer_electrolyte_processed.csv', index=False)
print("\n处理后数据已保存 → polymer_electrolyte_processed.csv")
print("（新增列：log_CONDUCTIVITY, log_Li_Diffusivity,")
print("          log_TFSI_Diffusivity, log_Poly_Diffusivity）")
print("\n✓ Part 1 项目全部完成。")
