import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
 
 
# ════════════════════════════════════════════════════════════════
# 1. 读取 & 特征工程
# ════════════════════════════════════════════════════════════════
df = pd.read_csv('polymer_electrolyte.csv')
 
# 对数变换（扩散系数跨越多个数量级，取log后更线性）
df['log_CONDUCTIVITY']     = np.log10(df['CONDUCTIVITY'])
df['log_Li_Diffusivity']   = np.log10(df['Li Diffusivity'])
df['log_TFSI_Diffusivity'] = np.log10(df['TFSI Diffusivity'])
df['log_Poly_Diffusivity'] = np.log10(df['Poly Diffusivity'])
 
# 特征列 & 目标列
# 为什么选这8个特征？
#   - Molality/Density/MW/DP : 配方参数，可直接设计
#   - Transference Number    : 迁移数，表征 Li⁺ 比例
#   - 三种 log 扩散系数       : Part1发现与电导率强相关(r≈0.64)
FEATURES = [
    'Molality',
    'Monomer Molecular Weight',
    'Degree of Polymerization',
    'Density',
    'Transference Number',
    'log_Li_Diffusivity',
    'log_TFSI_Diffusivity',
    'log_Poly_Diffusivity',
]
TARGET = 'log_CONDUCTIVITY'
 
X = df[FEATURES]
y = df[TARGET]
 
# 划分训练集 / 测试集（80% / 20%）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"训练集：{X_train.shape[0]} 条  |  测试集：{X_test.shape[0]} 条")
 
 
# ════════════════════════════════════════════════════════════════
# 2. 训练三个模型
# ════════════════════════════════════════════════════════════════
# 为什么训练三个？
#   Linear Regression  : 基线，理解线性关系的上限
#   Random Forest      : 集成树，抗噪声，特征重要性可解释
#   Gradient Boosting  : 通常比 RF 精度略高，但训练慢
 
models = {
    'Linear Regression' : LinearRegression(),
    'Random Forest'     : RandomForestRegressor(
                              n_estimators=100,   # 100棵树
                              random_state=42,
                              n_jobs=-1           # 用所有CPU核心
                          ),
    'Gradient Boosting' : GradientBoostingRegressor(
                              n_estimators=100,
                              random_state=42
                          ),
}
 
results = {}
print("\n" + "=" * 55)
print("  模型训练结果")
print("=" * 55)
print(f"{'模型':<22}  {'R²':>7}  {'RMSE':>7}  {'CV R²':>7}")
print("-" * 55)
 
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2     = r2_score(y_test, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    cv_r2  = cross_val_score(model, X, y, cv=5, scoring='r2').mean()
    results[name] = {
        'model'  : model,
        'y_pred' : y_pred,
        'R2'     : r2,
        'RMSE'   : rmse,
        'CV_R2'  : cv_r2,
    }
    print(f"{name:<22}  {r2:>7.4f}  {rmse:>7.4f}  {cv_r2:>7.4f}")
 
# ── 指标含义说明 ──────────────────────────────────────────────
print("""
指标说明：
  R²     : 模型解释方差的比例 (0~1, 越高越好, >0.7 可用)
  RMSE   : 预测误差 (log 单位, 越小越好)
  CV R²  : 5折交叉验证R² (衡量泛化能力，与R²差距大=过拟合)
""")
 
 
# ════════════════════════════════════════════════════════════════
# 3. 图1 — 模型评估 Dashboard (3×3)
# ════════════════════════════════════════════════════════════════
PALETTE = {
    'Linear Regression' : '#8B8BAE',
    'Random Forest'     : '#2E86AB',
    'Gradient Boosting' : '#F18F01',
}
 
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#F8F9FA')
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35)
 
names = list(results.keys())
r2s   = [results[n]['R2']    for n in names]
rmses = [results[n]['RMSE']  for n in names]
cvs   = [results[n]['CV_R2'] for n in names]
bcolors = [PALETTE[n] for n in names]
 
# ── 子图1：R² 对比
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(range(3), r2s, color=bcolors, alpha=0.85,
               edgecolor='white', width=0.5)
ax1.set_xticks(range(3))
ax1.set_xticklabels(['Linear\nRegr.', 'Random\nForest', 'Gradient\nBoosting'], fontsize=9)
ax1.set_ylabel('R² Score');  ax1.set_ylim(0, 1.05)
ax1.set_title('Model R² Comparison (Test Set)', fontweight='bold')
for bar, v in zip(bars, r2s):
    ax1.text(bar.get_x() + bar.get_width()/2, v + 0.01,
             f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')
ax1.axhline(0.7, color='red', linestyle='--', linewidth=1,
            alpha=0.5, label='R²=0.7 baseline')
ax1.legend(fontsize=8);  ax1.set_facecolor('#FFFFFF')
 
# ── 子图2：RMSE 对比
ax2 = fig.add_subplot(gs[0, 1])
bars2 = ax2.bar(range(3), rmses, color=bcolors, alpha=0.85,
                edgecolor='white', width=0.5)
ax2.set_xticks(range(3))
ax2.set_xticklabels(['Linear\nRegr.', 'Random\nForest', 'Gradient\nBoosting'], fontsize=9)
ax2.set_ylabel('RMSE  (log₁₀ scale)')
ax2.set_title('RMSE Comparison (lower = better)', fontweight='bold')
for bar, v in zip(bars2, rmses):
    ax2.text(bar.get_x() + bar.get_width()/2, v + 0.001,
             f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')
ax2.set_facecolor('#FFFFFF')
 
# ── 子图3：5折交叉验证 R²
ax3 = fig.add_subplot(gs[0, 2])
bars3 = ax3.bar(range(3), cvs, color=bcolors, alpha=0.85,
                edgecolor='white', width=0.5)
ax3.set_xticks(range(3))
ax3.set_xticklabels(['Linear\nRegr.', 'Random\nForest', 'Gradient\nBoosting'], fontsize=9)
ax3.set_ylabel('CV R²  (5-fold)');  ax3.set_ylim(0, 1.05)
ax3.set_title('Cross-Validation R² (generalization)', fontweight='bold')
for bar, v in zip(bars3, cvs):
    ax3.text(bar.get_x() + bar.get_width()/2, v + 0.01,
             f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')
ax3.set_facecolor('#FFFFFF')
 
# ── 子图4：Random Forest 预测 vs 实际
ax4 = fig.add_subplot(gs[1, 0])
rf_pred = results['Random Forest']['y_pred']
ax4.scatter(y_test, rf_pred, alpha=0.3, s=8, color='#2E86AB')
lims = [min(y_test.min(), rf_pred.min()) - 0.05,
        max(y_test.max(), rf_pred.max()) + 0.05]
ax4.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')
ax4.set_xlabel('Actual log₁₀(Conductivity)')
ax4.set_ylabel('Predicted')
ax4.set_title(f'Random Forest: Predicted vs Actual\n'
              f'(R²={results["Random Forest"]["R2"]:.4f})', fontweight='bold')
ax4.legend(fontsize=9);  ax4.set_facecolor('#FFFFFF')
 
# ── 子图5：Gradient Boosting 预测 vs 实际
ax5 = fig.add_subplot(gs[1, 1])
gb_pred = results['Gradient Boosting']['y_pred']
ax5.scatter(y_test, gb_pred, alpha=0.3, s=8, color='#F18F01')
ax5.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')
ax5.set_xlabel('Actual log₁₀(Conductivity)')
ax5.set_ylabel('Predicted')
ax5.set_title(f'Gradient Boosting: Predicted vs Actual\n'
              f'(R²={results["Gradient Boosting"]["R2"]:.4f})', fontweight='bold')
ax5.legend(fontsize=9);  ax5.set_facecolor('#FFFFFF')
 
# ── 子图6：残差图（Random Forest）
# 残差 = 实际值 - 预测值，好的模型残差应随机分布在0附近
ax6 = fig.add_subplot(gs[1, 2])
residuals = y_test.values - rf_pred
ax6.scatter(rf_pred, residuals, alpha=0.3, s=8, color='#A23B72')
ax6.axhline(0, color='red', linestyle='--', linewidth=1.5)
ax6.set_xlabel('Predicted log₁₀(Conductivity)')
ax6.set_ylabel('Residual  (Actual − Predicted)')
ax6.set_title('Residual Plot — Random Forest\n'
              '(good model: points random around 0)', fontweight='bold')
ax6.set_facecolor('#FFFFFF')
 
# ── 子图7：特征重要性（Random Forest）
# 特征重要性 = 该特征在所有树中平均减少不纯度的贡献
ax7 = fig.add_subplot(gs[2, :2])
rf_model = results['Random Forest']['model']
feat_imp  = pd.Series(rf_model.feature_importances_,
                      index=FEATURES).sort_values()
fi_colors = ['#C73E1D' if v > 0.15 else
             '#2E86AB' if v > 0.05 else '#8B8BAE'
             for v in feat_imp]
ax7.barh(feat_imp.index, feat_imp.values,
         color=fi_colors, alpha=0.85, edgecolor='white')
ax7.set_xlabel('Feature Importance')
ax7.set_title('Random Forest — Feature Importance\n'
              '(red > 0.15 = high,  blue > 0.05 = medium,  gray = low)',
              fontweight='bold')
ax7.set_facecolor('#FFFFFF')
for i, (name, val) in enumerate(feat_imp.items()):
    ax7.text(val + 0.002, i, f'{val:.3f}', va='center', fontsize=9)
 
# ── 子图8：学习曲线
# 学习曲线：随样本量增加，train/val R² 的变化
# 解读：两条线收敛 = 正常；gap大 = 过拟合；val一直低 = 欠拟合
ax8 = fig.add_subplot(gs[2, 2])
train_sizes, train_scores, val_scores = learning_curve(
    RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    X, y, cv=3,
    train_sizes=np.linspace(0.1, 1.0, 8),
    scoring='r2', n_jobs=-1
)
ax8.plot(train_sizes, train_scores.mean(axis=1),
         color='#2E86AB', label='Train R²', linewidth=2)
ax8.fill_between(train_sizes,
                 train_scores.mean(axis=1) - train_scores.std(axis=1),
                 train_scores.mean(axis=1) + train_scores.std(axis=1),
                 alpha=0.2, color='#2E86AB')
ax8.plot(train_sizes, val_scores.mean(axis=1),
         color='#C73E1D', label='Val R²', linewidth=2)
ax8.fill_between(train_sizes,
                 val_scores.mean(axis=1) - val_scores.std(axis=1),
                 val_scores.mean(axis=1) + val_scores.std(axis=1),
                 alpha=0.2, color='#C73E1D')
ax8.set_xlabel('Training samples');  ax8.set_ylabel('R²')
ax8.set_title('Learning Curve — Random Forest\n'
              '(gap = overfitting indicator)', fontweight='bold')
ax8.legend(fontsize=9);  ax8.set_facecolor('#FFFFFF')
 
fig.suptitle(
    'Week 2 — Conductivity Prediction Models\n'
    '(Polymer Electrolyte Dataset, n=6,270)',
    fontsize=14, fontweight='bold', y=0.98
)
plt.savefig('week2_model_evaluation.png', dpi=150,
            bbox_inches='tight', facecolor='#F8F9FA')
plt.close()
print("图1 保存 → Part2. Model_Evaluation.png")
 
 
# ════════════════════════════════════════════════════════════════
# 4. PCA 降维分析
# ════════════════════════════════════════════════════════════════
# PCA 之前必须标准化（否则量纲大的特征会主导主成分）
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# 全维PCA → 看解释方差（碎石图）
pca_full = PCA()
pca_full.fit(X_scaled)
 
# 2维PCA → 可视化
pca2  = PCA(n_components=2)
X_pca = pca2.fit_transform(X_scaled)
 
evr   = pca_full.explained_variance_ratio_
cumev = np.cumsum(evr)
 
print("\n" + "=" * 45)
print("  PCA 结果")
print("=" * 45)
for i, (e, c) in enumerate(zip(evr, cumev)):
    print(f"  PC{i+1}: {e*100:5.1f}%  (累计 {c*100:5.1f}%)")
print(f"\n  前3个主成分解释 {cumev[2]*100:.1f}% 的方差")
print(f"  前5个主成分解释 {cumev[4]*100:.1f}% 的方差")
 
# ── 图2：PCA 分析（1×3）
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
fig2.patch.set_facecolor('#F8F9FA')
fig2.suptitle('PCA Analysis — Polymer Electrolyte Features',
              fontsize=13, fontweight='bold')
 
# ── 子图1：碎石图（Scree Plot）
ax = axes2[0]
ax.bar(range(1, len(evr)+1), evr*100,
       color='#2E86AB', alpha=0.8, label='Each PC')
ax.plot(range(1, len(evr)+1), cumev*100,
        'r-o', linewidth=2, markersize=5, label='Cumulative')
ax.axhline(80, color='gray', linestyle='--', linewidth=1, label='80% line')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Explained Variance (%)')
ax.set_title('Scree Plot\n(how much info each PC holds)', fontweight='bold')
ax.legend(fontsize=8);  ax.set_facecolor('#FFFFFF')
for i, v in enumerate(evr*100):
    ax.text(i+1, v+0.5, f'{v:.1f}%', ha='center', fontsize=8)
 
# ── 子图2：PC1 vs PC2，按电导率着色
ax = axes2[1]
sc = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                c=y, cmap='RdYlGn', alpha=0.4, s=8)
plt.colorbar(sc, ax=ax, label='log₁₀(Conductivity)', shrink=0.8)
ax.set_xlabel(f'PC1  ({evr[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2  ({evr[1]*100:.1f}% variance)')
ax.set_title('PCA Projection (by Conductivity)\n'
             '(similar color = similar conductivity)',
             fontweight='bold')
ax.set_facecolor('#FFFFFF')
 
# ── 子图3：PC1 vs PC2，按聚合度着色（主要DP组）
ax = axes2[2]
dp_vals_u = sorted(df['Degree of Polymerization'].unique())
cmap_dp   = plt.cm.get_cmap('tab10', len(dp_vals_u))
main_dps  = [14, 17, 19, 26, 31]
for dp in main_dps:
    mask = df['Degree of Polymerization'].values == dp
    idx  = dp_vals_u.index(dp)
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               color=cmap_dp(idx), alpha=0.5, s=10,
               label=f'DP={int(dp)}')
ax.set_xlabel(f'PC1  ({evr[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2  ({evr[1]*100:.1f}% variance)')
ax.set_title('PCA Projection (by Degree of Polymerization)\n'
             '(clusters = structurally similar groups)',
             fontweight='bold')
ax.legend(fontsize=8, markerscale=2);  ax.set_facecolor('#FFFFFF')
 
plt.tight_layout()
plt.savefig('week2_pca.png', dpi=150,
            bbox_inches='tight', facecolor='#F8F9FA')
plt.close()
print("图2 保存 → Part2. PCA.png")
 
print("\n✓ Week 2 项目全部完成。")
 
