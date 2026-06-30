"""Ridge/RF/GBR对比 - 1500W真实数据作为验证集"""
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, r'D:\ML-Laser-JG1-Q355\ml-laser-jg1-q355\utils')
from plot_style import setup_plot_style, style_axes, add_subplot_label, save_fig
setup_plot_style()

BASE = r'D:\ML-Laser-JG1-Q355\ml-laser-jg1-q355'
OUTPUT = os.path.join(BASE, 'analysis_output')
FIG_DIR = os.path.join(OUTPUT, 'figures')

# 1. Load data
df = pd.read_csv(os.path.join(OUTPUT, 'fenicsx_training.csv'))
features = ['power_w', 'scan_speed_mm_s', 'T_max_C', 'pool_width_mm', 'pool_depth_mm']
target = 'hardness_HV'

# 2. Split: 1500W real as validation, rest as training
train = df[df['source'] != 'real'].copy()
real_all = df[df['source'] == 'real'].copy()
val = real_all[real_all['power_w'] == 1500].copy()
train_real = real_all[real_all['power_w'] != 1500].copy()
train_full = pd.concat([train, train_real], ignore_index=True)

print("=== Dataset ===")
print("Train: %d (100 sim + 3 real)" % len(train_full))
print("Val: 1 real 1500W (HV=%.1f)" % val[target].values[0])

X_train = train_full[features].values
y_train = train_full[target].values
X_val = val[features].values
y_val = val[target].values

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)

# 3. Models
models = {
    'Ridge': Ridge(alpha=1.0),
    'RFR': RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=3, random_state=42),
    'GBR': GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, min_samples_leaf=3, random_state=42),
}

print("\n=== Model Comparison ===")
print("%-10s %-10s %-10s %-10s %-10s %-8s" % ("Model", "TrainR2", "ValPred", "RealHV", "Error", "Err%"))
print("-" * 60)

results = []
best_name, best_model, best_err = None, None, 999

for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred_train = model.predict(X_train_s)
    y_pred_val = model.predict(X_val_s)

    train_r2 = r2_score(y_train, y_pred_train)
    err = abs(y_pred_val[0] - y_val[0])
    err_pct = err / y_val[0] * 100

    print("%-10s %-10.3f %-10.1f %-10.1f %-10.1f %-8.1f%%" % (name, train_r2, y_pred_val[0], y_val[0], err, err_pct))
    results.append({'name': name, 'train_r2': train_r2, 'val_pred': y_pred_val[0], 'err': err, 'err_pct': err_pct})

    if err < best_err:
        best_err = err
        best_name = name
        best_model = model

print("\nBest: %s (error=%.1f HV)" % (best_name, best_err))

# 4. Feature importance (RF)
rf = models['RFR']
print("\n=== Feature Importance (RFR) ===")
for f, imp in sorted(zip(features, rf.feature_importances_), key=lambda x: -x[1]):
    bar = '#' * int(imp * 40)
    print("  %-25s %.3f %s" % (f, imp, bar))

# 5. Full prediction across power range at v=10mm/s
print("\n=== Predictions at v=10mm/s ===")
power_range = np.arange(600, 2500, 100)
speed_fixed = 10.0
# Interpolate T_max and pool from FEniCSx data for unseen powers
t_data = train_full.groupby('power_w')['T_max_C'].mean()
pool_w_data = train_full.groupby('power_w')['pool_width_mm'].mean()
pool_d_data = train_full.groupby('power_w')['pool_depth_mm'].mean()

pred_data = []
for p in power_range:
    t_est = np.interp(p, t_data.index, t_data.values)
    pw_est = np.interp(p, pool_w_data.index, pool_w_data.values)
    pd_est = np.interp(p, pool_d_data.index, pool_d_data.values)
    x = np.array([[p, speed_fixed, t_est, pw_est, pd_est]])
    x_s = scaler.transform(x)
    pred_ridge = models['Ridge'].predict(x_s)[0]
    pred_rf = models['RFR'].predict(x_s)[0]
    pred_gbr = models['GBR'].predict(x_s)[0]
    pred_data.append({'power': int(p), 'Ridge': pred_ridge, 'RFR': pred_rf, 'GBR': pred_gbr})

pred_df = pd.DataFrame(pred_data)

# 6. Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# (a) Model comparison at 1500W
ax = axes[0]
names = [r['name'] for r in results]
errs = [r['err'] for r in results]
colors = ['#27ae60' if e < 30 else '#f39c12' if e < 60 else '#e74c3c' for e in errs]
bars = ax.bar(names, errs, color=colors, edgecolor='black', linewidth=0.8)
for bar, e in zip(bars, errs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            '%.1f HV' % e, ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('Validation Error (HV)', fontsize=12, fontweight='bold')
ax.set_title('1500W Holdout Validation', fontsize=14, fontweight='bold')
ax.axhline(y=20, color='green', linestyle='--', alpha=0.5, label='Good threshold')
ax.legend(fontsize=10)
style_axes(ax)
add_subplot_label(ax, '(a)')

# (b) Power-Hardness curves
ax = axes[1]
ax.plot(pred_df['power'], pred_df['Ridge'], 'b-o', linewidth=2, markersize=4, label='Ridge')
ax.plot(pred_df['power'], pred_df['RFR'], 'r-s', linewidth=2, markersize=4, label='RFR')
ax.plot(pred_df['power'], pred_df['GBR'], 'g-^', linewidth=2, markersize=4, label='GBR')
# Real data points
real_ps = real_all['power_w'].values
real_hvs = real_all['hardness_HV'].values
ax.scatter(real_ps, real_hvs, s=150, c='black', edgecolors='gold', linewidth=2, zorder=5, label='Real (anchor)')
# Highlight held-out 1500W
ax.scatter([1500], [val[target].values[0]], s=200, c='red', edgecolors='white', linewidth=2, zorder=6, marker='*', label='Validation (1500W)')
ax.set_xlabel('Power (W)', fontsize=12, fontweight='bold')
ax.set_ylabel('Hardness (HV)', fontsize=12, fontweight='bold')
ax.set_title('Predicted Hardness vs Power', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
style_axes(ax)
add_subplot_label(ax, '(b)')

# (c) Feature importance
ax = axes[2]
feat_names = features
importances = models['RFR'].feature_importances_
idx_sorted = np.argsort(importances)
ax.barh([feat_names[i] for i in idx_sorted], importances[idx_sorted], color='#3498db', edgecolor='black', linewidth=0.8)
ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
ax.set_title('RFR Feature Importance', fontsize=14, fontweight='bold')
style_axes(ax)
add_subplot_label(ax, '(c)')

plt.tight_layout()
save_fig(fig, 'ml_comparison_1500w_holdout', FIG_DIR, dpi=300)
print("\nFigure saved: %s" % os.path.join(FIG_DIR, 'ml_comparison_1500w_holdout.png'))
