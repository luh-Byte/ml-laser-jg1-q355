"""重建data_full.csv + 跑Ridge回归预测硬度"""
import pandas as pd
import numpy as np
import os, re
from docx import Document
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = r'D:\ML-Laser-JG1-Q355\ml-laser-jg1-q355'
OUTPUT = os.path.join(BASE, 'analysis_output')
FIG_DIR = os.path.join(OUTPUT, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

import sys
sys.path.insert(0, os.path.join(BASE, 'utils'))
from plot_style import setup_plot_style, style_axes, add_subplot_label, save_fig
setup_plot_style()

# ===== 1. Build data_full.csv =====
metal_csv = os.path.join(OUTPUT, '金相定量表征数据汇总.csv')
df_metal = pd.read_csv(metal_csv, encoding='utf-8-sig')
df_metal['power_w'] = df_metal.apply(lambda row: float(str(row['激光功率']).replace('W', '')), axis=1)
metal_feats = ['熔覆层组织面积占比(%)', '析出相/碳化物面积占比(%)', '气孔缺陷面积占比(%)',
               '熔覆层平均晶粒尺寸(μm)', '基体稀释率(%)']
metal_avg = df_metal.groupby('power_w')[metal_feats].mean().reset_index()

eis = {900: (48.1, 1114.4, 1192.0, -39.1), 1200: (45.8, 2647.2, 2698.0, -45.2),
       1500: (46.9, 4464.4, 4678.3, -56.2), 1800: (48.0, 2941.0, 3010.0, -52.1)}
xrd = {900: (44.67, 4972, 51520), 1200: (44.75, 4745, 49800),
       1500: (43.71, 3495, 38500), 1800: (43.62, 4058, 42100)}
wear = {900: (0.2567, 0.0143), 1200: (0.2095, 0.0068),
        1500: (0.3003, 0.0281), 1800: (0.1044, 0.0211)}

rows = []
for pw in [900, 1200, 1500, 1800]:
    doc = Document(os.path.join(BASE, 'data', 'microhardness-data', f'{pw}W.docx'))
    hvs = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if 'HV=' in text:
            parts = text.split('HV=')
            if len(parts) > 1:
                try:
                    hvs.append(float(parts[1].strip()))
                except:
                    pass
    m = metal_avg[metal_avg['power_w'] == pw]
    for i, hv in enumerate(hvs):
        pos = i + 1
        region = 'cladding' if pos <= 5 else 'substrate'
        row = {'power_w': pw, 'position': pos, 'region': region, 'mh_hv': hv}
        for f in metal_feats:
            row[f] = m[f].values[0] if len(m) > 0 else np.nan
        row['eis_Rs_ohm'] = eis[pw][0]
        row['eis_Rct_ohm'] = eis[pw][1]
        row['eis_Z_max_ohm'] = eis[pw][2]
        row['eis_theta_min_deg'] = eis[pw][3]
        row['xrd_main_peak_2theta'] = xrd[pw][0]
        row['xrd_main_peak_intensity'] = xrd[pw][1]
        row['xrd_peak_44_area'] = xrd[pw][2]
        row['wear_friction_steady'] = wear[pw][0]
        row['wear_friction_std'] = wear[pw][1]
        rows.append(row)

df = pd.DataFrame(rows)
csv_path = os.path.join(OUTPUT, 'data_full.csv')
df.to_csv(csv_path, index=False, encoding='utf-8-sig')

# ===== 2. Ridge Regression =====
print('=== Ridge Regression: Predict Hardness ===')
print(f'Data: {len(df)} points, 4 power groups')
print(f'Hardness range: [{df["mh_hv"].min():.1f}, {df["mh_hv"].max():.1f}]')
print()

# Feature sets to test
feature_sets = {
    'M1: power only': ['power_w'],
    'M2: power + eis_Rct': ['power_w', 'eis_Rct_ohm'],
    'M3: power + wear': ['power_w', 'wear_friction_steady'],
    'M4: power + grain': ['power_w', '熔覆层平均晶粒尺寸(μm)'],
    'M5: power + eis + wear': ['power_w', 'eis_Rct_ohm', 'wear_friction_steady'],
    'M6: power + eis + xrd': ['power_w', 'eis_Rct_ohm', 'xrd_main_peak_intensity'],
    'M7: all useful': ['power_w', 'eis_Rct_ohm', 'wear_friction_steady',
                       'xrd_main_peak_intensity', '熔覆层平均晶粒尺寸(μm)'],
}

groups = df['power_w'].values
y = df['mh_hv'].values
logo = LeaveOneGroupOut()

results = []
print(f'{"Model":<35s} {"R2":<10s} {"RMSE":<10s} {"MAE":<10s} {"Verdict"}')
print('-' * 80)

for name, feats in feature_sets.items():
    valid = [f for f in feats if f in df.columns and df[f].notna().all()]
    if len(valid) != len(feats):
        print(f'{name:<35s} SKIP (missing features)')
        continue

    X = df[valid].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y_pred = np.zeros_like(y)
    for train_idx, test_idx in logo.split(X_scaled, y, groups):
        model = Ridge(alpha=1.0)
        model.fit(X_scaled[train_idx], y[train_idx])
        y_pred[test_idx] = model.predict(X_scaled[test_idx])

    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)

    if r2 > 0.5:
        verdict = 'GOOD'
    elif r2 > 0.2:
        verdict = 'usable'
    elif r2 > 0:
        verdict = 'weak'
    else:
        verdict = 'fail'

    results.append({'name': name, 'feats': valid, 'r2': r2, 'rmse': rmse, 'mae': mae,
                    'y_pred': y_pred, 'verdict': verdict})
    print(f'{name:<35s} {r2:<10.3f} {rmse:<10.1f} {mae:<10.1f} {verdict}')

# ===== 3. Plot best result =====
best = max(results, key=lambda r: r['r2'])
print(f'\nBest model: {best["name"]} (R2={best["r2"]:.3f})')

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
labels_map = {'(a)': 0, '(b)': 1, '(c)': 2}

# (a) All models R2 comparison
ax = axes[0]
names = [r['name'].split(': ')[1] for r in results]
r2s = [r['r2'] for r in results]
colors = ['#27ae60' if r > 0.3 else '#f39c12' if r > 0 else '#e74c3c' for r in r2s]
bars = ax.barh(names, r2s, color=colors, edgecolor='black', linewidth=0.8)
ax.axvline(x=0, color='black', linewidth=1, linestyle='--')
for bar, val in zip(bars, r2s):
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=10, fontweight='bold')
ax.set_xlabel('LOGO-CV R2', fontsize=12, fontweight='bold')
ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
ax.invert_yaxis()
style_axes(ax)
add_subplot_label(ax, '(a)')

# (b) Best model: predicted vs actual
ax = axes[1]
power_colors = {900: '#2980b9', 1200: '#e67e22', 1500: '#27ae60', 1800: '#c0392b'}
for pw in [900, 1200, 1500, 1800]:
    mask = df['power_w'].values == pw
    ax.scatter(y[mask], best['y_pred'][mask], s=80, c=power_colors[pw],
               edgecolors='black', linewidth=0.8, zorder=5, label=f'{pw}W')
lims = [min(y.min(), best['y_pred'].min()) - 20, max(y.max(), best['y_pred'].max()) + 20]
ax.plot(lims, lims, 'k--', linewidth=1.5, alpha=0.5, label='y=x')
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel('Measured HV', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted HV (LOGO-CV)', fontsize=12, fontweight='bold')
ax.set_title(f'{best["name"]}  R2={best["r2"]:.3f}', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
style_axes(ax)
add_subplot_label(ax, '(b)')

# (c) Residual distribution
ax = axes[2]
residuals = y - best['y_pred']
ax.hist(residuals, bins=15, color='#3498db', edgecolor='black', linewidth=0.8, alpha=0.8)
ax.axvline(x=0, color='red', linewidth=2, linestyle='--')
ax.axvline(x=residuals.mean(), color='black', linewidth=2, linestyle='-',
           label=f'Mean={residuals.mean():.1f}')
ax.set_xlabel('Residual (HV)', fontsize=12, fontweight='bold')
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_title('Residual Distribution', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
style_axes(ax)
add_subplot_label(ax, '(c)')

plt.tight_layout()
save_fig(fig, 'ridge_regression_results', FIG_DIR, dpi=300)
plt.close()

# ===== 4. Per-power summary =====
print()
print('=== Per-Power Prediction Summary ===')
print(f'{"Power":<8s} {"Measured":<12s} {"Predicted":<12s} {"Residual":<12s}')
print('-' * 50)
for pw in [900, 1200, 1500, 1800]:
    mask = df['power_w'].values == pw
    m_hv = y[mask].mean()
    p_hv = best['y_pred'][mask].mean()
    res = m_hv - p_hv
    print(f'{pw}W     {m_hv:<12.1f} {p_hv:<12.1f} {res:<+12.1f}')

print()
print('Done.')
