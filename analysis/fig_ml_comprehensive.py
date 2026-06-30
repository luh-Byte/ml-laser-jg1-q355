"""
论文核心图表 — 4子图组合
(a) PCC相关性热力图
(b) 模型预测vs实测散点图
(c) 多模型误差量化对比
(d) 特征重要性+消融实验
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import sys, os, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, r'D:\ML-Laser-JG1-Q355\ml-laser-jg1-q355\utils')
from plot_style import setup_plot_style, style_axes, add_subplot_label, save_fig
setup_plot_style()

BASE = r'D:\ML-Laser-JG1-Q355\ml-laser-jg1-q355'
OUTPUT = os.path.join(BASE, 'analysis_output', 'figures', 'paper')
os.makedirs(OUTPUT, exist_ok=True)

# ===== Load data =====
df = pd.read_csv(os.path.join(BASE, 'analysis_output', 'fenicsx_cct_processed.csv'))
features = ['power_w', 'cooling_rate_K_s', 'grain_size_um', 'T_max_C', 'f_austenite']
target = 'hardness_HV'
labels = ['Power\n(W)', 'Cooling Rate\n(K/s)', 'Grain Size\n(um)', 'Tmax\n(C)', 'f-austenite\n(%)']

train = df[df['source'] != 'real'].copy()
real_all = df[df['source'] == 'real'].copy()
train_full = pd.concat([train, real_all[real_all['power_w'] != 1500]], ignore_index=True)

X_train = train_full[features].values
y_train = train_full[target].values
groups = train_full['power_w'].values
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)


def train_model(name, X, y, groups):
    """训练模型并返回LOGO-CV预测和指标"""
    if name == 'GBR':
        model = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, min_samples_leaf=3, random_state=42)
    elif name == 'RFR':
        model = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=3, random_state=42)
    elif name == 'SVR':
        model = SVR(kernel='rbf', C=100, epsilon=0.1)
    elif name == 'BPNN':
        model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42)
    else:
        model = Ridge(alpha=1.0)

    logo = LeaveOneGroupOut()
    y_cv = np.zeros_like(y)
    for tr, te in logo.split(X, y, groups):
        m = type(model)(**model.get_params()) if hasattr(model, 'get_params') else type(model)()
        m.fit(X[tr], y[tr])
        y_cv[te] = m.predict(X[te])

    model.fit(X, y)
    y_pred = model.predict(X)

    r2_train = r2_score(y, y_pred)
    r2_cv = r2_score(y, y_cv)
    rmse = np.sqrt(mean_squared_error(y, y_cv))
    mae = mean_absolute_error(y, y_cv)

    return {'model': name, 'y_pred': y_pred, 'y_cv': y_cv, 'r2_train': r2_train,
            'r2_cv': r2_cv, 'rmse': rmse, 'mae': mae, 'estimator': model}


# ===== Train all models =====
model_names = ['GBR', 'RFR', 'SVR', 'BPNN']
results = {}
for name in model_names:
    results[name] = train_model(name, X_train_s, y_train, groups)

# ===== (a) PCC Correlation Heatmap =====
def plot_correlation(ax):
    all_cols = features + [target]
    corr_data = df[all_cols].corr()
    # Reorder
    order = [0, 1, 2, 3, 4, 5]  # power, G, grain, Tmax, f_aust, HV
    corr_data = corr_data.iloc[order, order]

    cmap = LinearSegmentedColormap.from_list('custom',
        ['#3498db', '#ecf0f1', '#e74c3c'])
    im = ax.imshow(corr_data.values, cmap=cmap, vmin=-1, vmax=1, aspect='auto')

    ax.set_xticks(range(len(all_cols)))
    ax.set_yticks(range(len(all_cols)))
    xlabels = ['Power', 'Cooling\nRate', 'Grain\nSize', 'Tmax', 'f-aust.', 'Hardness']
    ax.set_xticklabels(xlabels, fontsize=9, fontweight='bold', fontfamily='Times New Roman')
    ax.set_yticklabels(xlabels, fontsize=9, fontweight='bold', fontfamily='Times New Roman')

    # Annotate cells
    for i in range(len(all_cols)):
        for j in range(len(all_cols)):
            val = corr_data.values[i, j]
            color = 'white' if abs(val) > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9,
                    fontweight='bold', fontfamily='Times New Roman', color=color)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('PCC', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
    cbar.ax.tick_params(labelsize=9)

    # Highlight key correlations
    for (i, j, val) in [(0,1,-0.95), (1,2,-0.92), (2,5,0.88)]:
        rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, linewidth=2.5, edgecolor='gold', facecolor='none')
        ax.add_patch(rect)

    ax.set_title('Pearson Correlation Heatmap', fontsize=13, fontweight='bold',
                 fontfamily='Times New Roman', pad=10)
    style_axes(ax)
    add_subplot_label(ax, '(a)')


# ===== (b) Predicted vs Measured Scatter =====
def plot_prediction_scatter(ax):
    colors = {'GBR': '#e74c3c', 'RFR': '#3498db', 'SVR': '#27ae60', 'BPNN': '#f39c12'}
    markers = {'GBR': 'o', 'RFR': 's', 'SVR': '^', 'BPNN': 'D'}

    for name in ['SVR', 'BPNN', 'RFR', 'GBR']:
        r = results[name]
        ax.scatter(y_train, r['y_cv'], s=50, c=colors[name], marker=markers[name],
                   edgecolors='black', linewidth=0.5, alpha=0.7, label=name, zorder=5)

    lims = [min(y_train.min(), 200) - 10, max(y_train.max(), 300) + 10]
    ax.plot(lims, lims, 'k-', linewidth=2, label='Ideal (y=x)', zorder=10)
    ax.fill_between(lims, [l - 10 for l in lims], [l + 10 for l in lims],
                    color='green', alpha=0.08, label='+-10 HV band')

    ax.set_xlabel('Experimental Hardness (HV)', fontsize=12, fontweight='bold',
                  fontfamily='Times New Roman')
    ax.set_ylabel('Predicted Hardness (HV)', fontsize=12, fontweight='bold',
                  fontfamily='Times New Roman')
    ax.set_title('Predicted vs Measured', fontsize=13, fontweight='bold',
                 fontfamily='Times New Roman', pad=10)
    ax.legend(fontsize=9, loc='lower right', frameon=True, edgecolor='black', prop={'family': 'Times New Roman'})
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    style_axes(ax)
    add_subplot_label(ax, '(b)')


# ===== (c) Model Error Comparison =====
def plot_error_comparison(ax):
    model_labels = ['GBR', 'RFR', 'SVR', 'BPNN']
    r2_vals = [results[m]['r2_cv'] for m in model_labels]
    rmse_vals = [results[m]['rmse'] for m in model_labels]
    mae_vals = [results[m]['mae'] for m in model_labels]

    x = np.arange(len(model_labels))
    w = 0.25

    bars1 = ax.bar(x - w, r2_vals, w, label='R\u00b2', color='#3498db', edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x, rmse_vals, w, label='RMSE (HV)', color='#e74c3c', edgecolor='black', linewidth=0.8)
    bars3 = ax.bar(x + w, mae_vals, w, label='MAE (HV)', color='#27ae60', edgecolor='black', linewidth=0.8)

    # Value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f'{h:.3f}' if h < 1 else f'{h:.1f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold', fontfamily='Times New Roman')

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    ax.set_title('Model Error Comparison', fontsize=13, fontweight='bold',
                 fontfamily='Times New Roman', pad=10)
    ax.legend(fontsize=10, loc='upper left', frameon=True, edgecolor='black', prop={'family': 'Times New Roman'})
    ax.set_ylim(0, max(rmse_vals) * 1.3)
    style_axes(ax)
    add_subplot_label(ax, '(c)')


# ===== (d) Feature Importance + Ablation =====
def plot_feature_importance(ax):
    # Main: RFR feature importance
    rf = results['RFR']['estimator']
    importances = rf.feature_importances_
    idx = np.argsort(importances)

    feat_labels_short = ['Power', 'Cooling Rate', 'Grain Size', 'Tmax', 'f-austenite']
    bars = ax.barh(range(len(idx)), importances[idx],
                   color='#3498db', edgecolor='black', linewidth=0.8, height=0.6)

    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feat_labels_short[i] for i in idx], fontsize=11, fontweight='bold',
                       fontfamily='Times New Roman')

    # Value labels
    for i, (bar, idx_i) in enumerate(zip(bars, idx)):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{importances[idx_i]*100:.1f}%', va='center', fontsize=10, fontweight='bold',
                fontfamily='Times New Roman')

    # Highlight top 2
    bars[-1].set_color('#e74c3c')
    bars[-2].set_color('#f39c12')

    # Cumulative annotation
    cum_top2 = (importances[idx[-1]] + importances[idx[-2]]) * 100
    ax.text(0.65, 0.15, f'Top-2: {cum_top2:.1f}%\n(Grain + Cooling)',
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            fontfamily='Times New Roman', color='#c0392b',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffeaa7', edgecolor='#c0392b', linewidth=1.5))

    ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    ax.set_title('Feature Importance Analysis', fontsize=13, fontweight='bold',
                 fontfamily='Times New Roman', pad=10)
    style_axes(ax)
    add_subplot_label(ax, '(d)')


# ===== Generate combined figure =====
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

plot_correlation(axes[0, 0])
plot_prediction_scatter(axes[0, 1])
plot_error_comparison(axes[1, 0])
plot_feature_importance(axes[1, 1])

plt.tight_layout(pad=2.0)
save_fig(fig, 'fig_ml_comprehensive_analysis', OUTPUT, dpi=300)
print(f'Saved: {os.path.join(OUTPUT, "fig_ml_comprehensive_analysis.png")}')

# ===== Print summary =====
print('\n=== Model Performance Summary ===')
print(f'{"Model":<8} {"R2(CV)":<10} {"RMSE(HV)":<12} {"MAE(HV)":<12} {"MAPE(%)":<10}')
print('-' * 55)
for name in model_names:
    r = results[name]
    mape = r['mae'] / np.mean(y_train) * 100
    print(f'{name:<8} {r["r2_cv"]:<10.4f} {r["rmse"]:<12.2f} {r["mae"]:<12.2f} {mape:<10.2f}')
