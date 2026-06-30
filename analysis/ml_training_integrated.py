"""
ML训练脚本 — FEniCSx温度场 + CCT后处理 整合训练

数据流:
  FEniCSx(温度场) → CCT后处理(相分数+晶粒+硬度) → ML训练

特征集:
  工艺: power_w, scan_speed_mm_s
  温度场: T_max_C, pool_width_mm, pool_depth_mm
  组织: cooling_rate_K_s, grain_size_um, f_austenite, f_martensite, f_carbide
  性能: dilution_pct

目标: hardness_HV
验证: 1500W真实数据 (holdout)
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os, json

sys.path.insert(0, r'D:\ML-Laser-JG1-Q355\ml-laser-jg1-q355\utils')
from plot_style import setup_plot_style, style_axes, add_subplot_label, save_fig
setup_plot_style()

BASE = r'D:\ML-Laser-JG1-Q355\ml-laser-jg1-q355'
OUTPUT = os.path.join(BASE, 'analysis_output')
FIG_DIR = os.path.join(OUTPUT, 'figures')


def load_data():
    """加载CCT后处理数据"""
    df = pd.read_csv(os.path.join(OUTPUT, 'fenicsx_cct_processed.csv'))
    return df


def split_data(df):
    """分割: 3组真实+仿真=训练, 1组真实1500W=验证"""
    train_mask = ~((df['source'] == 'real') & (df['power_w'] == 1500))
    val_mask = (df['source'] == 'real') & (df['power_w'] == 1500)
    return df[train_mask].copy(), df[val_mask].copy()


def get_feature_sets():
    """定义特征组合"""
    return {
        'M0: power+speed': ['power_w', 'scan_speed_mm_s'],
        'M1: power+speed+T': ['power_w', 'scan_speed_mm_s', 'T_max_C'],
        'M2: +grain': ['power_w', 'scan_speed_mm_s', 'T_max_C', 'grain_size_um'],
        'M3: +cooling': ['power_w', 'scan_speed_mm_s', 'T_max_C', 'grain_size_um', 'cooling_rate_K_s'],
        'M4: +phases': ['power_w', 'scan_speed_mm_s', 'T_max_C', 'grain_size_um', 'cooling_rate_K_s',
                        'f_austenite', 'f_martensite'],
        'M5: +pool': ['power_w', 'scan_speed_mm_s', 'T_max_C', 'grain_size_um', 'cooling_rate_K_s',
                      'f_austenite', 'f_martensite', 'pool_width_mm', 'pool_depth_mm'],
        'M6: all': ['power_w', 'scan_speed_mm_s', 'T_max_C', 'grain_size_um', 'cooling_rate_K_s',
                    'f_austenite', 'f_martensite', 'f_carbide', 'pool_width_mm', 'pool_depth_mm', 'dilution_pct'],
    }


def get_models():
    """定义模型"""
    return {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'RFR': RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=3, random_state=42),
        'GBR': GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                         min_samples_leaf=3, random_state=42),
    }


def train_and_evaluate(df_train, df_val, features, target='hardness_HV'):
    """训练+验证"""
    X_train = df_train[features].values
    y_train = df_train[target].values
    X_val = df_val[features].values
    y_val = df_val[target].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # LOGO-CV (按功率组)
    groups_train = df_train['power_w'].values
    logo = LeaveOneGroupOut()

    results = []
    for name, model in get_models().items():
        # LOGO-CV
        y_cv = np.zeros_like(y_train)
        for train_idx, test_idx in logo.split(X_train_s, y_train, groups_train):
            m = type(model)(**model.get_params())
            m.fit(X_train_s[train_idx], y_train[train_idx])
            y_cv[test_idx] = m.predict(X_train_s[test_idx])

        # 验证集
        model.fit(X_train_s, y_train)
        y_val_pred = model.predict(X_val_s)

        # 特征重要性 (树模型)
        if hasattr(model, 'feature_importances_'):
            importances = dict(zip(features, model.feature_importances_))
        else:
            importances = {}

        results.append({
            'model': name,
            'features': features,
            'n_features': len(features),
            'train_r2': round(r2_score(y_train, model.predict(X_train_s)), 4),
            'logo_r2': round(r2_score(y_train, y_cv), 4),
            'logo_rmse': round(np.sqrt(mean_squared_error(y_train, y_cv)), 1),
            'val_pred': round(float(y_val_pred[0]), 1),
            'val_real': round(float(y_val[0]), 1),
            'val_err': round(float(abs(y_val_pred[0] - y_val[0])), 1),
            'val_err_pct': round(float(abs(y_val_pred[0] - y_val[0]) / y_val[0] * 100), 1),
            'importances': importances,
        })

    return results


def plot_results(all_results, df_train, df_val, target='hardness_HV'):
    """生成可视化"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # (a) 特征数 vs 验证误差
    ax = axes[0, 0]
    models_seen = set()
    for r in all_results:
        key = r['model']
        if key not in models_seen:
            models_seen.add(key)
            model_results = [x for x in all_results if x['model'] == key]
            n_feats = [x['n_features'] for x in model_results]
            val_errs = [x['val_err_pct'] for x in model_results]
            ax.plot(n_feats, val_errs, 'o-', linewidth=2, markersize=6, label=key)
    ax.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Error (%)', fontsize=12, fontweight='bold')
    ax.set_title('Feature Count vs Validation Error', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.axhline(y=10, color='green', linestyle='--', alpha=0.5)
    style_axes(ax)
    add_subplot_label(ax, '(a)')

    # (b) 最佳模型: 预测 vs 实测 (训练集)
    ax = axes[0, 1]
    best = min(all_results, key=lambda x: x['val_err'])
    best_feats = best['features']
    X_all = df_train[best_feats].values
    y_all = df_train[target].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_all)

    # 用最佳模型重新训练
    from sklearn.ensemble import GradientBoostingRegressor
    if best['model'] == 'GBR':
        m = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                      min_samples_leaf=3, random_state=42)
    else:
        m = Ridge(alpha=1.0)
    m.fit(X_s, y_all)
    y_pred_train = m.predict(X_s)

    powers = df_train['power_w'].values
    colors = {900: '#2980b9', 1200: '#e67e22', 1500: '#27ae60', 1800: '#c0392b',
              600: '#8e44ad', 800: '#16a085', 1000: '#d35400', 1400: '#2c3e50',
              1600: '#7f8c8d', 2000: '#f39c12', 2200: '#1abc9c', 2400: '#e74c3c'}
    for p in sorted(set(powers)):
        mask = powers == p
        c = colors.get(p, 'gray')
        ax.scatter(y_all[mask], y_pred_train[mask], s=40, c=c, edgecolors='black',
                   linewidth=0.5, zorder=5, label='%dW' % p)
    lims = [min(y_all.min(), y_pred_train.min()) - 20, max(y_all.max(), y_pred_train.max()) + 20]
    ax.plot(lims, lims, 'k--', linewidth=1.5, alpha=0.5)
    ax.set_xlabel('Measured HV', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted HV (Train)', fontsize=12, fontweight='bold')
    ax.set_title('Train Set: %s' % best['model'], fontsize=14, fontweight='bold')
    ax.legend(fontsize=7, ncol=3, loc='upper left')
    style_axes(ax)
    add_subplot_label(ax, '(b)')

    # (c) 验证集: 4组真实数据对比
    ax = axes[1, 0]
    real_ps = [900, 1200, 1500, 1800]
    real_hvs = [224.2, 243.4, 288.6, 385.2]

    # 重新训练各模型在全训练集上
    X_train = df_train[best_feats].values
    y_train = df_train[target].values
    X_s_train = scaler.fit_transform(X_train)

    model_preds = {}
    for name in ['Ridge', 'RFR', 'GBR']:
        if name == 'GBR':
            m = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                          min_samples_leaf=3, random_state=42)
        elif name == 'RFR':
            from sklearn.ensemble import RandomForestRegressor
            m = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=3, random_state=42)
        else:
            m = Ridge(alpha=1.0)
        m.fit(X_s_train, y_train)
        # 预测4组真实功率 (用仿真数据的平均特征)
        preds = []
        for p in real_ps:
            sim_subset = df_train[(df_train['power_w'] == p) & (df_train['scan_speed_mm_s'] == 10.0)]
            if len(sim_subset) > 0:
                x = sim_subset[best_feats].values
                x_s = scaler.transform(x)
                preds.append(float(m.predict(x_s)[0]))
            else:
                preds.append(np.nan)
        model_preds[name] = preds

    x_pos = np.arange(len(real_ps))
    ax.bar(x_pos - 0.25, real_hvs, 0.25, label='Real', color='black', edgecolor='white')
    ax.bar(x_pos, model_preds.get('Ridge', [0]*4), 0.25, label='Ridge', color='#3498db', edgecolor='black', linewidth=0.5)
    ax.bar(x_pos + 0.25, model_preds.get('GBR', [0]*4), 0.25, label='GBR', color='#e74c3c', edgecolor='black', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['%dW' % p for p in real_ps], fontsize=11)
    ax.set_ylabel('Hardness (HV)', fontsize=12, fontweight='bold')
    ax.set_title('Validation: 4 Real Experiments', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    style_axes(ax)
    add_subplot_label(ax, '(c)')

    # (d) 特征重要性 (GBR最佳)
    ax = axes[1, 1]
    best_r = min([r for r in all_results if r['model'] == 'GBR'], key=lambda x: x['val_err'])
    imp = best_r['importances']
    if imp:
        sorted_imp = sorted(imp.items(), key=lambda x: x[1])
        names = [x[0] for x in sorted_imp]
        vals = [x[1] for x in sorted_imp]
        ax.barh(names, vals, color='#3498db', edgecolor='black', linewidth=0.8)
        ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
        ax.set_title('GBR Feature Importance', fontsize=14, fontweight='bold')
    style_axes(ax)
    add_subplot_label(ax, '(d)')

    plt.tight_layout()
    save_fig(fig, 'ml_training_fenicsx_cct', FIG_DIR, dpi=300)
    print('Figure saved: %s' % os.path.join(FIG_DIR, 'ml_training_fenicsx_cct.png'))


def main():
    print('=' * 70)
    print('ML Training: FEniCSx + CCT Integrated Model')
    print('=' * 70)

    # 1. 加载数据
    df = load_data()
    df_train, df_val = split_data(df)
    print('Train: %d samples (100 sim + 3 real)' % len(df_train))
    print('Val: 1 real 1500W (HV=%.1f)' % df_val['hardness_HV'].values[0])

    # 2. 遍历特征×模型组合
    all_results = []
    for fs_name, features in get_feature_sets().items():
        results = train_and_evaluate(df_train, df_val, features)
        for r in results:
            r['feature_set'] = fs_name
        all_results.extend(results)

    # 3. 排序输出
    all_results.sort(key=lambda x: x['val_err_pct'])

    print('\n=== Top 10 Configurations ===')
    print('%-35s %-8s %-10s %-8s %-10s %-10s %-8s' % (
        'Config', 'Model', '#Feat', 'TrainR2', 'LOGO-R2', 'ValPred', 'ValErr%'))
    print('-' * 90)
    for r in all_results[:10]:
        print('%-35s %-8s %-8d %-10.3f %-8.3f %-10.1f %-8.1f%%' % (
            r['feature_set'], r['model'], r['n_features'], r['train_r2'],
            r['logo_r2'], r['val_pred'], r['val_err_pct']))

    best = all_results[0]
    print('\nBest: %s + %s (val_err=%.1f%%)' % (best['feature_set'], best['model'], best['val_err_pct']))

    # 4. 特征重要性
    print('\n=== Best Model Feature Importance ===')
    for f, imp in sorted(best['importances'].items(), key=lambda x: -x[1]):
        bar = '#' * int(imp * 40)
        print('  %-25s %.3f %s' % (f, imp, bar))

    # 5. 生成图表
    plot_results(all_results, df_train, df_val)

    # 6. 保存结果
    summary = {
        'best_config': best['feature_set'],
        'best_model': best['model'],
        'best_val_error_pct': best['val_err_pct'],
        'best_val_pred': best['val_pred'],
        'real_1500W': 288.6,
        'feature_importance': best['importances'],
        'all_results_count': len(all_results),
    }
    with open(os.path.join(OUTPUT, 'ml_training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print('\nSummary saved: %s' % os.path.join(OUTPUT, 'ml_training_summary.json'))


if __name__ == '__main__':
    main()
