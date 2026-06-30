"""Ridge vs RFR vs GBR 对比"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

df = pd.read_csv(r'D:\ML-Laser-JG1-Q355\ml-laser-jg1-q355\analysis_output\data_full.csv', encoding='utf-8-sig')
groups = df['power_w'].values
y = df['mh_hv'].values
logo = LeaveOneGroupOut()

feature_sets = {
    'M1: power': ['power_w'],
    'M2: power+eis': ['power_w', 'eis_Rct_ohm'],
    'M3: power+eis+wear': ['power_w', 'eis_Rct_ohm', 'wear_friction_steady'],
    'M4: power+eis+xrd': ['power_w', 'eis_Rct_ohm', 'xrd_main_peak_intensity'],
    'M5: all': ['power_w', 'eis_Rct_ohm', 'wear_friction_steady',
                'xrd_main_peak_intensity', '熔覆层平均晶粒尺寸(μm)'],
}

models = {
    'Ridge': lambda: Ridge(alpha=1.0),
    'RFR': lambda: RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=2, random_state=42),
    'GBR': lambda: GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, min_samples_leaf=2, subsample=0.8, random_state=42),
}

header = "{:<12s} {:<25s} {:<8s} {:<8s} {:<8s}".format("Model", "Features", "R2", "RMSE", "MAE")
print("LOGO-CV Results (40 points, 4 groups)")
print()
print(header)
print("-" * 65)

best_r2 = -999
best_result = None

for model_name, model_fn in models.items():
    for fs_name, feats in feature_sets.items():
        valid = [f for f in feats if f in df.columns and df[f].notna().all()]
        if len(valid) != len(feats):
            continue
        X = df[valid].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        y_pred = np.zeros_like(y)
        for train_idx, test_idx in logo.split(X_scaled, y, groups):
            model = model_fn()
            model.fit(X_scaled[train_idx], y[train_idx])
            y_pred[test_idx] = model.predict(X_scaled[test_idx])

        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)

        if r2 > best_r2:
            best_r2 = r2
            best_result = {'model': model_name, 'feats': fs_name, 'r2': r2,
                          'rmse': rmse, 'mae': mae, 'y_pred': y_pred.copy()}

        print("{:<12s} {:<25s} {:<8.3f} {:<8.1f} {:<8.1f}".format(model_name, fs_name, r2, rmse, mae))

print()
print("Best: {} + {}  R2={:.3f}  RMSE={:.1f} HV".format(best_result["model"], best_result["feats"], best_result["r2"], best_result["rmse"]))

print()
print("Per-Power Summary (best model):")
for pw in [900, 1200, 1500, 1800]:
    mask = df['power_w'].values == pw
    m_hv = y[mask].mean()
    p_hv = best_result['y_pred'][mask].mean()
    res = m_hv - p_hv
    print("  {}W: measured={:.1f}  predicted={:.1f}  residual={:+.1f}".format(pw, m_hv, p_hv, res))
