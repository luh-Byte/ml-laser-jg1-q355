"""
交互式硬度预测工具
整合自 random_forest_template 的交互式预测 + 特征响应可视化
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.plot_style import setup_plot_style, style_axes, add_subplot_label, save_fig
from utils.model_utils import HardnessModel
setup_plot_style()

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(BASE, 'analysis_output')
FIG_DIR = os.path.join(OUTPUT, 'figures')

FEATURE_NAMES = ['Power (W)', 'Speed (mm/s)', 'Tmax (C)', 'Grain (um)',
                 'Cooling Rate (K/s)', 'f-austenite', 'f-martensite']
FEATURE_KEYS = ['power_w', 'scan_speed_mm_s', 'T_max_C', 'grain_size_um',
                'cooling_rate_K_s', 'f_austenite', 'f_martensite']


def plot_feature_response(model, data, input_params=None, save_path=None, fig_title=None):
    """2x2特征响应曲线"""
    X = data[FEATURE_KEYS].values
    y = data['hardness_HV'].values
    n = len(y)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    labels = ['(a)', '(b)', '(c)', '(d)']
    sorts = [np.arange(n), np.argsort(y), np.argsort(X[:, 0]), np.argsort(X[:, 3])]
    titles = ['By Sample', 'By True Value', 'By Power', 'By Grain Size']
    for idx in range(4):
        ax = axes.flat[idx]
        si = sorts[idx]
        ax.scatter(range(n), y[si], marker='^', color='red', s=60, label='True', zorder=3)
        if input_params is not None:
            pred = model.predict(np.array([input_params]))[0]
            ax.scatter(n, pred, marker='*', color='gold', s=200, edgecolors='black',
                       linewidth=1.5, zorder=5, label='Prediction')
        ax.set_xlabel('Samples', fontsize=12, fontfamily='Times New Roman', fontweight='bold')
        ax.set_ylabel('Hardness (HV)', fontsize=12, fontfamily='Times New Roman', fontweight='bold')
        ax.set_title(f'{labels[idx]} {titles[idx]}', fontsize=13, fontweight='bold',
                     fontfamily='Times New Roman', loc='left')
        ax.legend(loc='lower right', fontsize=10, prop={'family': 'Times New Roman'})
        ax.set_xlim(-1, n + 1)
        style_axes(ax)
        add_subplot_label(ax, labels[idx])
    if fig_title:
        fig.suptitle(fig_title, fontsize=15, fontweight='bold', fontfamily='Times New Roman', y=1.01)
    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path, FIG_DIR, dpi=300)
    plt.close()


def interactive_predict(model, df):
    """交互式预测循环"""
    print("\n" + "=" * 70)
    print("  Hardness Predictor")
    print("=" * 70)
    defaults = [1500, 15, 2000, 20, 100, 0.5, 0.2]
    while True:
        print("\nEnter parameters (Enter=default, q=quit):")
        values = []
        for i, name in enumerate(FEATURE_NAMES):
            user_input = input(f"  {name} [{defaults[i]}]: ").strip()
            if user_input.lower() == 'q':
                print("Bye!")
                return
            values.append(float(user_input) if user_input else float(defaults[i]))
        pred, ci_low, ci_high = model.predict_with_ci(np.array([values]))
        print(f"\n  Prediction: {pred[0]:.1f} HV")
        print(f"  95% CI: [{ci_low[0]:.1f}, {ci_high[0]:.1f}] HV")
        hv_min, hv_max = df['hardness_HV'].min(), df['hardness_HV'].max()
        in_range = hv_min <= pred[0] <= hv_max
        print(f"  Range: [{hv_min:.1f}, {hv_max:.1f}] -> {'IN RANGE' if in_range else 'EXTRAPOLATION'}")
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        plot_feature_response(model, df, input_params=values,
                              save_path=f'predictor_response_{timestamp}',
                              fig_title='Hardness Prediction Response')


def main():
    df = pd.read_csv(os.path.join(OUTPUT, 'fenicsx_cct_processed.csv'))
    X = df[FEATURE_KEYS].values
    y = df['hardness_HV'].values

    model = HardnessModel('GBR')
    m = model.train(X, y, feature_names=FEATURE_NAMES)
    print(f"Model: R2(train)={m['train_r2']:.4f} R2(test)={m['test_r2']:.4f}")
    print(f"RMSE={m['rmse']:.2f} HV  MAE={m['mae']:.2f} HV  MAPE={m['mape']:.2f}%")

    imp = model.feature_importance()
    print("\nFeature Importance:")
    for _, row in imp.iterrows():
        bar = '#' * int(row['Importance'] * 40)
        print(f"  {row['Feature']:<25s} {row['Importance']:.3f} {bar}")

    model.save(os.path.join(OUTPUT, 'hardness_model.joblib'))
    print(f"\nModel saved: {os.path.join(OUTPUT, 'hardness_model.joblib')}")

    interactive_predict(model, df)


if __name__ == '__main__':
    main()
