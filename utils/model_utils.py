"""
模型工具类 — 整合自 random_forest_template 的优秀部分
封装训练/评估/预测/保存/加载/对比
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

MODEL_REGISTRY = {
    'Ridge': lambda: Ridge(alpha=1.0),
    'RFR': lambda: RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=3, random_state=42),
    'GBR': lambda: GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, min_samples_leaf=3, random_state=42),
    'SVR': lambda: SVR(kernel='rbf', C=100, epsilon=0.1),
    'BPNN': lambda: MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42),
}


class HardnessModel:
    """硬度预测模型封装类"""

    def __init__(self, model_name='GBR', random_state=42):
        self.model_name = model_name
        self.random_state = random_state
        self.model = MODEL_REGISTRY[model_name]()
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        self.train_metrics = None

    def train(self, X, y, feature_names=None, test_size=0.2):
        self.feature_names = feature_names or [f'f_{i}' for i in range(X.shape[1])]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state)
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)
        self.model.fit(X_train_s, y_train)
        self.is_fitted = True

        y_pred_tr = self.model.predict(X_train_s)
        y_pred_te = self.model.predict(X_test_s)
        self.train_metrics = {
            'train_r2': r2_score(y_train, y_pred_tr),
            'test_r2': r2_score(y_test, y_pred_te),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_te)),
            'mae': mean_absolute_error(y_test, y_pred_te),
            'mape': np.mean(np.abs((y_test - y_pred_te) / y_test)) * 100,
            'n_train': len(y_train), 'n_test': len(y_test),
        }
        return self.train_metrics

    def logo_cv(self, X, y, groups):
        logo = LeaveOneGroupOut()
        y_cv = np.zeros_like(y)
        for tr, te in logo.split(X, y, groups):
            m = MODEL_REGISTRY[self.model_name]()
            X_tr = self.scaler.fit_transform(X[tr])
            X_te = self.scaler.transform(X[te])
            m.fit(X_tr, y[tr])
            y_cv[te] = m.predict(X_te)
        self.is_fitted = True
        return {'r2': r2_score(y, y_cv), 'rmse': np.sqrt(mean_squared_error(y, y_cv)),
                'mae': mean_absolute_error(y, y_cv), 'y_cv': y_cv}

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        return self.model.predict(self.scaler.transform(X))

    def predict_with_ci(self, X, confidence=0.95):
        y_pred = self.predict(X)
        z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)
        rmse = self.train_metrics['rmse']
        return y_pred, y_pred - z * rmse, y_pred + z * rmse

    def feature_importance(self):
        if not hasattr(self.model, 'feature_importances_'):
            return None
        return pd.DataFrame({'Feature': self.feature_names,
                             'Importance': self.model.feature_importances_}).sort_values('Importance', ascending=False)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return {'r2': r2_score(y, y_pred), 'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'mae': mean_absolute_error(y, y_pred),
                'mape': np.mean(np.abs((y - y_pred) / y)) * 100, 'y_pred': y_pred}

    def save(self, filepath):
        joblib.dump({'model': self.model, 'scaler': self.scaler,
                     'model_name': self.model_name, 'feature_names': self.feature_names,
                     'train_metrics': self.train_metrics}, filepath)

    def load(self, filepath):
        data = joblib.load(filepath)
        self.model, self.scaler, self.model_name = data['model'], data['scaler'], data['model_name']
        self.feature_names, self.train_metrics = data['feature_names'], data['train_metrics']
        self.is_fitted = True


def compare_models(X, y, groups=None, model_names=None):
    if model_names is None:
        model_names = list(MODEL_REGISTRY.keys())
    results = []
    for name in model_names:
        m = HardnessModel(name)
        m.train(X, y)
        metrics = m.train_metrics
        if groups is not None:
            logo = m.logo_cv(X, y, groups)
            metrics['logo_r2'], metrics['logo_rmse'] = logo['r2'], logo['rmse']
        results.append({k: v for k, v in metrics.items() if k not in ('n_train', 'n_test')})
        results[-1]['model'] = name
    return pd.DataFrame(results).sort_values('test_r2', ascending=False)
