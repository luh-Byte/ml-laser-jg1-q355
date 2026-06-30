"""分析硬度梯度数据并做SVM分类"""
import pandas as pd
import numpy as np
from docx import Document
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score

data_dir = r'D:\ML-Laser-JG1-Q355\ml-laser-jg1-q355\data\microhardness-data'

all_data = []
for power in ['900W', '1200W', '1500W', '1800W']:
    doc = Document(os.path.join(data_dir, f'{power}.docx'))
    pw = int(power.replace('W', ''))
    for para in doc.paragraphs:
        text = para.text.strip()
        if 'HV=' in text:
            parts = text.split('HV=')
            if len(parts) > 1:
                try:
                    hv = float(parts[1].strip())
                    # extract position number from the pattern like (01) or 01
                    import re
                    m = re.search(r'(\d{2})', text)
                    pos = int(m.group(1)) if m else 0
                    region = 'cladding' if pos <= 5 else 'substrate'
                    all_data.append({'power': pw, 'position': pos, 'region': region, 'hv': hv})
                except:
                    pass

df = pd.DataFrame(all_data)
print(f'Total data points: {len(df)}')
print()

# Per-power summary
for pw in sorted(df['power'].unique()):
    sub = df[df['power'] == pw]
    clad = sub[sub['region'] == 'cladding']
    subr = sub[sub['region'] == 'substrate']
    c_mean = clad['hv'].mean()
    c_std = clad['hv'].std()
    s_mean = subr['hv'].mean()
    s_std = subr['hv'].std()
    print(f'  {pw}W: clad={c_mean:.1f}+/-{c_std:.1f}  sub={s_mean:.1f}+/-{s_std:.1f}  range=[{sub["hv"].min():.0f}-{sub["hv"].max():.0f}]')

# Hardness class distribution
bins = [0, 200, 300, 400, 600]
labels_b = ['low', 'mid', 'high', 'vhigh']
df['hv_class'] = pd.cut(df['hv'], bins=bins, labels=labels_b)
print()
print('=== Hardness Class Distribution ===')
ct = df.groupby(['power', 'hv_class']).size().unstack(fill_value=0)
print(ct)

# SVM Classification
print()
print('=== SVM 1: predict hardness class from power ===')
X1 = df[['power']].values
y1 = df['hv_class'].cat.codes.values
scaler1 = StandardScaler()
X1s = scaler1.fit_transform(X1)
loo = LeaveOneOut()

for kernel in ['linear', 'rbf']:
    model = SVC(kernel=kernel, C=1.0, gamma='scale', random_state=42)
    y_pred = cross_val_predict(model, X1s, y1, cv=loo)
    acc = accuracy_score(y1, y_pred)
    print(f'  {kernel:<8s} LOO-CV Accuracy = {acc:.1%}')

print()
print('=== SVM 2: predict power from hardness ===')
X2 = df[['hv']].values
y2 = df['power'].values
scaler2 = StandardScaler()
X2s = scaler2.fit_transform(X2)

for kernel in ['linear', 'rbf']:
    model = SVC(kernel=kernel, C=1.0, gamma='scale', random_state=42)
    y_pred = cross_val_predict(model, X2s, y2, cv=loo)
    acc = accuracy_score(y2, y_pred)
    print(f'  {kernel:<8s} LOO-CV Accuracy = {acc:.1%}')

print()
print('=== SVM 3: predict region (cladding/substrate) from power ===')
X3 = df[['power']].values
y3 = (df['region'] == 'cladding').astype(int).values
scaler3 = StandardScaler()
X3s = scaler3.fit_transform(X3)

for kernel in ['linear', 'rbf']:
    model = SVC(kernel=kernel, C=1.0, gamma='scale', random_state=42)
    y_pred = cross_val_predict(model, X3s, y3, cv=loo)
    acc = accuracy_score(y3, y_pred)
    print(f'  {kernel:<8s} LOO-CV Accuracy = {acc:.1%}')
