import pandas as pd, numpy as np
df = pd.read_csv('analysis_output/data_full.csv', encoding='utf-8-sig')
df['power_w'] = df['激光功率'].apply(lambda x: float(str(x).replace('W', '')))

feats = [
    '熔覆层组织面积占比(%)',
    '析出相/碳化物面积占比(%)',
    '气孔孔隙率(%)',
    '熔覆层平均晶粒尺寸(μm)',
    '基体稀释率(%)'
]

print('各功率组的金相特征均值:')
print()
for p in sorted(df['激光功率'].unique(), key=lambda x: int(x.replace('W',''))):
    g = df[df['激光功率']==p]
    parts = [p + ': 硬度=' + str(round(g['mh_mean_hv'].iloc[0],1)) + ' HV']
    for f in feats:
        name = f[:8]
        val = str(round(g[f].mean(),2)) + '±' + str(round(g[f].std(),2))
        parts.append(name + '=' + val)
    print('  '.join(parts))

print()
print('特征与硬度的皮尔逊相关系数:')
for f in feats:
    r = np.corrcoef(df[f], df['mh_mean_hv'])[0,1]
    print('  ' + f + ': r=' + str(round(r,3)))

print()
print('特征与功率的皮尔逊相关系数:')
for f in feats:
    r = np.corrcoef(df[f], df['power_w'])[0,1]
    print('  ' + f + ': r=' + str(round(r,3)))
