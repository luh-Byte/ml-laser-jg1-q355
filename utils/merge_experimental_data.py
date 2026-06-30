import os
import re
import math
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAVE_DIR = os.path.join(BASE_DIR, "analysis_output")
os.makedirs(SAVE_DIR, exist_ok=True)


def parse_eis(txt_path):
    """解析简单的 EIS 数值文件，返回高频实部、低频实部、估计 Rct、低频 |Z| 与相位（rad）。"""
    try:
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [l.strip() for l in f if l.strip()]
    except Exception:
        return None

    # 找到首个包含三列数字的行索引
    data_lines = []
    for l in lines:
        parts = re.split(r"\s+|\t|,", l)
        parts = [p for p in parts if p != '']
        if len(parts) >= 3:
            # 判定是否为数字行
            try:
                float(parts[0])
                float(parts[1])
                float(parts[2])
                data_lines.append(parts)
            except Exception:
                continue

    if not data_lines:
        return None

    # 取第一行为高频，最后一行为低频
    first = data_lines[0]
    last = data_lines[-1]
    try:
        zf_hf = float(first[1])
        zf_lf = float(last[1])
        zf_lf_im = float(last[2])
    except Exception:
        return None

    rct_est = zf_lf - zf_hf
    abs_lf = math.hypot(zf_lf, zf_lf_im)
    phase_lf = math.atan2(zf_lf_im, zf_lf)
    return {
        'EIS_R_high': zf_hf,
        'EIS_R_low': zf_lf,
        'EIS_Rct_est': rct_est,
        'EIS_low_abs': abs_lf,
        'EIS_low_phase_rad': phase_lf
    }


def parse_wear(txt_path):
    try:
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception:
        return None

    # 提取逗号分割的数值行（时间, 值, ...）
    values = []
    for l in content.splitlines():
        if ',' in l:
            parts = [p.strip() for p in l.split(',')]
            try:
                v = float(parts[1])
                values.append(v)
            except Exception:
                continue

    if not values:
        # 退化：提取行内单个浮点数作为最终磨损
        nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", content)
        if nums:
            try:
                return {'wear_last': float(nums[-1]), 'wear_mean': float(np.mean([float(x) for x in nums]))}
            except Exception:
                return None
        return None

    return {'wear_last': values[-1], 'wear_mean': float(np.mean(values))}


def parse_xrd(txt_path):
    try:
        data = np.loadtxt(txt_path, dtype=float, comments='?', encoding='utf-8')
    except Exception:
        # 退化读取
        try:
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                rows = []
                for l in f:
                    parts = re.split(r"\s+|,", l.strip())
                    if len(parts) >= 2:
                        try:
                            a = float(parts[0]); b = float(parts[1])
                            rows.append((a, b))
                        except Exception:
                            continue
            data = np.array(rows)
        except Exception:
            return None

    if data.size == 0:
        return None
    if data.ndim == 1:
        return None
    angles = data[:, 0]
    ints = data[:, 1]
    idx = np.nanargmax(ints)
    return {'xrd_peak_angle': float(angles[idx]), 'xrd_peak_intensity': float(ints[idx])}


def parse_docx_hardness(docx_path):
    try:
        from docx import Document
    except Exception:
        return None
    try:
        doc = Document(docx_path)
    except Exception:
        return None
    nums = []
    for p in doc.paragraphs:
        nums += re.findall(r"[-+]?[0-9]*\.?[0-9]+", p.text)
    for tbl in doc.tables:
        for row in tbl.rows:
            for cell in row.cells:
                nums += re.findall(r"[-+]?[0-9]*\.?[0-9]+", cell.text)
    if not nums:
        return None
    vals = [float(x) for x in nums]
    return {'mh_mean': float(np.mean(vals)), 'mh_std': float(np.std(vals)), 'mh_count': len(vals)}


def main():
    src_csv = os.path.join(SAVE_DIR, "金相定量表征数据汇总.csv")
    if not os.path.exists(src_csv):
        print("错误: 找不到汇总 CSV：", src_csv)
        return

    df = pd.read_csv(src_csv)
    powers = sorted(df['激光功率'].unique())
    print("Found power groups:", powers)

    # prepare columns
    add_cols = ['EIS_R_high','EIS_R_low','EIS_Rct_est','EIS_low_abs','EIS_low_phase_rad',
                'wear_last','wear_mean','mh_mean','mh_std','mh_count','xrd_peak_angle','xrd_peak_intensity']
    for c in add_cols:
        df[c] = np.nan

    # scan and parse per power
    for p in powers:
        key = str(p)
        power_root = key.replace('W','') if key.endswith('W') else key
        print('Processing', key)

        # EIS
        eis_dir = os.path.join(DATA_DIR, 'electrochemical-impedance')
        eis_match = None
        for fn in os.listdir(eis_dir):
            if fn.lower().startswith(power_root.lower()):
                if 'eis' in fn.lower() or 'eis' in fn.lower():
                    eis_match = os.path.join(eis_dir, fn)
                    break
        if eis_match and os.path.exists(eis_match):
            res = parse_eis(eis_match)
            if res:
                for k,v in res.items():
                    df.loc[df['激光功率']==key, k] = v

        # 磨损
        wear_dir = os.path.join(DATA_DIR, 'wear-data')
        wear_match = None
        for fn in os.listdir(wear_dir):
            if power_root in fn.lower():
                wear_match = os.path.join(wear_dir, fn)
                break
        if wear_match and os.path.exists(wear_match):
            res = parse_wear(wear_match)
            if res:
                for k,v in res.items():
                    df.loc[df['激光功率']==key, k] = v

        # 显微硬度
        mh_dir = os.path.join(DATA_DIR, 'microhardness-data')
        mh_match = None
        if os.path.isdir(mh_dir):
            for fn in os.listdir(mh_dir):
                if power_root in fn.lower():
                    mh_match = os.path.join(mh_dir, fn)
                    break
        if mh_match and os.path.exists(mh_match):
            res = parse_docx_hardness(mh_match)
            if res:
                for k,v in res.items():
                    df.loc[df['激光功率']==key, {'mh_mean':'mh_mean','mh_std':'mh_std','mh_count':'mh_count'}[k] if k in ['mh_mean','mh_std','mh_count'] else k] = v

        # XRD
        xrd_dir = os.path.join(DATA_DIR, 'xrd-data')
        xrd_match = None
        for fn in os.listdir(xrd_dir):
            if power_root in fn.lower():
                xrd_match = os.path.join(xrd_dir, fn)
                break
        if xrd_match and os.path.exists(xrd_match):
            res = parse_xrd(xrd_match)
            if res:
                for k,v in res.items():
                    df.loc[df['激光功率']==key, k] = v

    out_csv = os.path.join(SAVE_DIR, '金相定量表征数据汇总_enriched.csv')
    df.to_csv(out_csv, index=False)
    print('Saved enriched csv ->', out_csv)


if __name__ == '__main__':
    main()
