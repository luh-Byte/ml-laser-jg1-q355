"""
数据格式统一脚本
统一data文件夹中原始数据的文件名和格式
"""

import os
import shutil
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def unify_wear_data():
    """统一磨损数据文件名格式"""
    print("\n=== 统一磨损数据格式 ===")
    wear_dir = os.path.join(DATA_DIR, "wear-data")
    
    files = os.listdir(wear_dir)
    renamed = []
    
    for f in files:
        # 统一为大写W格式
        if f.endswith('.txt'):
            new_name = f.replace('w.txt', 'W.txt')
            if new_name != f:
                old_path = os.path.join(wear_dir, f)
                new_path = os.path.join(wear_dir, new_name)
                os.rename(old_path, new_path)
                renamed.append(f"  {f} -> {new_name}")
    
    if renamed:
        print("  重命名文件:")
        for r in renamed:
            print(r)
    else:
        print("  文件名已统一")
    
    # 验证
    files = os.listdir(wear_dir)
    print(f"  当前文件: {files}")


def unify_eis_data():
    """统一EIS数据格式，只保留拟合数据"""
    print("\n=== 统一EIS数据格式 ===")
    eis_dir = os.path.join(DATA_DIR, "electrochemical-impedance")
    
    files = os.listdir(eis_dir)
    
    # 删除Q335相关文件
    q335_files = [f for f in files if 'Q335' in f or 'q335' in f]
    for f in q335_files:
        fpath = os.path.join(eis_dir, f)
        os.remove(fpath)
        print(f"  删除: {f}")
    
    # 删除非必要文件（保留拟合数据和原始数据）
    keep_patterns = ['EIS-拟合数据', 'EIS-原始数据', '.xlsx']
    remove_files = []
    
    for f in files:
        if 'Q335' in f or 'q335' in f:
            continue
        if not any(p in f for p in keep_patterns):
            # 检查是否是简写版EIS文件（如900-EIS.txt）
            if f.endswith('.txt') and '-EIS.' not in f and '拟合' not in f and '原始' not in f:
                remove_files.append(f)
    
    for f in remove_files:
        fpath = os.path.join(eis_dir, f)
        if os.path.exists(fpath):
            os.remove(fpath)
            print(f"  删除冗余: {f}")
    
    # 统一文件名：拟合数据改为标准格式
    files = os.listdir(eis_dir)
    for f in files:
        if '拟合数据' in f:
            # 提取功率
            power = f.split('-')[0]
            new_name = f"{power}-EIS-拟合数据.txt"
            if f != new_name:
                old_path = os.path.join(eis_dir, f)
                new_path = os.path.join(eis_dir, new_name)
                if os.path.exists(old_path) and not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    print(f"  统一: {f} -> {new_name}")
    
    # 显示最终文件列表
    files = os.listdir(eis_dir)
    print(f"\n  最终文件列表:")
    for f in sorted(files):
        size = os.path.getsize(os.path.join(eis_dir, f))
        print(f"    {f} ({size:,} bytes)")


def unify_xrd_data():
    """统一XRD数据格式"""
    print("\n=== 统一XRD数据格式 ===")
    xrd_dir = os.path.join(DATA_DIR, "xrd-data")
    
    files = os.listdir(xrd_dir)
    print(f"  当前文件: {files}")
    
    # XRD数据格式已经比较统一，只需检查
    for f in files:
        fpath = os.path.join(xrd_dir, f)
        if f.endswith('.txt'):
            with open(fpath, 'r', encoding='utf-8', errors='ignore') as file:
                first_line = file.readline().strip()
                print(f"  {f}: 首行='{first_line}'")


def check_hardness_data():
    """检查显微硬度数据格式"""
    print("\n=== 检查显微硬度数据 ===")
    from docx import Document
    
    mh_dir = os.path.join(DATA_DIR, "microhardness-data")
    files = os.listdir(mh_dir)
    
    for f in files:
        fpath = os.path.join(mh_dir, f)
        doc = Document(fpath)
        
        # 统计硬度值数量
        hv_count = 0
        for p in doc.paragraphs:
            if 'HV=' in p.text:
                hv_count += 1
        
        status = "OK" if hv_count == 10 else f"WARN: {hv_count}个点"
        print(f"  {f}: {status}")


def create_unified_readme():
    """创建统一格式说明"""
    readme_path = os.path.join(DATA_DIR, "README_DATA_FORMAT.md")
    
    content = """# 原始数据格式说明 (已统一)

## 文件命名规范
- 功率单位: W (大写)
- 文件格式: .txt (文本), .docx (Word), .xlsx (Excel), .tif (图像)

---

## 1. 显微硬度数据 (microhardness-data/)

**文件**: 900W.docx, 1200W.docx, 1500W.docx, 1800W.docx

**格式**: Word文档，每文件10个测量点
- 位置01-05: 熔覆层区域 (从熔覆层顶部到底部)
- 位置06-10: 基体区域 (从界面到基体深处)
- 数据格式: `【XX】  D1= xx.xx  D2= xx.xx  HV= xxx.x`

---

## 2. 磨损数据 (wear-data/)

**文件**: 900W.txt, 1200W.txt, 1500W.txt, 1800W.txt

**格式**: 
- 前28行: 元数据
- 第29行起: CSV数据 (时间,摩擦系数,累积距离,0)
- 编码: GBK

---

## 3. 电化学阻抗数据 (electrochemical-impedance/)

**文件**: 
- {功率}-EIS-拟合数据.txt - 主要使用的拟合数据
- {功率}-EIS-原始数据.txt - 原始测量数据
- {功率}w.xlsx - Excel格式完整数据

**EIS格式**: 制表符分隔，3列 (频率, Z', Z'')

---

## 4. XRD数据 (xrd-data/)

**文件**: 1-900.txt, 2-1200.txt, 3-1500.txt, 4-1800.txt

**格式**: 
- 第1行: 样品名称
- 第3行起: 空格分隔 (2θ角度, 强度)

---

## 5. 工艺参数 (paper_data/)

**关键参数**:
- 激光功率: 900W, 1200W, 1500W, 1800W
- 扫描速度: 10 mm/s = 600 mm/min
- 送粉速率: 10 g/min
"""
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"\n[OK] 格式说明已更新: {readme_path}")


def main():
    print("=" * 60)
    print("数据格式统一脚本")
    print("=" * 60)
    
    unify_wear_data()
    unify_eis_data()
    unify_xrd_data()
    check_hardness_data()
    create_unified_readme()
    
    print("\n" + "=" * 60)
    print("格式统一完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
