"""
缓存清理与结果管理模块
提供图表缓存、模型缓存、报告数据的清理和重新生成功能
"""

import os
import shutil
import glob
from datetime import datetime

# 基础路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_output")
CACHE_DIR = os.path.join(OUTPUT_DIR, "model_cache")
MODEL_CACHE_FILE = os.path.join(CACHE_DIR, "trained_models.joblib")
DATA_HASH_CACHE_FILE = os.path.join(CACHE_DIR, "data_hash.txt")

# 结果输出子文件夹
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results_by_date")
CACHE_CLEAN_MODULES = [
    ("模型缓存", MODEL_CACHE_FILE),
    ("数据哈希缓存", DATA_HASH_CACHE_FILE),
]

def get_all_cache_items():
    """获取所有缓存项及其大小"""
    cache_items = []
    total_size = 0
    
    # 检查各缓存项
    for name, path in CACHE_CLEAN_MODULES:
        if os.path.exists(path):
            size = os.path.getsize(path)
            cache_items.append({
                'name': name,
                'path': path,
                'size': size,
                'type': 'file'
            })
            total_size += size
    
    # 检查cache目录中的其他文件
    if os.path.exists(CACHE_DIR):
        for f in os.listdir(CACHE_DIR):
            fpath = os.path.join(CACHE_DIR, f)
            if os.path.isfile(fpath) and f not in [os.path.basename(MODEL_CACHE_FILE), os.path.basename(DATA_HASH_CACHE_FILE)]:
                size = os.path.getsize(fpath)
                cache_items.append({
                    'name': f'缓存文件: {f}',
                    'path': fpath,
                    'size': size,
                    'type': 'file'
                })
                total_size += size
    
    return cache_items, total_size

def get_output_items():
    """获取所有输出文件及其大小"""
    output_items = []
    total_size = 0
    
    if os.path.exists(OUTPUT_DIR):
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for f in files:
                fpath = os.path.join(root, f)
                rel_path = os.path.relpath(fpath, OUTPUT_DIR)
                size = os.path.getsize(fpath)
                output_items.append({
                    'name': rel_path,
                    'path': fpath,
                    'size': size,
                    'modified': datetime.fromtimestamp(os.path.getmtime(fpath)).strftime('%Y-%m-%d %H:%M')
                })
                total_size += size
    
    return output_items, total_size

def format_size(size_bytes):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"

def clean_cache_item(path):
    """清理单个缓存项"""
    try:
        if os.path.exists(path):
            os.remove(path)
            return True, "已删除"
        return False, "文件不存在"
    except Exception as e:
        return False, str(e)

def clean_all_cache():
    """清理所有缓存"""
    results = []
    for name, path in CACHE_CLEAN_MODULES:
        success, msg = clean_cache_item(path)
        results.append((name, success, msg))
    
    # 清理cache目录中的其他文件
    if os.path.exists(CACHE_DIR):
        for f in os.listdir(CACHE_DIR):
            fpath = os.path.join(CACHE_DIR, f)
            if os.path.isfile(fpath):
                success, msg = clean_cache_item(fpath)
                results.append((f, success, msg))
    
    return results

def clean_output_items(patterns=None):
    """清理指定的输出文件
    
    Args:
        patterns: 文件路径模式列表，如 ['*.csv', '*.png']
    
    Note:
        会跳过results_by_date目录，避免误删备份结果
    """
    results = []
    
    if patterns is None:
        # 清理所有输出文件
        patterns = ['*.csv', '*.png', '*.txt', '*.xlsx', '*.jpg']
    
    for pattern in patterns:
        for fpath in glob.glob(os.path.join(OUTPUT_DIR, '**', pattern), recursive=True):
            # 跳过results_by_date目录，避免误删备份结果
            if 'results_by_date' in fpath:
                continue
            # 跳过主数据文件，防止误删
            basename = os.path.basename(fpath)
            if '金相定量表征数据汇总' in basename:
                continue
            
            try:
                os.remove(fpath)
                rel_path = os.path.relpath(fpath, OUTPUT_DIR)
                results.append((rel_path, True, "已删除"))
            except Exception as e:
                results.append((fpath, False, str(e)))
    
    return results

def create_results_folder():
    """创建带时间戳的结果文件夹"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"results_{timestamp}"
    folder_path = os.path.join(RESULTS_DIR, folder_name)
    
    try:
        os.makedirs(folder_path, exist_ok=True)
        
        # 创建子文件夹结构
        subfolders = [
            'quantitative_data',
            'charts',
            'reports',
            'models',
            'thermodynamics'
        ]
        for subfolder in subfolders:
            os.makedirs(os.path.join(folder_path, subfolder), exist_ok=True)
        
        return True, folder_path
    except Exception as e:
        return False, str(e)

def backup_current_results():
    """备份当前结果 — 只保留最新一次备份"""
    # 先清理旧备份
    if os.path.exists(RESULTS_DIR):
        for item in os.listdir(RESULTS_DIR):
            item_path = os.path.join(RESULTS_DIR, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path, ignore_errors=True)

    success, folder_path = create_results_folder()

    if not success:
        return success, folder_path

    # 复制输出文件到新文件夹
    files_copied = 0
    try:
        if os.path.exists(OUTPUT_DIR):
            for root, dirs, files in os.walk(OUTPUT_DIR):
                # 跳过results_by_date目录
                if 'results_by_date' in root:
                    continue

                for f in files:
                    src = os.path.join(root, f)
                    rel_path = os.path.relpath(src, OUTPUT_DIR)
                    dst = os.path.join(folder_path, rel_path)

                    # 创建目标目录
                    os.makedirs(os.path.dirname(dst), exist_ok=True)

                    # 复制文件
                    shutil.copy2(src, dst)
                    files_copied += 1

        return True, f"已备份到: {folder_path} ({files_copied} 个文件)"
    except Exception as e:
        return False, str(e)

def get_statistics():
    """获取缓存和输出的统计信息"""
    cache_items, cache_size = get_all_cache_items()
    output_items, output_size = get_output_items()
    
    return {
        'cache_count': len(cache_items),
        'cache_size': cache_size,
        'cache_size_formatted': format_size(cache_size),
        'output_count': len(output_items),
        'output_size': output_size,
        'output_size_formatted': format_size(output_size),
        'results_folder_exists': os.path.exists(RESULTS_DIR),
        'cache_items': cache_items,
        'output_items': output_items
    }

def save_to_results_subfolder(subfolder_name, file_name, data, data_type='csv'):
    """保存数据到results子文件夹
    
    Args:
        subfolder_name: 子文件夹名称 (如 'quantitative_data', 'charts', 'reports')
        file_name: 文件名
        data: 数据 (DataFrame for csv, or bytes for images)
        data_type: 'csv', 'png', 'txt'
    """
    target_dir = os.path.join(RESULTS_DIR, 'current', subfolder_name)
    os.makedirs(target_dir, exist_ok=True)
    
    file_path = os.path.join(target_dir, file_name)
    
    try:
        if data_type == 'csv':
            data.to_csv(file_path, index=False, encoding='utf-8-sig')
        elif data_type == 'png':
            data.savefig(file_path, dpi=150, bbox_inches='tight')
        elif data_type == 'txt':
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(data)
        
        return True, file_path
    except Exception as e:
        return False, str(e)

if __name__ == "__main__":
    print("=" * 80)
    print("缓存清理与结果管理模块 - 诊断工具")
    print("=" * 80)
    
    stats = get_statistics()
    
    print(f"\n缓存统计:")
    print(f"  缓存项数量: {stats['cache_count']}")
    print(f"  缓存总大小: {stats['cache_size_formatted']}")
    
    print(f"\n输出文件统计:")
    print(f"  输出文件数量: {stats['output_count']}")
    print(f"  输出总大小: {stats['output_size_formatted']}")
    
    print("\n缓存项详情:")
    for item in stats['cache_items']:
        print(f"  - {item['name']}: {format_size(item['size'])}")
    
    print("\n输出文件详情 (前10个):")
    for item in stats['output_items'][:10]:
        print(f"  - {item['name']}: {format_size(item['size'])} ({item['modified']})")
    
    if len(stats['output_items']) > 10:
        print(f"  ... 还有 {len(stats['output_items']) - 10} 个文件")