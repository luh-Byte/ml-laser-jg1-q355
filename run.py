"""
JG-1铁基合金激光熔覆参数优化 - 根目录入口脚本
调用 pipeline/run_pipeline.py 执行完整四阶段流程
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from pipeline.run_pipeline import run_pipeline, show_help, run_single_stage

def _try_backup():
    try:
        from utils.cache_manager import backup_current_results
        success, msg = backup_current_results()
        if success:
            print(f"\n[备份] {msg}")
    except Exception as e:
        print(f"\n[备份] 自动备份失败: {e}")

if __name__ == "__main__":
    args = sys.argv[1:]
    
    if "--help" in args or "-h" in args:
        show_help()
        sys.exit(0)
    
    if "--stage" in args:
        try:
            idx = args.index("--stage")
            if idx + 1 < len(args):
                stage_index = int(args[idx + 1])
                success = run_single_stage(stage_index)
                if success:
                    _try_backup()
                sys.exit(0 if success else 1)
        except ValueError:
            pass
        print("  [ERROR] 无效的阶段参数")
        show_help()
        sys.exit(1)
    
    success = run_pipeline()
    if success:
        _try_backup()
    sys.exit(0 if success else 1)