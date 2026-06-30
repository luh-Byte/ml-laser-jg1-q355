"""
统一命令行入口脚本
按顺序执行四个阶段：数据整合 → 响应面建模 → 性能建模 → 优化
"""

import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEPS = [
    ("阶段1: 数据整合", os.path.join("pipeline", "data_integration.py")),
    ("阶段2: 功率→组织响应面", os.path.join("pipeline", "power_response_model.py")),
    ("阶段3: 组织→性能模型", os.path.join("pipeline", "property_model.py")),
    ("阶段4: 链式优化", os.path.join("pipeline", "optimized_pipeline.py")),
]

def run_pipeline():
    print("=" * 80)
    print("JG-1铁基合金激光熔覆参数优化 - 完整流程")
    print("=" * 80)
    
    for i, (name, script) in enumerate(STEPS, 1):
        print(f"\n[{i}/4] {name}")
        print("-" * 60)
        
        script_path = os.path.join(BASE_DIR, script)
        if not os.path.exists(script_path):
            print(f"  [ERROR] 脚本文件不存在: {script}")
            return False
        
        result = subprocess.run([sys.executable, script_path],
                                capture_output=True, text=True, cwd=BASE_DIR,
                                encoding="utf-8", errors="replace")

        if result.stdout:
            print(result.stdout.strip())
        if result.stderr:
            print(f"  [WARN] {result.stderr.strip()}")
        
        if result.returncode == 0:
            print(f"  [OK] {name} 完成")
        else:
            print(f"  [ERROR] {name} 失败 (退出码: {result.returncode})")
            return False
    
    print("\n" + "=" * 80)
    print("全部4个阶段完成!")
    print("输出目录: analysis_output/")
    print("=" * 80)
    return True

def run_single_stage(stage_index):
    if stage_index < 1 or stage_index > len(STEPS):
        print(f"  [ERROR] 无效的阶段索引: {stage_index} (有效范围: 1-4)")
        return False
    
    name, script = STEPS[stage_index - 1]
    print(f"\n执行: {name}")
    print("-" * 60)
    
    script_path = os.path.join(BASE_DIR, script)
    if not os.path.exists(script_path):
        print(f"  [ERROR] 脚本文件不存在: {script}")
        return False
    
    result = subprocess.run([sys.executable, script_path],
                            capture_output=True, text=True, cwd=BASE_DIR,
                            encoding="utf-8", errors="replace")

    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(f"  [WARN] {result.stderr.strip()}")
    
    if result.returncode == 0:
        print(f"  [OK] {name} 完成")
        return True
    else:
        print(f"  [ERROR] {name} 失败 (退出码: {result.returncode})")
        return False

def show_help():
    print("=" * 80)
    print("JG-1铁基合金激光熔覆参数优化 - 命令行工具")
    print("=" * 80)
    print("\n用法:")
    print("  python -m pipeline              # 执行完整四阶段流程")
    print("  python -m pipeline --stage N    # 执行单个阶段 (N=1,2,3,4)")
    print("  python -m pipeline --help       # 显示帮助信息")
    print("\n阶段说明:")
    print("  1 - 数据整合: 整合显微硬度、磨损、EIS、XRD实测数据")
    print("  2 - 响应面建模: 建立功率→组织GPR响应面")
    print("  3 - 性能建模: 建立组织→硬度GBR模型")
    print("  4 - 链式优化: 多目标Pareto优化")

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
                sys.exit(0 if success else 1)
        except ValueError:
            pass
        print("  [ERROR] 无效的阶段参数")
        show_help()
        sys.exit(1)
    
    success = run_pipeline()
    sys.exit(0 if success else 1)