"""
激光熔覆ML优化Pipeline - 命令行入口
"""

from .run_pipeline import run_pipeline, run_single_stage, show_help

if __name__ == "__main__":
    import sys
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
