import os
import shutil
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

desktop_path = Path(os.path.expanduser("~/Desktop"))
summary_folder = desktop_path / "ML-Laser-JG1-Q355-图表汇总"
summary_folder.mkdir(parents=True, exist_ok=True)

# ==================== 主工程 ====================
main_proj = summary_folder / "主工程_ml-laser-jg1-q355"
main_figs = main_proj / "图表"
main_data = main_proj / "数据"
main_figs.mkdir(parents=True, exist_ok=True)
main_data.mkdir(parents=True, exist_ok=True)

print("复制主工程图表...")
source_figs = Path(os.path.join(BASE_DIR, "analysis_output", "figures", "paper"))
for f in source_figs.iterdir():
    if f.is_file():
        shutil.copy2(f, main_figs / f.name)
print(f"  复制了 {len(list(main_figs.iterdir()))} 个图表文件")

print("复制主工程数据...")
source_data_dir = Path(r"d:\ML-Laser-JG1-Q355\ml-laser-jg1-q355\analysis_output")
copied_count = 0
for f in source_data_dir.rglob("*.csv"):
    rel = f.relative_to(source_data_dir)
    safe_name = "_".join(rel.parts)
    shutil.copy2(f, main_data / safe_name)
    copied_count += 1
for f in source_data_dir.rglob("*.xlsx"):
    rel = f.relative_to(source_data_dir)
    safe_name = "_".join(rel.parts)
    shutil.copy2(f, main_data / safe_name)
    copied_count += 1
print(f"  复制了 {copied_count} 个数据文件")

# ==================== 新工程 ====================
new_proj = summary_folder / "新工程_ml-laser-jg1-q355-new"
new_figs = new_proj / "图表"
new_data = new_proj / "数据"
new_figs.mkdir(parents=True, exist_ok=True)
new_data.mkdir(parents=True, exist_ok=True)

print("复制新工程图表...")
source_figs_new = Path(r"d:\ML-Laser-JG1-Q355\ml-laser-jg1-q355-new\reports\figures")
for f in source_figs_new.iterdir():
    if f.is_file():
        shutil.copy2(f, new_figs / f.name)
print(f"  复制了 {len(list(new_figs.iterdir()))} 个图表文件")

print("复制新工程数据...")
source_data_new = Path(r"d:\ML-Laser-JG1-Q355\ml-laser-jg1-q355-new\reports")
copied_count_new = 0
for f in source_data_new.rglob("*.csv"):
    rel = f.relative_to(source_data_new)
    safe_name = "_".join(rel.parts)
    shutil.copy2(f, new_data / safe_name)
    copied_count_new += 1
for f in source_data_new.rglob("*.xlsx"):
    rel = f.relative_to(source_data_new)
    safe_name = "_".join(rel.parts)
    shutil.copy2(f, new_data / safe_name)
    copied_count_new += 1
print(f"  复制了 {copied_count_new} 个数据文件")

print("\n" + "=" * 60)
print(f"汇总完成! 文件夹位置: {summary_folder}")
print("=" * 60)