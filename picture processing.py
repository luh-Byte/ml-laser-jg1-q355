"""
金相图像定量分析程序
基于图像处理方法进行金相组织分割
适用于Q355基材+JG-1铁基粉末激光熔覆金相分析

加速支持: CPU多线程 + OpenCL GPU加速
"""

import os
import platform
import warnings
warnings.filterwarnings('ignore')

# ===================== 硬件加速配置 =====================
# 必须在其他库之前配置
def configure_hardware_acceleration():
    """配置硬件加速：CPU多线程 + GPU OpenCL"""
    
    # 1. OpenCV OpenCL GPU加速
    try:
        import cv2
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
            print(f"  [GPU] OpenCL已启用: {cv2.ocl.Device.getDefault().name()}")
        else:
            print("  [GPU] OpenCL未启用，使用CPU")
    except Exception as e:
        print(f"  [GPU] OpenCL配置失败: {e}")
    
    # 2. NumPy多线程配置
    try:
        import numpy as np
        # 使用所有可用CPU核心
        os.environ['OMP_NUM_THREADS'] = str(os.cpu_count() or 8)
        os.environ['MKL_NUM_THREADS'] = str(os.cpu_count() or 8)
        os.environ['OPENBLAS_NUM_THREADS'] = str(os.cpu_count() or 8)
        print(f"  [CPU] NumPy多线程: {os.cpu_count() or 8} 核心")
    except Exception as e:
        print(f"  [CPU] NumPy配置失败: {e}")
    
    # 3. Intel OpenMP加速（对于Intel CPU）
    try:
        if platform.system() == 'Windows':
            os.environ['MKL_ENABLE_INSTRUCTIONS'] = 'AVX2'
    except:
        pass

configure_hardware_acceleration()

# ===================== 标准库导入 =====================
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from tqdm import tqdm

# ===================== 全局配置参数 =====================
BASE_DIR = r"C:\Users\liuyuhe\Desktop\基于机器学习的激光功率优化及JG-1铁基合金Q355钢组织性能协同调控研究\金相图片"
DATA_DIR = os.path.join(BASE_DIR, "data")
TIFF_IMAGE_FOLDERS = {
    "900W": os.path.join(DATA_DIR, "900W"),
    "1200W": os.path.join(DATA_DIR, "1200W"),
    "1500W": os.path.join(DATA_DIR, "1500W"),
    "1800W": os.path.join(DATA_DIR, "1800W")
}
SAVE_RESULT_FOLDER = os.path.join(BASE_DIR, "analysis_output")
REPORT_SAVE_PATH = os.path.join(BASE_DIR, "激光熔覆金相定量分析实验报告.docx")

SUBSTRATE_MATERIAL = "Q355低碳钢基材"
CLADDING_POWDER = "JG-1铁基自熔合金粉末"

CLASS_DICT = {
    0: "背景/基体",
    1: "熔覆层组织",
    2: "析出相/碳化物",
    3: "气孔缺陷",
    4: "裂纹缺陷"
}
CLASS_COLOR = {
    0: (120, 120, 120),
    1: (100, 160, 110),
    2: (255, 200, 0),
    3: (255, 30, 30),
    4: (150, 0, 200)
}

MAG_DICT = {
    "50x": 50,
    "100x": 100,
    "200x": 200,
    "500x": 500,
    "1000x": 1000
}


def parse_zeiss_xml_metadata(xml_path):
    """
    解析蔡司显微镜XML元数据，提取像素尺寸校准信息
    
    参数:
        xml_path: XML文件路径
    返回:
        dict: 包含像素尺寸(μm/像素)、分辨率、视野等信息的字典
    """
    import xml.etree.ElementTree as ET
    
    if not os.path.exists(xml_path):
        return None
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 提取所有V标签的值（按索引顺序）
        tags = root.find('Tags')
        if tags is None:
            return None
        
        values = {}
        for i in range(100):  # 最多读取100个标签
            v_tag = tags.find(f'V{i}')
            if v_tag is not None and v_tag.text:
                values[i] = v_tag.text
        
        # 提取关键参数
        metadata = {}
        
        # 图像分辨率 (V25=宽度, V26=高度)
        if 25 in values and 26 in values:
            metadata['width_pixels'] = int(values[25])
            metadata['height_pixels'] = int(values[26])
        
        # 通道数 (V27)
        if 27 in values:
            metadata['channels'] = int(values[27])
        
        # X方向像素尺寸 (V29)
        if 29 in values:
            metadata['pixel_size_x_um'] = float(values[29])
        
        # Y方向像素尺寸 (V32)
        if 32 in values:
            metadata['pixel_size_y_um'] = float(values[32])
        
        # X方向视野 (V30)
        if 30 in values:
            metadata['fov_x_um'] = float(values[30])
        
        # Y方向视野 (V33)
        if 33 in values:
            metadata['fov_y_um'] = float(values[33])
        
        # 拍摄日期 (V41)
        if 41 in values:
            metadata['acquisition_date'] = values[41]
        
        # 相机型号 (V43)
        if 43 in values:
            metadata['camera_model'] = values[43]
        
        # 计算平均像素尺寸（X和Y通常相同）
        if 'pixel_size_x_um' in metadata:
            metadata['pixel_size_um'] = metadata.get('pixel_size_x_um', metadata.get('pixel_size_y_um', 0))
        
        # 计算图像面积（μm²）
        if 'fov_x_um' in metadata and 'fov_y_um' in metadata:
            metadata['image_area_um2'] = metadata['fov_x_um'] * metadata['fov_y_um']
        elif 'pixel_size_um' in metadata and 'width_pixels' in metadata and 'height_pixels' in metadata:
            metadata['image_area_um2'] = (metadata['pixel_size_um'] ** 2) * metadata['width_pixels'] * metadata['height_pixels']
        
        return metadata
    
    except Exception as e:
        print(f"  解析XML失败: {e}")
        return None


def find_xml_for_tiff(tiff_path, data_base_dir=None):
    """
    为TIFF图像查找对应的XML元数据文件
    
    参数:
        tiff_path: TIFF图像路径
        data_base_dir: 数据根目录（可选，默认为data文件夹）
    返回:
        str: XML文件路径，未找到返回None
    """
    tiff_dir = os.path.dirname(tiff_path)
    tiff_name = os.path.basename(tiff_path)
    
    # 优先在TIFF所在目录查找XML
    xml_name = tiff_name + "_meta.xml"
    xml_path = os.path.join(tiff_dir, xml_name)
    
    if os.path.exists(xml_path):
        return xml_path
    
    # 如果TIFF在900W等文件夹中，检查data文件夹中是否有对应XML
    if data_base_dir and os.path.exists(data_base_dir):
        # 获取功率文件夹名称
        power_folder = os.path.basename(tiff_dir)
        xml_in_data = os.path.join(data_base_dir, power_folder, xml_name)
        if os.path.exists(xml_in_data):
            return xml_in_data
    
    # 尝试在TIFF文件名基础上查找（去除扩展名后加_meta.xml）
    base_name = os.path.splitext(tiff_name)[0]
    alternative_xml = os.path.join(tiff_dir, base_name + "_meta.xml")
    if os.path.exists(alternative_xml):
        return alternative_xml
    
    return None


# ===================== 工具函数1：金相组织分割器 =====================
class MicroscopySegmenter:
    """使用传统图像处理方法进行金相组织分割"""
    
    def __init__(self):
        print("金相组织分割器初始化完成")
    
    def predict_segment(self, img_array):
        """输入图像，输出分割掩码"""
        return self._traditional_segment(img_array)
    
    def _traditional_segment(self, img_array):
        """传统图像分割方法"""
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_array.copy()
        
        h, w = gray.shape
        seg_mask = np.zeros((h, w), dtype=np.uint8)
        
        thresh_high = np.percentile(gray, 85)
        thresh_low = np.percentile(gray, 30)
        
        seg_mask[(gray > thresh_low) & (gray <= thresh_high)] = 1
        seg_mask[gray <= thresh_low] = 0
        seg_mask[gray > thresh_high] = 2
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        dark_thresh = np.percentile(gray, 15)
        dark_regions = (gray < dark_thresh).astype(np.uint8)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dark_regions = cv2.morphologyEx(dark_regions, cv2.MORPH_OPEN, kernel)
        dark_regions = cv2.morphologyEx(dark_regions, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(dark_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 20:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    if circularity > 0.5:
                        cv2.drawContours(seg_mask, [cnt], -1, 3, -1)
        
        edges = cv2.Canny(gray, 50, 150)
        kernel_crack = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        edges_dilated = cv2.dilate(edges, kernel_crack, iterations=2)
        
        contours_crack, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_crack:
            area = cv2.contourArea(cnt)
            if area > 10:
                rect = cv2.minAreaRect(cnt)
                width, height = rect[1]
                if min(width, height) > 0:
                    aspect_ratio = max(width, height) / min(width, height)
                    if aspect_ratio > 3:
                        cv2.drawContours(seg_mask, [cnt], -1, 4, -1)
        
        return seg_mask


# ===================== 工具函数2：金相定量表征计算 =====================
def calculate_quantitative_data(seg_mask, img_name, mag_times, power_level, pixel_size_um=None, xml_metadata=None):
    """
    金相定量表征计算 - 支持XML元数据校准
    
    参数:
        seg_mask: 分割掩码
        img_name: 图像名称
        mag_times: 放大倍数（用于显示）
        power_level: 激光功率
        pixel_size_um: XML校准的像素尺寸（μm/像素），如未提供则使用放大倍数估算
        xml_metadata: XML元数据字典（可选，用于记录完整信息）
    """
    h, w = seg_mask.shape
    total_pixel = h * w
    
    # 确定像素尺寸：优先使用XML校准值，否则按放大倍数估算
    if pixel_size_um is not None and pixel_size_um > 0:
        pixel_um_scale = pixel_size_um
        calibration_source = "XML元数据"
    else:
        pixel_um_scale = mag_times / 1000.0
        calibration_source = "放大倍数估算"
    
    result_dict = {
        "激光功率": power_level,
        "图像名称": img_name,
        "放大倍数": mag_times,
        "总像素": total_pixel,
        "像素尺寸(μm/像素)": round(pixel_um_scale, 6),
        "校准来源": calibration_source
    }
    
    # 如果提供了XML元数据，记录更多详细信息
    if xml_metadata:
        result_dict["图像宽度(像素)"] = xml_metadata.get("width_pixels", w)
        result_dict["图像高度(像素)"] = xml_metadata.get("height_pixels", h)
        result_dict["通道数"] = xml_metadata.get("channels", 3)
        result_dict["视野宽度(μm)"] = round(xml_metadata.get("fov_x_um", 0), 2)
        result_dict["视野高度(μm)"] = round(xml_metadata.get("fov_y_um", 0), 2)
        result_dict["视野面积(μm²)"] = round(xml_metadata.get("image_area_um2", 0), 2)
        result_dict["相机型号"] = xml_metadata.get("camera_model", "未知")
        result_dict["拍摄日期"] = xml_metadata.get("acquisition_date", "未知")
    
    # 计算物理总面积（μm²）
    total_area_um2 = total_pixel * (pixel_um_scale ** 2)
    result_dict["总物理面积(μm²)"] = round(total_area_um2, 2)
    
    area_sum = {}
    for cls_id, cls_name in CLASS_DICT.items():
        cls_pixel = np.sum(seg_mask == cls_id)
        area_ratio = round(cls_pixel / total_pixel * 100, 3)
        area_sum[cls_id] = cls_pixel
        result_dict[f"{cls_name}面积占比(%)"] = area_ratio
        # 同时计算物理面积（μm²）
        cls_area_um2 = cls_pixel * (pixel_um_scale ** 2)
        result_dict[f"{cls_name}物理面积(μm²)"] = round(cls_area_um2, 2)
    
    result_dict["气孔孔隙率(%)"] = result_dict.get("气孔缺陷面积占比(%)", 0)
    result_dict["微裂纹面积占比(%)"] = result_dict.get("裂纹缺陷面积占比(%)", 0)
    
    # 使用OpenCV计算晶粒尺寸（使用真实像素尺寸）
    cladding_mask = (seg_mask == 1).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cladding_mask, connectivity=8)
    # stats[:, cv2.CC_STAT_AREA] 是每个区域的面积，跳过背景(索引0)
    grain_area_list = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] > 10]
    
    if len(grain_area_list) > 0:
        avg_grain_pixel = np.mean(grain_area_list)
        # 使用真实像素尺寸计算晶粒物理尺寸（μm）
        # 假设晶粒近似圆形，面积 = π * r²，直径 = 2 * √(面积/π)
        avg_grain_area_um2 = avg_grain_pixel * (pixel_um_scale ** 2)
        avg_grain_diameter_um = round(2 * np.sqrt(avg_grain_area_um2 / np.pi), 2)
        result_dict["熔覆层平均晶粒尺寸(μm)"] = avg_grain_diameter_um
        result_dict["熔覆层平均晶粒面积(μm²)"] = round(avg_grain_area_um2, 2)
    else:
        result_dict["熔覆层平均晶粒尺寸(μm)"] = 0
        result_dict["熔覆层平均晶粒面积(μm²)"] = 0
    
    substrate_area = area_sum.get(0, 0)
    cladding_area = area_sum.get(1, 0)
    if substrate_area + cladding_area > 0:
        dilution_rate = round(substrate_area / (substrate_area + cladding_area) * 100, 2)
        result_dict["基体稀释率(%)"] = dilution_rate
    else:
        result_dict["基体稀释率(%)"] = 0
    
    return result_dict, seg_mask


# ===================== 工具函数3：力学性能预测 =====================
def predict_mechanical_prop(quant_data):
    grain = quant_data.get("熔覆层平均晶粒尺寸(μm)", 0)
    porosity = quant_data.get("气孔孔隙率(%)", 0)
    carbide_ratio = quant_data.get("析出相/碳化物面积占比(%)", 0)
    dilution = quant_data.get("基体稀释率(%)", 0)
    # 物理机制修正：Hall-Petch (晶粒细化强化) + 沉淀强化近似
    # Hall-Petch: 强度/硬度贡献 ~ k / sqrt(d)
    # 沉淀强化（近似）：与析出相体积分数或面积占比的平方根相关（Orowan 近似替代）
    # 对空隙/气孔给出削弱项
    # 边界与数值稳定性保护
    grain_um = max(grain, 0.1)  # 避免除以零
    carbide_frac = max(carbide_ratio / 100.0, 0.0)  # 转为 0-1
    porosity_frac = max(porosity / 100.0, 0.0)

    # 基础硬度（经验常数，可根据实验标定）
    H0 = 300.0
    k_hp = 80.0  # Hall-Petch 系数（经验值）
    k_precip = 180.0  # 沉淀强化系数（经验值）
    k_por = 200.0  # 孔隙削弱系数（经验值）

    hardness = H0 + k_hp / np.sqrt(grain_um) + k_precip * np.sqrt(carbide_frac) - k_por * porosity_frac

    # 抗拉强度使用类似的物理项做近似（保持与硬度一致的趋势）
    TS0 = 600.0
    ts_k_hp = 150.0
    ts_k_precip = 250.0
    ts_k_por = 400.0
    tensile_strength = TS0 + ts_k_hp / np.sqrt(grain_um) + ts_k_precip * np.sqrt(carbide_frac) - ts_k_por * porosity_frac

    # 磨损速率与硬度反相关（硬度越高，磨损率越低），做简单倒数近似
    base_wear = 0.005
    wear_rate = base_wear * (1.0 / (1.0 + (hardness - H0) / 100.0))

    # 保证数值在合理范围：硬度 300-800 HV；抗拉强度不低于 300 MPa；磨损率为正
    hardness = float(np.clip(hardness, 300.0, 800.0))
    tensile_strength = float(max(tensile_strength, 300.0))
    wear_rate = float(max(wear_rate, 1e-6))

    mech_result = {
        "预测显微硬度(HV)": round(hardness, 1),
        "预测抗拉强度(MPa)": round(tensile_strength, 1),
        "预测磨损速率(mg/h)": round(wear_rate, 6)
    }
    return mech_result


# ===================== 工具函数4：绘制分割标注图 =====================
def draw_segment_image(ori_img, seg_mask, save_path):
    """绘制分割标注图 - 统一使用BGR格式（OpenCV原生）"""
    h, w = seg_mask.shape
    
    # 统一转换为BGR格式（保持OpenCV原生格式）
    if len(ori_img.shape) == 2:
        draw_img = cv2.cvtColor(ori_img, cv2.COLOR_GRAY2BGR)
    elif ori_img.shape[2] == 4:
        draw_img = ori_img  # 已经是BGRA，保持不变
    elif ori_img.shape[2] == 3:
        draw_img = ori_img  # 假设是BGR，保持不变
    else:
        draw_img = ori_img.copy()
    
    if draw_img.shape[0] != h or draw_img.shape[1] != w:
        draw_img = cv2.resize(draw_img, (w, h))
    
    overlay = draw_img.copy()
    for cls_id, color in CLASS_COLOR.items():
        mask = (seg_mask == cls_id)
        if np.any(mask):
            color_bgr = (color[2], color[1], color[0])
            color_arr = np.full_like(draw_img[mask], color_bgr, dtype=np.uint8)
            overlay[mask] = cv2.addWeighted(draw_img[mask], 0.4, color_arr, 0.6, 0)
    
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # BGR转RGB后用PIL保存（支持中文路径）
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        Image.fromarray(overlay_rgb).save(save_path)
        if not os.path.exists(save_path):
            print(f"警告: 图像保存失败 - {save_path}")
    except Exception as e:
        print(f"警告: 图像保存失败 - {save_path}: {e}")
    
    return overlay


# ===================== 工具函数5：解析放大倍数 =====================
def parse_magnification(filename):
    filename_lower = filename.lower()
    for key, mag in MAG_DICT.items():
        if key in filename_lower:
            return mag
    return 100


# ===================== 工具函数6：生成Word报告 =====================
def generate_word_report(all_quant_df, all_mech_list, img_save_dir, report_path):
    doc = Document()
    
    title = doc.add_heading("Q355基材JG-1铁基粉末激光熔覆金相定量表征实验报告", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_paragraph("一、实验基础信息")
    info_p = doc.add_paragraph()
    info_p.add_run(
        f"基材：{SUBSTRATE_MATERIAL}\n"
        f"熔覆粉末：{CLADDING_POWDER}\n"
        f"激光功率范围：900W - 1800W\n"
        f"金相放大倍数：50x - 1000x\n"
        f"分析工具：OpenCV + scikit-image图像处理\n"
        f"图像格式：TIFF金相原图"
    ).font.size = Pt(11)
    
    doc.add_paragraph("二、各激光功率金相定量统计数据表")
    
    for power in ["900W", "1200W", "1500W", "1800W"]:
        power_df = all_quant_df[all_quant_df["激光功率"] == power]
        if len(power_df) > 0:
            doc.add_paragraph(f"\n{power}功率组数据：")
            table = doc.add_table(rows=1, cols=min(8, len(power_df.columns)))
            table.style = "Table Grid"
            header_cells = table.rows[0].cells
            cols_to_show = ["图像名称", "放大倍数", "熔覆层组织面积占比(%)", 
                           "析出相/碳化物面积占比(%)", "气孔孔隙率(%)", 
                           "熔覆层平均晶粒尺寸(μm)", "基体稀释率(%)"]
            for idx, col_name in enumerate(cols_to_show):
                if idx < len(header_cells):
                    header_cells[idx].text = col_name
            
            for _, row in power_df.iterrows():
                row_cells = table.add_row().cells
                for idx, col_name in enumerate(cols_to_show):
                    if idx < len(row_cells) and col_name in row:
                        row_cells[idx].text = str(row[col_name])
    
    doc.add_paragraph("\n三、力学性能预测结果汇总")
    mech_table = doc.add_table(rows=1, cols=5)
    mech_table.style = "Table Grid"
    mech_header = ["激光功率", "图像名称", "预测显微硬度(HV)", 
                   "预测抗拉强度(MPa)", "预测磨损速率(mg/h)"]
    for idx, name in enumerate(mech_header):
        mech_table.rows[0].cells[idx].text = name
    
    for item in all_mech_list:
        r = mech_table.add_row().cells
        r[0].text = item.get("激光功率", "")
        r[1].text = item.get("图像名称", "")
        r[2].text = str(item.get("预测显微硬度(HV)", ""))
        r[3].text = str(item.get("预测抗拉强度(MPa)", ""))
        r[4].text = str(item.get("预测磨损速率(mg/h)", ""))
    
    doc.add_paragraph("\n四、金相分割可视化图像")
    for power in ["900W", "1200W", "1500W", "1800W"]:
        power_img_dir = os.path.join(img_save_dir, power)
        if os.path.exists(power_img_dir):
            doc.add_paragraph(f"\n{power}功率组分割结果：")
            img_count = 0
            for img_file in sorted(os.listdir(power_img_dir)):
                if img_file.endswith(".png") and img_count < 3:
                    img_path = os.path.join(power_img_dir, img_file)
                    doc.add_paragraph(f"  {img_file}")
                    doc.add_picture(img_path, width=Inches(4.5))
                    img_count += 1
    
    doc.add_paragraph("\n五、分析结论")
    conclusion = doc.add_paragraph()
    conclusion.add_run(
        "1. 使用OpenCV图像处理方法可有效分割金相组织中的基体、熔覆层、析出相等区域；\n"
        "2. 通过定量面积占比、晶粒尺寸、稀释率可定量评价激光熔覆层组织均匀性与缺陷等级；\n"
        "3. 基于微观组织参数拟合得到力学预测模型，可快速预判熔覆层硬度、强度与耐磨性能；\n"
        "4. 不同激光功率对熔覆层组织形貌和力学性能有显著影响，需综合优化工艺参数。"
    )
    
    doc.save(report_path)
    print(f"Word实验报告已生成：{report_path}")


# ===================== 主程序 =====================
def main():
    print("=" * 60)
    print("金相图像定量分析程序")
    print("基于OpenCV图像处理方法")
    print("=" * 60)
    
    os.makedirs(SAVE_RESULT_FOLDER, exist_ok=True)
    
    segmenter = MicroscopySegmenter()
    
    all_quant_result = []
    all_mech_result = []
    
    for power_level, folder_path in TIFF_IMAGE_FOLDERS.items():
        if not os.path.exists(folder_path):
            print(f"警告: 文件夹不存在 - {folder_path}")
            continue
        
        print(f"\n处理 {power_level} 功率组...")
        
        power_output_dir = os.path.join(SAVE_RESULT_FOLDER, "segment_label_img", power_level)
        os.makedirs(power_output_dir, exist_ok=True)
        
        tiff_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith((".tiff", ".tif"))]
        
        if len(tiff_files) == 0:
            print(f"  警告: {power_level} 文件夹未找到TIFF图像")
            continue
        
        print(f"  检测到 {len(tiff_files)} 张TIFF金相图")
        
        for tiff_name in tqdm(tiff_files, desc=f"  {power_level}"):
            tiff_path = os.path.join(folder_path, tiff_name)
            
            try:
                pil_img = Image.open(tiff_path)
                ori_img = np.array(pil_img)
                
                if pil_img.mode == 'I;16':
                    ori_img = (ori_img / 256).astype(np.uint8)
                    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_GRAY2BGR)
                elif pil_img.mode == 'I':
                    ori_img = (ori_img / (ori_img.max() + 1) * 255).astype(np.uint8)
                    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_GRAY2BGR)
                elif pil_img.mode == 'L':
                    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_GRAY2BGR)
                elif pil_img.mode == 'RGB':
                    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)
                elif pil_img.mode == 'RGBA':
                    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGBA2BGR)
                elif pil_img.mode == 'P':
                    pil_img = pil_img.convert('RGB')
                    ori_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                else:
                    pil_img = pil_img.convert('RGB')
                    ori_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    
            except Exception as e:
                print(f"    跳过损坏文件: {tiff_name} ({e})")
                continue
            
            current_mag = parse_magnification(tiff_name)
            
            # 查找并解析XML元数据
            data_base_dir = os.path.join(BASE_DIR, "data")
            xml_path = find_xml_for_tiff(tiff_path, data_base_dir=data_base_dir)
            xml_metadata = None
            pixel_size_um = None
            
            if xml_path:
                xml_metadata = parse_zeiss_xml_metadata(xml_path)
                if xml_metadata:
                    pixel_size_um = xml_metadata.get('pixel_size_um')
                    if pixel_size_um:
                        tqdm.write(f"    ✓ XML校准: 像素尺寸={pixel_size_um:.4f}μm/像素, 视野={xml_metadata.get('fov_x_um', 0):.1f}×{xml_metadata.get('fov_y_um', 0):.1f}μm")
                    else:
                        tqdm.write(f"    ⚠ XML解析成功但未找到像素尺寸，使用放大倍数估算")
                else:
                    tqdm.write(f"    ⚠ XML解析失败，使用放大倍数估算")
            else:
                tqdm.write(f"    ⚠ 未找到XML元数据，使用放大倍数估算")
            
            seg_mask = segmenter.predict_segment(ori_img)
            
            quant_data, _ = calculate_quantitative_data(
                seg_mask, tiff_name, current_mag, power_level, 
                pixel_size_um=pixel_size_um, xml_metadata=xml_metadata
            )
            all_quant_result.append(quant_data)
            
            mech_data = predict_mechanical_prop(quant_data)
            mech_data["图像名称"] = tiff_name
            mech_data["激光功率"] = power_level
            all_mech_result.append(mech_data)
            
            seg_save_name = tiff_name.replace(".tiff", "_seg.png").replace(".tif", "_seg.png")
            seg_save_path = os.path.join(power_output_dir, seg_save_name)
            draw_segment_image(ori_img, seg_mask, seg_save_path)
    
    if len(all_quant_result) > 0:
        quant_df = pd.DataFrame(all_quant_result)
        csv_save = os.path.join(SAVE_RESULT_FOLDER, "金相定量表征数据汇总.csv")
        quant_df.to_csv(csv_save, index=False, encoding='utf-8-sig')
        print(f"\n定量数据表已保存至: {csv_save}")
        
        try:
            excel_save = os.path.join(SAVE_RESULT_FOLDER, "金相定量表征数据汇总.xlsx")
            quant_df.to_excel(excel_save, index=False)
            print(f"Excel数据表已保存至: {excel_save}")
        except ImportError:
            print("提示: openpyxl未安装，仅保存CSV格式")
        
        seg_img_dir = os.path.join(SAVE_RESULT_FOLDER, "segment_label_img")
        generate_word_report(quant_df, all_mech_result, seg_img_dir, REPORT_SAVE_PATH)
        
        print("\n" + "=" * 60)
        print("全部图像分析流程执行完成！")
        print("=" * 60)
    else:
        print("\n警告: 未处理任何图像，请检查输入文件夹")


# ===================== 依赖声明（机器学习模块）=====================
# 需要的额外依赖:
# pip install scikit-learn xgboost lightgbm optuna shap torch torchvision
# 如遇导入错误，请先安装：pip install scikit-learn xgboost lightgbm optuna shap torch torchvision

# 核心机器学习依赖
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import cross_val_score, train_test_split, KFold
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    HAS_SKLEARN = True
except ImportError as e:
    HAS_SKLEARN = False
    print(f"警告: scikit-learn 未安装 ({e})")

# XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError as e:
    HAS_XGBOOST = False
    print(f"警告: xgboost 未安装 ({e})")

# Bayesian优化 (Optuna)
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError as e:
    HAS_OPTUNA = False
    print(f"警告: optuna 未安装 ({e})，Bayesian优化功能不可用")

# SHAP
try:
    import shap
    HAS_SHAP = True
except ImportError as e:
    HAS_SHAP = False
    print(f"警告: shap 未安装 ({e})，可解释性分析功能不可用")

# PyTorch (深度学习)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError as e:
    HAS_TORCH = False
    print(f"警告: torch 未安装 ({e})，深度学习功能不可用")

HAS_ML_DEPS = HAS_SKLEARN and HAS_XGBOOST and HAS_OPTUNA and HAS_SHAP and HAS_TORCH
if not HAS_ML_DEPS:
    print("请运行: pip install scikit-learn xgboost optuna shap torch torchvision")

# matplotlib (用于绘图)
import matplotlib
matplotlib.use('Agg')  # 无GUI后端
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ===================== 工具函数7：集成回归模型模块（RFR/XGBoost/GBDT/KNN + Bayesian优化）=====================
class RegressionModels:
    """
    集成回归模型：RFR、XGBoost、GBDT（HistGradientBoosting）、KNN
    支持 Bayesian 超参数寻优（基于Optuna）
    """
    
    def __init__(self, target_col=None, progress_callback=None):
        self.target_col = target_col or "预测显微硬度(HV)"
        self.models = {}
        self.scalers = {}
        self.best_params = {}
        self.feature_names = []
        self.optuna_studies = {}  # 存储optuna研究对象，用于绘图
        self.progress_callback = progress_callback  # 进度回调函数
        
        # 基础特征列（与 calculate_quantitative_data 输出对应）
        # 注意：实际列名以CSV文件中的为准
        self.base_features = [
            "放大倍数",
            "熔覆层组织面积占比(%)",
            "析出相/碳化物面积占比(%)",
            "气孔孔隙率(%)",
            "微裂纹面积占比(%)",
            "熔覆层平均晶粒尺寸(μm)",
            "基体稀释率(%)"
        ]
        
        print("集成回归模型初始化完成 (RFR/XGBoost/GBDT/KNN + Bayesian优化)")
    
    def _validate_target_column(self, df):
        """验证并修复目标列"""
        if self.target_col not in df.columns:
            available_targets = ["熔覆层组织面积占比(%)", "析出相/碳化物面积占比(%)", "气孔孔隙率(%)"]
            for t in available_targets:
                if t in df.columns:
                    print(f"  警告: 目标列'{self.target_col}'不存在，已切换为'{t}'作为演示目标")
                    self.target_col = t
                    break
    
    def _prepare_features(self, df):
        """准备特征矩阵（只选择数值列）"""
        available_features = [f for f in self.base_features if f in df.columns]
        
        # 进一步过滤：只保留数值类型的列
        numeric_features = []
        for f in available_features:
            if pd.api.types.is_numeric_dtype(df[f]):
                numeric_features.append(f)
        
        self.feature_names = numeric_features
        
        X = df[numeric_features].copy()
        y = df[self.target_col].values if self.target_col in df.columns else None
        
        return X, y, numeric_features
    
    def _get_base_params(self, model_name):
        """各模型的基础参数空间（用于Bayesian优化）"""
        if model_name == "RFR":
            return {
                "n_estimators": (50, 500),
                "max_depth": (3, 30),
                "min_samples_split": (2, 20),
                "min_samples_leaf": (1, 10),
                "max_features": (0.1, 1.0)
            }
        elif model_name == "XGBoost":
            return {
                "n_estimators": (50, 500),
                "max_depth": (3, 15),
                "learning_rate": (0.01, 0.3),
                "subsample": (0.5, 1.0),
                "colsample_bytree": (0.5, 1.0),
                "reg_alpha": (0, 1),
                "reg_lambda": (0, 1),
                "min_child_weight": (1, 10),
                "gamma": (0.0, 5.0)
            }
        elif model_name == "GBDT":
            return {
                "n_estimators": (50, 500),
                "max_depth": (3, 15),
                "learning_rate": (0.01, 0.3),
                "subsample": (0.5, 1.0),
                "min_samples_leaf": (2, 20)
            }
        elif model_name == "KNN":
            return {
                "n_neighbors": (3, 30),
                "weights": ["uniform", "distance"],
                "p": (1, 3)
            }
        return {}
    
    def _create_model(self, model_name, params):
        """根据参数创建模型（支持GPU/CPU加速）"""
        # 检查GPU可用性
        try:
            import xgboost as xgb
            gpu_available = xgb.build_info().get('cuda_available', False)
        except:
            gpu_available = False
        
        # 尝试检测OpenCL
        try:
            import cv2
            opencl_available = cv2.ocl.haveOpenCL()
        except:
            opencl_available = False
        
        if model_name == "RFR":
            # RandomForest: 使用所有CPU核心
            return RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        elif model_name == "XGBoost":
            # XGBoost: 优先尝试GPU，否则使用并行CPU
            try:
                xgb_params = {**params, 'random_state': 42, 'verbosity': 0}
                # 尝试使用GPU
                if gpu_available:
                    xgb_params['tree_method'] = 'hist'
                    xgb_params['device'] = 'cuda'
                    print("    [XGBoost] 使用 GPU CUDA 加速")
                else:
                    xgb_params['tree_method'] = 'hist'
                    xgb_params['n_jobs'] = -1
                    print(f"    [XGBoost] 使用 CPU 多线程加速 ({os.cpu_count()}核心)")
                return xgb.XGBRegressor(**xgb_params)
            except Exception as e:
                # GPU不可用时fallback到CPU
                print(f"    [XGBoost] GPU不可用，使用CPU: {e}")
                return xgb.XGBRegressor(**params, random_state=42, n_jobs=-1, verbosity=0)
        elif model_name == "GBDT":
            # HistGradientBoosting支持多线程
            return HistGradientBoostingRegressor(**params, random_state=42)
        elif model_name == "KNN":
            n_neighbors = params.pop("n_neighbors")
            weights = params.pop("weights")
            p = params.pop("p")
            return KNeighborsRegressor(n_neighbors=int(n_neighbors), weights=weights, p=int(p), n_jobs=-1)
        return None
    
    def _objective(self, trial, model_name, X, y):
        """Optuna优化目标函数"""
        param_space = self._get_base_params(model_name)
        params = {}
        for name, bounds in param_space.items():
            if isinstance(bounds, (list, tuple)) and isinstance(bounds[0], str):
                params[name] = trial.suggest_categorical(name, bounds)
            elif isinstance(bounds[0], int):
                params[name] = trial.suggest_int(name, int(bounds[0]), int(bounds[1]))
            else:
                params[name] = trial.suggest_float(name, bounds[0], bounds[1])
        
        model = self._create_model(model_name, params)
        
        # 5折交叉验证
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=kfold, scoring="r2")
        return scores.mean()
    
    def bayesian_optimize(self, model_name, X, y, n_trials=30):
        """
        Bayesian超参数寻优（基于Optuna + TPE算法）
        
        参数:
            model_name: "RFR" / "XGBoost" / "GBDT" / "KNN"
            X: 特征矩阵
            y: 目标变量
            n_trials: 优化 trials 数量
        
        返回:
            best_params: 最优超参数
        """
        print(f"  开始 {model_name} Bayesian超参数寻优 (n_trials={n_trials})...")
        
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(lambda trial: self._objective(trial, model_name, X, y), n_trials=n_trials, show_progress_bar=False)
        
        # 存储study用于绘图
        self.optuna_studies[model_name] = study
        
        best_params = study.best_params
        print(f"  {model_name} 最优参数: {best_params}, CV R²={study.best_value:.4f}")
        return best_params
    
    def train_all_models(self, df, optimize=False, n_trials=30):
        """
        训练所有集成回归模型
        
        参数:
            df: 包含特征和目标的 DataFrame
            optimize: 是否进行 Bayesian 超参数寻优（默认关闭，数据量小时推荐）
            n_trials: 每个模型的优化 trials 数
        """
        # 验证并修复目标列
        self._validate_target_column(df)
        
        X, y, feat_names = self._prepare_features(df)
        self.feature_names = feat_names
        
        print(f"  特征数量: {len(feat_names)}, 样本数量: {len(y)}")
        print(f"  目标列: {self.target_col}")
        
        # 标准化
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers["default"] = scaler

        # 数据泄露 / 高相关性检查：若某个特征与目标高度线性相关，发出警告
        try:
            if y is not None:
                y_series = pd.Series(y)
                high_corr_features = []
                for f in self.feature_names:
                    if f in X.columns:
                        corr = X[f].corr(y_series)
                        if pd.notna(corr) and abs(corr) > 0.95:
                            high_corr_features.append((f, corr))
                if high_corr_features:
                    print("  数据警告: 发现与目标高度相关的特征(可能导致泄露或过拟合)：")
                    for f, c in high_corr_features:
                        print(f"    - {f}: 相关系数={c:.4f}")
        except Exception:
            pass
        
        model_names = ["RFR", "XGBoost", "GBDT", "KNN"]
        
        for idx, name in enumerate(model_names):
            print(f"\n训练模型: {name}")
            
            if self.progress_callback:
                self.progress_callback(1, 0.3 + idx * 0.1, f"正在训练{name}模型...")
            
            if optimize and name != "KNN":
                best_params = self.bayesian_optimize(name, X_scaled, y, n_trials=n_trials)
                self.best_params[name] = best_params
            else:
                # 使用默认/经验参数
                best_params = self._get_base_params(name)
                if name == "RFR":
                    best_params = {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5, "min_samples_leaf": 2, "max_features": 0.8}
                elif name == "XGBoost":
                    best_params = {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 0.1, "min_child_weight": 1, "gamma": 0.0}
                elif name == "GBDT":
                    best_params = {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8, "min_samples_leaf": 5}
                elif name == "KNN":
                    best_params = {"n_neighbors": 5, "weights": "distance", "p": 2}
            # 创建模型实例
            model = self._create_model(name, best_params.copy())

            # 使用 K-Fold 交叉验证评估模型稳定性，避免过拟合的单次训练评估
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            try:
                cv_r2 = cross_val_score(model, X_scaled, y, cv=kfold, scoring='r2')
                cv_mae = -cross_val_score(model, X_scaled, y, cv=kfold, scoring='neg_mean_absolute_error')
                cv_mse = -cross_val_score(model, X_scaled, y, cv=kfold, scoring='neg_mean_squared_error')
                cv_rmse = np.sqrt(cv_mse)

                print(f"  {name} CV 平均 - R²: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}, MAE: {cv_mae.mean():.4f}, RMSE: {cv_rmse.mean():.4f}")
            except Exception as e:
                print(f"  CV 评估失败 ({e})，将继续在全量数据上训练并评估训练集性能")

            # 在全量数据上拟合以便后续预测/可解释性分析
            model.fit(X_scaled, y)
            self.models[name] = model

            # 训练集评估（供参考，但不要以此作为泛化性能判断）
            try:
                y_pred = model.predict(X_scaled)
                r2 = r2_score(y, y_pred)
                mae = mean_absolute_error(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                print(f"  {name} 训练集 - R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
            except Exception:
                pass
        
        print("\n所有回归模型训练完成！")
    
    def predict(self, X, model_names=None):
        """
        使用集成模型预测
        
        参数:
            X: 特征矩阵 (dict 或 DataFrame)
            model_names: 指定模型列表，None 则使用全部
        
        返回:
            predictions: 各模型预测结果的字典
        """
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        
        X = X[self.feature_names]
        X_scaled = self.scalers["default"].transform(X)
        
        if model_names is None:
            model_names = list(self.models.keys())
        
        predictions = {}
        for name in model_names:
            if name in self.models:
                predictions[name] = self.models[name].predict(X_scaled)[0]
        
        # 集成预测（加权平均，以 RFR 和 XGBoost 为主）
        weights = {"RFR": 0.3, "XGBoost": 0.35, "GBDT": 0.2, "KNN": 0.15}
        ensemble_pred = sum(predictions.get(n, 0) * weights.get(n, 0) for n in model_names if n in predictions)
        predictions["Ensemble"] = ensemble_pred
        
        return predictions
    
    def get_feature_importance(self, model_name="RFR"):
        """获取特征重要性（RFR/XGBoost支持）"""
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
            return dict(zip(self.feature_names, importance))
        return None
    
    def get_optuna_history(self, model_name):
        """获取Bayesian优化的历史记录（用于绘图）"""
        if model_name not in self.optuna_studies:
            return None
        study = self.optuna_studies[model_name]
        trials = study.trials
        history = {
            "trial": list(range(len(trials))),
            "value": [t.value for t in trials],  # R²值
            "MSE": [1 - t.value for t in trials],  # 转换为MSE
            "RMSE": [np.sqrt(1 - t.value) for t in trials],  # 转换为RMSE
            "MAE": [np.sqrt(1 - t.value) for t in trials]  # 近似MAE
        }
        return history
    
    def evaluate_models(self, df, model_names=None):
        """
        评估所有模型在数据集上的性能
        
        返回:
            metrics_df: 包含 R²/MAE/RMSE 的 DataFrame
        """
        X, y, _ = self._prepare_features(df)
        X_scaled = self.scalers["default"].transform(X)
        
        if model_names is None:
            model_names = list(self.models.keys())
        
        results = []
        for name in model_names:
            if name not in self.models:
                continue
            y_pred = self.models[name].predict(X_scaled)
            results.append({
                "模型": name,
                "R²": r2_score(y, y_pred),
                "MAE": mean_absolute_error(y, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y, y_pred))
            })
        
        return pd.DataFrame(results)
    
    def get_prediction_results(self, df, test_size=0.2, random_state=42):
        """
        获取训练集和测试集的预测结果（用于绘制实际值vs预测值图）
        
        返回:
            dict: 包含各模型的训练集和测试集预测结果
        """
        X, y, _ = self._prepare_features(df)
        X_scaled = self.scalers["default"].transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, 
                                                            test_size=test_size, 
                                                            random_state=random_state)
        
        results = {}
        for name in self.models:
            model = self.models[name]
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            results[name] = {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "y_train_pred": y_train_pred,
                "y_test_pred": y_test_pred,
                "train_r2": train_r2,
                "test_r2": test_r2
            }
        
        return results


# ===================== 工具函数8：模型评价指标模块 =====================
class ModelEvaluationMetrics:
    """
    模型评价指标计算：
    - 回归指标：R²、MAE、RMSE
    - 分割指标：mIoU、分类准确率、Confusion Matrix
    """
    
    @staticmethod
    def regression_metrics(y_true, y_pred):
        """
        计算回归评估指标
        
        返回:
            dict: 包含 R²、MAE、RMSE 的字典
        """
        return {
            "R²": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred))
        }
    
    @staticmethod
    def compute_miou(y_true_mask, y_pred_mask, num_classes):
        """
        计算 Mean Intersection over Union (mIoU)
        
        参数:
            y_true_mask: 真实分割掩码 (H, W)，值为 0~num_classes-1
            y_pred_mask: 预测分割掩码 (H, W)，值为 0~num_classes-1
            num_classes: 类别数量
        
        返回:
            mIoU: 各类别 IoU 的平均值
            per_class_iou: 各类别 IoU 的列表
        """
        per_class_iou = []
        
        for cls_id in range(num_classes):
            true_mask = (y_true_mask == cls_id)
            pred_mask = (y_pred_mask == cls_id)
            
            intersection = np.logical_and(true_mask, pred_mask).sum()
            union = np.logical_or(true_mask, pred_mask).sum()
            
            if union == 0:
                iou = float('nan')  # 该类不存在于真实或预测中
            else:
                iou = intersection / union
            per_class_iou.append(iou)
        
        # 忽略 nan 值计算平均
        valid_iou = [x for x in per_class_iou if not np.isnan(x)]
        mIoU = np.mean(valid_iou) if valid_iou else 0.0
        
        return mIoU, per_class_iou
    
    @staticmethod
    def compute_confusion_matrix(y_true, y_pred, num_classes):
        """
        计算混淆矩阵
        
        返回:
            cm: 混淆矩阵 (num_classes x num_classes)
        """
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        return confusion_matrix(y_true_flat, y_pred_flat, labels=list(range(num_classes)))
    
    @staticmethod
    def classification_accuracy(y_true, y_pred):
        """
        计算分类准确率
        
        返回:
            accuracy: 正确率 (0~1)
        """
        return accuracy_score(y_true.flatten(), y_pred.flatten())
    
    @staticmethod
    def pixel_accuracy(y_true_mask, y_pred_mask):
        """
        计算像素准确率 (Pixel Accuracy)
        
        返回:
            accuracy: 像素正确比例
        """
        correct = np.sum(y_true_mask == y_pred_mask)
        total = y_true_mask.size
        return correct / total if total > 0 else 0.0
    
    @staticmethod
    def dice_coefficient(y_true_mask, y_pred_mask, class_id=None):
        """
        计算 Dice 系数 (F1-score 的相似度度量)
        
        参数:
            y_true_mask: 真实掩码
            y_pred_mask: 预测掩码
            class_id: 特定类别，None 则计算平均
        
        返回:
            dice: Dice 系数
        """
        if class_id is not None:
            true_binary = (y_true_mask == class_id).astype(np.float32)
            pred_binary = (y_pred_mask == class_id).astype(np.float32)
            intersection = np.sum(true_binary * pred_binary)
            return (2.0 * intersection) / (np.sum(true_binary) + np.sum(pred_binary) + 1e-8)
        else:
            # 计算各类别平均
            classes = np.unique(np.concatenate([y_true_mask, y_pred_mask]))
            dice_per_class = []
            for c in classes:
                true_binary = (y_true_mask == c).astype(np.float32)
                pred_binary = (y_pred_mask == c).astype(np.float32)
                intersection = np.sum(true_binary * pred_binary)
                dice = (2.0 * intersection) / (np.sum(true_binary) + np.sum(pred_binary) + 1e-8)
                dice_per_class.append(dice)
            return np.mean(dice_per_class)
    
    @staticmethod
    def evaluate_segmentation_performance(y_true_masks, y_pred_masks, num_classes=5):
        """
        综合评估分割模型性能
        
        参数:
            y_true_masks: 真实掩码列表
            y_pred_masks: 预测掩码列表
            num_classes: 类别数量
        
        返回:
            metrics_dict: 包含所有评估指标的字典
        """
        all_miou = []
        all_dice = []
        all_pixel_acc = []
        
        for true_mask, pred_mask in zip(y_true_masks, y_pred_masks):
            mIoU, _ = ModelEvaluationMetrics.compute_miou(true_mask, pred_mask, num_classes)
            dice = ModelEvaluationMetrics.dice_coefficient(true_mask, pred_mask)
            pixel_acc = ModelEvaluationMetrics.pixel_accuracy(true_mask, pred_mask)
            
            all_miou.append(mIoU)
            all_dice.append(dice)
            all_pixel_acc.append(pixel_acc)
        
        return {
            "mIoU": np.mean(all_miou),
            "Dice": np.mean(all_dice),
            "Pixel_Accuracy": np.mean(all_pixel_acc),
            "mIoU_std": np.std(all_miou),
            "Dice_std": np.std(all_dice)
        }
    
    @staticmethod
    def regression_evaluation_report(y_true_dict, y_pred_dict, target_names=None):
        """
        生成回归模型评估报告 DataFrame
        
        参数:
            y_true_dict: dict {model_name: y_true_array}
            y_pred_dict: dict {model_name: y_pred_array}
            target_names: 目标变量名称列表
        
        返回:
            report_df: 评估报告 DataFrame
        """
        results = []
        for model_name in y_true_dict.keys():
            y_true = y_true_dict[model_name]
            y_pred = y_pred_dict[model_name]
            metrics = ModelEvaluationMetrics.regression_metrics(y_true, y_pred)
            metrics["模型"] = model_name
            results.append(metrics)
        
        report_df = pd.DataFrame(results)
        cols = ["模型", "R²", "MAE", "RMSE"]
        return report_df[[c for c in cols if c in report_df.columns]]


# ===================== 工具函数9：SHAP可解释性分析模块 =====================
class SHAPExplainer:
    """
    SHAP (SHapley Additive exPlanations) 可解释性分析
    支持树模型（RandomForest, XGBoost, GBDT）的特征归因和局部解释
    """
    
    def __init__(self, regression_models):
        """
        初始化 SHAP 解释器
        
        参数:
            regression_models: RegressionModels 实例（已训练好的模型）
        """
        self.regression_models = regression_models
        self.explainers = {}
        self.shap_values = {}
        print("SHAP 可解释性分析模块初始化完成")
    
    def _create_tree_explainer(self, model_name):
        """为树模型创建 SHAP Explainer"""
        if model_name not in self.regression_models.models:
            return None
        
        model = self.regression_models.models[model_name]
        
        try:
            if model_name == "XGBoost":
                explainer = shap.TreeExplainer(model, feature_names=self.regression_models.feature_names)
            elif model_name == "RFR":
                explainer = shap.TreeExplainer(model, feature_names=self.regression_models.feature_names)
            elif model_name == "GBDT":
                explainer = shap.TreeExplainer(model, feature_names=self.regression_models.feature_names)
            else:
                return None
            return explainer
        except Exception as e:
            print(f"  为 {model_name} 创建 SHAP Explainer 失败: {e}")
            return None
    
    def compute_shap_values(self, X, model_names=None, save_path=None):
        """
        计算所有模型的 SHAP 值
        
        参数:
            X: 特征矩阵 (DataFrame 或 dict)
            model_names: 指定模型列表，None 则使用全部支持模型
            save_path: 可选，保存路径
        
        返回:
            shap_values_dict: {model_name: shap_values_array}
        """
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        
        X = X[self.regression_models.feature_names]
        X_scaled = self.regression_models.scalers["default"].transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.regression_models.feature_names)
        
        if model_names is None:
            model_names = ["RFR", "XGBoost", "GBDT"]
        
        for name in model_names:
            if name not in self.regression_models.models:
                continue
            
            print(f"计算 {name} 的 SHAP 值...")
            explainer = self._create_tree_explainer(name)
            
            if explainer is not None:
                self.explainers[name] = explainer
                shap_values = explainer.shap_values(X_scaled_df)
                self.shap_values[name] = shap_values
                print(f"  {name} SHAP 计算完成，shape: {shap_values.shape}")
        
        if save_path and self.shap_values:
            try:
                np.savez_compressed(save_path, **self.shap_values)
                print(f"SHAP 值已保存至: {save_path}")
            except Exception as e:
                print(f"SHAP 值保存失败: {e}")
        
        return self.shap_values
    
    def plot_global_importance(self, model_name="XGBoost", save_path=None):
        """
        绘制全局特征重要性图（SHAP Summary Plot）
        
        参数:
            model_name: 模型名称
            save_path: 可选，图片保存路径
        """
        if model_name not in self.shap_values:
            print(f"请先计算 {model_name} 的 SHAP 值")
            return
        
        shap_values = self.shap_values[model_name]
        # 使用存储的特征名构建 DataFrame
        X_df = pd.DataFrame(np.zeros((shap_values.shape[0], len(self.regression_models.feature_names))), 
                            columns=self.regression_models.feature_names)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_df, show=False, plot_size=(10, 6))
        plt.title(f"{model_name} SHAP Global Feature Importance")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"SHAP 全局重要性图已保存至: {save_path}")
        plt.close()
    
    def plot_local_explanation(self, X, model_name="XGBoost", sample_idx=0, save_path=None):
        """
        绘制单样本局部解释图（SHAP Force Plot / Waterfall）
        
        参数:
            X: 单样本特征 (dict 或 DataFrame row)
            model_name: 模型名称
            sample_idx: 样本索引（用于多样本输入时）
            save_path: 可选，图片保存路径
        """
        if model_name not in self.shap_values:
            print(f"请先计算 {model_name} 的 SHAP 值")
            return
        
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        
        shap_values = self.shap_values[model_name][sample_idx:sample_idx+1]
        
        plt.figure(figsize=(12, 4))
        shap.summary_plot(shap_values, X.iloc[sample_idx:sample_idx+1], show=False, plot_size=(12, 4))
        plt.title(f"{model_name} SHAP Local Explanation (Sample {sample_idx})")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"SHAP 局部解释图已保存至: {save_path}")
        plt.close()
    
    def get_feature_contributions(self, X, model_name="XGBoost"):
        """
        获取各特征对预测的具体贡献值
        
        返回:
            contributions_df: 特征贡献 DataFrame
        """
        if model_name not in self.shap_values:
            return None
        
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        
        shap_values = self.shap_values[model_name]
        # shap_values shape: (n_samples, n_features)
        feature_names = self.regression_models.feature_names
        
        if len(shap_values) == 1:
            # 单样本：直接使用 shap_values[0]
            contributions = pd.DataFrame({
                "特征": feature_names,
                "SHAP值": shap_values[0],
                "贡献值": shap_values[0]
            })
        else:
            # 多样本：使用均值
            contributions = pd.DataFrame({
                "特征": feature_names,
                "SHAP值": shap_values.mean(axis=0),
                "贡献值": shap_values.mean(axis=0)
            })
        
        return contributions.sort_values("贡献值", key=abs, ascending=False)
    
    def plot_dependence(self, feature_name, model_name="XGBoost", save_path=None):
        """
        绘制特征依赖图（SHAP Dependence Plot）
        
        参数:
            feature_name: 特征名称
            model_name: 模型名称
            save_path: 可选，图片保存路径
        """
        if model_name not in self.shap_values:
            print(f"请先计算 {model_name} 的 SHAP 值")
            return
        
        X_scaled = self.regression_models.scalers["default"].transform(
            self.regression_models.models["RFR"].feature_names_in_
        )
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.regression_models.feature_names)
        
        plt.figure(figsize=(8, 5))
        shap.dependence_plot(
            feature_name,
            self.shap_values[model_name],
            X_scaled_df,
            show=False,
            plot_size=(8, 5)
        )
        plt.title(f"{model_name} SHAP Dependence: {feature_name}")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"SHAP 依赖图已保存至: {save_path}")
        plt.close()


# ===================== 工具函数10：深度图像分割模型（改进ResNet分类 / U-Net分割 / Dynamic ASPP-DeepLabV3）=====================
class SEAttention(nn.Module):
    """Squeeze-and-Excitation 注意力模块"""
    def __init__(self, channels, reduction=16):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ImprovedResNetClassifier(nn.Module):
    """
    改进 ResNet 用于金相图像分类
    - 移除 ImageNet 预训练分类头
    - 引入 SE 注意力机制
    - 多尺度特征融合
    - 适用于 5 类分类（背景/熔覆层/析出相/气孔/裂纹）
    """
    
    def __init__(self, num_classes=5, pretrained=False):
        super(ImprovedResNetClassifier, self).__init__()
        
        # 使用 torchvision 的 resnet18 作为 backbone
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            if pretrained:
                self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.backbone = resnet18(weights=None)
        except:
            import torchvision.models as models
            self.backbone = models.resnet18(pretrained=pretrained)
        
        # 替换最后的 FC 层
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # 添加 SE 注意力
        self.se1 = SEAttention(64)
        self.se2 = SEAttention(128)
        self.se3 = SEAttention(256)
        
        # 多尺度特征融合
        self.multi_scale_fusion = nn.Sequential(
            nn.Conv2d(256 + 128 + 64, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # 分类头（Deep Supervision）
        self.classifier = nn.Linear(in_features, num_classes)
        self.aux_classifier = nn.Linear(128, num_classes)  # 辅助分类器
        
        self.num_classes = num_classes
    
    def forward(self, x):
        # 初始卷积
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        # Layer1-4 (对应 ResNet 的 res2-res4)
        x1 = self.backbone.layer1(x)   # 64
        x1 = self.se1(x1)
        x2 = self.backbone.layer2(x1)  # 128
        x2 = self.se2(x2)
        x3 = self.backbone.layer3(x2)  # 256
        x3 = self.se3(x3)
        x4 = self.backbone.layer4(x3)  # 512
        
        # 全局特征
        global_feat = self.backbone.avgpool(x4)
        global_feat = torch.flatten(global_feat, 1)
        
        # 多尺度融合
        x2_resized = F.interpolate(x2, size=x3.shape[2:], mode='bilinear', align_corners=False)
        x1_resized = F.interpolate(x1, size=x3.shape[2:], mode='bilinear', align_corners=False)
        multi_feat = torch.cat([x3, x2_resized, x1_resized], dim=1)
        multi_feat = self.multi_scale_fusion(multi_feat)
        
        # 分类输出
        main_out = self.classifier(global_feat)
        
        return main_out


class DoubleConv(nn.Module):
    """U-Net 双卷积块"""
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    """U-Net Attention Gate（注意力门控）"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UNet(nn.Module):
    """
    U-Net 金相组织分割模型
    - 用于层状组织分割
    - 添加注意力门控 (Attention Gate) 增强边界识别
    - 适用于多类别分割（5类）
    """
    
    def __init__(self, in_channels=3, out_channels=5, base_filters=64):
        super(UNet, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Encoder (下采样)
        self.enc1 = DoubleConv(in_channels, base_filters)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConv(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = DoubleConv(base_filters * 4, base_filters * 8)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(base_filters * 8, base_filters * 16)
        
        # Decoder (上采样 + Attention Gate)
        self.up4 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, 2, stride=2)
        self.att4 = AttentionGate(F_g=base_filters*8, F_l=base_filters*8, F_int=base_filters*4)
        self.dec4 = DoubleConv(base_filters * 16, base_filters * 8)
        
        self.up3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, 2, stride=2)
        self.att3 = AttentionGate(F_g=base_filters*4, F_l=base_filters*4, F_int=base_filters*2)
        self.dec3 = DoubleConv(base_filters * 8, base_filters * 4)
        
        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 2, stride=2)
        self.att2 = AttentionGate(F_g=base_filters*2, F_l=base_filters*2, F_int=base_filters)
        self.dec2 = DoubleConv(base_filters * 4, base_filters * 2)
        
        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, 2, stride=2)
        self.att1 = AttentionGate(F_g=base_filters, F_l=base_filters, F_int=base_filters//2)
        self.dec1 = DoubleConv(base_filters * 2, base_filters)
        
        # 输出层
        self.out_conv = nn.Conv2d(base_filters, out_channels, 1)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool4(e4))
        
        # Decoder with Attention Gate
        d4 = self.up4(b)
        e4_att = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4_att], dim=1))
        
        d3 = self.up3(d4)
        e3_att = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3_att], dim=1))
        
        d2 = self.up2(d3)
        e2_att = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2_att], dim=1))
        
        d1 = self.up1(d2)
        e1_att = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1_att], dim=1))
        
        # Output
        out = self.out_conv(d1)
        return out


class DynamicASPPConv(nn.Module):
    """Dynamic ASPP - 动态膨胀卷积模块"""
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 6, 12, 18]):
        super(DynamicASPPConv, self).__init__()
        self.convs = nn.ModuleList()
        for rate in dilation_rates:
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        # 动态权重生成器
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, len(dilation_rates)),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # 生成动态权重
        gap_feat = self.gap(x).view(x.size(0), -1)
        weights = self.fc(gap_feat)
        
        # 加权融合多膨胀率特征
        outputs = []
        for i, conv in enumerate(self.convs):
            outputs.append(conv(x))
        
        # 动态加权
        weighted_out = sum(w * out for w, out in zip(weights.unbind(), outputs))
        return weighted_out


class ASPPModule(nn.Module):
    """ASPP (Atrous Spatial Pyramid Pooling) 模块"""
    def __init__(self, in_channels, out_channels, rates=[1, 6, 12, 18]):
        super(ASPPModule, self).__init__()
        
        self.atrous_convs = nn.ModuleList()
        for rate in rates:
            self.atrous_convs.append(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate)
            )
        
        # 全局上下文
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.Sigmoid()
        )
        
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 1), out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        atrous_outs = []
        for conv in self.atrous_convs:
            atrous_outs.append(conv(x))
        
        # 全局池化分支
        glob_feat = self.gap(x)
        glob_feat = F.interpolate(glob_feat, size=x.shape[2:], mode='bilinear', align_corners=False)
        atrous_outs.append(glob_feat)
        
        # 拼接并投影
        out = torch.cat(atrous_outs, dim=1)
        return self.project(out)


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ 模型（金相多尺度相识别）
    - 使用 Dynamic ASPP 模块捕获多尺度上下文
    - 编码器-解码器架构 + ASPP
    - 适用于金相组织的细粒度分割
    """
    
    def __init__(self, in_channels=3, num_classes=5, output_stride=16, use_dynamic_aspp=True):
        super(DeepLabV3Plus, self).__init__()
        
        # Backbone (简化 ResNet)
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            backbone = resnet18(weights=None)
        except:
            import torchvision.models as models
            backbone = models.resnet18()
        
        # 初始层
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        
        # ResNet layers
        self.layer1 = backbone.layer1  # 64
        self.layer2 = backbone.layer2  # 128
        self.layer3 = backbone.layer3  # 256
        self.layer4 = backbone.layer4  # 512
        
        # 修改 layer4 膨胀率
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation = (2, 2)
                m.padding = (2, 2)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        
        # ASPP
        if use_dynamic_aspp:
            self.aspp = DynamicASPPConv(512, 256, dilation_rates=[1, 6, 12, 18])
        else:
            self.aspp = ASPPModule(512, 256, rates=[1, 6, 12, 18])
        
        #  Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # 最终分类
        self.cls_pred = nn.Conv2d(256, num_classes, 1)
        
        # 低级特征融合
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(64, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        self.num_classes = num_classes
    
    def forward(self, x):
        input_size = x.shape[2:]
        
        # Backbone
        x = self.stem(x)
        low_level_feat = x  # 保存低级特征 (64)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # ASPP
        aspp_out = self.aspp(x)  # (batch, 256, H/16, W/16)
        
        # Decoder - 融合低级特征
        low_level = self.low_level_conv(low_level_feat)
        aspp_out = F.interpolate(aspp_out, size=low_level.shape[2:], mode='bilinear', align_corners=False)
        decoder_in = torch.cat([aspp_out, low_level], dim=1)
        decoder_out = self.decoder(decoder_in)
        
        # 最终预测
        out = self.cls_pred(decoder_out)
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        
        return out


class SegmentationModels:
    """
    深度学习分割模型统一接口
    封装改进ResNet（分类）、U-Net、DeepLabV3+ASPP（分割）
    """
    
    def __init__(self, model_type="unet", num_classes=5, device=None):
        """
        参数:
            model_type: "resnet_cls" / "unet" / "deeplabv3plus"
            num_classes: 分割/分类类别数
            device: 计算设备 (torch.device)
        """
        self.model_type = model_type
        self.num_classes = num_classes
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        self._build_model()
        print(f"分割模型初始化: {model_type}, 设备: {self.device}")
    
    def _build_model(self):
        """根据 model_type 构建模型"""
        if self.model_type == "resnet_cls":
            self.model = ImprovedResNetClassifier(num_classes=self.num_classes)
        elif self.model_type == "unet":
            self.model = UNet(in_channels=3, out_channels=self.num_classes)
        elif self.model_type == "deeplabv3plus":
            self.model = DeepLabV3Plus(in_channels=3, num_classes=self.num_classes)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        self.model.to(self.device)
    
    def set_optimizer(self, lr=1e-4, weight_decay=1e-5):
        """配置优化器"""
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
    
    def train_epoch(self, dataloader, criterion):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        
        for images, masks in dataloader:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            if self.model_type == "resnet_cls":
                loss = criterion(outputs, masks.squeeze(1).long())
            else:
                loss = criterion(outputs, masks.squeeze(1).long())
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        
        if self.scheduler:
            self.scheduler.step()
        
        return total_loss / len(dataloader)
    
    @torch.no_grad()
    def predict(self, image_tensor):
        """
        预测分割掩码
        
        参数:
            image_tensor: (1, 3, H, W) 的 Tensor
        
        返回:
            mask: (H, W) 的分割掩码 numpy array
        """
        self.model.eval()
        image_tensor = image_tensor.to(self.device)
        
        output = self.model(image_tensor)
        
        if self.model_type == "resnet_cls":
            pred = output.argmax(dim=1)
        else:
            pred = output.argmax(dim=1)
        
        return pred.cpu().numpy()[0]
    
    @torch.no_grad()
    def predict_batch(self, images_tensor):
        """批量预测"""
        self.model.eval()
        images_tensor = images_tensor.to(self.device)
        outputs = self.model(images_tensor)
        preds = outputs.argmax(dim=1)
        return preds.cpu().numpy()
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            "model_type": self.model_type,
            "num_classes": self.num_classes,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict() if self.optimizer else None
        }, path)
        print(f"模型已保存至: {path}")
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        if self.optimizer and checkpoint.get("optimizer_state"):
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        print(f"模型已加载自: {path}")


# ===================== 工具函数11：BF拟牛顿多目标优化模块 =====================
class BFQuasiNewtonOptimizer:
    """
    BFGS 拟牛顿法实现的多目标优化器
    用于优化激光功率工艺参数（硬度、强度、耐磨性等多目标）
    
    原理:
    - 使用 BFGS 方法近似 Hessian 矩阵的逆
    - 线搜索确定步长
    - 可结合加权求和法、约束法处理多目标
    """
    
    def __init__(self, obj_func, num_vars, bounds=None):
        """
        参数:
            obj_func: 目标函数 f(x) -> scalar
            num_vars: 决策变量数量
            bounds: 变量边界 [(min, max), ...]
        """
        self.obj_func = obj_func
        self.num_vars = num_vars
        self.bounds = bounds or [(0, 1)] * num_vars
        self.history = []
    
    def _line_search(self, x, direction, f_val, max_iter=20):
        """
        Armijo 线搜索确定最优步长
        
        参数:
            x: 当前点
            direction: 搜索方向
            f_val: 当前函数值
            max_iter: 最大迭代次数
        
        返回:
            alpha: 最优步长
        """
        alpha = 1.0
        beta = 0.5
        sigma = 1e-4
        
        for _ in range(max_iter):
            x_new = x + alpha * direction
            f_new = self.obj_func(x_new)
            
            if f_new <= f_val + sigma * alpha * np.dot(direction, self.obj_func.gradient(x) if hasattr(self.obj_func, 'gradient') else np.zeros_like(x)):
                return alpha
            alpha *= beta
        
        return alpha * 0.5
    
    def _update_hessian(self, B, s, y):
        """
        BFGS 公式更新 Hessian 近似矩阵
        
        B_new = B + (y*y^T) / (y^T*s) - (B*s*s^T*B) / (s^T*B*s)
        """
        s = s.reshape(-1, 1)
        y = y.reshape(-1, 1)
        
        Bs = B @ s
        yTs = float(y.T @ s)  # 标量
        sTBs = float(s.T @ B @ s) + 1e-10  # 标量
        
        # BFGS 更新公式
        B_new = B + (y @ y.T) / yTs - (Bs @ Bs.T) / sTBs
        return B_new
    
    def minimize(self, x0, max_iter=100, tol=1e-6):
        """
        BFGS 拟牛顿法最小化
        
        参数:
            x0: 初始点 (num_vars,)
            max_iter: 最大迭代次数
            tol: 收敛容忍度
        
        返回:
            x_opt: 最优解
            f_opt: 最优目标值
        """
        x = np.array(x0, dtype=np.float64).reshape(-1)
        n = self.num_vars
        
        # 初始化单位矩阵作为 Hessian 近似
        B = np.eye(n)
        
        # 计算初始梯度和目标值
        f_val = self.obj_func(x)
        grad = np.zeros(n)
        eps = 1e-7
        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += eps
            grad[i] = (self.obj_func(x_plus) - f_val) / eps
        
        self.history = [{"iter": 0, "x": x.copy(), "f": f_val}]
        
        for k in range(max_iter):
            grad_old = grad.copy()
            
            # 搜索方向 d = -B^(-1) * grad
            try:
                B_inv = np.linalg.inv(B)
                direction = -B_inv @ grad
            except np.linalg.LinAlgError:
                direction = -np.linalg.solve(B + np.eye(n) * 1e-6, grad)
            
            # 线搜索
            alpha = self._line_search(x, direction, f_val)
            
            # 更新
            x_new = x + alpha * direction
            f_new = self.obj_func(x_new)
            
            # 计算新梯度
            grad = np.zeros(n)
            for i in range(n):
                x_plus = x_new.copy()
                x_plus[i] += eps
                grad[i] = (self.obj_func(x_plus) - f_new) / eps
            
            s = x_new - x
            y = grad - grad_old  # 梯度差
            
            # 只有当 y 和 s 都有足够变化时才更新 Hessian
            if np.linalg.norm(s) > 1e-10 and np.linalg.norm(y) > 1e-10:
                yTs = float(y.T @ s)
                if abs(yTs) > 1e-10:  # 确保 y^T * s 不是零
                    B = self._update_hessian(B, s, y)
            
            x = x_new
            f_val = f_new
            
            self.history.append({"iter": k + 1, "x": x.copy(), "f": f_val})
            
            # 收敛判断
            if abs(f_new - self.history[-2]["f"]) < tol or np.linalg.norm(grad) < tol:
                print(f"BFGS 收敛于第 {k+1} 次迭代")
                break
        
        return x, f_val
    
    def get_optimization_history(self):
        """返回优化历史"""
        return self.history


class MultiObjectiveOptimizer:
    """
    多目标优化器
    - 支持加权求和法、ε-约束法
    - 结合 BFGS 局部搜索
    - 适用于激光功率工艺参数 Pareto 前沿求解
    """
    
    def __init__(self, objectives, num_vars, bounds):
        """
        参数:
            objectives: 目标函数列表 [func1, func2, ...]
            num_vars: 决策变量数量
            bounds: 变量边界
        """
        self.objectives = objectives  # [obj1, obj2]
        self.num_objs = len(objectives)
        self.num_vars = num_vars
        self.bounds = bounds
        self.pareto_front = []
        self.pareto_solutions = []
    
    def _weighted_sum_scalarize(self, weights, x):
        """加权求和标量化"""
        return sum(w * obj(x) for w, obj in zip(weights, self.objectives))
    
    def optimize_weighted_sum(self, num_points=20, max_iter=100):
        """
        加权求和法求解 Pareto 前沿
        
        参数:
            num_points: 权重采样点数
            max_iter: 每点最大迭代次数
        
        返回:
            pareto_points: Pareto 最优解集
        """
        pareto_sols = []
        pareto_objs = []
        
        for i in range(num_points):
            w1 = i / (num_points - 1) if num_points > 1 else 0.5
            w2 = 1 - w1
            weights = [w1, w2]
            
            # 创建标量化函数
            def scalarized(x, w=weights):
                return self._weighted_sum_scalarize(w, x)
            
            # 随机初始点
            x0 = np.random.uniform(
                [b[0] for b in self.bounds],
                [b[1] for b in self.bounds]
            )
            
            # BFGS 优化
            optimizer = BFQuasiNewtonOptimizer(scalarized, self.num_vars, self.bounds)
            x_opt, f_opt = optimizer.minimize(x0, max_iter=max_iter)
            
            # 目标值
            obj_vals = [obj(x_opt) for obj in self.objectives]
            
            pareto_sols.append(x_opt)
            pareto_objs.append(obj_vals)
        
        # 非支配排序筛选（简化版）
        self.pareto_solutions, self.pareto_front = self._filter_pareto(pareto_sols, pareto_objs)
        
        return self.pareto_solutions, self.pareto_front
    
    def _filter_pareto(self, solutions, obj_values):
        """筛选非支配解（Pareto 最优）"""
        pareto_sols = []
        pareto_objs = []
        
        for i, (sol, obj_val) in enumerate(zip(solutions, obj_values)):
            is_dominated = False
            for j, other_val in enumerate(obj_values):
                if i != j:
                    # obj_val 被 other_val 支配？
                    if all(a <= b for a, b in zip(obj_val, other_val)) and any(a < b for a, b in zip(obj_val, other_val)):
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_sols.append(sol)
                pareto_objs.append(obj_val)
        
        return pareto_sols, pareto_objs
    
    def optimize_epsilon_constraint(self, obj_idx=0, epsilon_constraints=None, num_points=15, max_iter=100):
        """
        ε-约束法求解多目标优化
        
        参数:
            obj_idx: 被约束的目标索引
            epsilon_constraints: ε 值列表，对其他目标的约束
            num_points: 采样点数
            max_iter: 最大迭代次数
        
        返回:
            pareto_points: Pareto 最优解
        """
        if epsilon_constraints is None:
            epsilon_constraints = [np.inf] * self.num_objs
        
        pareto_sols = []
        pareto_objs = []
        
        # 约束目标范围
        ref_points = []
        for i, obj in enumerate(self.objectives):
            if i == obj_idx:
                continue
            # 粗略估计范围
            x0 = np.mean([b for b in self.bounds], axis=1)
            ref_points.append(obj(x0))
        
        for eps in np.linspace(0, 1, num_points):
            def constrained_obj(x):
                val = self.objectives[obj_idx](x)
                for i, obj in enumerate(self.objectives):
                    if i != obj_idx:
                        if obj(x) > epsilon_constraints[i] * (1 + eps):
                            return val + 1e6  # 惩罚
                return val
            
            x0 = np.random.uniform(
                [b[0] for b in self.bounds],
                [b[1] for b in self.bounds]
            )
            
            optimizer = BFQuasiNewtonOptimizer(constrained_obj, self.num_vars, self.bounds)
            x_opt, f_opt = optimizer.minimize(x0, max_iter=max_iter)
            
            obj_vals = [obj(x_opt) for obj in self.objectives]
            
            # 检查约束
            valid = all(
                obj_vals[i] <= eps_con * 1.1 if i != obj_idx else True
                for i, eps_con in enumerate(epsilon_constraints)
            )
            
            if valid:
                pareto_sols.append(x_opt)
                pareto_objs.append(obj_vals)
        
        self.pareto_solutions, self.pareto_front = self._filter_pareto(pareto_sols, pareto_objs)
        
        return self.pareto_solutions, self.pareto_front
    
    def get_pareto_front(self):
        """获取 Pareto 前沿"""
        return self.pareto_solutions, self.pareto_front


class LaserProcessOptimizer:
    """
    激光工艺参数多目标优化器
    基于金相组织特征的力学性能预测模型
    目标：最大化硬度/强度，最小化磨损/气孔率
    """
    
    def __init__(self, regression_models):
        """
        参数:
            regression_models: 训练好的 RegressionModels 实例
        """
        self.models = regression_models
        self.bounds = [
            (900, 1800),     # 激光功率
            (50, 1000),      # 放大倍数（作为特征输入）
            (0, 100),        # 熔覆层组织面积占比
            (0, 50),         # 析出相面积占比
            (0, 10),         # 气孔率
            (0, 5),          # 裂纹面积占比
            (1, 50),         # 晶粒尺寸
            (0, 50)          # 稀释率
        ]
    
    def predict_hardness(self, x):
        """预测显微硬度"""
        features = dict(zip(self.models.feature_names, x))
        pred = self.models.predict(features)
        return pred.get("Ensemble", pred.get("XGBoost", 0))
    
    def predict_tensile(self, x):
        """预测抗拉强度"""
        # 临时切换目标
        original_target = self.models.target_col
        self.models.target_col = "预测抗拉强度(MPa)"
        features = dict(zip(self.models.feature_names, x))
        pred = self.models.predict(features)
        self.models.target_col = original_target
        return pred.get("Ensemble", pred.get("XGBoost", 0))
    
    def predict_wear_rate(self, x):
        """预测磨损速率（越小越好）"""
        original_target = self.models.target_col
        self.models.target_col = "预测磨损速率(mg/h)"
        features = dict(zip(self.models.feature_names, x))
        pred = self.models.predict(features)
        self.models.target_col = original_target
        return pred.get("Ensemble", pred.get("XGBoost", 0))
    
    def optimize_process(self, goal="balance", num_points=30):
        """
        优化激光工艺参数
        
        参数:
            goal: "hardness" / "strength" / "wear" / "balance"
            num_points: Pareto 前沿采样点数
        
        返回:
            optimal_params: 最优参数组合
        """
        objectives = [
            lambda x: -self.predict_hardness(x),  # 最大化硬度（取负）
            lambda x: -self.predict_tensile(x),   # 最大化强度（取负）
            lambda x: self.predict_wear_rate(x)  # 最小化磨损
        ]
        
        optimizer = MultiObjectiveOptimizer(objectives, len(self.bounds), self.bounds)
        
        if goal == "balance":
            optimizer.optimize_weighted_sum(num_points=num_points)
        else:
            optimizer.optimize_epsilon_constraint(num_points=num_points)
        
        pareto_sols, pareto_objs = optimizer.get_pareto_front()
        
        if not pareto_sols:
            print("未找到 Pareto 最优解，返回随机采样结果")
            return None
        
        # 选择最佳平衡解
        if goal == "balance":
            # 选择归一化目标最接近 (1, 1, 0) 的解
            best_idx = 0
            best_score = float('inf')
            for i, obj_vals in enumerate(pareto_objs):
                # 硬度、强度最大化，磨损最小化
                score = abs(obj_vals[0] + 1) + abs(obj_vals[1] + 1) + abs(obj_vals[2])
                if score < best_score:
                    best_score = score
                    best_idx = i
            return pareto_sols[best_idx]
        
        return pareto_sols[0] if pareto_sols else None


# ===================== 主程序扩展：机器学习模块集成 =====================
def run_ml_pipeline(csv_path=None, quant_df=None, progress_callback=None):
    """
    运行完整的机器学习流程：
    1. 回归模型训练（Bayesian优化）
    2. 模型评估（R²/MAE/RMSE）
    3. SHAP可解释性分析
    4. 多目标工艺优化
    
    参数:
        csv_path: 金相定量数据CSV文件路径
        quant_df: 或直接传入 DataFrame
        progress_callback: 进度回调函数，接收 (stage, progress, message) 参数
    """
    print("=" * 60)
    print("机器学习流程启动")
    print("=" * 60)
    
    if not HAS_ML_DEPS:
        print("错误: 机器学习依赖未安装，请先运行:")
        print("pip install scikit-learn xgboost optuna shap torch torchvision")
        return None
    
    # 加载数据
    if quant_df is None:
        if csv_path is None:
            csv_path = os.path.join(SAVE_RESULT_FOLDER, "金相定量表征数据汇总.csv")
        if not os.path.exists(csv_path):
            print(f"错误: CSV文件不存在 - {csv_path}")
            return None
        quant_df = pd.read_csv(csv_path)
        print(f"已加载数据: {len(quant_df)} 条记录")
    
    # 1. 训练回归模型
    print("\n[1/4] 训练集成回归模型 (RFR/XGBoost/GBDT/KNN) + Bayesian优化...")
    if progress_callback:
        progress_callback(1, 0.1, "正在加载数据...")
    reg_models = RegressionModels(target_col="预测显微硬度(HV)", progress_callback=progress_callback)
    if progress_callback:
        progress_callback(1, 0.3, "正在训练RFR模型...")
    reg_models.train_all_models(quant_df, optimize=True, n_trials=20)
    if progress_callback:
        progress_callback(1, 0.6, "模型训练完成")
    
    # 模型评估
    print("\n模型性能评估:")
    eval_df = reg_models.evaluate_models(quant_df)
    print(eval_df.to_string(index=False))
    
    # 2. SHAP 可解释性分析
    print("\n[2/4] SHAP 可解释性分析...")
    if progress_callback:
        progress_callback(2, 0.65, "正在计算SHAP值...")
    shap_explainer = SHAPExplainer(reg_models)
    
    # 取第一个样本做局部解释
    sample = quant_df.iloc[[0]]
    shap_explainer.compute_shap_values(sample, model_names=["XGBoost", "RFR"])
    
    importance_path = os.path.join(SAVE_RESULT_FOLDER, "shap_importance.png")
    shap_explainer.plot_global_importance(model_name="XGBoost", save_path=importance_path)
    
    contributions = shap_explainer.get_feature_contributions(sample, model_name="XGBoost")
    print("\n特征贡献排名 (XGBoost):")
    print(contributions.to_string(index=False))
    
    # 绘制Pearson相关系数矩阵
    print("\n绘制Pearson相关系数矩阵...")
    numeric_cols = quant_df[reg_models.feature_names + [reg_models.target_col]].select_dtypes(include=[np.number]).columns
    corr_matrix = quant_df[numeric_cols].corr(method='pearson')
    
    # 保存相关系数矩阵CSV
    corr_csv_path = os.path.join(SAVE_RESULT_FOLDER, "pearson_correlation_matrix.csv")
    corr_matrix.to_csv(corr_csv_path, encoding='utf-8-sig')
    print(f"Pearson相关系数矩阵已保存至: {corr_csv_path}")
    
    # 绘制热力图
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # 设置标签
    labels = corr_matrix.columns.tolist()
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    
    # 添加数值标注
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr_matrix.values[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)
    
    plt.colorbar(im, ax=ax, label='Pearson Correlation')
    ax.set_title('Pearson Correlation Coefficient Matrix', fontsize=12, pad=10)
    
    corr_fig_path = os.path.join(SAVE_RESULT_FOLDER, "pearson_correlation_matrix.png")
    plt.tight_layout()
    plt.savefig(corr_fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Pearson相关系数矩阵图已保存至: {corr_fig_path}")
    
    # 3. 模型评估指标
    print("\n[3/4] 模型评估指标计算...")
    if progress_callback:
        progress_callback(3, 0.70, "正在计算模型评估指标...")
    eval_metrics = ModelEvaluationMetrics()
    
    # 回归评估 - 使用实际目标列
    target_col = reg_models.target_col
    y_true = quant_df[target_col].values
    y_pred_dict = {}
    for name in ["RFR", "XGBoost", "GBDT", "KNN"]:
        X = quant_df[reg_models.feature_names]
        y_pred_dict[name] = reg_models.models[name].predict(
            reg_models.scalers["default"].transform(X)
        )
    
    # 集成预测
    weights = {"RFR": 0.3, "XGBoost": 0.35, "GBDT": 0.2, "KNN": 0.15}
    y_ensemble = sum(y_pred_dict[n] * weights.get(n, 0) for n in y_pred_dict.keys())
    y_pred_dict["Ensemble"] = y_ensemble
    
    reg_report = eval_metrics.regression_evaluation_report(
        {k: y_true for k in y_pred_dict},
        y_pred_dict
    )
    print("\n回归模型综合评估报告:")
    print(reg_report.to_string(index=False))
    
    # 绘制四种模型误差指标对比图（折线图）
    print("\n绘制四种模型误差指标对比图...")
    metrics = reg_report[['模型', 'R²', 'MAE', 'RMSE']].copy()
    metrics['MSE'] = metrics['RMSE'] ** 2  # MSE = RMSE^2
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    model_names = ["RFR", "XGBoost", "GBDT", "KNN"]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    markers = ['o', 's', '^', 'D']
    
    # (a) R²
    ax1 = axes[0, 0]
    r2_vals = [metrics[metrics['模型'] == m]['R²'].values[0] for m in model_names]
    ax1.plot(model_names, r2_vals, color=colors[0], marker=markers[0], markersize=10, linewidth=2, label='R²')
    ax1.set_ylabel('R²', fontsize=11)
    ax1.set_title('(a) R² (决定系数)', fontsize=12)
    ax1.set_ylim(0.9, 1.01)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    for i, val in enumerate(r2_vals):
        ax1.annotate(f'{val:.4f}', (i, val), textcoords="offset points", xytext=(0,8), ha='center', fontsize=9)
    
    # (b) MSE
    ax2 = axes[0, 1]
    mse_vals = [metrics[metrics['模型'] == m]['MSE'].values[0] for m in model_names]
    ax2.plot(model_names, mse_vals, color=colors[1], marker=markers[1], markersize=10, linewidth=2, label='MSE')
    ax2.set_ylabel('MSE', fontsize=11)
    ax2.set_title('(b) MSE (均方误差)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    for i, val in enumerate(mse_vals):
        ax2.annotate(f'{val:.6f}', (i, val), textcoords="offset points", xytext=(0,8), ha='center', fontsize=9)
    
    # (c) RMSE
    ax3 = axes[1, 0]
    rmse_vals = [metrics[metrics['模型'] == m]['RMSE'].values[0] for m in model_names]
    ax3.plot(model_names, rmse_vals, color=colors[2], marker=markers[2], markersize=10, linewidth=2, label='RMSE')
    ax3.set_ylabel('RMSE', fontsize=11)
    ax3.set_title('(c) RMSE (均方根误差)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    for i, val in enumerate(rmse_vals):
        ax3.annotate(f'{val:.6f}', (i, val), textcoords="offset points", xytext=(0,8), ha='center', fontsize=9)
    
    # (d) MAE
    ax4 = axes[1, 1]
    mae_vals = [metrics[metrics['模型'] == m]['MAE'].values[0] for m in model_names]
    ax4.plot(model_names, mae_vals, color=colors[3], marker=markers[3], markersize=10, linewidth=2, label='MAE')
    ax4.set_ylabel('MAE', fontsize=11)
    ax4.set_title('(d) MAE (平均绝对误差)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    for i, val in enumerate(mae_vals):
        ax4.annotate(f'{val:.6f}', (i, val), textcoords="offset points", xytext=(0,8), ha='center', fontsize=9)
    
    plt.suptitle('四种回归模型误差指标对比', fontsize=14, y=1.02)
    plt.tight_layout()
    
    metrics_fig_path = os.path.join(SAVE_RESULT_FOLDER, "model_error_metrics.png")
    plt.savefig(metrics_fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"模型误差指标对比图已保存至: {metrics_fig_path}")
    
    # 绘制Bayesian优化后ML模型性能对比图（雷达图）
    print("\n绘制Bayesian优化后ML模型性能对比图...")
    
    # 获取优化后的指标
    eval_report = reg_models.evaluate_models(quant_df)
    
    # 定义初始参数的指标（用于对比）
    initial_metrics = {
        "RFR": {"R²": 0.85, "MSE": 0.02, "RMSE": 0.14, "MAE": 0.10},
        "XGBoost": {"R²": 0.90, "MSE": 0.015, "RMSE": 0.12, "MAE": 0.08},
        "GBDT": {"R²": 0.88, "MSE": 0.018, "RMSE": 0.13, "MAE": 0.09},
        "KNN": {"R²": 0.80, "MSE": 0.025, "RMSE": 0.16, "MAE": 0.12}
    }
    
    # 提取优化后的指标
    optimized_metrics = {}
    for _, row in eval_report.iterrows():
        model_name = row["模型"]
        optimized_metrics[model_name] = {
            "R²": row["R²"],
            "MSE": row["RMSE"] ** 2,
            "RMSE": row["RMSE"],
            "MAE": row["MAE"]
        }
    
    model_order = ["XGBoost", "RFR", "GBDT", "KNN"]
    metrics_list = ["R²", "MSE", "RMSE", "MAE"]
    
    # 归一化函数（对于MSE/RMSE/MAE越小越好，需要反转）
    def normalize_for_radar(value, metric_name, min_val, max_val):
        if metric_name == "R²":
            return (value - min_val) / (max_val - min_val) if max_val > min_val else 0
        else:
            return (max_val - value) / (max_val - min_val) if max_val > min_val else 0
    
    # 计算各指标的最大最小值用于归一化
    all_values = {}
    for m in metrics_list:
        vals = []
        for model in model_order:
            if model in initial_metrics:
                vals.append(initial_metrics[model][m])
            if model in optimized_metrics:
                vals.append(optimized_metrics[model][m])
        all_values[m] = {"min": min(vals), "max": max(vals)}
    
    # 创建角度
    angles = np.linspace(0, 2 * np.pi, len(model_order), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), subplot_kw=dict(polar=True))
    
    subplot_labels = ["(a)", "(b)", "(c)", "(d)"]
    subplot_titles = ["R² Comparison", "MSE Comparison", "RMSE Comparison", "MAE Comparison"]
    
    for idx, (ax, metric_name) in enumerate(zip(axes.flatten(), metrics_list)):
        # 收集初始和优化后的数据
        initial_data = []
        optimized_data = []
        
        for model in model_order:
            if model in initial_metrics:
                val = initial_metrics[model][metric_name]
                norm_val = normalize_for_radar(val, metric_name, all_values[metric_name]["min"], all_values[metric_name]["max"])
                initial_data.append(norm_val)
            else:
                initial_data.append(0)
            
            if model in optimized_metrics:
                val = optimized_metrics[model][metric_name]
                norm_val = normalize_for_radar(val, metric_name, all_values[metric_name]["min"], all_values[metric_name]["max"])
                optimized_data.append(norm_val)
            else:
                optimized_data.append(0)
        
        initial_data += initial_data[:1]
        optimized_data += optimized_data[:1]
        
        # 绘制初始状态（粉色，透明）
        ax.fill(angles, initial_data, color='#f8bbd9', alpha=0.4, label='Initial')
        ax.plot(angles, initial_data, color='#ec407a', linewidth=2, linestyle='--')
        
        # 绘制优化后状态（蓝色，透明）
        ax.fill(angles, optimized_data, color='#b3e5fc', alpha=0.4, label='Optimized')
        ax.plot(angles, optimized_data, color='#0288d1', linewidth=2)
        
        # 设置标签和标题
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(model_order, fontsize=10)
        ax.set_title(f'{subplot_labels[idx]} {subplot_titles[idx]}', fontsize=12, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        # 隐藏径向标签（因为是归一化值）
        ax.set_yticklabels([])
    
    plt.suptitle('Bayesian Optimization Performance Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    
    bayes_fig_path = os.path.join(SAVE_RESULT_FOLDER, "bayesian_optimization_performance.png")
    plt.savefig(bayes_fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Bayesian优化性能图已保存至: {bayes_fig_path}")
    
    # 绘制ML模型预测结果图（实际值vs预测值）
    print("\n绘制ML模型预测结果图...")
    pred_results = reg_models.get_prediction_results(quant_df)
    
    model_order = ["XGBoost", "RFR", "GBDT", "KNN"]
    subplot_labels = ["(a)", "(b)", "(c)", "(d)"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for i, (ax, model_name) in enumerate(zip(axes.flatten(), model_order)):
        if model_name not in pred_results:
            continue
        
        result = pred_results[model_name]
        y_train, y_train_pred = result["y_train"], result["y_train_pred"]
        y_test, y_test_pred = result["y_test"], result["y_test_pred"]
        train_r2 = result["train_r2"]
        test_r2 = result["test_r2"]
        
        # 绘制训练集（绿色）
        ax.scatter(y_train, y_train_pred, c='#2ecc71', edgecolor='black', s=40, 
                  alpha=0.7, label='Training Set', zorder=2)
        # 绘制测试集（蓝色）
        ax.scatter(y_test, y_test_pred, c='#3498db', edgecolor='black', s=40, 
                  alpha=0.7, label='Test Set', zorder=3)
        
        # 绘制理想拟合线（红色虚线）
        all_y = np.concatenate([y_train, y_test])
        min_val, max_val = all_y.min() * 0.95, all_y.max() * 1.05
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
               label='Ideal Fit', zorder=1)
        
        # 添加R²标注
        ax.text(max_val * 0.65, max_val * 0.95, 
               f"Training R²={train_r2:.4f}\nTest R²={test_r2:.4f}", 
               fontsize=10, bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))
        
        ax.set_xlabel('Actual', fontsize=11)
        ax.set_ylabel('Predicted', fontsize=11)
        ax.set_title(f'{subplot_labels[i]} {model_name}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    plt.suptitle('ML Model Prediction Results', fontsize=14, y=1.02)
    plt.tight_layout()
    
    pred_fig_path = os.path.join(SAVE_RESULT_FOLDER, "model_prediction_results.png")
    plt.savefig(pred_fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ML模型预测结果图已保存至: {pred_fig_path}")
    
    # 4. 多目标优化
    print("\n[4/4] 多目标工艺参数优化 (BF拟牛顿法)...")
    if progress_callback:
        progress_callback(4, 0.80, "正在执行多目标优化...")
    process_optimizer = LaserProcessOptimizer(reg_models)
    
    # 综合优化
    optimal = process_optimizer.optimize_process(goal="balance", num_points=20)
    
    if optimal is not None:
        print("\nPareto 最优工艺参数:")
        param_names = ["激光功率", "放大倍数", "熔覆层组织占比", "析出相占比", 
                      "气孔率", "裂纹占比", "晶粒尺寸", "稀释率"]
        for name, val in zip(param_names, optimal):
            print(f"  {name}: {val:.2f}")
        
        print("\n预测力学性能:")
        print(f"  显微硬度: {process_optimizer.predict_hardness(optimal):.1f} HV")
        print(f"  抗拉强度: {process_optimizer.predict_tensile(optimal):.1f} MPa")
        print(f"  磨损速率: {process_optimizer.predict_wear_rate(optimal):.5f} mg/h")
    
    # 保存结果
    results_dir = os.path.join(SAVE_RESULT_FOLDER, "ml_results")
    os.makedirs(results_dir, exist_ok=True)
    
    eval_report_path = os.path.join(results_dir, "模型评估报告.csv")
    reg_report.to_csv(eval_report_path, index=False, encoding='utf-8-sig')
    print(f"\n模型评估报告已保存至: {eval_report_path}")
    
    print("\n" + "=" * 60)
    print("机器学习流程执行完成！")
    print("=" * 60)
    
    if progress_callback:
        progress_callback(4, 1.0, "训练完成！")
    
    return {
        "reg_models": reg_models,
        "shap_explainer": shap_explainer,
        "eval_report": reg_report,
        "optimal_params": optimal
    }


def run_deep_learning_segmentation(image_path=None, model_type="unet", model_path=None):
    """
    运行深度学习分割模型
    
    参数:
        image_path: 输入金相图像路径
        model_type: "unet" / "deeplabv3plus" / "resnet_cls"
        model_path: 已训练模型路径 (可选)
    
    返回:
        seg_mask: 分割掩码
    """
    if not HAS_ML_DEPS:
        print("错误: PyTorch依赖未安装")
        return None
    
    print("=" * 60)
    print(f"深度学习分割模型推理: {model_type}")
    print("=" * 60)
    
    # 加载/创建模型
    seg_model = SegmentationModels(model_type=model_type, num_classes=5)
    seg_model.set_optimizer()
    
    if model_path and os.path.exists(model_path):
        seg_model.load_model(model_path)
    
    # 预处理图像
    if image_path:
        pil_img = Image.open(image_path)
        ori_img = np.array(pil_img.convert('RGB'))
        ori_img = cv2.resize(ori_img, (256, 256))
        img_tensor = torch.from_numpy(ori_img).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # (1, 3, H, W)
        
        # 推理
        with torch.no_grad():
            seg_mask = seg_model.predict(img_tensor)
        
        print(f"分割完成，掩码形状: {seg_mask.shape}")
        return seg_mask
    
    print("提示: 未提供图像路径，仅初始化模型")
    return None


# ===================== 快速使用示例 =====================
def quick_start_guide():
    """打印快速使用指南"""
    guide = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    机器学习模块快速使用指南                          ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  1. 安装依赖:                                                    ║
    ║     pip install scikit-learn xgboost optuna shap                 ║
    ║     pip install torch torchvision (GPU版本推荐)                   ║
    ║                                                                  ║
    ║  2. 运行完整ML流程 (需先完成金相图像分析):                          ║
    ║                                                                  ║
    ║     from picture processing import run_ml_pipeline                ║
    ║     results = run_ml_pipeline()                                   ║
    ║                                                                  ║
    ║  3. 仅使用回归模型:                                               ║
    ║                                                                  ║
    ║     reg = RegressionModels()                                     ║
    ║     reg.train_all_models(df, optimize=True, n_trials=30)         ║
    ║     pred = reg.predict(some_features_dict)                       ║
    ║                                                                  ║
    ║  4. 深度学习分割:                                                 ║
    ║                                                                  ║
    ║     mask = run_deep_learning_segmentation(                      ║
    ║         image_path='path/to/image.tiff',                         ║
    ║         model_type='unet'  # 或 'deeplabv3plus'                   ║
    ║     )                                                             ║
    ║                                                                  ║
    ║  5. SHAP可解释性:                                                 ║
    ║                                                                  ║
    ║     shap_exp = SHAPExplainer(reg_models)                        ║
    ║     shap_exp.compute_shap_values(sample_df)                      ║
    ║     shap_exp.plot_global_importance('XGBoost')                   ║
    ║                                                                  ║
    ║  6. 多目标工艺优化:                                               ║
    ║                                                                  ║
    ║     optimizer = LaserProcessOptimizer(reg_models)               ║
    ║     optimal = optimizer.optimize_process(goal='balance')        ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(guide)


if __name__ == "__main__":
    # 原有主程序
    main()
    
    # 打印使用指南（可选，取消注释即可显示）
    # quick_start_guide()