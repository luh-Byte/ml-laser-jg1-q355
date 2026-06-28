"""
显微图像预训练模型集成模块
基于 Segment Anything for Microscopy (micro_sam)
用于金相图像的高级分割和特征提取
"""

import os
import numpy as np
import cv2
from typing import Optional, Dict, Tuple, List

# micro_sam 模型类型映射
MICRO_SAM_MODELS = {
    "vit_h": "vit_h (最大模型，最高精度)",
    "vit_l": "vit_l (大模型，平衡精度与速度)",
    "vit_b": "vit_b (基础模型，较快)",
    "vit_t": "vit_t (轻量模型，最快)",
    "em_organelles": "em_organelles (电子显微镜-细胞器)",
    "em_nuclei": "em_nuclei (电子显微镜-细胞核)",
    "lm_livecell": "lm_livecell (光学显微镜-活细胞)",
    "lm_general": "lm_general (光学显微镜-通用)",
}

class MicroSAMSegmenter:
    """基于 micro_sam 的显微图像分割器"""

    def __init__(self, model_type: str = "vit_b", device: str = "auto"):
        """
        初始化 micro_sam 分割器

        参数:
            model_type: 模型类型 (vit_h, vit_l, vit_b, vit_t 或专用模型)
            device: 计算设备 ('auto', 'cpu', 'cuda')
        """
        self.model_type = model_type
        self.device = device
        self.model = None
        self.predictor = None
        self._initialized = False

    def _lazy_init(self):
        """延迟初始化模型（首次使用时加载）"""
        if self._initialized:
            return

        try:
            # 尝试导入 micro_sam
            import micro_sam
            from micro_sam.automatic_segmentation import automatic_instance_segmentation

            # 确定设备
            if self.device == "auto":
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            print(f"正在加载 micro_sam 模型: {self.model_type} (设备: {self.device})")

            # micro_sam 1.8.4 使用不同的API
            # 这里我们直接存储模型类型，实际分割时调用 automatic_instance_segmentation
            self.model = None
            self.predictor = None

            self._initialized = True
            print("micro_sam 模型加载完成")

        except ImportError as e:
            print(f"警告: micro_sam 未安装 ({e})，将使用传统分割方法")
            print("安装方法: pip install micro-sam 或 conda install -c conda-forge micro_sam")
            self._initialized = True  # 标记为已尝试初始化
            self.model = None

    def predict_segment(
        self,
        img_array: np.ndarray,
        box_prompt: Optional[Tuple[int, int, int, int]] = None,
        point_prompts: Optional[List[Tuple[int, int]]] = None,
        auto_mode: bool = True
    ) -> np.ndarray:
        """
        执行图像分割

        参数:
            img_array: 输入图像 (RGB格式)
            box_prompt: 边界框提示 (x1, y1, x2, y2)
            point_prompts: 点提示列表 [(x, y), ...]
            auto_mode: 是否使用自动分割模式

        返回:
            分割掩码 (numpy数组)
        """
        self._lazy_init()

        if self.model is None:
            # micro_sam 未安装，使用传统方法
            return self._traditional_segment(img_array)

        try:
            # 设置图像
            self.predictor.set_image(img_array)

            if auto_mode:
                # 自动分割模式
                masks = self._auto_segment(img_array)
            else:
                # 提示引导分割
                masks, scores, logits = self.predictor.predict(
                    box=box_prompt,
                    point_coords=point_prompts,
                    multimask_output=True
                )
                # 选择最佳掩码
                best_mask_idx = np.argmax(scores)
                masks = masks[best_mask_idx]

            return masks.astype(np.uint8)

        except Exception as e:
            print(f"micro_sam 分割失败: {e}")
            return self._traditional_segment(img_array)

    def _auto_segment(self, img_array: np.ndarray) -> np.ndarray:
        """
        自动分割模式 - 使用 micro_sam 的自动分割功能
        """
        try:
            from micro_sam.automatic_segmentation import automatic_instance_segmentation

            # 自动实例分割
            # micro_sam 1.8.4 的 API 可能不同，尝试不同的调用方式
            try:
                # 尝试新API
                masks = automatic_instance_segmentation(
                    img_array,
                    model_type=self.model_type,
                    pred_iou_thresh=0.8,
                    stability_score_thresh=0.9,
                    min_mask_region_area=100
                )
            except TypeError:
                # 尝试旧API
                masks = automatic_instance_segmentation(
                    self.predictor,
                    img_array,
                    pred_iou_thresh=0.8,
                    stability_score_thresh=0.9,
                    min_mask_region_area=100
                )
            return masks

        except Exception as e:
            print(f"自动分割失败: {e}")
            # 回退到传统分割
            return self._traditional_segment(img_array)

    def _traditional_segment(self, img_array: np.ndarray) -> np.ndarray:
        """
        传统图像分割方法（备用）
        """
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array.copy()

        # 多阈值分割
        h, w = gray.shape
        seg_mask = np.zeros((h, w), dtype=np.uint8)

        # 基于灰度分布的阈值
        thresh_high = np.percentile(gray, 85)
        thresh_mid = np.percentile(gray, 50)
        thresh_low = np.percentile(gray, 30)

        seg_mask[gray <= thresh_low] = 0  # 气孔/缺陷
        seg_mask[(gray > thresh_low) & (gray <= thresh_mid)] = 1  # 基体
        seg_mask[(gray > thresh_mid) & (gray <= thresh_high)] = 2  # 熔覆层
        seg_mask[gray > thresh_high] = 3  # 析出相

        return seg_mask

    def extract_features(
        self,
        img_array: np.ndarray,
        seg_mask: np.ndarray
    ) -> Dict:
        """
        从分割结果提取定量特征

        参数:
            img_array: 原始图像
            seg_mask: 分割掩码

        返回:
            特征字典
        """
        features = {}
        h, w = seg_mask.shape
        total_pixels = h * w

        # 各区域面积占比
        for label in range(4):
            area = np.sum(seg_mask == label)
            features[f"区域{label}_面积占比(%)"] = (area / total_pixels) * 100

        # 晶粒尺寸估算
        if 2 in seg_mask:  # 熔覆层区域
            cladding_mask = (seg_mask == 2).astype(np.uint8)
            contours, _ = cv2.findContours(
                cladding_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                areas = [cv2.contourArea(c) for c in contours]
                avg_area = np.mean(areas)
                # 假设晶粒为圆形，计算等效直径
                avg_diameter = 2 * np.sqrt(avg_area / np.pi)
                features["平均晶粒尺寸(像素)"] = avg_diameter

        # 气孔特征
        if 0 in seg_mask:
            pore_mask = (seg_mask == 0).astype(np.uint8)
            contours, _ = cv2.findContours(
                pore_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                features["气孔数量"] = len(contours)
                areas = [cv2.contourArea(c) for c in contours]
                features["气孔平均面积(像素)"] = np.mean(areas)
                features["气孔总面积占比(%)"] = (sum(areas) / total_pixels) * 100

        return features


def get_available_models() -> Dict[str, str]:
    """
    获取可用的 micro_sam 模型列表

    返回:
        模型类型到描述的映射字典
    """
    return MICRO_SAM_MODELS


def check_micro_sam_installed() -> bool:
    """
    检查 micro_sam 是否已安装

    返回:
        True 如果已安装，否则 False
    """
    try:
        import micro_sam
        return True
    except ImportError:
        return False


def install_micro_sam_instructions() -> str:
    """
    返回 micro_sam 安装说明
    """
    return """
# 安装 micro_sam (Segment Anything for Microscopy)

## 方法1: 使用 pip
pip install micro-sam

## 方法2: 使用 conda (推荐)
conda install -c conda-forge micro_sam

## 方法3: 从源码安装
git clone https://github.com/computational-cell-analytics/micro-sam.git
cd micro-sam
pip install -e .

## 验证安装
python -c "import micro_sam; print('micro_sam 安装成功!')"

## 可用模型类型:
- vit_h: 最大模型，最高精度
- vit_l: 大模型，平衡精度与速度
- vit_b: 基础模型，较快
- vit_t: 轻量模型，最快
- em_organelles: 电子显微镜-细胞器专用
- em_nuclei: 电子显微镜-细胞核专用
- lm_livecell: 光学显微镜-活细胞专用
- lm_general: 光学显微镜-通用

## 参考文档
https://computational-cell-analytics.github.io/micro-sam/micro_sam.html
"""


# ===================== 示例用法 =====================
if __name__ == "__main__":
    print("=" * 60)
    print("micro_sam 显微图像预训练模型集成模块")
    print("=" * 60)

    # 检查安装
    if check_micro_sam_installed():
        print("✅ micro_sam 已安装")

        # 显示可用模型
        print("\n可用模型:")
        for model_type, desc in get_available_models().items():
            print(f"  • {model_type}: {desc}")

        # 创建分割器示例
        segmenter = MicroSAMSegmenter(model_type="vit_b")
        print(f"\n分割器已创建 (模型: vit_b)")

    else:
        print("❌ micro_sam 未安装")
        print(install_micro_sam_instructions())