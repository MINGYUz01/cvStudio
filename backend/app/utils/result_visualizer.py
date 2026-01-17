"""
结果可视化工具
在图像上绘制推理结果（检测框、分类标签）
"""

from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from loguru import logger


# 预定义颜色（BGR格式）
DEFAULT_COLORS = [
    (0, 255, 0),      # 绿色
    (255, 0, 0),      # 蓝色
    (0, 0, 255),      # 红色
    (255, 255, 0),    # 青色
    (255, 0, 255),    # 洋红
    (0, 255, 255),    # 黄色
    (128, 0, 128),    # 紫色
    (255, 165, 0),    # 橙色
    (255, 192, 203),  # 粉色
    (0, 128, 128),    # 深青
]


class ResultVisualizer:
    """
    推理结果可视化器

    在图像上绘制检测框、分类标签等
    """

    def __init__(self, colors: Optional[List[Tuple[int, int, int]]] = None):
        """
        初始化可视化器

        Args:
            colors: 自定义颜色列表（BGR格式）
        """
        self.colors = colors or DEFAULT_COLORS
        self.logger = logger.bind(component="result_visualizer")

    def draw_detections(
        self,
        image: np.ndarray,
        results: List[Dict[str, Any]],
        class_names: Optional[List[str]] = None,
        show_confidence: bool = True,
        show_label: bool = True,
        thickness: int = 2,
        font_scale: float = 0.6
    ) -> np.ndarray:
        """
        绘制检测框和标签

        Args:
            image: 输入图像（BGR格式）
            results: 检测结果列表
            class_names: 类别名称列表
            show_confidence: 是否显示置信度
            show_label: 是否显示标签
            thickness: 边框线宽
            font_scale: 字体大小

        Returns:
            绘制后的图像
        """
        output = image.copy()

        for idx, result in enumerate(results):
            if result.get('type') != 'detection':
                continue

            bbox = result.get('bbox')
            if not bbox or len(bbox) < 4:
                continue

            x1, y1, x2, y2 = map(int, bbox[:4])

            # 获取颜色
            class_id = result.get('class_id', idx)
            color = self._get_color(class_id)

            # 绘制边框
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

            # 绘制标签
            if show_label or show_confidence:
                label_parts = []
                if show_label:
                    label = result.get('label', f'class_{class_id}')
                    label_parts.append(label)
                if show_confidence:
                    conf = result.get('confidence', 0)
                    label_parts.append(f"{conf:.2f}")

                label_text = ' '.join(label_parts)
                output = self._draw_label(output, label_text, (x1, y1), color, font_scale)

        self.logger.debug(f"绘制了{len(results)}个检测框")
        return output

    def draw_classification(
        self,
        image: np.ndarray,
        results: List[Dict[str, Any]],
        position: str = "top_left",
        bg_color: Tuple[int, int, int] = (0, 0, 0),
        text_color: Tuple[int, int, int] = (255, 255, 255),
        font_scale: float = 1.0,
        show_top_k: int = 5
    ) -> np.ndarray:
        """
        绘制分类结果

        Args:
            image: 输入图像（BGR格式）
            results: 分类结果列表
            position: 标签位置 (top_left, top_right, bottom_left, bottom_right)
            bg_color: 背景颜色
            text_color: 文字颜色
            font_scale: 字体大小
            show_top_k: 显示前K个结果

        Returns:
            绘制后的图像
        """
        output = image.copy()

        # 获取分类结果
        class_results = [r for r in results if r.get('type') == 'classification']

        if not class_results:
            return output

        # 准备文字
        lines = []
        for i, result in enumerate(class_results[:show_top_k]):
            label = result.get('label', f"class_{result.get('class_id', i)}")
            conf = result.get('confidence', 0)
            lines.append(f"{i+1}. {label}: {conf:.3f}")

        # 绘制背景和文字
        output = self._draw_text_box(output, lines, position, bg_color, text_color, font_scale)

        self.logger.debug(f"绘制了分类结果（{len(lines)}条）")
        return output

    def draw_all(
        self,
        image: np.ndarray,
        results: List[Dict[str, Any]],
        task_type: str,
        **kwargs
    ) -> np.ndarray:
        """
        根据任务类型自动绘制结果

        Args:
            image: 输入图像
            results: 推理结果
            task_type: 任务类型
            **kwargs: 额外参数

        Returns:
            绘制后的图像
        """
        if task_type == 'detection':
            return self.draw_detections(image, results, **kwargs)
        elif task_type == 'classification':
            return self.draw_classification(image, results, **kwargs)
        else:
            self.logger.warning(f"未知任务类型: {task_type}")
            return image

    def _get_color(self, class_id: int) -> Tuple[int, int, int]:
        """根据类别ID获取颜色"""
        return self.colors[class_id % len(self.colors)]

    def _draw_label(
        self,
        image: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int],
        font_scale: float = 0.6
    ) -> np.ndarray:
        """
        绘制标签背景和文字

        Args:
            image: 图像
            text: 标签文字
            position: 位置 (x, y)
            color: 颜色
            font_scale: 字体大小

        Returns:
            绘制后的图像
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1

        # 获取文字尺寸
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        # 绘制背景
        x, y = position
        cv2.rectangle(
            image,
            (x, y - text_height - baseline * 2),
            (x + text_width + 4, y),
            color,
            -1
        )

        # 绘制文字
        cv2.putText(
            image,
            text,
            (x + 2, y - baseline),
            font,
            font_scale,
            (255, 255, 255),
            thickness
        )

        return image

    def _draw_text_box(
        self,
        image: np.ndarray,
        lines: List[str],
        position: str,
        bg_color: Tuple[int, int, int],
        text_color: Tuple[int, int, int],
        font_scale: float = 1.0
    ) -> np.ndarray:
        """
        绘制多行文字框

        Args:
            image: 图像
            lines: 文字行列表
            position: 位置
            bg_color: 背景颜色
            text_color: 文字颜色
            font_scale: 字体大小

        Returns:
            绘制后的图像
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        padding = 10
        line_height = int(30 * font_scale)

        # 获取最大宽度
        max_width = 0
        for line in lines:
            (w, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
            max_width = max(max_width, w)

        # 计算框大小
        box_width = max_width + padding * 2
        box_height = len(lines) * line_height + padding * 2

        # 计算位置
        img_h, img_w = image.shape[:2]
        if position == "top_left":
            x, y = 10, 10
        elif position == "top_right":
            x, y = img_w - box_width - 10, 10
        elif position == "bottom_left":
            x, y = 10, img_h - box_height - 10
        elif position == "bottom_right":
            x, y = img_w - box_width - 10, img_h - box_height - 10
        else:
            x, y = 10, 10

        # 绘制背景
        cv2.rectangle(
            image,
            (x, y),
            (x + box_width, y + box_height),
            bg_color,
            -1
        )

        # 绘制文字
        for i, line in enumerate(lines):
            text_y = y + padding + (i + 1) * line_height - 5
            cv2.putText(
                image,
                line,
                (x + padding, text_y),
                font,
                font_scale,
                text_color,
                thickness
            )

        return image

    def _mask_to_overlay(self, mask: np.ndarray) -> np.ndarray:
        """
        将掩码转换为彩色叠加层

        Args:
            mask: 掩码数组

        Returns:
            彩色叠加层（BGR格式）
        """
        # 创建彩色掩码
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)

        # 为每个类别分配颜色
        unique_ids = np.unique(mask)
        for class_id in unique_ids:
            if class_id == 0:  # 背景跳过
                continue
            color = self._get_color(int(class_id))
            colored[mask == class_id] = color

        return colored

    @staticmethod
    def create_result_comparison(
        original: np.ndarray,
        result: np.ndarray,
        task_type: str
    ) -> np.ndarray:
        """
        创建原图和结果的对比图

        Args:
            original: 原始图像
            result: 结果图像
            task_type: 任务类型

        Returns:
            拼接后的对比图
        """
        h1, w1 = original.shape[:2]
        h2, w2 = result.shape[:2]

        # 确保高度一致
        if h1 != h2:
            if h1 > h2:
                result = cv2.resize(result, (w2, h1))
            else:
                original = cv2.resize(original, (w1, h2))

        # 水平拼接
        comparison = np.hstack([original, result])

        # 添加标题
        comparison = ResultVisualizer._add_titles(comparison, w1, task_type)

        return comparison

    @staticmethod
    def _add_titles(image: np.ndarray, split_x: int, task_type: str) -> np.ndarray:
        """添加标题文字"""
        output = image.copy()
        h, w = output.shape[:2]

        # 添加标题区域
        title_height = 40
        extended = cv2.copyMakeBorder(
            output, title_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        # 绘制标题
        cv2.putText(
            extended,
            "Original",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            extended,
            f"{task_type.capitalize()} Result",
            (split_x + 10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        return extended


# PIL版本的可视化器（用于Web应用）
class PILResultVisualizer:
    """
    基于PIL的结果可视化器

    更适合Web应用，支持更丰富的字体和样式
    """

    def __init__(self, colors: Optional[List[Tuple[int, int, int]]] = None):
        """
        初始化PIL可视化器

        Args:
            colors: 自定义颜色列表（RGB格式）
        """
        # 转换BGR到RGB
        if colors:
            self.colors = [tuple(reversed(c)) for c in colors]
        else:
            self.colors = [
                (0, 255, 0),      # 绿色
                (0, 0, 255),      # 红色
                (255, 0, 0),      # 蓝色
                (255, 255, 0),    # 黄色
                (255, 0, 255),    # 洋红
                (0, 255, 255),    # 青色
            ]

    def draw_detections_pil(
        self,
        image: Image.Image,
        results: List[Dict[str, Any]],
        class_names: Optional[List[str]] = None
    ) -> Image.Image:
        """
        使用PIL绘制检测框

        Args:
            image: PIL图像
            results: 检测结果
            class_names: 类别名称

        Returns:
            绘制后的PIL图像
        """
        draw = ImageDraw.Draw(image)

        for idx, result in enumerate(results):
            if result.get('type') != 'detection':
                continue

            bbox = result.get('bbox')
            if not bbox or len(bbox) < 4:
                continue

            x1, y1, x2, y2 = bbox[:4]

            # 获取颜色
            class_id = result.get('class_id', idx)
            color = self._get_color_pil(class_id)

            # 绘制边框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # 绘制标签
            label = result.get('label', f'class_{class_id}')
            conf = result.get('confidence', 0)
            text = f"{label} {conf:.2f}"

            # 绘制标签背景
            text_bbox = draw.textbbox((x1, y1), text)
            draw.rectangle(text_bbox, fill=color)

            # 绘制文字
            draw.text((x1 + 2, y1 + 2), text, fill=(255, 255, 255))

        return image

    def _get_color_pil(self, class_id: int) -> Tuple[int, int, int]:
        """获取颜色（RGB）"""
        return self.colors[class_id % len(self.colors)]


# 导出
__all__ = [
    'ResultVisualizer',
    'PILResultVisualizer',
]
