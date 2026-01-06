"""
结果后处理器
处理推理结果，包括过滤、可视化、格式转换等
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import cv2
from loguru import logger


class ResultPostprocessor:
    """
    结果后处理器

    功能：
    - 置信度过滤
    - 检测框绘制
    - 格式转换（COCO、YOLO）
    - 统计信息
    """

    def __init__(self, class_names: Optional[List[str]] = None):
        """
        初始化后处理器

        Args:
            class_names: 类别名称列表
        """
        self.class_names = class_names or []
        self.logger = logger.bind(component="result_postprocessor")

        # 颜色映射（用于绘制不同类别的检测框）
        self.colors = self._generate_colors(len(class_names) if class_names else 80)

    def _generate_colors(self, num_classes: int) -> List[Tuple[int, int, int]]:
        """
        生成类别颜色

        Args:
            num_classes: 类别数量

        Returns:
            RGB颜色列表
        """
        colors = []
        for i in range(num_classes):
            # 使用HSV到RGB的转换生成不同颜色
            hue = i * 137.508  # 黄金角度
            saturation = 0.7
            value = 0.9

            # HSV转RGB
            import colorsys
            rgb = colorsys.hsv_to_rgb(hue / 360, saturation, value)
            colors.append(tuple(int(c * 255) for c in rgb))

        return colors

    def filter_by_confidence(
        self,
        results: List[Dict[str, Any]],
        threshold: float
    ) -> List[Dict[str, Any]]:
        """
        按置信度过滤结果

        Args:
            results: 检测结果列表
            threshold: 置信度阈值

        Returns:
            过滤后的结果列表
        """
        filtered = [
            r for r in results
            if r.get('confidence', 0) >= threshold
        ]

        self.logger.debug(
            f"置信度过滤: {len(results)} -> {len(filtered)} "
            f"(阈值={threshold})"
        )

        return filtered

    def filter_by_class(
        self,
        results: List[Dict[str, Any]],
        class_ids: Optional[List[int]] = None,
        class_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        按类别过滤结果

        Args:
            results: 检测结果列表
            class_ids: 类别ID列表（可选）
            class_names: 类别名称列表（可选）

        Returns:
            过滤后的结果列表
        """
        if class_ids is None and class_names is None:
            return results

        filtered = []
        for r in results:
            label = r.get('label', '')

            # 按类别ID过滤
            if class_ids:
                # 从label中提取类别ID（例如 'class_0' -> 0）
                if label.startswith('class_'):
                    class_id = int(label.split('_')[1])
                    if class_id in class_ids:
                        filtered.append(r)
                        continue

            # 按类别名称过滤
            if class_names and label in class_names:
                filtered.append(r)

        self.logger.debug(
            f"类别过滤: {len(results)} -> {len(filtered)}"
        )

        return filtered

    def filter_by_size(
        self,
        results: List[Dict[str, Any]],
        min_area: Optional[float] = None,
        max_area: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        按检测框大小过滤结果

        Args:
            results: 检测结果列表
            min_area: 最小面积（可选）
            max_area: 最大面积（可选）

        Returns:
            过滤后的结果列表
        """
        filtered = []
        for r in results:
            bbox = r.get('bbox', [])
            if len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
                area = (x2 - x1) * (y2 - y1)

                if min_area and area < min_area:
                    continue
                if max_area and area > max_area:
                    continue

                filtered.append(r)

        self.logger.debug(
            f"大小过滤: {len(results)} -> {len(filtered)}"
        )

        return filtered

    def draw_bboxes(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        show_confidence: bool = True,
        show_label: bool = True,
        line_width: int = 2,
        font_size: int = 20
    ) -> np.ndarray:
        """
        在图像上绘制检测框

        Args:
            image: 输入图像（numpy数组）
            detections: 检测结果列表
            show_confidence: 是否显示置信度
            show_label: 是否显示标签
            line_width: 线宽
            font_size: 字体大小

        Returns:
            绘制后的图像
        """
        # 转换为RGB（如果是BGR）
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image.copy()

        # 转换为PIL图像
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)

        try:
            # 尝试加载字体
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            # 如果字体文件不存在，使用默认字体
            font = ImageFont.load_default()

        # 绘制每个检测框
        for detection in detections:
            bbox = detection.get('bbox', [])
            label = detection.get('label', '')
            confidence = detection.get('confidence', 0)

            if len(bbox) < 4:
                continue

            x1, y1, x2, y2 = [int(c) for c in bbox[:4]]

            # 获取类别颜色
            color = self._get_color_for_label(label)

            # 绘制边界框
            draw.rectangle(
                [(x1, y1), (x2, y2)],
                outline=color,
                width=line_width
            )

            # 绘制标签和置信度
            if show_label or show_confidence:
                text_parts = []
                if show_label:
                    text_parts.append(label)
                if show_confidence:
                    text_parts.append(f"{confidence:.2f}")

                text = " ".join(text_parts)

                # 计算文本大小
                bbox_text = draw.textbbox((0, 0), text, font=font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]

                # 绘制文本背景
                draw.rectangle(
                    [(x1, y1 - text_height - 4), (x1 + text_width + 4, y1)],
                    fill=color
                )

                # 绘制文本
                draw.text((x1 + 2, y1 - text_height - 2), text, fill="white", font=font)

        # 转回numpy数组
        annotated_image = np.array(pil_image)

        # 转回BGR（如果需要）
        if len(annotated_image.shape) == 3 and annotated_image.shape[2] == 3:
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        return annotated_image

    def _get_color_for_label(self, label: str) -> Tuple[int, int, int]:
        """
        根据标签获取颜色

        Args:
            label: 标签字符串

        Returns:
            RGB颜色元组
        """
        # 从label中提取类别ID
        if label.startswith('class_'):
            class_id = int(label.split('_')[1])
            color_index = class_id % len(self.colors)
            return self.colors[color_index]
        else:
            # 根据字符串哈希生成颜色
            hash_val = hash(label) % len(self.colors)
            return self.colors[hash_val]

    def save_annotated_image(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        output_path: str,
        **draw_kwargs
    ) -> str:
        """
        保存标注图像

        Args:
            image: 输入图像
            detections: 检测结果
            output_path: 输出路径
            **draw_kwargs: 绘制参数

        Returns:
            保存的文件路径
        """
        try:
            # 绘制检测框
            annotated_image = self.draw_bboxes(image, detections, **draw_kwargs)

            # 确保输出目录存在
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 保存图像
            cv2.imwrite(str(output_path), annotated_image)

            self.logger.info(f"已保存标注图像: {output_path}")

            return str(output_path)

        except Exception as e:
            self.logger.error(f"保存标注图像失败: {e}")
            raise

    def format_to_coco(
        self,
        results: List[Dict[str, Any]],
        image_id: int,
        image_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        转换为COCO格式

        Args:
            results: 检测结果列表
            image_id: 图像ID
            image_info: 图像信息（可选）

        Returns:
            COCO格式的字典
        """
        annotations = []

        for idx, result in enumerate(results):
            bbox = result.get('bbox', [])
            label = result.get('label', '')
            confidence = result.get('confidence', 0)

            if len(bbox) < 4:
                continue

            x1, y1, x2, y2 = bbox[:4]
            width = x2 - x1
            height = y2 - y1

            # 提取类别ID
            if label.startswith('class_'):
                category_id = int(label.split('_')[1])
            else:
                # 使用标签的哈希值作为类别ID
                category_id = hash(label) % 1000

            annotation = {
                "id": idx + 1,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x1, y1, width, height],
                "area": width * height,
                "iscrowd": 0,
                "score": confidence
            }

            annotations.append(annotation)

        coco_result = {
            "annotations": annotations
        }

        if image_info:
            coco_result["image_info"] = image_info

        return coco_result

    def format_to_yolo(
        self,
        results: List[Dict[str, Any]],
        image_width: int,
        image_height: int
    ) -> List[str]:
        """
        转换为YOLO格式

        Args:
            results: 检测结果列表
            image_width: 图像宽度
            image_height: 图像高度

        Returns:
            YOLO格式的字符串列表
        """
        yolo_lines = []

        for result in results:
            bbox = result.get('bbox', [])
            label = result.get('label', '')
            confidence = result.get('confidence', 0)

            if len(bbox) < 4:
                continue

            x1, y1, x2, y2 = bbox[:4]

            # 计算中心点和宽高（归一化）
            center_x = ((x1 + x2) / 2) / image_width
            center_y = ((y1 + y2) / 2) / image_height
            width = (x2 - x1) / image_width
            height = (y2 - y1) / image_height

            # 提取类别ID
            if label.startswith('class_'):
                class_id = int(label.split('_')[1])
            else:
                class_id = 0

            # YOLO格式: class_id center_x center_y width height [confidence]
            yolo_line = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"

            # 如果有置信度，添加到末尾
            if confidence > 0:
                yolo_line += f" {confidence:.6f}"

            yolo_lines.append(yolo_line)

        return yolo_lines

    def save_results_json(
        self,
        results: List[Dict[str, Any]],
        output_path: str,
        image_path: Optional[str] = None
    ) -> str:
        """
        保存结果为JSON格式

        Args:
            results: 检测结果列表
            output_path: 输出路径
            image_path: 图像路径（可选）

        Returns:
            保存的文件路径
        """
        try:
            output_data = {
                "image_path": image_path,
                "num_detections": len(results),
                "detections": results
            }

            # 确保输出目录存在
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 保存JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"已保存JSON结果: {output_path}")

            return str(output_path)

        except Exception as e:
            self.logger.error(f"保存JSON结果失败: {e}")
            raise

    def save_results_coco(
        self,
        results: List[Dict[str, Any]],
        output_path: str,
        image_id: int = 0
    ) -> str:
        """
        保存结果为COCO格式

        Args:
            results: 检测结果列表
            output_path: 输出路径
            image_id: 图像ID

        Returns:
            保存的文件路径
        """
        try:
            coco_data = self.format_to_coco(results, image_id)

            # 确保输出目录存在
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 保存JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(coco_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"已保存COCO结果: {output_path}")

            return str(output_path)

        except Exception as e:
            self.logger.error(f"保存COCO结果失败: {e}")
            raise

    def save_results_yolo(
        self,
        results: List[Dict[str, Any]],
        output_path: str,
        image_width: int,
        image_height: int
    ) -> str:
        """
        保存结果为YOLO格式

        Args:
            results: 检测结果列表
            output_path: 输出路径
            image_width: 图像宽度
            image_height: 图像高度

        Returns:
            保存的文件路径
        """
        try:
            yolo_lines = self.format_to_yolo(results, image_width, image_height)

            # 确保输出目录存在
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 保存文本文件
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(yolo_lines))

            self.logger.info(f"已保存YOLO结果: {output_path}")

            return str(output_path)

        except Exception as e:
            self.logger.error(f"保存YOLO结果失败: {e}")
            raise

    def calculate_statistics(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        计算检测结果的统计信息

        Args:
            results: 检测结果列表

        Returns:
            统计信息字典
        """
        if not results:
            return {
                "total_detections": 0,
                "avg_confidence": 0,
                "class_distribution": {}
            }

        # 总检测数
        total_detections = len(results)

        # 平均置信度
        confidences = [r.get('confidence', 0) for r in results]
        avg_confidence = np.mean(confidences) if confidences else 0

        # 类别分布
        class_counts = {}
        for r in results:
            label = r.get('label', 'unknown')
            class_counts[label] = class_counts.get(label, 0) + 1

        # 置信度分布
        confidence_ranges = {
            "high (>0.8)": sum(1 for c in confidences if c > 0.8),
            "medium (0.5-0.8)": sum(1 for c in confidences if 0.5 <= c <= 0.8),
            "low (<0.5)": sum(1 for c in confidences if c < 0.5)
        }

        stats = {
            "total_detections": total_detections,
            "avg_confidence": float(avg_confidence),
            "min_confidence": float(np.min(confidences)),
            "max_confidence": float(np.max(confidences)),
            "std_confidence": float(np.std(confidences)),
            "class_distribution": class_counts,
            "confidence_distribution": confidence_ranges
        }

        return stats
