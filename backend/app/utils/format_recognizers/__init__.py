"""
数据集格式识别器模块
"""

from .yolo import YOLORecognizer
from .coco import COCORecognizer
from .voc import VOCRecognizer
from .classification import ClassificationRecognizer


class DatasetFormatRecognizer:
    """数据集格式识别器统一接口"""

    def __init__(self):
        self.recognizers = {
            'yolo': YOLORecognizer(),
            'coco': COCORecognizer(),
            'voc': VOCRecognizer(),
            'classification': ClassificationRecognizer()
        }

    def recognize_format(self, dataset_path: str) -> dict:
        """
        识别数据集格式

        Args:
            dataset_path: 数据集路径

        Returns:
            dict: 识别结果
        """
        results = {}

        # 运行所有识别器
        for format_name, recognizer in self.recognizers.items():
            try:
                result = recognizer.recognize(dataset_path)
                results[format_name] = result
            except Exception as e:
                results[format_name] = {
                    "format": format_name,
                    "confidence": 0,
                    "error": f"识别器运行失败: {str(e)}"
                }

        # 选择置信度最高的格式
        best_format = self._select_best_format(results)

        return {
            "best_format": best_format,
            "all_results": results,
            "dataset_path": dataset_path
        }

    def _select_best_format(self, results: dict) -> dict:
        """选择置信度最高的格式"""
        best_format = {
            "format": "unknown",
            "confidence": 0,
            "details": {},
            "error": "无法识别数据集格式"
        }

        for format_name, result in results.items():
            confidence = result.get("confidence", 0)
            if confidence > best_format["confidence"]:
                best_format = {
                    "format": result.get("format", format_name),
                    "confidence": confidence,
                    "details": result.get("details", {}),
                    "error": result.get("error")
                }

        return best_format


# 导出主要接口
__all__ = ['DatasetFormatRecognizer', 'YOLORecognizer', 'COCORecognizer', 'VOCRecognizer', 'ClassificationRecognizer']