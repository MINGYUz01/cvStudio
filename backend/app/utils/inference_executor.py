"""
推理执行器
执行单图和批量推理，包括预处理、推理和后处理
"""

import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from PIL import Image
import numpy as np
import torch
import cv2
from loguru import logger


class InferenceExecutor:
    """
    推理执行器

    功能：
    - 单图推理
    - 批量推理
    - 图像预处理和后处理
    - 性能指标收集
    - 支持PyTorch和ONNX模型
    """

    def __init__(
        self,
        model: Any,
        model_type: str,
        device: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化推理执行器

        Args:
            model: 加载的模型
            model_type: 模型类型（'pytorch'或'onnx'）
            device: 推理设备
            config: 推理配置
        """
        self.model = model
        self.model_type = model_type
        self.device = device
        self.config = config or {}

        # 性能统计
        self.metrics = {
            'total_inferences': 0,
            'total_time': 0.0,
            'preprocessing_time': 0.0,
            'postprocessing_time': 0.0
        }

        logger.bind(component="inference_executor").info(
            f"推理执行器初始化完成，模型类型={model_type}，设备={device}"
        )

    def _preprocess(self, image_path: str) -> tuple:
        """
        图像预处理

        Args:
            image_path: 图像路径

        Returns:
            (预处理后的图像, 原始图像, 图像信息)
        """
        start_time = time.time()

        # 读取图像
        image = Image.open(image_path)

        # 转换为RGB（如果需要）
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # 获取原始尺寸
        original_size = image.size  # (width, height)

        # 根据模型类型进行预处理
        if self.model_type == 'pytorch':
            # PyTorch模型预处理
            processed = self._preprocess_pytorch(image)
        else:
            # ONNX模型预处理
            processed = self._preprocess_onnx(image)

        preprocessing_time = (time.time() - start_time) * 1000  # 转换为毫秒

        return processed, image, {
            'original_size': original_size,
            'preprocessing_time': preprocessing_time
        }

    def _preprocess_pytorch(self, image: Image.Image) -> torch.Tensor:
        """
        PyTorch模型预处理

        Args:
            image: PIL图像

        Returns:
            预处理后的tensor
        """
        # 调整大小（YOLO默认640x640）
        target_size = self.config.get('target_size', 640)
        image = image.resize((target_size, target_size), Image.BILINEAR)

        # 转换为tensor
        tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        # 添加batch维度
        tensor = tensor.unsqueeze(0)

        # 移动到设备
        if 'cuda' in self.device or 'mps' in self.device:
            tensor = tensor.to(self.device)

        return tensor

    def _preprocess_onnx(self, image: Image.Image) -> np.ndarray:
        """
        ONNX模型预处理

        Args:
            image: PIL图像

        Returns:
            预处理后的numpy数组
        """
        # 调整大小
        target_size = self.config.get('target_size', 640)
        image = image.resize((target_size, target_size), Image.BILINEAR)

        # 转换为numpy数组
        array = np.array(image).astype(np.float32)

        # HWC -> CHW
        array = array.transpose(2, 0, 1)

        # 归一化
        array = array / 255.0

        # 添加batch维度
        array = np.expand_dims(array, axis=0)

        return array

    def _run_inference(self, preprocessed_input: Union[torch.Tensor, np.ndarray]) -> Any:
        """
        执行推理

        Args:
            preprocessed_input: 预处理后的输入

        Returns:
            模型输出
        """
        if self.model_type == 'pytorch':
            with torch.no_grad():
                output = self.model(preprocessed_input)
            return output
        else:
            # ONNX推理
            input_name = self.model.get_inputs()[0].name
            output = self.model.run(None, {input_name: preprocessed_input})
            return output[0]

    def _postprocess(
        self,
        output: Any,
        image_info: Dict[str, Any],
        confidence_threshold: float,
        iou_threshold: float
    ) -> List[Dict[str, Any]]:
        """
        后处理模型输出

        Args:
            output: 模型输出
            image_info: 图像信息
            confidence_threshold: 置信度阈值
            iou_threshold: IOU阈值

        Returns:
            检测结果列表
        """
        start_time = time.time()

        results = []

        # 根据模型类型进行后处理
        if self.model_type == 'pytorch':
            if isinstance(output, (list, tuple)):
                output = output[0]

            # 转换为numpy
            if torch.is_tensor(output):
                output = output.cpu().numpy()

            # YOLO格式输出: [batch, detections, 6]
            # detections格式: [x1, y1, x2, y2, confidence, class_id]
            if len(output.shape) == 3 and output.shape[2] >= 6:
                detections = output[0]  # 移除batch维度

                # 过滤低置信度检测
                mask = detections[:, 4] >= confidence_threshold
                detections = detections[mask]

                # NMS
                if len(detections) > 0:
                    keep_indices = self._nms(
                        detections[:, :4],
                        detections[:, 4],
                        iou_threshold
                    )
                    detections = detections[keep_indices]

                # 转换为结果格式
                for det in detections:
                    x1, y1, x2, y2, conf, class_id = det[:6]
                    results.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'label': f'class_{int(class_id)}',
                        'confidence': float(conf)
                    })

        else:
            # ONNX输出后处理
            if isinstance(output, np.ndarray):
                if len(output.shape) == 3 and output.shape[2] >= 6:
                    detections = output[0]

                    # 过滤和NMS
                    mask = detections[:, 4] >= confidence_threshold
                    detections = detections[mask]

                    if len(detections) > 0:
                        keep_indices = self._nms(
                            detections[:, :4],
                            detections[:, 4],
                            iou_threshold
                        )
                        detections = detections[keep_indices]

                    for det in detections:
                        x1, y1, x2, y2, conf, class_id = det[:6]
                        results.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'label': f'class_{int(class_id)}',
                            'confidence': float(conf)
                        })

        postprocessing_time = (time.time() - start_time) * 1000

        # 更新图像信息
        image_info['postprocessing_time'] = postprocessing_time

        return results

    def _nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float
    ) -> np.ndarray:
        """
        非极大值抑制

        Args:
            boxes: 边界框 [N, 4]
            scores: 置信度 [N]
            iou_threshold: IOU阈值

        Returns:
            保留的索引
        """
        if len(boxes) == 0:
            return np.array([], dtype=np.int32)

        # 计算面积
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        # 按置信度排序
        order = scores.argsort()[::-1]

        keep = []
        while len(order) > 0:
            # 保留最高置信度的框
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            # 计算IoU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            # 保留IoU小于阈值的框
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return np.array(keep, dtype=np.int32)

    async def infer_single_image(
        self,
        image_path: str,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45
    ) -> Dict[str, Any]:
        """
        单图推理

        Args:
            image_path: 图像路径
            confidence_threshold: 置信度阈值
            iou_threshold: IOU阈值

        Returns:
            推理结果字典
        """
        inference_start = time.time()

        try:
            # 预处理
            preprocessed, original_image, image_info = self._preprocess(image_path)

            # 推理
            inference_start = time.time()
            output = self._run_inference(preprocessed)
            inference_time = (time.time() - inference_start) * 1000

            # 后处理
            results = self._postprocess(
                output,
                image_info,
                confidence_threshold,
                iou_threshold
            )

            # 计算总时间
            total_time = (time.time() - inference_start) * 1000

            # 更新统计
            self.metrics['total_inferences'] += 1
            self.metrics['total_time'] += total_time

            return {
                'results': results,
                'metrics': {
                    'inference_time': inference_time,
                    'preprocessing_time': image_info.get('preprocessing_time', 0),
                    'postprocessing_time': image_info.get('postprocessing_time', 0),
                    'total_time': total_time,
                    'device': self.device,
                    'image_size': list(image_info['original_size'])
                },
                'image_path': image_path
            }

        except Exception as e:
            logger.bind(component="inference_executor").error(
                f"单图推理失败: {image_path}, 错误: {e}"
            )
            raise

    async def infer_batch(
        self,
        image_paths: List[str],
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        批量推理

        Args:
            image_paths: 图像路径列表
            confidence_threshold: 置信度阈值
            iou_threshold: IOU阈值
            progress_callback: 进度回调函数

        Returns:
            推理结果列表
        """
        total_images = len(image_paths)
        results = []
        batch_start = time.time()

        logger.bind(component="inference_executor").info(
            f"开始批量推理，共{total_images}张图像"
        )

        for idx, image_path in enumerate(image_paths):
            try:
                result = await self.infer_single_image(
                    image_path,
                    confidence_threshold,
                    iou_threshold
                )
                results.append(result)

                # 进度回调
                if progress_callback:
                    await progress_callback({
                        'processed': idx + 1,
                        'total': total_images,
                        'progress': ((idx + 1) / total_images) * 100
                    })

            except Exception as e:
                logger.error(f"批量推理中处理图像失败: {image_path}, 错误: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e),
                    'results': []
                })

        # 计算平均FPS
        total_time = time.time() - batch_start
        avg_fps = total_images / total_time if total_time > 0 else 0

        logger.bind(component="inference_executor").success(
            f"批量推理完成，共{total_images}张图像，平均FPS: {avg_fps:.2f}"
        )

        return results

    def get_metrics(self) -> Dict[str, Any]:
        """
        获取性能指标

        Returns:
            性能指标字典
        """
        if self.metrics['total_inferences'] > 0:
            avg_time = self.metrics['total_time'] / self.metrics['total_inferences']
            fps = 1000 / avg_time if avg_time > 0 else 0
        else:
            avg_time = 0
            fps = 0

        return {
            'total_inferences': self.metrics['total_inferences'],
            'avg_inference_time': avg_time,
            'avg_fps': fps,
            'device': self.device
        }
