"""
模型加载器
支持PyTorch(.pt)和ONNX(.onnx)格式的模型加载、缓存和设备管理
"""

import torch
import onnxruntime as ort
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
from collections import OrderedDict
from loguru import logger

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logger.warning("pynvml未安装，GPU监控功能将受限")


class ModelLoader:
    """
    模型加载器

    功能：
    - 支持PyTorch和ONNX格式
    - LRU缓存机制（最多缓存3个模型）
    - 自动设备选择（CUDA > MPS > CPU）
    - CUDA OOM自动降级到CPU
    """

    def __init__(self, cache_size: int = 3):
        """
        初始化模型加载器

        Args:
            cache_size: 缓存大小，默认3个模型
        """
        self.model_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.cache_size = cache_size
        self.device = self._auto_select_device()
        self.model_info_cache: Dict[str, Dict[str, Any]] = {}

        logger.bind(component="model_loader").info(
            f"模型加载器初始化完成，设备={self.device}，缓存大小={cache_size}"
        )

    def _auto_select_device(self) -> str:
        """
        自动选择最佳设备

        优先级：CUDA > MPS (Apple Silicon) > CPU

        Returns:
            设备字符串
        """
        if torch.cuda.is_available():
            device = "cuda:0"
            logger.bind(component="model_loader").info(f"检测到CUDA，使用设备: {device}")
            if PYNVML_AVAILABLE:
                try:
                    pynvml.nvmlInit()
                    gpu_count = pynvml.nvmlDeviceGetCount()
                    for i in range(gpu_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        name = pynvml.nvmlDeviceGetName(handle)
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        logger.bind(component="model_loader").info(
                            f"GPU {i}: {name}, "
                            f"总内存: {mem_info.total / 1024**3:.2f}GB, "
                            f"可用内存: {mem_info.free / 1024**3:.2f}GB"
                        )
                except Exception as e:
                    logger.warning(f"获取GPU信息失败: {e}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            logger.bind(component="model_loader").info("检测到Apple Silicon MPS，使用设备: mps")
        else:
            device = "cpu"
            logger.bind(component="model_loader").info("使用CPU设备")

        return device

    def _check_model_path(self, model_path: str) -> Path:
        """
        检查模型文件是否存在

        Args:
            model_path: 模型路径

        Returns:
            Path对象

        Raises:
            FileNotFoundError: 模型文件不存在
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        return path

    def _get_model_type(self, model_path: str) -> str:
        """
        根据文件扩展名判断模型类型

        Args:
            model_path: 模型路径

        Returns:
            模型类型：'pytorch' 或 'onnx'
        """
        suffix = Path(model_path).suffix.lower()
        if suffix in ['.pt', '.pth', '.pkl']:
            return 'pytorch'
        elif suffix == '.onnx':
            return 'onnx'
        else:
            raise ValueError(f"不支持的模型格式: {suffix}")

    def _load_pytorch_model(
        self,
        model_path: str,
        device: Optional[str] = None
    ) -> torch.nn.Module:
        """
        加载PyTorch模型

        Args:
            model_path: 模型路径
            device: 目标设备

        Returns:
            加载的模型
        """
        if device is None:
            device = self.device

        logger.bind(component="model_loader").debug(f"加载PyTorch模型: {model_path}")

        try:
            # 先加载到CPU避免设备映射问题（特别是"tagged with auto"错误）
            # 然后再移动到目标设备
            checkpoint = torch.load(model_path, map_location='cpu')

            # 尝试不同的checkpoint结构
            if isinstance(checkpoint, dict):
                # 情况1: {'model': model, 'optimizer': ...}
                if 'model' in checkpoint:
                    model = checkpoint['model']
                # 情况2: {'state_dict': ...}
                elif 'state_dict' in checkpoint:
                    model = checkpoint['state_dict']
                # 情况3: 整个模型
                else:
                    model = checkpoint
            else:
                # 直接是模型对象
                model = checkpoint

            # 移动到指定设备
            if hasattr(model, 'to'):
                model = model.to(device)
            elif isinstance(model, dict):
                # state_dict情况，需要先创建模型
                # 这里假设已经是在外层处理好的
                pass

            # 设置为评估模式
            if hasattr(model, 'eval'):
                model.eval()

            logger.bind(component="model_loader").success(
                f"PyTorch模型加载成功: {model_path}, 设备={device}"
            )

            return model

        except Exception as e:
            logger.bind(component="model_loader").error(
                f"加载PyTorch模型失败: {model_path}, 错误: {e}"
            )
            raise

    def _load_onnx_model(
        self,
        model_path: str,
        device: Optional[str] = None
    ) -> ort.InferenceSession:
        """
        加载ONNX模型

        Args:
            model_path: 模型路径
            device: 目标设备

        Returns:
            ONNX推理会话
        """
        logger.bind(component="model_loader").debug(f"加载ONNX模型: {model_path}")

        # 确定提供者
        if device is None:
            device = self.device

        providers = []
        if 'cuda' in device or 'gpu' in device.lower():
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')

        try:
            session = ort.InferenceSession(
                model_path,
                providers=providers
            )

            logger.bind(component="model_loader").success(
                f"ONNX模型加载成功: {model_path}, "
                f"提供者={session.get_providers()}"
            )

            return session

        except Exception as e:
            logger.bind(component="model_loader").error(
                f"加载ONNX模型失败: {model_path}, 错误: {e}"
            )
            raise

    def load_model(
        self,
        model_path: str,
        model_type: Optional[str] = None,
        device: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        加载模型（带缓存）

        Args:
            model_path: 模型路径
            model_type: 模型类型（'pytorch'或'onnx'），None则自动判断
            device: 目标设备，None则使用默认设备

        Returns:
            包含模型和元信息的字典
        """
        # 检查文件是否存在
        path = self._check_model_path(model_path)
        model_path = str(path)

        # 检查缓存
        if model_path in self.model_cache:
            # LRU: 移到末尾
            self.model_cache.move_to_end(model_path)
            logger.bind(component="model_loader").debug(f"从缓存加载模型: {model_path}")
            return self.model_cache[model_path]

        # 判断模型类型
        if model_type is None:
            model_type = self._get_model_type(model_path)

        # 加载模型
        try:
            if model_type == 'pytorch':
                model = self._load_pytorch_model(model_path, device)
            elif model_type == 'onnx':
                model = self._load_onnx_model(model_path, device)
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")

            # 获取模型信息
            info = self._get_model_info(model, model_type)

            # 缓存管理
            if len(self.model_cache) >= self.cache_size:
                # 移除最久未使用的模型
                oldest_path, oldest_model = self.model_cache.popitem(last=False)
                logger.bind(component="model_loader").info(
                    f"缓存已满，卸载最久未使用的模型: {oldest_path}"
                )
                # 清理GPU内存
                self._cleanup_model(oldest_model)

            # 添加到缓存
            model_data = {
                'model': model,
                'type': model_type,
                'path': model_path,
                'device': device if device else self.device,
                'info': info
            }
            self.model_cache[model_path] = model_data
            self.model_cache.move_to_end(model_path)

            logger.bind(component="model_loader").success(
                f"模型已加载并缓存: {model_path}, "
                f"类型={model_type}, 缓存大小={len(self.model_cache)}/{self.cache_size}"
            )

            return model_data

        except RuntimeError as e:
            if 'out of memory' in str(e).lower() and device and 'cuda' in device:
                logger.bind(component="model_loader").warning(
                    f"CUDA OOM，尝试降级到CPU: {model_path}"
                )
                return self.load_model(model_path, model_type, 'cpu')
            else:
                raise

    def get_model(self, model_path: str) -> Optional[Dict[str, Any]]:
        """
        从缓存获取模型

        Args:
            model_path: 模型路径

        Returns:
            模型数据字典，如果不存在返回None
        """
        if model_path in self.model_cache:
            self.model_cache.move_to_end(model_path)
            return self.model_cache[model_path]
        return None

    def unload_model(self, model_path: str) -> bool:
        """
        卸载模型释放内存

        Args:
            model_path: 模型路径

        Returns:
            是否成功卸载
        """
        if model_path in self.model_cache:
            model_data = self.model_cache.pop(model_path)
            self._cleanup_model(model_data)
            logger.bind(component="model_loader").info(f"模型已卸载: {model_path}")
            return True
        return False

    def _cleanup_model(self, model_data: Dict[str, Any]):
        """
        清理模型资源

        Args:
            model_data: 模型数据字典
        """
        try:
            model = model_data['model']
            model_type = model_data['type']

            if model_type == 'pytorch':
                # 删除模型对象
                del model
                # 清空CUDA缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            elif model_type == 'onnx':
                # ONNX Runtime会自动管理内存
                del model

        except Exception as e:
            logger.warning(f"清理模型资源时出错: {e}")

    def get_model_info(self, model_path: str) -> Optional[Dict[str, Any]]:
        """
        获取模型信息（从缓存）

        Args:
            model_path: 模型路径

        Returns:
            模型信息字典
        """
        if model_path in self.model_cache:
            return self.model_cache[model_path]['info']

        # 如果不在缓存中，尝试获取基本信息
        if model_path in self.model_info_cache:
            return self.model_info_cache[model_path]

        return None

    def _get_model_info(
        self,
        model: Any,
        model_type: str
    ) -> Dict[str, Any]:
        """
        提取模型元信息

        Args:
            model: 模型对象
            model_type: 模型类型

        Returns:
            模型信息字典
        """
        info = {
            'type': model_type,
        }

        if model_type == 'pytorch':
            if isinstance(model, torch.nn.Module):
                # 计算参数量
                total_params = sum(
                    p.numel() for p in model.parameters()
                )
                info['total_params'] = total_params
                info['trainable_params'] = sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                )

                # 获取输入尺寸（如果有）
                if hasattr(model, 'input_shape'):
                    info['input_shape'] = model.input_shape

        elif model_type == 'onnx':
            # ONNX模型信息
            info['inputs'] = [
                {'name': inp.name, 'shape': inp.shape}
                for inp in model.get_inputs()
            ]
            info['outputs'] = [
                {'name': out.name, 'shape': out.shape}
                for out in model.get_outputs()
            ]

        return info

    def clear_cache(self):
        """清空所有缓存的模型"""
        logger.bind(component="model_loader").info("清空模型缓存")
        for model_path in list(self.model_cache.keys()):
            self.unload_model(model_path)

    def get_cache_info(self) -> Dict[str, Any]:
        """
        获取缓存信息

        Returns:
            缓存信息字典
        """
        return {
            'cache_size': len(self.model_cache),
            'max_cache_size': self.cache_size,
            'cached_models': [
                {
                    'path': path,
                    'type': data['type'],
                    'device': data['device']
                }
                for path, data in self.model_cache.items()
            ]
        }

    def __del__(self):
        """析构函数，清理所有缓存的模型"""
        self.clear_cache()
