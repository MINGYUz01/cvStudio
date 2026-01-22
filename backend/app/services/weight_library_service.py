"""
权重库服务
管理权重文件的上传、存储、版本分析和任务类型检测
"""

import os
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from sqlalchemy.orm import Session

from fastapi import UploadFile
from loguru import logger

from app.models.weight_library import WeightLibrary
from app.models.user import User
from app.models.augmentation import AugmentationStrategy
from app.utils.model_loader import ModelLoader
from app.utils.task_detector import TaskTypeDetector


def _clean_model_name(name: str) -> str:
    """
    清理模型名称中的版本后缀

    例如: "ResNet50 v1.0" -> "ResNet50"
         "YOLOv8 v2.1.0" -> "YOLOv8"

    Args:
        name: 原始模型名称

    Returns:
        清理后的模型名称
    """
    import re
    # 匹配版本模式: v + 数字格式 (如 v1.0, v2.1.0)
    pattern = r'\s+v\d+(?:\.\d+)*\.?\s*$'
    cleaned = re.sub(pattern, '', name)
    return cleaned.strip()


class WeightLibraryService:
    """
    权重库服务

    功能：
    - 权重文件上传和验证
    - 任务类型自动检测
    - 版本管理
    - 权重查询和删除
    """

    # 权重文件存储根目录
    STORAGE_PATH = Path("data/weights")

    # 支持的文件扩展名
    SUPPORTED_EXTENSIONS = {'.pt', '.pth', '.pkl', '.onnx'}

    # 最大文件大小（2GB）
    MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024

    def __init__(self):
        """初始化权重库服务"""
        self.logger = logger.bind(component="weight_library_service")
        self.model_loader = ModelLoader()

        # 确保存储目录存在
        self._ensure_directories()

    def _ensure_directories(self):
        """创建必要的存储目录"""
        for task_type in ['classification', 'detection']:
            task_dir = self.STORAGE_PATH / task_type
            task_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"权重存储目录已准备: {self.STORAGE_PATH}")

    async def upload_weight(
        self,
        file: UploadFile,
        name: str,
        task_type: str,
        description: Optional[str] = None,
        input_size: Optional[List[int]] = None,
        class_names: Optional[List[str]] = None,
        normalize_params: Optional[Dict] = None,
        uploaded_by: Optional[int] = None,
        db: Session = None
    ) -> WeightLibrary:
        """
        上传权重文件

        Args:
            file: 上传的文件
            name: 权重名称
            task_type: 任务类型 (classification/detection/auto)
            description: 描述
            input_size: 输入尺寸 [height, width]
            class_names: 类别名称列表
            normalize_params: 归一化参数
            uploaded_by: 上传用户ID
            db: 数据库会话

        Returns:
            创建的权重库记录

        Raises:
            ValueError: 当文件验证失败或上传失败时
        """
        try:
            # 1. 验证文件
            await self._validate_upload_file(file)

            # 2. 生成存储路径
            file_ext = Path(file.filename).suffix.lower()
            storage_filename = self._generate_storage_filename(name, file_ext)
            temp_file_path = self.STORAGE_PATH / "temp" / storage_filename

            # 确保临时目录存在
            temp_file_path.parent.mkdir(parents=True, exist_ok=True)

            # 3. 保存到临时位置
            await self._save_file(file, temp_file_path)

            self.logger.info(f"文件已保存到临时位置: {temp_file_path}")

            # 4. 分析权重文件
            metadata = await self._analyze_weight_file(temp_file_path, file_ext)

            # 5. 处理任务类型
            final_task_type, is_auto_detected = await self._determine_task_type(
                temp_file_path, file_ext, task_type, metadata
            )

            # 6. 移动到最终位置
            final_dir = self.STORAGE_PATH / final_task_type
            final_path = final_dir / storage_filename
            shutil.move(str(temp_file_path), str(final_path))

            # 7. 创建数据库记录
            weight = WeightLibrary(
                name=name,
                description=description,
                task_type=final_task_type,
                version="1.0",
                file_path=str(final_path),
                file_name=file.filename,
                file_size=final_path.stat().st_size,
                framework="pytorch" if file_ext != '.onnx' else "onnx",
                input_size=input_size or metadata.get('input_size'),
                class_names=class_names,
                normalize_params=normalize_params,
                extra_metadata=metadata,
                is_auto_detected=is_auto_detected,
                uploaded_by=uploaded_by
            )

            db.add(weight)
            db.commit()
            db.refresh(weight)

            self.logger.success(
                f"权重文件上传成功: {name} ({final_task_type}), "
                f"ID={weight.id}, 大小={weight.file_size / 1024 / 1024:.2f}MB"
            )

            return weight

        except Exception as e:
            self.logger.error(f"上传权重文件失败: {e}")
            # 清理临时文件
            if temp_file_path.exists():
                temp_file_path.unlink()
            raise ValueError(f"上传权重文件失败: {str(e)}")

    async def _validate_upload_file(self, file: UploadFile):
        """验证上传的文件"""
        if not file.filename:
            raise ValueError("文件名为空")

        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支持的文件格式: {file_ext}")

        # 读取文件内容以获取大小
        content = await file.read()
        file_size = len(content)

        if file_size > self.MAX_FILE_SIZE:
            raise ValueError(f"文件过大: {file_size / 1024 / 1024:.2f}MB (最大 {self.MAX_FILE_SIZE / 1024 / 1024}MB)")

        if file_size == 0:
            raise ValueError("文件为空")

        # 重置文件指针
        await file.seek(0)

        self.logger.debug(f"文件验证通过: {file.filename}, 大小={file_size}")

    def _generate_storage_filename(self, name: str, ext: str) -> str:
        """生成存储文件名（唯一）"""
        # 使用时间戳和名称哈希确保唯一性
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_hash = hashlib.md5(name.encode()).hexdigest()[:8]
        return f"{name}_{timestamp}_{name_hash}{ext}"

    async def _save_file(self, file: UploadFile, path: Path):
        """保存上传的文件"""
        try:
            with open(path, "wb") as f:
                content = await file.read()
                f.write(content)
        except Exception as e:
            raise ValueError(f"保存文件失败: {e}")

    async def _analyze_weight_file(
        self,
        file_path: Path,
        file_ext: str
    ) -> Dict[str, Any]:
        """分析权重文件获取元信息"""
        metadata = {}

        try:
            self.logger.info(f"开始分析权重文件: {file_path}")

            if file_ext == '.onnx':
                # ONNX模型分析
                model_data = self.model_loader.load_model(str(file_path))
                info = model_data.get('info', {})

                if 'inputs' in info and info['inputs']:
                    input_shape = info['inputs'][0].get('shape')
                    metadata['input_size'] = self._parse_input_shape(input_shape)
                    metadata['input_name'] = info['inputs'][0].get('name')

                if 'outputs' in info and info['outputs']:
                    output_shape = info['outputs'][0].get('shape')
                    metadata['output_shape'] = output_shape

                metadata['output_names'] = [o.get('name') for o in info.get('outputs', [])]

            else:
                # PyTorch模型分析
                model_data = self.model_loader.load_model(str(file_path))
                model = model_data.get('model')

                # 尝试获取参数量
                if hasattr(model, 'parameters'):
                    try:
                        total_params = sum(p.numel() for p in model.parameters())
                        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                        metadata['total_params'] = total_params
                        metadata['trainable_params'] = trainable_params
                        metadata['size_mb'] = total_params * 4 / (1024 * 1024)  # 假设float32
                    except Exception:
                        pass

                # 尝试获取输入尺寸（有些模型有这个属性）
                if hasattr(model, 'input_shape'):
                    metadata['input_size'] = list(model.input_shape)

            self.logger.success(f"权重文件分析完成: {metadata}")
            return metadata

        except Exception as e:
            self.logger.warning(f"分析权重文件失败: {e}")
            return metadata

    async def _determine_task_type(
        self,
        file_path: Path,
        file_ext: str,
        task_type: str,
        metadata: Dict[str, Any]
    ) -> Tuple[str, bool]:
        """
        确定任务类型

        Returns:
            (最终任务类型, 是否自动检测)
        """
        if task_type != 'auto':
            return task_type, False

        # 自动检测
        try:
            self.logger.info("开始自动检测任务类型")

            if file_ext == '.onnx':
                model_data = self.model_loader.load_model(str(file_path))
                detected_type = TaskTypeDetector.detect(model_data['model'], 'onnx')
            else:
                model_data = self.model_loader.load_model(str(file_path))
                detected_type = TaskTypeDetector.detect(
                    model_data['model'],
                    'pytorch',
                    input_size=(1, 3, 224, 224)
                )

            if detected_type == 'unknown':
                # 如果检测失败，根据输出形状推测
                output_shape = metadata.get('output_shape')
                if output_shape:
                    detected_type = TaskTypeDetector.detect_from_shape(tuple(output_shape))

            if detected_type == 'unknown':
                self.logger.warning("无法自动检测任务类型，默认为分类")
                detected_type = 'classification'

            self.logger.success(f"自动检测到任务类型: {detected_type}")
            return detected_type, True

        except Exception as e:
            self.logger.error(f"自动检测任务类型失败: {e}")
            return 'classification', True

    def _parse_input_shape(self, shape: List) -> Optional[List[int]]:
        """解析输入形状"""
        if not shape:
            return None

        # 处理动态维度
        clean_shape = []
        for dim in shape:
            if isinstance(dim, str) or dim is None:
                clean_shape.append(224)  # 默认值
            else:
                clean_shape.append(int(dim))

        # NCHW格式：返回 [H, W]
        if len(clean_shape) >= 4:
            return [clean_shape[2], clean_shape[3]]
        elif len(clean_shape) == 3:
            return [clean_shape[1], clean_shape[2]]

        return clean_shape

    def get_weights(
        self,
        db: Session,
        task_type: Optional[str] = None,
        is_active: Optional[bool] = True,
        skip: int = 0,
        limit: int = 100
    ) -> List[WeightLibrary]:
        """
        获取权重列表

        Args:
            db: 数据库会话
            task_type: 过滤任务类型
            is_active: 是否只返回活跃权重
            skip: 跳过记录数
            limit: 返回记录数

        Returns:
            权重列表
        """
        try:
            query = db.query(WeightLibrary)

            if task_type:
                query = query.filter(WeightLibrary.task_type == task_type)

            if is_active is not None:
                status = "active" if is_active else "deprecated"
                query = query.filter(WeightLibrary.is_active == status)

            weights = query.order_by(
                WeightLibrary.created_at.desc()
            ).offset(skip).limit(limit).all()

            self.logger.debug(f"获取权重列表: {len(weights)} 条记录")
            return weights

        except Exception as e:
            self.logger.error(f"获取权重列表失败: {e}")
            return []

    def get_weight(self, db: Session, weight_id: int) -> Optional[WeightLibrary]:
        """
        获取单个权重

        Args:
            db: 数据库会话
            weight_id: 权重ID

        Returns:
            权重对象，不存在返回None
        """
        try:
            weight = db.query(WeightLibrary).filter(
                WeightLibrary.id == weight_id
            ).first()
            return weight
        except Exception as e:
            self.logger.error(f"获取权重失败: {e}")
            return None

    def delete_weight(self, db: Session, weight_id: int) -> bool:
        """
        删除权重（包括所有子节点）

        Args:
            db: 数据库会话
            weight_id: 权重ID

        Returns:
            是否删除成功
        """
        try:
            weight = db.query(WeightLibrary).filter(
                WeightLibrary.id == weight_id
            ).first()

            if not weight:
                return False

            # 递归删除所有子节点
            def delete_recursive(w):
                # 找到所有子节点（parent_version_id 指向当前节点的）
                children = db.query(WeightLibrary).filter(
                    WeightLibrary.parent_version_id == w.id
                ).all()

                # 先递归删除子节点
                for child in children:
                    delete_recursive(child)

                # 删除当前节点的文件
                file_path = Path(w.file_path)
                if file_path.exists():
                    try:
                        file_path.unlink()
                        self.logger.info(f"已删除权重文件: {file_path}")
                    except Exception as e:
                        self.logger.warning(f"删除权重文件失败: {file_path}, 错误: {e}")

                # 删除数据库记录
                db.delete(w)

            # 执行递归删除
            delete_recursive(weight)

            db.commit()

            self.logger.info(f"权重及其子节点已删除: ID={weight_id}")
            return True

        except Exception as e:
            self.logger.error(f"删除权重失败: {e}")
            db.rollback()
            return False

    def create_new_version(
        self,
        db: Session,
        parent_weight_id: int,
        file_path: str,
        description: Optional[str] = None
    ) -> Optional[WeightLibrary]:
        """
        创建权重的新版本

        Args:
            db: 数据库会话
            parent_weight_id: 父版本ID
            file_path: 新权重文件路径
            description: 版本描述

        Returns:
            新版本权重对象
        """
        try:
            parent = db.query(WeightLibrary).filter(
                WeightLibrary.id == parent_weight_id
            ).first()

            if not parent:
                raise ValueError(f"父版本不存在: {parent_weight_id}")

            # 递增版本号
            parent_version = parent.version
            try:
                major, minor = map(int, parent_version.split('.'))
                new_version = f"{major}.{minor + 1}"
            except ValueError:
                new_version = "2.0"

            # 创建新版本记录
            new_weight = WeightLibrary(
                name=parent.name,
                description=description or parent.description,
                task_type=parent.task_type,
                version=new_version,
                parent_version_id=parent.id,
                file_path=file_path,
                file_name=Path(file_path).name,
                file_size=Path(file_path).stat().st_size,
                framework=parent.framework,
                input_size=parent.input_size,
                class_names=parent.class_names,
                normalize_params=parent.normalize_params,
                extra_metadata=parent.extra_metadata,
                uploaded_by=parent.uploaded_by
            )

            db.add(new_weight)
            db.commit()
            db.refresh(new_weight)

            self.logger.info(f"创建新版本: {parent.name} {new_version}")
            return new_weight

        except Exception as e:
            self.logger.error(f"创建新版本失败: {e}")
            db.rollback()
            return None

    def get_version_history(self, db: Session, weight_id: int) -> List[WeightLibrary]:
        """
        获取权重版本历史

        Args:
            db: 数据库会话
            weight_id: 权重ID

        Returns:
            版本历史列表（按时间顺序）
        """
        try:
            weight = db.query(WeightLibrary).filter(
                WeightLibrary.id == weight_id
            ).first()

            if not weight:
                return []

            # 获取所有版本
            versions = []

            # 添加当前版本
            versions.append(weight)

            # 查找父版本
            current = weight
            while current.parent_version_id:
                parent = db.query(WeightLibrary).filter(
                    WeightLibrary.id == current.parent_version_id
                ).first()
                if parent:
                    versions.append(parent)
                    current = parent
                else:
                    break

            # 查找子版本
            children = db.query(WeightLibrary).filter(
                WeightLibrary.parent_version_id == weight_id
            ).all()
            versions.extend(children)

            # 按版本号排序
            versions.sort(key=lambda x: x.version)

            return versions

        except Exception as e:
            self.logger.error(f"获取版本历史失败: {e}")
            return []

    async def auto_detect_task_type(
        self,
        file_path: str,
        file_ext: str
    ) -> str:
        """
        自动检测权重文件的任务类型

        Args:
            file_path: 权重文件路径
            file_ext: 文件扩展名

        Returns:
            任务类型
        """
        try:
            model_data = self.model_loader.load_model(file_path)
            model_type = 'onnx' if file_ext == '.onnx' else 'pytorch'

            if model_type == 'onnx':
                task_type = TaskTypeDetector.detect(model_data['model'], 'onnx')
            else:
                task_type = TaskTypeDetector.detect(
                    model_data['model'],
                    'pytorch',
                    input_size=(1, 3, 224, 224)
                )

            return task_type

        except Exception as e:
            self.logger.error(f"自动检测任务类型失败: {e}")
            return 'unknown'

    # ==================== 权重树形结构相关方法 ====================

    def get_root_weights(
        self,
        db: Session,
        task_type: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[WeightLibrary]:
        """
        获取根节点权重列表

        Args:
            db: 数据库会话
            task_type: 过滤任务类型
            skip: 跳过记录数
            limit: 返回记录数

        Returns:
            根节点权重列表（is_root=True）
        """
        try:
            query = db.query(WeightLibrary).filter(WeightLibrary.is_root == True)

            if task_type:
                query = query.filter(WeightLibrary.task_type == task_type)

            weights = query.order_by(
                WeightLibrary.created_at.desc()
            ).offset(skip).limit(limit).all()

            self.logger.debug(f"获取根节点权重列表: {len(weights)} 条记录")
            return weights

        except Exception as e:
            self.logger.error(f"获取根节点权重列表失败: {e}")
            return []

    def build_weight_tree(self, db: Session) -> List[Dict[str, Any]]:
        """
        构建完整的权重树形结构

        Args:
            db: 数据库会话

        Returns:
            权重树列表（每个根节点包含其所有子节点）
        """
        try:
            # 获取所有根节点
            roots = db.query(WeightLibrary).filter(
                WeightLibrary.is_root == True
            ).order_by(WeightLibrary.created_at.desc()).all()

            trees = []
            for root in roots:
                trees.append(self._build_subtree(db, root))

            return trees

        except Exception as e:
            self.logger.error(f"构建权重树失败: {e}")
            return []

    def _build_subtree(self, db: Session, weight: WeightLibrary) -> Dict[str, Any]:
        """
        递归构建子树

        Args:
            db: 数据库会话
            weight: 当前权重节点

        Returns:
            包含子节点的字典
        """
        # 获取 display_name，处理可能的 AttributeError
        display_name = f"{weight.name} v{weight.version}"
        try:
            display_name = weight.display_name
        except (AttributeError, TypeError):
            pass

        node = {
            "id": weight.id,
            "name": weight.name,
            "display_name": display_name,
            "description": weight.description,
            "task_type": weight.task_type,
            "version": weight.version,
            "file_name": weight.file_name,
            "file_size_mb": round(weight.file_size / 1024 / 1024, 2) if weight.file_size else None,
            "framework": weight.framework,
            "is_auto_detected": weight.is_auto_detected,
            "is_root": weight.is_root,
            "source_type": weight.source_type,
            "source_training_id": weight.source_training_id,
            "architecture_id": weight.architecture_id,
            "parent_version_id": weight.parent_version_id,
            "created_at": weight.created_at,
            "children": []
        }

        # 获取子节点
        children = db.query(WeightLibrary).filter(
            WeightLibrary.parent_version_id == weight.id
        ).all()

        for child in children:
            node["children"].append(self._build_subtree(db, child))

        return node

    def get_weight_subtree(
        self,
        db: Session,
        weight_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        获取指定权重的子树

        Args:
            db: 数据库会话
            weight_id: 权重ID

        Returns:
            子树字典，如果权重不存在返回None
        """
        try:
            weight = db.query(WeightLibrary).filter(
                WeightLibrary.id == weight_id
            ).first()

            if not weight:
                return None

            return self._build_subtree(db, weight)

        except Exception as e:
            self.logger.error(f"获取权重子树失败: {e}")
            return None

    def get_weights_for_training(
        self,
        db: Session,
        architecture_id: Optional[int] = None,
        task_type: Optional[str] = None
    ) -> List[WeightLibrary]:
        """
        获取可用于训练的权重列表

        Args:
            db: 数据库会话
            architecture_id: 模型架构ID筛选
            task_type: 任务类型筛选

        Returns:
            可用于训练的权重列表
        """
        try:
            query = db.query(WeightLibrary).filter(
                WeightLibrary.is_active == "active"
            )

            if architecture_id:
                query = query.filter(WeightLibrary.architecture_id == architecture_id)

            if task_type:
                query = query.filter(WeightLibrary.task_type == task_type)

            weights = query.order_by(
                WeightLibrary.created_at.desc()
            ).all()

            self.logger.debug(f"获取可用于训练的权重: {len(weights)} 条记录")
            return weights

        except Exception as e:
            self.logger.error(f"获取可用于训练的权重失败: {e}")
            return []

    def get_weight_tree_by_architecture(
        self,
        db: Session,
        architecture_id: Optional[int] = None,
        model_code_id: Optional[int] = None,
        task_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        获取按架构筛选的权重树形结构

        Args:
            db: 数据库会话
            architecture_id: 模型架构ID筛选（旧逻辑，保留兼容）
            model_code_id: 模型ID（Model表），筛选该模型训练产生的权重
            task_type: 任务类型筛选

        Returns:
            权重树列表（每个根节点包含其所有子节点）
        """
        try:
            self.logger.info(f"[权重筛选] API调用: model_code_id={model_code_id}, architecture_id={architecture_id}, task_type={task_type}")

            # 如果提供了 model_code_id，筛选该模型训练产生的权重
            if model_code_id:
                from app.models.training import TrainingRun
                from app.models.model import Model

                # 验证模型是否存在（Model表）
                model = db.query(Model).filter(
                    Model.id == model_code_id,
                    Model.is_active == "active"
                ).first()

                if not model:
                    self.logger.warning(f"模型不存在: model_code_id={model_code_id}")
                    return []

                # 找到使用此模型训练的所有任务（TrainingRun.model_id 指向 Model.id）
                training_runs = db.query(TrainingRun).filter(
                    TrainingRun.model_id == model_code_id
                ).all()
                training_ids = [tr.id for tr in training_runs]

                self.logger.info(f"[权重筛选] model_code_id={model_code_id}, 模型名={model.name}, 找到 {len(training_ids)} 个训练任务: {training_ids}")

                # 打印每个训练任务的详细信息
                for tr in training_runs:
                    self.logger.info(f"[权重筛选]   训练任务: id={tr.id}, name={tr.name}, model_id={tr.model_id}")

                if not training_ids:
                    self.logger.debug(f"模型 {model.name} 尚未训练过")
                    return []

                # 找到这些训练任务产生的权重（只返回根节点）
                query = db.query(WeightLibrary).filter(
                    WeightLibrary.source_training_id.in_(training_ids),
                    WeightLibrary.is_root == True
                )

                self.logger.info(f"[权重筛选] 筛选条件: source_training_id IN {training_ids}, is_root=True")

            else:
                # 原有逻辑，使用 architecture_id 筛选（兼容旧版）
                query = db.query(WeightLibrary).filter(
                    WeightLibrary.is_root == True
                )

                if architecture_id:
                    query = query.filter(WeightLibrary.architecture_id == architecture_id)

            if task_type:
                query = query.filter(WeightLibrary.task_type == task_type)

            roots = query.order_by(WeightLibrary.created_at.desc()).all()

            # 记录查询结果
            if model_code_id:
                self.logger.info(f"[权重筛选] 实际找到 {len(roots)} 个权重根节点")
                for r in roots:
                    self.logger.info(f"[权重筛选]   - 权重: {r.name} (id={r.id}, source_training_id={r.source_training_id})")

            trees = []
            for root in roots:
                trees.append(self._build_subtree(db, root))

            self.logger.debug(f"获取权重树: {len(trees)} 个根节点")
            return trees

        except Exception as e:
            self.logger.error(f"获取权重树失败: {e}")
            return []

    def get_weight_training_config(
        self,
        db: Session,
        weight_id: int
    ) -> Dict[str, Any]:
        """
        获取权重的训练配置信息（完整版）

        返回包含数据集、模型架构、预训练权重、数据增强等完整信息

        Args:
            db: 数据库会话
            weight_id: 权重ID

        Returns:
            包含权重信息和完整训练配置的字典
        """
        try:
            weight = db.query(WeightLibrary).filter(
                WeightLibrary.id == weight_id
            ).first()

            if not weight:
                return {
                    "weight_id": weight_id,
                    "weight_name": "",
                    "training_config": None,
                    "source_training": None,
                    "dataset": None,
                    "model_architecture": None,
                    "pretrained_weight": None,
                    "augmentation": None
                }

            result = {
                "weight_id": weight_id,
                "weight_name": weight.display_name,
                "training_config": None,
                "source_training": None,
                "dataset": None,
                "model_architecture": None,
                "pretrained_weight": None,
                "augmentation": None
            }

            # 如果权重来自训练，获取完整训练配置
            if weight.source_training_id:
                from app.models.training import TrainingRun
                from app.models.dataset import Dataset
                from app.models.generated_code import GeneratedCode

                training = db.query(TrainingRun).filter(
                    TrainingRun.id == weight.source_training_id
                ).first()

                if training:
                    # 基础训练配置
                    result["training_config"] = training.hyperparams or {}
                    result["source_training"] = {
                        "id": training.id,
                        "name": training.name,
                        "status": training.status,
                        "created_at": training.created_at.isoformat() if training.created_at else None
                    }

                    # 获取数据集信息
                    if training.dataset_id:
                        dataset = db.query(Dataset).filter(
                            Dataset.id == training.dataset_id
                        ).first()
                        if dataset:
                            result["dataset"] = {
                                "id": dataset.id,
                                "name": dataset.name,
                                "format": dataset.format,
                                "num_images": dataset.num_images,
                                "num_classes": dataset.num_classes,
                                "classes": dataset.classes,
                                "path": dataset.path
                            }

                    # 获取模型架构信息
                    if training.model_id:
                        # 优先从 GeneratedCode 获取
                        model = db.query(GeneratedCode).filter(
                            GeneratedCode.id == training.model_id,
                            GeneratedCode.is_active == "active"
                        ).first()

                        if model:
                            meta = model.meta if isinstance(model.meta, dict) else {}
                            result["model_architecture"] = {
                                "id": model.id,
                                "name": _clean_model_name(model.name),
                                "description": None,  # GeneratedCode 没有description字段
                                "file_path": model.file_path,
                                "input_size": meta.get("input_size"),
                                "task_type": meta.get("task_type")
                            }
                        else:
                            # 兼容 Model 表
                            from app.models.model import Model
                            old_model = db.query(Model).filter(
                                Model.id == training.model_id
                            ).first()
                            if old_model:
                                import json
                                graph_json = old_model.graph_json if isinstance(old_model.graph_json, dict) else \
                                    json.loads(old_model.graph_json) if old_model.graph_json else {}
                                result["model_architecture"] = {
                                    "id": old_model.id,
                                    "name": _clean_model_name(old_model.name),
                                    "description": old_model.description,
                                    "file_path": old_model.code_path,
                                    "input_size": graph_json.get("input_size"),
                                    "task_type": graph_json.get("task_type")
                                }

                    # 获取预训练权重信息
                    if training.pretrained_weight_id:
                        pretrained = db.query(WeightLibrary).filter(
                            WeightLibrary.id == training.pretrained_weight_id
                        ).first()
                        if pretrained:
                            result["pretrained_weight"] = {
                                "id": pretrained.id,
                                "name": pretrained.name,
                                "display_name": pretrained.display_name,
                                "task_type": pretrained.task_type,
                                "version": pretrained.version,
                                "source_type": pretrained.source_type
                            }

                    # 解析数据增强配置
                    hyperparams = training.hyperparams or {}
                    aug_strategy_id = hyperparams.get("augmentation_strategy_id")

                    if aug_strategy_id:
                        # 通过 augmentation_strategy_id 获取策略详情
                        aug_strategy = db.query(AugmentationStrategy).filter(
                            AugmentationStrategy.id == aug_strategy_id
                        ).first()
                        if aug_strategy:
                            result["augmentation"] = {
                                "enabled": True,
                                "strategy": aug_strategy.name,
                                "strategy_id": aug_strategy.id,
                                "description": aug_strategy.description,
                                "config": aug_strategy.pipeline
                            }
                        else:
                            result["augmentation"] = {"enabled": False}
                    elif "augmentation" in hyperparams:
                        # 兼容旧的 augmentation 对象格式
                        aug_config = hyperparams["augmentation"]
                        result["augmentation"] = {
                            "enabled": aug_config.get("enabled", False),
                            "strategy": aug_config.get("strategy"),
                            "config": aug_config
                        }
                    else:
                        result["augmentation"] = {"enabled": False}

            return result

        except Exception as e:
            self.logger.error(f"获取权重训练配置失败: {e}")
            return {
                "weight_id": weight_id,
                "weight_name": "",
                "training_config": None,
                "source_training": None,
                "dataset": None,
                "model_architecture": None,
                "pretrained_weight": None,
                "augmentation": None
            }
