"""
数据集服务类
"""

import os
import shutil
import asyncio
import zipfile
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter
from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session

from app.models.dataset import Dataset
from app.utils.format_recognizers import DatasetFormatRecognizer
from app.core.config import settings

# 支持的压缩格式
SUPPORTED_ARCHIVE_FORMATS = {
    '.zip': 'zip',
    '.tar': 'tar',
    '.tar.gz': 'tar.gz',
    '.tgz': 'tar.gz',
    '.tar.bz2': 'tar.bz2',
    '.tbz2': 'tar.bz2',
    '.tar.xz': 'tar.xz',
    '.txz': 'tar.xz',
    '.7z': '7z',
}

# 标准格式置信度阈值
STANDARD_FORMAT_THRESHOLD = 0.7
NON_STANDARD_THRESHOLD = 0.3


class DatasetService:
    """数据集服务类"""

    def __init__(self):
        self.format_recognizer = DatasetFormatRecognizer()
        self.storage_path = Path(settings.DATASET_STORAGE_PATH)
        self.thumbnail_path = Path(settings.THUMBNAIL_STORAGE_PATH)

        # 确保存储目录存在
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.thumbnail_path.mkdir(parents=True, exist_ok=True)

    async def create_dataset_from_upload(self,
                                       db: Session,
                                       name: str,
                                       description: str,
                                       files: List[UploadFile],
                                       user_id: int = None) -> Dataset:
        """
        从上传的文件创建数据集

        Args:
            db: 数据库会话
            name: 数据集名称
            description: 数据集描述
            files: 上传的文件列表
            user_id: 用户ID

        Returns:
            Dataset: 创建的数据集对象
        """
        # 检查数据集名称是否已存在
        existing_dataset = db.query(Dataset).filter(Dataset.name == name).first()
        if existing_dataset:
            raise HTTPException(status_code=400, detail=f"数据集名称 '{name}' 已存在")

        # 创建数据集存储目录
        dataset_dir = self.storage_path / name
        dataset_dir.mkdir(exist_ok=True)

        try:
            # 保存上传的文件
            saved_files = []
            for file in files:
                file_path = dataset_dir / file.filename
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                saved_files.append(str(file_path))

            # 识别数据集格式
            format_result = self.format_recognizer.recognize_format(str(dataset_dir))
            best_format = format_result["best_format"]

            # 提取数据集元信息
            metadata = self._extract_metadata(format_result, dataset_dir)

            # 创建数据集记录
            dataset = Dataset(
                name=name,
                description=description,
                path=str(dataset_dir),
                format=best_format["format"],
                num_images=metadata.get("num_images", 0),
                num_classes=metadata.get("num_classes", 0),
                classes=metadata.get("classes", []),
                meta=metadata,
                is_standard=metadata.get("is_standard", False)
            )

            db.add(dataset)
            db.commit()
            db.refresh(dataset)

            # 生成缩略图（异步执行）
            asyncio.create_task(self._generate_thumbnails_async(dataset.id, str(dataset_dir)))

            return dataset

        except Exception as e:
            # 如果失败，清理已创建的目录
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)
            raise HTTPException(status_code=500, detail=f"创建数据集失败: {str(e)}")

    def register_existing_dataset(self,
                                db: Session,
                                name: str,
                                description: str,
                                dataset_path: str,
                                user_id: int = None) -> Dataset:
        """
        注册现有数据集

        Args:
            db: 数据库会话
            name: 数据集名称
            description: 数据集描述
            dataset_path: 数据集路径
            user_id: 用户ID

        Returns:
            Dataset: 创建的数据集对象
        """
        # 检查路径是否存在
        path = Path(dataset_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"数据集路径不存在: {dataset_path}")

        # 检查数据集名称是否已存在
        existing_dataset = db.query(Dataset).filter(Dataset.name == name).first()
        if existing_dataset:
            raise HTTPException(status_code=400, detail=f"数据集名称 '{name}' 已存在")

        try:
            # 识别数据集格式
            format_result = self.format_recognizer.recognize_format(dataset_path)
            best_format = format_result["best_format"]

            if best_format["confidence"] < 0.3:
                raise HTTPException(status_code=400, detail=f"无法识别数据集格式，置信度太低: {best_format.get('error')}")

            # 提取数据集元信息
            metadata = self._extract_metadata(format_result, Path(dataset_path))

            # 创建数据集记录
            dataset = Dataset(
                name=name,
                description=description,
                path=dataset_path,
                format=best_format["format"],
                num_images=metadata.get("num_images", 0),
                num_classes=metadata.get("num_classes", 0),
                classes=metadata.get("classes", []),
                meta=metadata,
                is_standard=metadata.get("is_standard", False)
            )

            db.add(dataset)
            db.commit()
            db.refresh(dataset)

            # 生成缩略图（异步执行）
            asyncio.create_task(self._generate_thumbnails_async(dataset.id, dataset_path))

            return dataset

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"注册数据集失败: {str(e)}")

    async def create_dataset_from_archive(self,
                                         db: Session,
                                         name: str,
                                         description: str,
                                         archive: UploadFile,
                                         user_id: int = None) -> Dataset:
        """
        从上传的压缩包创建数据集

        Args:
            db: 数据库会话
            name: 数据集名称
            description: 数据集描述
            archive: 上传的压缩包文件
            user_id: 用户ID

        Returns:
            Dataset: 创建的数据集对象

        Raises:
            HTTPException: 当压缩格式不支持或解压失败时
        """
        # 检查数据集名称是否已存在
        existing_dataset = db.query(Dataset).filter(Dataset.name == name).first()
        if existing_dataset:
            raise HTTPException(status_code=400, detail=f"数据集名称 '{name}' 已存在")

        # 检查压缩格式
        filename = archive.filename or ""
        archive_format = None
        for ext, fmt in SUPPORTED_ARCHIVE_FORMATS.items():
            if filename.lower().endswith(ext):
                archive_format = fmt
                break

        if not archive_format:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的压缩格式。支持的格式: {', '.join(SUPPORTED_ARCHIVE_FORMATS.keys())}"
            )

        # 创建数据集存储目录
        dataset_dir = self.storage_path / name
        if dataset_dir.exists():
            raise HTTPException(status_code=400, detail=f"数据集目录 '{name}' 已存在")

        dataset_dir.mkdir(parents=True, exist_ok=True)

        # 创建临时目录保存压缩包
        temp_dir = Path(tempfile.gettempdir()) / f"dataset_upload_{name}"
        temp_dir.mkdir(exist_ok=True)

        try:
            # 保存上传的压缩包到临时目录
            temp_archive_path = temp_dir / Path(filename).name
            with open(temp_archive_path, "wb") as buffer:
                content = await archive.read()
                buffer.write(content)

            # 解压压缩包
            await self._extract_archive(temp_archive_path, dataset_dir, archive_format)

            # 验证解压后的目录是否有内容
            if not any(dataset_dir.iterdir()):
                raise HTTPException(status_code=400, detail="压缩包解压后为空，请检查压缩包内容")

            # 识别数据集格式
            format_result = self.format_recognizer.recognize_format(str(dataset_dir))
            best_format = format_result["best_format"]

            # 提取数据集元信息
            metadata = self._extract_metadata(format_result, dataset_dir)

            # 创建数据集记录
            dataset = Dataset(
                name=name,
                description=description,
                path=str(dataset_dir),
                format=best_format["format"],
                num_images=metadata.get("num_images", 0),
                num_classes=metadata.get("num_classes", 0),
                classes=metadata.get("classes", []),
                meta=metadata,
                is_standard=metadata.get("is_standard", False)
            )

            db.add(dataset)
            db.commit()
            db.refresh(dataset)

            # 生成缩略图（异步执行）
            asyncio.create_task(self._generate_thumbnails_async(dataset.id, str(dataset_dir)))

            return dataset

        except HTTPException:
            # 如果是HTTP异常，清理目录后重新抛出
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)
            raise
        except Exception as e:
            # 如果失败，清理已创建的目录
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)
            raise HTTPException(status_code=500, detail=f"创建数据集失败: {str(e)}")
        finally:
            # 清理临时目录
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    async def _extract_archive(self, archive_path: Path, extract_to: Path, archive_format: str) -> None:
        """
        解压压缩包到指定目录

        Args:
            archive_path: 压缩包路径
            extract_to: 解压目标目录
            archive_format: 压缩格式

        Raises:
            HTTPException: 当解压失败时
        """
        try:
            if archive_format == 'zip':
                self._extract_zip(archive_path, extract_to)
            elif archive_format in ('tar', 'tar.gz', 'tar.bz2', 'tar.xz'):
                self._extract_tar(archive_path, extract_to)
            elif archive_format == '7z':
                await self._extract_7z(archive_path, extract_to)
            else:
                raise HTTPException(status_code=400, detail=f"不支持的压缩格式: {archive_format}")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"解压文件失败: {str(e)}")

    def _extract_zip(self, archive_path: Path, extract_to: Path) -> None:
        """解压ZIP文件（扁平化解压，移除单层根目录）"""
        import locale
        import chardet

        # 获取系统默认编码
        system_encoding = locale.getpreferredencoding(False) or 'utf-8'

        def fix_filename_encoding(filename: str) -> str:
            """
            修复ZIP文件中的文件名编码问题

            某些ZIP工具（特别是Windows上的老版本工具）使用本地编码（如GBK）
            而不是UTF-8来存储非ASCII文件名，导致解压时出现乱码
            """
            # 如果文件名只包含ASCII字符，直接返回
            try:
                filename.encode('ascii')
                return filename
            except UnicodeEncodeError:
                pass

            # 尝试检测并修复编码
            # 方法1: 检查是否是UTF-8编码的字节被错误地用CP437解码了（Python zipfile的默认行为）
            try:
                # zipfile在Windows上通常用CP437解码非UTF-8的文件名
                # 尝试反向编码后再用正确编码解码
                decoded = filename.encode('cp437').decode('utf-8')
                return decoded
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass

            # 方法2: 尝试用系统编码（Windows上通常是GBK）
            try:
                decoded = filename.encode('cp437').decode(system_encoding)
                return decoded
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass

            # 方法3: 使用chardet检测编码
            try:
                raw_bytes = filename.encode('cp437')
                detected = chardet.detect(raw_bytes)
                if detected['encoding']:
                    decoded = raw_bytes.decode(detected['encoding'])
                    return decoded
            except Exception:
                pass

            # 如果所有方法都失败，返回原文件名
            return filename

        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            # 检查ZIP文件是否有安全风险（路径遍历攻击）
            for member in zip_ref.namelist():
                if ".." in member or member.startswith("/"):
                    raise HTTPException(status_code=400, detail="压缩包包含不安全的路径")

            # 获取所有成员并修复文件名编码
            all_members = zip_ref.namelist()
            if not all_members:
                raise HTTPException(status_code=400, detail="压缩包为空")

            # 修复所有文件名的编码
            fixed_members = [fix_filename_encoding(name) for name in all_members]

            # 检查是否只有一个根目录
            root_dirs = set()
            for name in fixed_members:
                if '/' in name:
                    root_dirs.add(name.split('/')[0])
                elif '\\' in name:
                    root_dirs.add(name.split('\\')[0])

            # 如果只有一个根目录，需要扁平化解压
            strip_prefix = None
            if len(root_dirs) == 1:
                strip_prefix = list(root_dirs)[0] + ('/' if '/' in fixed_members[0] else '\\')

            # 解压所有文件
            for i, member in enumerate(zip_ref.infolist()):
                # 跳过目录本身
                if member.is_dir():
                    continue

                # 使用修复后的文件名
                target_path = fixed_members[i]
                if strip_prefix and target_path.startswith(strip_prefix):
                    target_path = target_path[len(strip_prefix):]

                # 安全检查
                if ".." in target_path or target_path.startswith("/") or target_path.startswith("\\"):
                    continue

                output_path = extract_to / target_path

                # 确保目标目录存在
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # 解压文件
                with zip_ref.open(member) as source, open(output_path, "wb") as target:
                    target.write(source.read())

    def _extract_tar(self, archive_path: Path, extract_to: Path) -> None:
        """解压TAR文件（扁平化解压，移除单层根目录）"""
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            # 检查TAR文件是否有安全风险
            for member in tar_ref.getnames():
                if ".." in member or member.startswith("/"):
                    raise HTTPException(status_code=400, detail="压缩包包含不安全的路径")

            # 获取所有成员，检查是否只有一个根目录
            all_members = [m for m in tar_ref.getnames() if m]
            if not all_members:
                raise HTTPException(status_code=400, detail="压缩包为空")

            # 检查是否只有一个根目录
            root_dirs = set()
            for name in all_members:
                if '/' in name:
                    root_dirs.add(name.split('/')[0])

            # 如果只有一个根目录，需要扁平化解压
            strip_prefix = None
            if len(root_dirs) == 1:
                strip_prefix = list(root_dirs)[0] + '/'

            # 解压所有文件
            for member in tar_ref.getmembers():
                if member.isdir():
                    continue

                # 处理路径
                target_path = member.name
                if strip_prefix and target_path.startswith(strip_prefix):
                    target_path = target_path[len(strip_prefix):]

                # 安全检查
                if ".." in target_path or target_path.startswith("/"):
                    continue

                output_path = extract_to / target_path

                # 确保目标目录存在
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # 解压文件
                file = tar_ref.extractfile(member)
                if file:
                    with open(output_path, "wb") as f:
                        f.write(file.read())

    async def _extract_7z(self, archive_path: Path, extract_to: Path) -> None:
        """解压7z文件（扁平化解压，移除单层根目录）"""
        try:
            import py7zr
            import shutil

            # 先解压到临时目录
            temp_extract_dir = archive_path.parent / f"temp_extract_{archive_path.stem}"
            temp_extract_dir.mkdir(exist_ok=True)

            with py7zr.SevenZipFile(archive_path, mode='r') as seven_zip:
                # 检查文件列表安全性
                all_names = seven_zip.getnames()
                for member in all_names:
                    if ".." in member or member.startswith("/"):
                        raise HTTPException(status_code=400, detail="压缩包包含不安全的路径")

                # 解压到临时目录
                seven_zip.extractall(path=str(temp_extract_dir))

            # 检查是否只有一个根目录
            contents = list(temp_extract_dir.iterdir())
            if len(contents) == 1 and contents[0].is_dir():
                # 只有一个根目录，扁平化处理：移动其内容到目标目录
                single_root = contents[0]
                for item in single_root.iterdir():
                    dest = extract_to / item.name
                    if dest.exists():
                        # 目标已存在，删除或跳过
                        if dest.is_dir():
                            shutil.rmtree(dest, ignore_errors=True)
                        else:
                            dest.unlink()
                    shutil.move(str(item), str(dest))
            else:
                # 多个根目录或文件，直接移动所有内容
                for item in contents:
                    dest = extract_to / item.name
                    if dest.exists():
                        if dest.is_dir():
                            shutil.rmtree(dest, ignore_errors=True)
                        else:
                            dest.unlink()
                    shutil.move(str(item), str(dest))

            # 清理临时目录
            shutil.rmtree(temp_extract_dir, ignore_errors=True)

        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="7z格式支持未安装，请运行: pip install py7zr"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"解压7z文件失败: {str(e)}")

    def get_dataset(self, db: Session, dataset_id: int) -> Optional[Dataset]:
        """
        获取数据集

        Args:
            db: 数据库会话
            dataset_id: 数据集ID

        Returns:
            Optional[Dataset]: 数据集对象
        """
        return db.query(Dataset).filter(Dataset.id == dataset_id, Dataset.is_active == "active").first()

    def get_datasets(self, db: Session, skip: int = 0, limit: int = 100,
                    format_filter: str = None) -> List[Dataset]:
        """
        获取数据集列表

        Args:
            db: 数据库会话
            skip: 跳过记录数
            limit: 限制记录数
            format_filter: 格式过滤器

        Returns:
            List[Dataset]: 数据集列表
        """
        query = db.query(Dataset).filter(Dataset.is_active == "active")

        if format_filter:
            query = query.filter(Dataset.format == format_filter)

        return query.offset(skip).limit(limit).all()

    def update_dataset(self, db: Session, dataset_id: int,
                      name: str = None, description: str = None) -> Optional[Dataset]:
        """
        更新数据集信息

        Args:
            db: 数据库会话
            dataset_id: 数据集ID
            name: 新名称
            description: 新描述

        Returns:
            Optional[Dataset]: 更新后的数据集对象
        """
        dataset = self.get_dataset(db, dataset_id)
        if not dataset:
            return None

        if name:
            # 检查名称是否重复
            existing_dataset = db.query(Dataset).filter(
                Dataset.name == name, Dataset.id != dataset_id
            ).first()
            if existing_dataset:
                raise HTTPException(status_code=400, detail=f"数据集名称 '{name}' 已存在")
            dataset.name = name

        if description is not None:
            dataset.description = description

        db.commit()
        db.refresh(dataset)
        return dataset

    def delete_dataset(self, db: Session, dataset_id: int, physical: bool = True) -> bool:
        """
        删除数据集（物理删除，同时删除数据库记录和文件）

        Args:
            db: 数据库会话
            dataset_id: 数据集ID
            physical: 是否物理删除文件，默认为True

        Returns:
            bool: 是否删除成功
        """
        dataset = self.get_dataset(db, dataset_id)
        if not dataset:
            return False

        dataset_path = Path(dataset.path)

        # 清理缩略图
        dataset_thumbnail_dir = self.thumbnail_path / str(dataset_id)
        if dataset_thumbnail_dir.exists():
            shutil.rmtree(dataset_thumbnail_dir)

        # 物理删除：删除数据集目录
        if physical and dataset_path.exists():
            try:
                shutil.rmtree(dataset_path)
            except Exception as e:
                # 记录日志但继续删除数据库记录
                print(f"警告：删除数据集目录失败: {e}")

        # 从数据库中删除记录
        db.delete(dataset)
        db.commit()

        return True

    def cleanup_missing_datasets(self, db: Session) -> List[int]:
        """
        清理文件不存在的数据集记录

        Args:
            db: 数据库会话

        Returns:
            List[int]: 被清理的数据集ID列表
        """
        from app.models.dataset import Dataset

        # 获取所有活跃的数据集
        all_datasets = db.query(Dataset).filter(Dataset.is_active == "active").all()
        cleaned_ids = []

        for dataset in all_datasets:
            dataset_path = Path(dataset.path)
            if not dataset_path.exists():
                # 文件不存在，删除数据库记录
                cleaned_ids.append(dataset.id)
                db.delete(dataset)

        if cleaned_ids:
            db.commit()

        return cleaned_ids

    def rescan_dataset(self, db: Session, dataset_id: int) -> Optional[Dataset]:
        """
        重新扫描数据集，更新元信息

        Args:
            db: 数据库会话
            dataset_id: 数据集ID

        Returns:
            Optional[Dataset]: 更新后的数据集对象
        """
        dataset = self.get_dataset(db, dataset_id)
        if not dataset:
            return None

        try:
            # 重新识别数据集格式
            format_result = self.format_recognizer.recognize_format(dataset.path)
            best_format = format_result["best_format"]

            # 提取数据集元信息
            metadata = self._extract_metadata(format_result, Path(dataset.path))

            # 更新数据集信息
            dataset.format = best_format["format"]
            dataset.num_images = metadata.get("num_images", 0)
            dataset.num_classes = metadata.get("num_classes", 0)
            dataset.classes = metadata.get("classes", [])
            dataset.meta = metadata
            dataset.is_standard = metadata.get("is_standard", False)

            db.commit()
            db.refresh(dataset)

            # 重新生成缩略图
            asyncio.create_task(self._generate_thumbnails_async(dataset.id, dataset.path))

            return dataset

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"重新扫描数据集失败: {str(e)}")

    def get_directory_structure(self, dataset_path: str, max_depth: int = 5) -> Dict:
        """
        获取数据集目录结构

        Args:
            dataset_path: 数据集路径
            max_depth: 最大递归深度

        Returns:
            Dict: 目录树结构
        """
        path = Path(dataset_path)
        if not path.exists():
            return {"name": "root", "type": "error", "message": "目录不存在"}

        def build_tree(current_path: Path, current_name: str, depth: int) -> Dict:
            """递归构建目录树"""
            if depth > max_depth:
                return {"name": current_name, "type": "folder", "children": [], "truncated": True}

            try:
                if current_path.is_dir():
                    children = []
                    try:
                        # 获取目录内容，排序：文件夹在前，文件在后
                        items = sorted(current_path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
                        # 限制每个目录最多显示的子项数量
                        for item in items[:100]:  # 最多100个子项
                            if item.name.startswith('.'):  # 跳过隐藏文件
                                continue
                            children.append(build_tree(item, item.name, depth + 1))
                    except PermissionError:
                        pass

                    return {
                        "name": current_name,
                        "type": "folder",
                        "path": str(current_path.relative_to(path)) if current_path != path else "",
                        "children": children,
                        "child_count": len(children)
                    }
                else:
                    # 判断是否为图片文件
                    is_image = current_path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff')
                    return {
                        "name": current_name,
                        "type": "file",
                        "extension": current_path.suffix,
                        "is_image": is_image,
                        "size": current_path.stat().st_size if current_path.exists() else 0
                    }
            except Exception:
                return {"name": current_name, "type": "error", "message": "无法访问"}

        return build_tree(path, path.name, 0)

    def _calculate_dataset_size(self, dataset_path: Path) -> int:
        """
        计算数据集总大小（字节）

        Args:
            dataset_path: 数据集路径

        Returns:
            int: 数据集总大小（字节）
        """
        total_size = 0
        try:
            for item in dataset_path.rglob('*'):
                if item.is_file():
                    total_size += item.stat().st_size
        except Exception as e:
            print(f"计算数据集大小时出错: {e}")

        return total_size

    def _is_standard_format(self, format_result: Dict, dataset_path: Path = None) -> bool:
        """
        判断数据集是否为标准格式（可直接用于训练）

        标准格式判定规则：
        1. 置信度 >= 0.7
        2. 有足够的图像数量（>=10张）
        3. 对于检测任务，有足够的标注（标注率 >= 50%）

        Args:
            format_result: 格式识别结果
            dataset_path: 数据集路径

        Returns:
            bool: 是否为标准格式
        """
        best_format = format_result["best_format"]
        confidence = best_format.get("confidence", 0)

        # 基础置信度检查
        if confidence < STANDARD_FORMAT_THRESHOLD:
            return False

        details = best_format.get("details", {})

        # 图像数量检查
        num_images = details.get("num_images", 0)
        if num_images < 10:
            return False

        # 对于检测格式（YOLO、COCO、VOC），检查标注质量
        format_name = best_format.get("format", "")
        if format_name in ["yolo", "coco", "voc"]:
            # 检查标注率
            label_stats = details.get("label_stats", {})
            annotated_images = label_stats.get("annotated_images",
                              label_stats.get("valid_labels", 0))
            annotation_rate = annotated_images / num_images if num_images > 0 else 0

            if annotation_rate < 0.5:
                return False

        return True

    def _extract_metadata(self, format_result: Dict, dataset_path: Path = None) -> Dict:
        """
        从格式识别结果中提取元信息

        Args:
            format_result: 格式识别结果
            dataset_path: 数据集路径（可选，用于计算大小）

        Returns:
            Dict: 提取的元信息
        """
        best_format = format_result["best_format"]
        details = best_format.get("details", {})

        metadata = {
            "format_confidence": best_format.get("confidence", 0),
            "is_standard": self._is_standard_format(format_result, dataset_path) if dataset_path else False,
            "recognition_error": best_format.get("error"),
            "all_recognition_results": format_result.get("all_results", {})
        }

        # 基础信息
        metadata["num_images"] = details.get("num_images", 0)
        metadata["num_classes"] = details.get("num_classes", 0)
        metadata["classes"] = details.get("classes", [])

        # 计算数据集大小
        if dataset_path:
            metadata["size"] = self._calculate_dataset_size(dataset_path)

        # 格式特定信息
        if best_format["format"] == "yolo":
            metadata.update({
                "data_config": details.get("data_config", {}),
                "label_stats": details.get("label_stats", {}),
                "image_stats": details.get("image_stats", {}),
                "image_paths": details.get("image_paths", [])  # 保存图像路径列表
            })
        elif best_format["format"] == "coco":
            # COCO格式使用coco_data中的images列表
            coco_data = details.get("coco_data", {})
            metadata.update({
                "coco_data": coco_data,
                "validation": details.get("validation", {}),
                "image_stats": details.get("image_stats", {}),
                # 从COCO数据中提取图像路径
                "image_paths": details.get("image_paths", [])
            })
        elif best_format["format"] == "voc":
            metadata.update({
                "xml_stats": details.get("xml_stats", {}),
                "structure": details.get("structure", {}),
                "imagesets": details.get("imagesets", {}),
                "image_stats": details.get("image_stats", {}),
                "image_paths": details.get("image_files", [])  # 保存图像路径列表
            })
        elif best_format["format"] == "classification":
            # 对于分类格式，构造完整的 image_stats 和其他统计信息
            metadata.update({
                "class_directories": details.get("class_directories", {}),
                "structure": details.get("structure", {}),
                "size_distribution": details.get("size_distribution", {}),
                "format_distribution": details.get("format_distribution", {}),
                # 使用完整的图像路径列表
                "image_paths": details.get("image_paths", []),
                # 构造 image_stats 供前端使用
                "image_stats": {
                    "avg_width": details.get("avg_width", 0),
                    "avg_height": details.get("avg_height", 0),
                    "width_range": details.get("width_range", [0, 0]),
                    "height_range": details.get("height_range", [0, 0]),
                    "total_images": details.get("total_images", 0),
                    "analyzed_images": details.get("analyzed_images", 0),
                },
                # 分类任务标注率总是100%
                "annotation_rate": 1.0
            })

        # 提取路径映射信息（所有格式通用）
        if "path_mapping" in details:
            metadata["path_mapping"] = details["path_mapping"]
            # 添加数据集路径以便后续计算
            if dataset_path:
                metadata["path_mapping"]["dataset_path"] = str(dataset_path)

        return metadata

    async def _generate_thumbnails_async(self, dataset_id: int, dataset_path: str):
        """
        异步生成缩略图

        Args:
            dataset_id: 数据集ID
            dataset_path: 数据集路径
        """
        try:
            await self._generate_thumbnails(dataset_id, dataset_path)
        except Exception as e:
            print(f"生成缩略图失败: {e}")

    async def get_dataset_preview(self, dataset_id: int, limit: int = 10) -> Dict:
        """
        获取数据集预览信息

        Args:
            dataset_id: 数据集ID
            limit: 预览图像数量限制

        Returns:
            Dict: 预览信息
        """
        from sqlalchemy.orm import Session
        from app.database import SessionLocal

        db = SessionLocal()
        try:
            dataset = self.get_dataset(db, dataset_id)
            if not dataset:
                raise HTTPException(status_code=404, detail=f"数据集 {dataset_id} 不存在")

            preview_data = {
                "dataset_id": dataset_id,
                "sample_images": [],
                "format_details": {},
                "statistics": {}
            }

            # 根据数据集格式获取样本图像
            sample_images = await self._get_sample_images(dataset, limit)
            preview_data["sample_images"] = sample_images

            # 添加格式特定详情
            preview_data["format_details"] = self._get_format_details(dataset)

            # 添加基础统计信息
            preview_data["statistics"] = {
                "format": dataset.format,
                "num_images": dataset.num_images,
                "num_classes": dataset.num_classes,
                "classes": dataset.classes[:10] if dataset.classes else []  # 只显示前10个类别
            }

            return preview_data

        finally:
            db.close()

    async def get_dataset_statistics(self, dataset_id: int) -> Dict:
        """
        获取数据集统计信息

        Args:
            dataset_id: 数据集ID

        Returns:
            Dict: 统计信息
        """
        from sqlalchemy.orm import Session
        from app.database import SessionLocal
        from app.utils.dataset_statistics import DatasetStatistics

        db = SessionLocal()
        try:
            dataset = self.get_dataset(db, dataset_id)
            if not dataset:
                raise HTTPException(status_code=404, detail=f"数据集 {dataset_id} 不存在")

            # 使用专门的统计分析工具
            stats_analyzer = DatasetStatistics(dataset.path)
            detailed_stats = stats_analyzer.analyze_dataset(dataset.format)

            # 构建返回数据
            stats_data = {
                "dataset_id": dataset_id,
                "basic_info": {
                    "num_images": dataset.num_images,
                    "num_classes": dataset.num_classes,
                    "format": dataset.format,
                    "path": dataset.path
                },
                "detailed_statistics": detailed_stats,
                "summary": {
                    "total_images": detailed_stats.get("image_statistics", {}).get("total_images", 0),
                    "valid_images": detailed_stats.get("image_statistics", {}).get("valid_images", 0),
                    "corrupted_images": detailed_stats.get("image_statistics", {}).get("corrupted_images", 0),
                    "quality_score": detailed_stats.get("quality_metrics", {}).get("overall_quality_score", 0),
                    "recommendations": detailed_stats.get("quality_metrics", {}).get("recommendations", [])
                }
            }

            return stats_data

        finally:
            db.close()

    async def validate_dataset(self, dataset_id: int) -> Dict:
        """
        验证数据集

        Args:
            dataset_id: 数据集ID

        Returns:
            Dict: 验证结果
        """
        from sqlalchemy.orm import Session
        from app.database import SessionLocal

        db = SessionLocal()
        try:
            dataset = self.get_dataset(db, dataset_id)
            if not dataset:
                raise HTTPException(status_code=404, detail=f"数据集 {dataset_id} 不存在")

            validation_result = {
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "suggestions": []
            }

            # 基础验证
            dataset_path = Path(dataset.path)
            if not dataset_path.exists():
                validation_result["is_valid"] = False
                validation_result["errors"].append("数据集路径不存在")
                return validation_result

            # 验证图像文件
            image_validation = await self._validate_image_files(dataset)
            validation_result["errors"].extend(image_validation["errors"])
            validation_result["warnings"].extend(image_validation["warnings"])

            # 验证标注文件（如果有）
            if dataset.format in ["yolo", "coco", "voc"]:
                annotation_validation = await self._validate_annotation_files(dataset)
                validation_result["errors"].extend(annotation_validation["errors"])
                validation_result["warnings"].extend(annotation_validation["warnings"])

            # 格式特定验证
            format_validation = await self._validate_dataset_format(dataset)
            validation_result["errors"].extend(format_validation["errors"])
            validation_result["warnings"].extend(format_validation["warnings"])
            validation_result["suggestions"].extend(format_validation["suggestions"])

            # 设置整体验证状态
            validation_result["is_valid"] = len(validation_result["errors"]) == 0

            return validation_result

        finally:
            db.close()

    async def _get_sample_images(self, dataset: Dataset, limit: int) -> List[Dict]:
        """获取样本图像信息"""
        sample_images = []
        dataset_path = Path(dataset.path)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

        try:
            # 根据数据集格式查找图像
            if dataset.format == "classification":
                # 从每个类别目录中获取样本
                class_dirs = []
                for item in dataset_path.iterdir():
                    if item.is_dir():
                        class_dirs.append(item)

                for class_dir in class_dirs[:min(len(class_dirs), limit)]:  # 每类一张图
                    for img_file in class_dir.iterdir():
                        if img_file.suffix.lower() in image_extensions:
                            sample_images.append({
                                "path": str(img_file),
                                "filename": img_file.name,
                                "class": class_dir.name,
                                "size": self._get_image_size(img_file)
                            })
                            break

            else:
                # 对于其他格式，查找所有图像文件
                all_images = []
                for ext in image_extensions:
                    all_images.extend(dataset_path.rglob(f"*{ext}"))

                for i, img_file in enumerate(all_images[:limit]):
                    sample_images.append({
                        "path": str(img_file),
                        "filename": img_file.name,
                        "class": "unknown",
                        "size": self._get_image_size(img_file)
                    })

        except Exception as e:
            print(f"获取样本图像失败: {e}")

        return sample_images

    def _get_image_size(self, img_path: Path) -> Dict:
        """获取图像尺寸信息"""
        try:
            from PIL import Image
            with Image.open(img_path) as img:
                return {"width": img.width, "height": img.height}
        except Exception:
            return {"width": 0, "height": 0}

    def _get_format_details(self, dataset: Dataset) -> Dict:
        """获取格式特定详情"""
        details = {
            "format": dataset.format,
            "confidence": 0,
            "specific_info": {}
        }

        if dataset.meta:
            details["confidence"] = dataset.meta.get("format_confidence", 0)
            details["specific_info"] = {
                k: v for k, v in dataset.meta.items()
                if k not in ["num_images", "num_classes", "classes"]
            }

        return details

    def _get_yolo_statistics(self, dataset: Dataset) -> Dict:
        """获取YOLO格式统计信息"""
        stats = {}
        if dataset.meta:
            stats["class_distribution"] = dataset.meta.get("label_stats", {}).get("class_distribution", {})
            stats["avg_objects_per_image"] = dataset.meta.get("label_stats", {}).get("avg_objects_per_image", 0)
            stats["image_size_distribution"] = dataset.meta.get("image_stats", {})
        return stats

    def _get_coco_statistics(self, dataset: Dataset) -> Dict:
        """获取COCO格式统计信息"""
        stats = {}
        if dataset.meta:
            coco_data = dataset.meta.get("coco_data", {})
            stats["class_distribution"] = coco_data.get("category_counts", {})
            stats["image_size_distribution"] = dataset.meta.get("image_stats", {})
            stats["coco_summary"] = {
                "num_images": coco_data.get("num_images", 0),
                "num_annotations": coco_data.get("num_annotations", 0),
                "num_categories": coco_data.get("num_categories", 0)
            }
        return stats

    def _get_voc_statistics(self, dataset: Dataset) -> Dict:
        """获取VOC格式统计信息"""
        stats = {}
        if dataset.meta:
            xml_stats = dataset.meta.get("xml_stats", {})
            stats["avg_objects_per_image"] = xml_stats.get("avg_objects_per_image", 0)
            stats["max_objects_per_image"] = xml_stats.get("max_objects_per_image", 0)
            stats["class_distribution"] = xml_stats.get("classes", {})
            stats["image_size_distribution"] = dataset.meta.get("image_stats", {})
        return stats

    def _get_classification_statistics(self, dataset: Dataset) -> Dict:
        """获取分类格式统计信息"""
        stats = {}
        if dataset.meta:
            class_directories = dataset.meta.get("class_directories", {})
            stats["class_distribution"] = {
                class_name: info.get("image_count", 0)
                for class_name, info in class_directories.items()
            }
            stats["size_distribution"] = dataset.meta.get("size_distribution", {})
            stats["format_distribution"] = dataset.meta.get("format_distribution", {})
        return stats

    async def _validate_image_files(self, dataset: Dataset) -> Dict:
        """验证图像文件"""
        validation = {"errors": [], "warnings": []}
        dataset_path = Path(dataset.path)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

        try:
            # 检查是否有图像文件
            image_count = 0
            corrupted_files = []

            for ext in image_extensions:
                images = list(dataset_path.rglob(f"*{ext}"))
                image_count += len(images)

                # 抽样检查图像完整性
                for img_file in images[:10]:  # 最多检查10张
                    try:
                        from PIL import Image
                        with Image.open(img_file) as img:
                            img.verify()
                    except Exception:
                        corrupted_files.append(img_file.name)

            if image_count == 0:
                validation["errors"].append("数据集中没有找到图像文件")
            elif image_count < 10:
                validation["warnings"].append("数据集图像数量较少，可能影响训练效果")

            if corrupted_files:
                validation["errors"].append(f"发现 {len(corrupted_files)} 个损坏的图像文件")

        except Exception as e:
            validation["errors"].append(f"验证图像文件时出错: {str(e)}")

        return validation

    async def _validate_annotation_files(self, dataset: Dataset) -> Dict:
        """验证标注文件"""
        validation = {"errors": [], "warnings": []}
        dataset_path = Path(dataset.path)

        try:
            if dataset.format == "yolo":
                # 验证YOLO标注文件
                annotation_count = 0
                for txt_file in dataset_path.rglob("*.txt"):
                    if txt_file.name not in ["obj.names", "classes.txt", "names.txt"]:
                        annotation_count += 1

                if annotation_count == 0:
                    validation["errors"].append("没有找到YOLO标注文件(.txt)")

            elif dataset.format == "coco":
                # 验证COCO标注文件
                json_files = list(dataset_path.rglob("*.json"))
                if not json_files:
                    validation["errors"].append("没有找到COCO标注文件(.json)")

            elif dataset.format == "voc":
                # 验证VOC标注文件
                xml_count = 0
                for xml_file in dataset_path.rglob("*.xml"):
                    xml_count += 1

                if xml_count == 0:
                    validation["errors"].append("没有找到VOC标注文件(.xml)")

        except Exception as e:
            validation["errors"].append(f"验证标注文件时出错: {str(e)}")

        return validation

    async def _validate_dataset_format(self, dataset: Dataset) -> Dict:
        """验证数据集格式"""
        validation = {"errors": [], "warnings": [], "suggestions": []}

        # 检查格式置信度
        if dataset.meta:
            confidence = dataset.meta.get("format_confidence", 0)
            if confidence < 0.5:
                validation["warnings"].append("数据集格式识别置信度较低")
                validation["suggestions"].append("建议检查数据集结构或手动指定格式")

        # 检查类别和图像的比例
        if dataset.num_images > 0 and dataset.num_classes > 0:
            avg_images_per_class = dataset.num_images / dataset.num_classes
            if avg_images_per_class < 5:
                validation["warnings"].append("部分类别样本数量较少，可能影响训练效果")
                validation["suggestions"].append("建议为每个类别至少收集50-100个样本")

        return validation

    async def _generate_thumbnails(self, dataset_id: int, dataset_path: str):
        """
        生成数据集缩略图

        Args:
            dataset_id: 数据集ID
            dataset_path: 数据集路径
        """
        from PIL import Image
        import os

        try:
            # 创建缩略图存储目录
            thumbnail_dir = self.thumbnail_path / str(dataset_id)
            thumbnail_dir.mkdir(parents=True, exist_ok=True)

            dataset_path = Path(dataset_path)
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            thumbnail_size = (256, 256)  # 缩略图尺寸

            # 查找图像文件
            all_images = []
            for ext in image_extensions:
                all_images.extend(dataset_path.rglob(f"*{ext}"))

            # 限制生成缩略图数量以避免性能问题
            max_thumbnails = 100
            processed_count = 0

            for img_path in all_images:
                if processed_count >= max_thumbnails:
                    break

                try:
                    # 生成缩略图文件路径
                    relative_path = img_path.relative_to(dataset_path)
                    thumbnail_path = thumbnail_dir / f"{relative_path.stem}_thumb{relative_path.suffix}"

                    # 确保缩略图目录存在
                    thumbnail_path.parent.mkdir(parents=True, exist_ok=True)

                    # 生成缩略图
                    with Image.open(img_path) as img:
                        # 转换为RGB（处理RGBA等格式）
                        if img.mode != 'RGB':
                            img = img.convert('RGB')

                        # 创建缩略图，保持宽高比
                        img.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)

                        # 保存缩略图
                        img.save(thumbnail_path, 'JPEG', quality=85)

                    processed_count += 1

                except Exception as e:
                    print(f"生成缩略图失败 {img_path}: {e}")
                    continue

            print(f"数据集 {dataset_id} 缩略图生成完成，共处理 {processed_count} 张图像")

        except Exception as e:
            print(f"生成数据集缩略图时出错: {e}")

    async def get_thumbnail(self, dataset_id: int, image_path: str) -> Optional[str]:
        """
        获取图像的缩略图路径

        Args:
            dataset_id: 数据集ID
            image_path: 原始图像路径

        Returns:
            Optional[str]: 缩略图路径，如果不存在则返回None
        """
        try:
            img_path = Path(image_path)
            thumbnail_dir = self.thumbnail_path / str(dataset_id)

            # 生成缩略图文件名
            relative_path = img_path.relative_to(img_path.parents[-2])  # 获取相对于数据集目录的路径
            thumbnail_name = f"{img_path.stem}_thumb.jpg"

            # 在缩略图目录中查找
            for thumbnail_file in thumbnail_dir.rglob(f"{img_path.stem}_thumb*"):
                if thumbnail_file.exists():
                    return str(thumbnail_file)

            return None

        except Exception as e:
            print(f"获取缩略图失败: {e}")
            return None

    async def generate_single_thumbnail(self, dataset_id: int, image_path: str) -> Optional[str]:
        """
        为单个图像生成缩略图

        Args:
            dataset_id: 数据集ID
            image_path: 图像路径

        Returns:
            Optional[str]: 生成的缩略图路径
        """
        from PIL import Image

        try:
            img_path = Path(image_path)
            if not img_path.exists():
                return None

            # 创建缩略图存储目录
            thumbnail_dir = self.thumbnail_path / str(dataset_id)
            thumbnail_dir.mkdir(parents=True, exist_ok=True)

            # 生成缩略图文件路径
            thumbnail_path = thumbnail_dir / f"{img_path.stem}_thumb.jpg"

            # 生成缩略图
            with Image.open(img_path) as img:
                # 转换为RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # 创建缩略图
                thumbnail_size = (256, 256)
                img.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)

                # 保存缩略图
                img.save(thumbnail_path, 'JPEG', quality=85)

            return str(thumbnail_path)

        except Exception as e:
            print(f"生成单个缩略图失败: {e}")
            return None

    def get_thumbnails_list(self, dataset_id: int) -> List[str]:
        """
        获取数据集的所有缩略图列表

        Args:
            dataset_id: 数据集ID

        Returns:
            List[str]: 缩略图路径列表
        """
        try:
            thumbnail_dir = self.thumbnail_path / str(dataset_id)
            if not thumbnail_dir.exists():
                return []

            thumbnails = []
            for thumbnail_file in thumbnail_dir.rglob("*_thumb.jpg"):
                if thumbnail_file.is_file():
                    thumbnails.append(str(thumbnail_file))

            return sorted(thumbnails)

        except Exception as e:
            print(f"获取缩略图列表失败: {e}")
            return []

    def clear_thumbnails(self, dataset_id: int) -> bool:
        """
        清理数据集的缩略图

        Args:
            dataset_id: 数据集ID

        Returns:
            bool: 是否清理成功
        """
        try:
            thumbnail_dir = self.thumbnail_path / str(dataset_id)
            if thumbnail_dir.exists():
                shutil.rmtree(thumbnail_dir)
            return True

        except Exception as e:
            print(f"清理缩略图失败: {e}")
            return False

    async def compare_datasets(self, db: Session, dataset_ids: List[int]) -> Dict:
        """
        比较多个数据集的统计信息

        Args:
            db: 数据库会话
            dataset_ids: 数据集ID列表

        Returns:
            Dict: 比较结果
        """
        try:
            if len(dataset_ids) < 2:
                raise HTTPException(status_code=400, detail="至少需要选择2个数据集进行比较")

            # 获取所有数据集信息
            datasets = []
            for dataset_id in dataset_ids:
                dataset = self.get_dataset(db, dataset_id)
                if not dataset:
                    raise HTTPException(status_code=404, detail=f"数据集 {dataset_id} 不存在")
                datasets.append(dataset)

            # 获取每个数据集的统计信息
            comparison_data = {
                "datasets": [],
                "comparison_metrics": {},
                "summary": {}
            }

            all_stats = []
            for dataset in datasets:
                stats = await self.get_dataset_statistics(dataset.id)
                comparison_data["datasets"].append({
                    "id": dataset.id,
                    "name": dataset.name,
                    "format": dataset.format,
                    "basic_info": stats["basic_info"],
                    "summary": stats["summary"]
                })
                all_stats.append(stats)

            # 比较指标
            comparison_metrics = {
                "image_count": {},
                "class_count": {},
                "quality_score": {},
                "format_distribution": {},
                "resolution_comparison": {}
            }

            # 图像数量比较
            image_counts = [(d["id"], d["basic_info"]["num_images"]) for d in comparison_data["datasets"]]
            comparison_metrics["image_count"] = {
                "values": dict(image_counts),
                "max": max(image_counts, key=lambda x: x[1])[0],
                "min": min(image_counts, key=lambda x: x[1])[0],
                "range": max(x[1] for x in image_counts) - min(x[1] for x in image_counts)
            }

            # 类别数量比较
            class_counts = [(d["id"], d["basic_info"]["num_classes"]) for d in comparison_data["datasets"]]
            comparison_metrics["class_count"] = {
                "values": dict(class_counts),
                "max": max(class_counts, key=lambda x: x[1])[0],
                "min": min(class_counts, key=lambda x: x[1])[0],
                "range": max(x[1] for x in class_counts) - min(x[1] for x in class_counts)
            }

            # 质量分数比较
            quality_scores = [(d["id"], d["summary"]["quality_score"]) for d in comparison_data["datasets"]]
            comparison_metrics["quality_score"] = {
                "values": dict(quality_scores),
                "best": max(quality_scores, key=lambda x: x[1])[0],
                "worst": min(quality_scores, key=lambda x: x[1])[0],
                "average": sum(x[1] for x in quality_scores) / len(quality_scores)
            }

            # 格式分布
            formats = [d["format"] for d in comparison_data["datasets"]]
            comparison_metrics["format_distribution"] = {
                "unique_formats": list(set(formats)),
                "format_counts": dict(Counter(formats))
            }

            comparison_data["comparison_metrics"] = comparison_metrics

            # 生成总结
            comparison_data["summary"] = {
                "total_datasets": len(datasets),
                "total_images": sum(d["basic_info"]["num_images"] for d in comparison_data["datasets"]),
                "unique_formats": len(set(d["format"] for d in datasets)),
                "recommendations": self._generate_comparison_recommendations(comparison_data)
            }

            return comparison_data

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"比较数据集失败: {str(e)}")

    def _generate_comparison_recommendations(self, comparison_data: Dict) -> List[str]:
        """生成数据集比较的建议"""
        recommendations = []

        try:
            # 基于质量分数的建议
            quality_scores = comparison_data["comparison_metrics"]["quality_score"]
            best_quality = quality_scores["best"]
            worst_quality = quality_scores["worst"]

            if quality_scores["average"] < 70:
                recommendations.append("整体数据集质量偏低，建议进行数据清洗和增强")

            if quality_scores["values"][best_quality] - quality_scores["values"][worst_quality] > 30:
                recommendations.append("数据集质量差异较大，建议以高质量数据集为基准进行改进")

            # 基于图像数量的建议
            image_count_range = comparison_data["comparison_metrics"]["image_count"]["range"]
            if image_count_range > 1000:
                recommendations.append("数据集规模差异较大，可能影响模型训练效果")

            # 基于格式一致性的建议
            unique_formats = comparison_data["comparison_metrics"]["format_distribution"]["unique_formats"]
            if len(unique_formats) > 1:
                recommendations.append(f"发现多种数据格式: {', '.join(unique_formats)}，建议统一格式以便于管理")

            # 基于类别数量的建议
            class_count_values = list(comparison_data["comparison_metrics"]["class_count"]["values"].values())
            if len(set(class_count_values)) > 1:
                recommendations.append("数据集类别数量不同，可能需要进行类别对齐")

        except Exception as e:
            print(f"生成比较建议失败: {e}")

        return recommendations

    async def get_images_by_class(
        self,
        dataset_path: str,
        class_name: str,
        page: int = 1,
        page_size: int = 20
    ) -> Dict[str, any]:
        """
        获取指定类别的图片（用于分类任务）

        Args:
            dataset_path: 数据集路径
            class_name: 类别名称
            page: 页码
            page_size: 每页大小

        Returns:
            图片列表和分页信息
        """
        from pathlib import Path
        from app.utils.image_processor import image_processor

        dataset_path = Path(dataset_path)
        class_dir = dataset_path / class_name

        if not class_dir.exists():
            return {
                'images': [],
                'total': 0,
                'page': page,
                'page_size': page_size,
                'total_pages': 0,
                'class_name': class_name
            }

        # 获取类别目录下的所有图片
        image_files = []
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

        for ext in supported_formats:
            image_files.extend(class_dir.glob(f"*{ext}"))

        # 去重（Windows文件系统不区分大小写，glob可能返回重复项）
        # 使用字典保持去重同时保留顺序
        image_files = list(dict.fromkeys(image_files))

        # 排序
        image_files.sort(key=lambda x: x.name)

        # 分页
        total = len(image_files)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_files = image_files[start_idx:end_idx]

        # 构建结果
        images = []
        for img_file in page_files:
            info = image_processor.get_image_info(str(img_file))
            if info:
                # 添加类别信息
                info['class_name'] = class_name
                info['relative_path'] = f"{class_name}/{img_file.name}"
                images.append(info)

        total_pages = (total + page_size - 1) // page_size if total > 0 else 0

        return {
            'images': images,
            'total': total,
            'page': page,
            'page_size': page_size,
            'total_pages': total_pages,
            'class_name': class_name
        }