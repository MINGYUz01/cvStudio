"""
模型服务类

提供模型架构和生成代码的数据库操作服务。
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from sqlalchemy.orm import Session

from app.models.model_architecture import ModelArchitecture
from app.models.generated_code import GeneratedCode
from app.schemas.model import (
    ModelArchitectureCreate,
    ModelArchitectureUpdate,
    GeneratedCodeCreate,
    GeneratedCodeUpdate
)


# 存储路径配置
ARCHITECTURE_DIR = Path("data/architectures")
MODEL_DIR = Path("data/models")
ARCHITECTURE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class ModelArchitectureService:
    """模型架构服务类"""

    def __init__(self):
        self.storage_path = ARCHITECTURE_DIR

    def _generate_filename(self, name: str) -> str:
        """生成安全的文件名"""
        # 规范化名称，移除特殊字符
        safe_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{safe_name}_{timestamp}.json"

    def create_architecture(
        self,
        db: Session,
        data: ModelArchitectureCreate,
        user_id: Optional[int] = None,
        overwrite: bool = False,
        target_filename: Optional[str] = None
    ) -> ModelArchitecture:
        """
        创建模型架构

        Args:
            db: 数据库会话
            data: 架构数据
            user_id: 用户ID
            overwrite: 是否覆盖同名文件
            target_filename: 目标ID或文件名（用于更新操作）

        Returns:
            ModelArchitecture: 创建的架构对象
        """
        file_name: str
        existing = None

        # 判断target_filename是ID还是文件名
        target_id = None
        if target_filename and target_filename.isdigit():
            # 是数字ID，通过ID查找
            target_id = int(target_filename)
            existing = db.query(ModelArchitecture).filter(
                ModelArchitecture.id == target_id,
                ModelArchitecture.is_active == "active"
            ).first()
        elif target_filename:
            # 是文件名，通过文件名查找
            existing = db.query(ModelArchitecture).filter(
                ModelArchitecture.file_name == target_filename,
                ModelArchitecture.is_active == "active"
            ).first()

        # 如果没有通过ID/文件名找到，再检查名称是否重复
        if not existing:
            existing = db.query(ModelArchitecture).filter(
                ModelArchitecture.name == data.name,
                ModelArchitecture.is_active == "active"
            ).first()

        if existing:
            # 更新现有架构
            file_name = existing.file_name
            file_path = self.storage_path / file_name

            # 更新数据库记录
            existing.name = data.name  # 名称也可能修改
            existing.description = data.description
            existing.type = data.type
            existing.node_count = len(data.nodes)
            existing.connection_count = len(data.connections)
            existing.meta = {
                "nodes": data.nodes,
                "connections": data.connections,
                "updated": datetime.now().isoformat()
            }
            existing.updated_at = datetime.now()
            db.commit()
            db.refresh(existing)

            # 保存文件
            self._save_architecture_file(file_path, data.model_dump())

            return existing

        elif not overwrite:
            raise ValueError(f"架构名称 '{data.name}' 已存在")

        else:
            # 创建新架构
            file_name = target_filename if target_filename else self._generate_filename(data.name)
            file_path = self.storage_path / file_name

            # 保存文件
            self._save_architecture_file(file_path, data.model_dump())

            # 创建数据库记录
            architecture = ModelArchitecture(
                name=data.name,
                description=data.description,
                type=data.type,
                file_path=str(file_path),
                file_name=file_name,
                node_count=len(data.nodes),
                connection_count=len(data.connections),
                meta={
                    "nodes": data.nodes,
                    "connections": data.connections,
                    "created": datetime.now().isoformat()
                },
                created_by=user_id
            )
            db.add(architecture)
            db.commit()
            db.refresh(architecture)

            return architecture

    def _save_architecture_file(self, file_path: Path, data: Dict[str, Any]) -> None:
        """保存架构文件"""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_architecture(self, db: Session, architecture_id: int) -> Optional[ModelArchitecture]:
        """获取单个架构"""
        return db.query(ModelArchitecture).filter(
            ModelArchitecture.id == architecture_id,
            ModelArchitecture.is_active == "active"
        ).first()

    def list_architectures(
        self,
        db: Session,
        skip: int = 0,
        limit: int = 100
    ) -> List[ModelArchitecture]:
        """获取架构列表"""
        return db.query(ModelArchitecture).filter(
            ModelArchitecture.is_active == "active"
        ).order_by(
            ModelArchitecture.updated_at.desc()
        ).offset(skip).limit(limit).all()

    def count_architectures(self, db: Session) -> int:
        """统计架构数量"""
        return db.query(ModelArchitecture).filter(
            ModelArchitecture.is_active == "active"
        ).count()

    def update_architecture(
        self,
        db: Session,
        architecture_id: int,
        data: ModelArchitectureUpdate
    ) -> Optional[ModelArchitecture]:
        """更新架构"""
        architecture = self.get_architecture(db, architecture_id)
        if not architecture:
            return None

        # 更新字段
        if data.name is not None:
            architecture.name = data.name
        if data.description is not None:
            architecture.description = data.description
        if data.type is not None:
            architecture.type = data.type

        # 更新节点和连接
        if data.nodes is not None or data.connections is not None:
            nodes = data.nodes if data.nodes is not None else architecture.meta.get("nodes", [])
            connections = data.connections if data.connections is not None else architecture.meta.get("connections", [])
            architecture.node_count = len(nodes)
            architecture.connection_count = len(connections)
            architecture.meta = {
                "nodes": nodes,
                "connections": connections,
                "updated": datetime.now().isoformat()
            }

            # 更新文件
            file_data = {
                "name": architecture.name,
                "description": architecture.description,
                "type": architecture.type,
                "nodes": nodes,
                "connections": connections,
                "updated": datetime.now().isoformat()
            }
            self._save_architecture_file(Path(architecture.file_path), file_data)

        db.commit()
        db.refresh(architecture)
        return architecture

    def delete_architecture(self, db: Session, architecture_id: int, physical: bool = True) -> bool:
        """
        删除架构（硬删除，从数据库中真正删除）

        Args:
            db: 数据库会话
            architecture_id: 架构ID
            physical: 是否同时删除物理文件
        """
        # 不使用 is_active 过滤，直接查找
        from app.models.model_architecture import ModelArchitecture
        architecture = db.query(ModelArchitecture).filter(
            ModelArchitecture.id == architecture_id
        ).first()

        if not architecture:
            return False

        # 删除物理文件
        if physical:
            file_path = Path(architecture.file_path)
            if file_path.exists():
                try:
                    file_path.unlink()
                    print(f"[ModelArchitectureService] 已删除架构文件: {file_path}")
                except Exception as e:
                    print(f"[ModelArchitectureService] 删除架构文件失败: {file_path}, 错误: {e}")

        # 硬删除：从数据库中真正删除记录
        db.delete(architecture)
        db.commit()

        print(f"[ModelArchitectureService] 架构已从数据库删除: ID={architecture_id}, 名称={architecture.name}")
        return True

    def load_architecture_file(self, architecture: ModelArchitecture) -> Dict[str, Any]:
        """从文件加载完整的架构数据"""
        file_path = Path(architecture.file_path)
        if not file_path.exists():
            # 如果文件不存在，从meta中返回数据
            return {
                "id": architecture.id,
                "name": architecture.name,
                "description": architecture.description or "",
                "type": architecture.type,
                "nodes": architecture.meta.get("nodes", []) if architecture.meta else [],
                "connections": architecture.meta.get("connections", []) if architecture.meta else [],
                "created": architecture.created_at.isoformat() if architecture.created_at else None,
                "updated": architecture.updated_at.isoformat() if architecture.updated_at else None,
            }

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 添加数据库信息
        data["id"] = architecture.id
        data["filename"] = architecture.file_name
        return data


class GeneratedCodeService:
    """生成代码服务类"""

    def __init__(self):
        self.storage_path = MODEL_DIR

    def _generate_filename(self, name: str) -> str:
        """生成安全的文件名"""
        safe_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{safe_name}_{timestamp}.py"

    def create_code(
        self,
        db: Session,
        name: str,
        code: str,
        user_id: Optional[int] = None,
        template_tag: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None
    ) -> GeneratedCode:
        """
        创建生成代码记录

        Args:
            db: 数据库会话
            name: 代码名称
            code: Python代码
            user_id: 用户ID
            template_tag: 模板标签
            meta: 元数据

        Returns:
            GeneratedCode: 创建的代码对象
        """
        file_name = self._generate_filename(name)
        file_path = self.storage_path / file_name

        # 保存代码文件
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)

        code_size = len(code.encode("utf-8"))

        # 创建数据库记录
        generated_code = GeneratedCode(
            name=name,
            file_path=str(file_path),
            file_name=file_name,
            code_size=code_size,
            template_tag=template_tag,
            meta=meta or {},
            created_by=user_id
        )
        db.add(generated_code)
        db.commit()
        db.refresh(generated_code)

        return generated_code

    def get_code(self, db: Session, code_id: int) -> Optional[GeneratedCode]:
        """获取单个代码记录"""
        return db.query(GeneratedCode).filter(
            GeneratedCode.id == code_id,
            GeneratedCode.is_active == "active"
        ).first()

    def list_codes(
        self,
        db: Session,
        skip: int = 0,
        limit: int = 100
    ) -> List[GeneratedCode]:
        """获取代码列表"""
        return db.query(GeneratedCode).filter(
            GeneratedCode.is_active == "active"
        ).order_by(
            GeneratedCode.created_at.desc()
        ).offset(skip).limit(limit).all()

    def count_codes(self, db: Session) -> int:
        """统计代码数量"""
        return db.query(GeneratedCode).filter(
            GeneratedCode.is_active == "active"
        ).count()

    def delete_code(self, db: Session, code_id: int, physical: bool = True) -> bool:
        """
        删除代码

        Args:
            db: 数据库会话
            code_id: 代码ID
            physical: 是否同时删除物理文件
        """
        code = self.get_code(db, code_id)
        if not code:
            return False

        # 删除物理文件
        if physical:
            file_path = Path(code.file_path)
            if file_path.exists():
                file_path.unlink()

        # 软删除
        code.is_active = "deleted"
        db.commit()
        return True

    def load_code_file(self, code: GeneratedCode) -> str:
        """从文件加载代码内容"""
        file_path = Path(code.file_path)
        if not file_path.exists():
            return ""

        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
