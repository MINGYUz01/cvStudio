"""
权重库数据模型
管理上传的权重文件，支持版本管理
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, ForeignKey, Boolean
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base


class WeightLibrary(Base):
    """
    权重库模型

    存储上传的权重文件信息，支持分类、检测两种任务类型
    """
    __tablename__ = "weight_library"

    # 主键
    id = Column(Integer, primary_key=True, index=True)

    # 基本信息
    name = Column(String(100), nullable=False, index=True, comment="权重名称")
    description = Column(Text, nullable=True, comment="权重描述")

    # 任务类型：classification/detection
    task_type = Column(String(50), nullable=False, index=True, comment="任务类型")

    # 版本管理
    version = Column(String(20), default="1.0", comment="版本号")
    parent_version_id = Column(Integer, ForeignKey("weight_library.id"), nullable=True, comment="父版本ID")
    is_active = Column(String(10), default="active", comment="状态：active/deprecated")

    # 文件信息
    file_path = Column(String(500), nullable=False, comment="权重文件存储路径")
    file_name = Column(String(255), nullable=False, comment="原始文件名")
    file_size = Column(Integer, nullable=True, comment="文件大小（字节）")
    framework = Column(String(50), default="pytorch", comment="框架类型：pytorch/onnx")

    # 模型元信息
    input_size = Column(JSON, nullable=True, comment="模型输入尺寸 [height, width] 或 [batch, channels, height, width]")
    class_names = Column(JSON, nullable=True, comment="类别名称列表 ['cat', 'dog', ...]")
    normalize_params = Column(JSON, nullable=True, comment="归一化参数 {'mean': [...], 'std': [...]}")
    extra_metadata = Column(JSON, nullable=True, comment="其他元数据（输出形状、参数量等）")

    # 自动检测标记
    is_auto_detected = Column(Boolean, default=False, comment="任务类型是否自动检测")

    # 版本管理增强
    source_type = Column(String(20), default="uploaded", comment="来源类型: uploaded/trained")
    source_training_id = Column(Integer, ForeignKey("training_runs.id"), nullable=True, comment="来源训练任务ID")
    is_root = Column(Boolean, default=True, comment="是否为根节点(导入权重或best权重)")
    architecture_id = Column(Integer, ForeignKey("model_architectures.id"), nullable=True, comment="关联的模型架构ID")

    # 用户信息
    uploaded_by = Column(Integer, ForeignKey("users.id"), nullable=True, comment="上传用户ID")

    # 时间戳
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), comment="更新时间")

    # 关联关系
    uploader = relationship("User", back_populates="uploaded_weights")
    parent_version = relationship("WeightLibrary", remote_side=[id], foreign_keys=[parent_version_id])
    child_versions = relationship("WeightLibrary", foreign_keys=[parent_version_id], remote_side=[id], overlaps="parent_version")
    source_training = relationship("TrainingRun", foreign_keys=[source_training_id])
    architecture = relationship("ModelArchitecture", back_populates="weights")

    def __repr__(self):
        return f"<WeightLibrary(id={self.id}, name='{self.name}', task_type='{self.task_type}', version='{self.version}')>"

    @property
    def display_name(self) -> str:
        """获取显示名称（包含版本）"""
        return f"{self.name} v{self.version}"

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "id": self.id,
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "task_type": self.task_type,
            "version": self.version,
            "file_name": self.file_name,
            "file_size": self.file_size,
            "file_size_mb": round(self.file_size / 1024 / 1024, 2) if self.file_size else None,
            "framework": self.framework,
            "input_size": self.input_size,
            "class_names": self.class_names,
            "is_auto_detected": self.is_auto_detected,
            "source_type": self.source_type,
            "source_training_id": self.source_training_id,
            "is_root": self.is_root,
            "architecture_id": self.architecture_id,
            "parent_version_id": self.parent_version_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
