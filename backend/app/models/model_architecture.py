"""
模型架构数据模型

存储用户通过 ModelBuilder 创建的模型架构信息。
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base


class ModelArchitecture(Base):
    """模型架构模型"""
    __tablename__ = "model_architectures"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, index=True)
    description = Column(Text, nullable=True)
    version = Column(String(20), default="v1.0")
    type = Column(String(50), default="Custom")  # 模型类型：ResNet, YOLO, Custom等
    file_path = Column(String(500), nullable=False)  # JSON配置文件存储路径
    file_name = Column(String(100), nullable=False)  # 原始文件名（用于兼容）
    node_count = Column(Integer, default=0)
    connection_count = Column(Integer, default=0)
    meta = Column(JSON, nullable=True)  # 存储完整的nodes/connections数据
    is_active = Column(String(10), default="active")  # active, deleted
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # 关联关系
    creator = relationship("User", back_populates="model_architectures")
    weights = relationship("WeightLibrary", back_populates="architecture")

    def __repr__(self):
        return f"<ModelArchitecture(id={self.id}, name='{self.name}', version='{self.version}')>"

    def to_dict(self):
        """转换为字典格式"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description or "",
            "version": self.version,
            "type": self.type,
            "file_path": self.file_path,
            "file_name": self.file_name,
            "node_count": self.node_count,
            "connection_count": self.connection_count,
            "created": self.created_at.isoformat() if self.created_at else None,
            "updated": self.updated_at.isoformat() if self.updated_at else None,
        }
