"""
数据集数据模型
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Boolean
from sqlalchemy.sql import func
from app.database import Base


class Dataset(Base):
    """数据集模型"""
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, index=True)
    description = Column(Text, nullable=True)
    path = Column(String(500), nullable=False)
    format = Column(String(20), nullable=False)  # yolo, coco, voc, classification
    num_images = Column(Integer, default=0)
    num_classes = Column(Integer, default=0)
    classes = Column(JSON, nullable=True)  # 类别信息
    meta = Column(JSON, nullable=True)  # 元数据（图像尺寸、统计信息等）
    is_active = Column(String(10), default="active")  # active, deleted
    is_standard = Column(Boolean, default=False)  # 是否为标准格式（可直接用于训练）
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    @property
    def format_confidence(self) -> float:
        """从meta中获取格式识别置信度"""
        if self.meta:
            return self.meta.get("format_confidence", 0)
        return 0

    def __repr__(self):
        return f"<Dataset(id={self.id}, name='{self.name}', format='{self.format}')>"