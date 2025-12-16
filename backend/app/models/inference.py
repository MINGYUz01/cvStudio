"""
推理任务数据模型
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, ForeignKey, Float
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base


class InferenceJob(Base):
    """推理任务模型"""
    __tablename__ = "inference_jobs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, index=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    input_path = Column(String(500), nullable=False)  # 输入文件/文件夹路径
    output_path = Column(String(500), nullable=True)  # 输出路径
    status = Column(String(20), default="pending")  # pending, running, completed, failed
    inference_type = Column(String(20), default="single")  # single, batch
    confidence_threshold = Column(Float, default=0.5)
    iou_threshold = Column(Float, default=0.45)
    batch_size = Column(Integer, default=1)
    device = Column(String(20), default="cpu")
    total_images = Column(Integer, default=0)
    processed_images = Column(Integer, default=0)
    fps = Column(Float, nullable=True)  # 推理速度
    error_message = Column(Text, nullable=True)
    results = Column(JSON, nullable=True)  # 推理结果摘要
    start_time = Column(DateTime(timezone=True), nullable=True)
    end_time = Column(DateTime(timezone=True), nullable=True)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # 关联关系
    model = relationship("Model")
    creator = relationship("User")

    def __repr__(self):
        return f"<InferenceJob(id={self.id}, name='{self.name}', status='{self.status}')>"