"""
训练任务数据模型
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, ForeignKey, Float
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base


class TrainingRun(Base):
    """训练任务模型"""
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, index=True)
    description = Column(Text, nullable=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    hyperparams = Column(JSON, nullable=True)  # 超参数配置
    status = Column(String(20), default="pending")  # pending, running, completed, failed, stopped
    progress = Column(Float, default=0.0)  # 进度百分比
    current_epoch = Column(Integer, default=0)
    total_epochs = Column(Integer, default=0)
    best_metric = Column(Float, nullable=True)  # 最佳指标值
    device = Column(String(20), default="cpu")  # 训练设备
    celery_task_id = Column(String(255), nullable=True)  # Celery任务ID
    experiment_dir = Column(String(500), nullable=True)  # 实验目录路径
    log_file = Column(String(500), nullable=True)  # 日志文件路径
    error_message = Column(Text, nullable=True)
    start_time = Column(DateTime(timezone=True), nullable=True)
    end_time = Column(DateTime(timezone=True), nullable=True)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # 关联关系
    model = relationship("Model", back_populates="training_runs")
    dataset = relationship("Dataset")
    creator = relationship("User")
    checkpoints = relationship("Checkpoint", back_populates="training_run")

    def __repr__(self):
        return f"<TrainingRun(id={self.id}, name='{self.name}', status='{self.status}')>"


class Checkpoint(Base):
    """检查点模型"""
    __tablename__ = "checkpoints"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("training_runs.id"), nullable=False)
    epoch = Column(Integer, nullable=False)
    metric_value = Column(Float, nullable=True)  # 主要指标值（如accuracy、mAP）
    metrics = Column(JSON, nullable=True)  # 所有指标
    path = Column(String(500), nullable=False)  # 检查点文件路径
    file_size = Column(Integer, nullable=True)  # 文件大小（字节）
    is_best = Column(String(10), default="false")  # 是否为最佳模型
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # 关联关系
    training_run = relationship("TrainingRun", back_populates="checkpoints")

    def __repr__(self):
        return f"<Checkpoint(id={self.id}, run_id={self.run_id}, epoch={self.epoch})>"