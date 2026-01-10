"""
数据增强策略数据库模型
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship

from app.database import Base


class AugmentationStrategy(Base):
    """数据增强策略预设模型"""
    __tablename__ = 'augmentation_strategies'

    id = Column(Integer, primary_key=True, index=True, comment='策略ID')
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, comment='用户ID')
    name = Column(String(100), nullable=False, comment='策略名称')
    description = Column(Text, nullable=True, comment='策略描述')
    pipeline = Column(JSON, nullable=False, default=list, comment='算子配置流水线')
    is_default = Column(Integer, default=0, comment='是否为默认策略')
    created_at = Column(DateTime, default=datetime.now, comment='创建时间')
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, comment='更新时间')

    # 关系
    user = relationship("User", back_populates="augmentation_strategies")
