"""
生成的模型代码数据模型

存储通过代码生成器生成的PyTorch模型代码文件信息。
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base


class GeneratedCode(Base):
    """生成的模型代码模型"""
    __tablename__ = "generated_codes"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, index=True)
    file_path = Column(String(500), nullable=False)  # .py文件存储路径
    file_name = Column(String(100), nullable=False)  # 原始文件名（用于兼容）
    code_size = Column(Integer, default=0)  # 代码文件大小（字节）
    template_tag = Column(String(50), nullable=True)  # 使用的代码模板
    meta = Column(JSON, nullable=True)  # 其他元数据（如图结构、输入形状等）
    is_active = Column(String(10), default="active")  # active, deleted
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # 关联关系
    creator = relationship("User", back_populates="generated_codes")

    def __repr__(self):
        return f"<GeneratedCode(id={self.id}, name='{self.name}', file_name='{self.file_name}')>"

    def to_dict(self):
        """转换为字典格式"""
        return {
            "id": self.id,
            "name": self.name,
            "file_path": self.file_path,
            "file_name": self.file_name,
            "code_size": self.code_size,
            "template_tag": self.template_tag,
            "created": self.created_at.isoformat() if self.created_at else None,
        }
