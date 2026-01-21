"""
删除 preset_models 表的迁移脚本

运行方式:
    python backend/scripts/drop_preset_models_table.py
"""

import sys
import os
from pathlib import Path

# 设置环境变量
os.environ.setdefault('SECRET_KEY', 'migration-secret-key')
os.environ.setdefault('DATABASE_URL', 'sqlite:///./backend/cvstudio.db')

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 添加 backend 目录到 Python 路径
backend_root = Path(__file__).parent
sys.path.insert(0, str(backend_root))

from sqlalchemy import text
from app.database import engine, SessionLocal


def drop_preset_models_table():
    """删除 preset_models 表"""
    print("Starting to drop preset_models table...")

    # 检查表是否存在
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='preset_models'"
        ))
        table_exists = result.fetchone() is not None

        if not table_exists:
            print("Table preset_models does not exist, no need to drop")
            return

    # 删除表
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS preset_models"))
        conn.commit()

    print("Table preset_models has been successfully dropped")


if __name__ == "__main__":
    drop_preset_models_table()
