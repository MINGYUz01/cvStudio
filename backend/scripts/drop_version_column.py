"""
删除 model_architectures 表的 version 列

运行方式:
    python backend/scripts/drop_version_column.py
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
from app.database import engine


def drop_version_column():
    """删除 model_architectures 表的 version 列"""
    print("Starting to drop version column from model_architectures table...")

    # SQLite 不支持直接 DROP COLUMN，需要重建表
    with engine.connect() as conn:
        # 1. 检查 version 列是否存在
        result = conn.execute(text("PRAGMA table_info(model_architectures)"))
        columns = [row[1] for row in result.fetchall()]
        if 'version' not in columns:
            print("Column 'version' does not exist, no need to drop")
            return

        # 2. 创建新表（不包含 version 列）
        conn.execute(text("""
            CREATE TABLE model_architectures_new (
                id INTEGER PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                description TEXT,
                type VARCHAR(50) DEFAULT 'Custom',
                file_path VARCHAR(500) NOT NULL,
                file_name VARCHAR(100) NOT NULL,
                node_count INTEGER DEFAULT 0,
                connection_count INTEGER DEFAULT 0,
                meta JSON,
                is_active VARCHAR(10) DEFAULT 'active',
                created_by INTEGER,
                created_at DATETIME,
                updated_at DATETIME,
                FOREIGN KEY(created_by) REFERENCES users(id)
            )
        """))

        # 3. 复制数据（跳过 version 列）
        conn.execute(text("""
            INSERT INTO model_architectures_new
            (id, name, description, type, file_path, file_name, node_count, connection_count, meta, is_active, created_by, created_at, updated_at)
            SELECT id, name, description, type, file_path, file_name, node_count, connection_count, meta, is_active, created_by, created_at, updated_at
            FROM model_architectures
        """))

        # 4. 删除旧表
        conn.execute(text("DROP TABLE model_architectures"))

        # 5. 重命名新表
        conn.execute(text("ALTER TABLE model_architectures_new RENAME TO model_architectures"))

        # 6. 重建索引
        conn.execute(text("CREATE INDEX ix_model_architectures_id ON model_architectures (id)"))
        conn.execute(text("CREATE INDEX ix_model_architectures_name ON model_architectures (name)"))

        conn.commit()

    print("Column 'version' has been successfully dropped from model_architectures table")


if __name__ == "__main__":
    drop_version_column()
