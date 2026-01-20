"""
权重库版本管理字段迁移脚本

为 weight_library 表添加新的字段，并设置现有数据的默认值
运行方式: python backend/scripts/add_weight_version_fields.py
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sqlalchemy import create_engine, text
from app.core.config import Settings

settings = Settings()


def migrate_weights_table():
    """为 weight_library 表添加版本管理相关字段"""

    # 连接数据库
    database_url = str(settings.DATABASE_URL).replace('sqlite:///', '')
    engine = create_engine(f'sqlite:///{database_url}')

    with engine.connect() as conn:
        # 检查表是否存在
        result = conn.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='weight_library'"
        ))
        if not result.fetchone():
            print("[X] weight_library 表不存在，请先运行数据库初始化")
            return False

        # 检查哪些列已经存在
        result = conn.execute(text("PRAGMA table_info(weight_library)"))
        existing_columns = {row[1] for row in result.fetchall()}

        print(f"[INFO] 现有列: {existing_columns}")

        # 添加新列（如果不存在）
        migrations = []

        if 'source_type' not in existing_columns:
            migrations.append("ALTER TABLE weight_library ADD COLUMN source_type VARCHAR(20) DEFAULT 'uploaded'")

        if 'source_training_id' not in existing_columns:
            migrations.append("ALTER TABLE weight_library ADD COLUMN source_training_id INTEGER")

        if 'is_root' not in existing_columns:
            migrations.append("ALTER TABLE weight_library ADD COLUMN is_root BOOLEAN DEFAULT 1")

        if 'architecture_id' not in existing_columns:
            migrations.append("ALTER TABLE weight_library ADD COLUMN architecture_id INTEGER")

        # 执行迁移
        for sql in migrations:
            try:
                print(f"[EXEC] {sql}")
                conn.execute(text(sql))
                conn.commit()
                print("[OK] 成功")
            except Exception as e:
                print(f"[ERR] 失败: {e}")

        # 更新现有数据的 NULL 值
        print("\n[INFO] 更新现有数据...")
        updates = [
            "UPDATE weight_library SET source_type = 'uploaded' WHERE source_type IS NULL",
            "UPDATE weight_library SET is_root = 1 WHERE is_root IS NULL"
        ]

        for sql in updates:
            try:
                result = conn.execute(text(sql))
                conn.commit()
                print(f"[OK] {sql} - 更新了 {result.rowcount} 行")
            except Exception as e:
                print(f"[ERR] 失败: {e}")

        # 检查 training_runs 表是否有 pretrained_weight_id 列
        print("\n[INFO] 检查 training_runs 表...")
        result = conn.execute(text("PRAGMA table_info(training_runs)"))
        training_columns = {row[1] for row in result.fetchall()}

        if 'pretrained_weight_id' not in training_columns:
            print("[EXEC] 添加 pretrained_weight_id 列...")
            try:
                conn.execute(text("ALTER TABLE training_runs ADD COLUMN pretrained_weight_id INTEGER"))
                conn.commit()
                print("[OK] 成功添加 pretrained_weight_id 列")
            except Exception as e:
                print(f"[ERR] 失败: {e}")
        else:
            print("[OK] pretrained_weight_id 列已存在")

        print("\n[OK] 迁移完成！")
        return True


if __name__ == "__main__":
    print("=" * 50)
    print("权重库版本管理字段迁移")
    print("=" * 50)
    migrate_weights_table()
