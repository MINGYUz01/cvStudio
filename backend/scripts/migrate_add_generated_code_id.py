"""
数据库迁移脚本：添加 generated_code_id 字段到 weight_library 表

运行方式：
    cd backend
    python -m scripts.migrate_add_generated_code_id
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from app.database import engine, SessionLocal


def migrate():
    """执行迁移"""
    print("开始迁移：添加 generated_code_id 字段到 weight_library 表...")

    # 获取数据库连接
    db = SessionLocal()

    try:
        # 检查字段是否已存在
        result = db.execute(text("PRAGMA table_info(weight_library)"))
        columns = [row[1] for row in result.fetchall()]

        if 'generated_code_id' in columns:
            print("✅ generated_code_id 字段已存在，跳过迁移")
            return

        # 添加新字段
        print("添加 generated_code_id 字段...")
        db.execute(text(
            "ALTER TABLE weight_library ADD COLUMN generated_code_id INTEGER REFERENCES generated_codes(id)"
        ))
        db.commit()

        print("✅ 迁移完成：成功添加 generated_code_id 字段")

        # 验证字段是否添加成功
        result = db.execute(text("PRAGMA table_info(weight_library)"))
        columns = [row[1] for row in result.fetchall()]
        if 'generated_code_id' in columns:
            print("✅ 验证通过：generated_code_id 字段已成功添加")
        else:
            print("❌ 验证失败：generated_code_id 字段未找到")

    except Exception as e:
        db.rollback()
        print(f"❌ 迁移失败: {e}")
        raise
    finally:
        db.close()


def rollback():
    """回滚迁移"""
    print("开始回滚：删除 generated_code_id 字段...")

    db = SessionLocal()

    try:
        # SQLite 不支持 DROP COLUMN，需要重建表
        # 这是一个简化的回滚，只清空数据
        print("⚠️  SQLite 不支持 DROP COLUMN，如需回滚请手动重建数据库")

    except Exception as e:
        print(f"❌ 回滚失败: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="数据库迁移")
    parser.add_argument("--rollback", action="store_true", help="回滚迁移")
    args = parser.parse_args()

    if args.rollback:
        rollback()
    else:
        migrate()
