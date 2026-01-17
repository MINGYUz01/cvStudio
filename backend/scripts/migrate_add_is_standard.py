"""
数据库迁移脚本：添加数据集 is_standard 字段

运行方式：
    python backend/scripts/migrate_add_is_standard.py
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text
from app.database import engine, Base
from app.models.dataset import Dataset


def migrate():
    """执行数据库迁移"""
    print("开始数据库迁移：添加 is_standard 字段...")

    try:
        with engine.connect() as conn:
            # 检查字段是否已存在
            result = conn.execute(text("PRAGMA table_info(datasets)"))
            columns = [row[1] for row in result.fetchall()]

            if 'is_standard' in columns:
                print("字段 is_standard 已存在，跳过迁移")
                return

            # 添加 is_standard 字段
            print("添加 is_standard 字段到 datasets 表...")
            conn.execute(text(
                "ALTER TABLE datasets ADD COLUMN is_standard BOOLEAN DEFAULT 0"
            ))

            # 为现有数据集更新 is_standard 值（基于 format_confidence）
            print("更新现有数据集的 is_standard 值...")
            conn.execute(text("""
                UPDATE datasets
                SET is_standard = CASE
                    WHEN CAST(json_extract(meta, '$.format_confidence') AS REAL) >= 0.7 THEN 1
                    ELSE 0
                END
            """))

            conn.commit()
            print("迁移完成！")

    except Exception as e:
        print(f"迁移失败: {e}")
        raise


def rollback():
    """回滚迁移"""
    print("开始回滚迁移：移除 is_standard 字段...")

    try:
        with engine.connect() as conn:
            # SQLite 不支持 DROP COLUMN，需要重建表
            print("警告: SQLite 不支持直接删除列")
            print("如需回滚，请手动重建数据库表")
            return

            # 以下代码适用于支持 DROP COLUMN 的数据库（如 PostgreSQL、MySQL）
            # conn.execute(text("ALTER TABLE datasets DROP COLUMN is_standard"))
            # conn.commit()
            # print("回滚完成！")

    except Exception as e:
        print(f"回滚失败: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="数据库迁移脚本")
    parser.add_argument("--rollback", action="store_true", help="回滚迁移")

    args = parser.parse_args()

    if args.rollback:
        rollback()
    else:
        migrate()
