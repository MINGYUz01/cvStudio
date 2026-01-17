"""
简单的数据库迁移脚本：添加数据集 is_standard 字段
直接使用 sqlite3 操作数据库

运行方式：
    python backend/scripts/add_is_standard_field.py
"""

import sqlite3
import os
from pathlib import Path


# 数据库文件路径
DB_PATH = Path(__file__).parent.parent / "cvstudio.db"


def migrate():
    """执行数据库迁移"""
    if not DB_PATH.exists():
        print(f"数据库文件不存在: {DB_PATH}")
        return

    print(f"连接数据库: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # 检查字段是否已存在
        cursor.execute("PRAGMA table_info(datasets)")
        columns = [row[1] for row in cursor.fetchall()]

        if 'is_standard' in columns:
            print("字段 is_standard 已存在，跳过迁移")
            return

        print("添加 is_standard 字段到 datasets 表...")
        cursor.execute(
            "ALTER TABLE datasets ADD COLUMN is_standard BOOLEAN DEFAULT 0"
        )

        # 为现有数据集更新 is_standard 值（基于 meta 中的 format_confidence）
        print("更新现有数据集的 is_standard 值...")
        cursor.execute("""
            UPDATE datasets
            SET is_standard = CASE
                WHEN CAST(json_extract(meta, '$.format_confidence') AS REAL) >= 0.7 THEN 1
                ELSE 0
            END
        """)

        conn.commit()
        print("迁移完成！")

        # 显示更新后的数据
        cursor.execute("SELECT id, name, format, is_standard FROM datasets")
        rows = cursor.fetchall()
        print(f"\n当前数据集数量: {len(rows)}")
        for row in rows:
            print(f"  - ID: {row[0]}, Name: {row[1]}, Format: {row[2]}, is_standard: {row[3]}")

    except Exception as e:
        print(f"迁移失败: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    migrate()
