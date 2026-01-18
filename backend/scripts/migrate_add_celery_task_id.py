"""
数据库迁移脚本：添加 training_runs 表的 celery_task_id 字段

运行方式：
    python backend/scripts/migrate_add_celery_task_id.py
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text
from app.database import engine


def migrate():
    """执行数据库迁移"""
    print("开始数据库迁移：添加 celery_task_id 字段...")

    try:
        with engine.connect() as conn:
            # 检查字段是否已存在
            result = conn.execute(text("PRAGMA table_info(training_runs)"))
            columns = [row[1] for row in result.fetchall()]

            if 'celery_task_id' in columns:
                print("字段 celery_task_id 已存在，跳过迁移")
                return

            # 添加 celery_task_id 字段
            print("添加 celery_task_id 字段到 training_runs 表...")
            conn.execute(text(
                "ALTER TABLE training_runs ADD COLUMN celery_task_id VARCHAR(255)"
            ))

            conn.commit()
            print("迁移完成！")

    except Exception as e:
        print(f"迁移失败: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    migrate()
