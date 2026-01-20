"""
添加 experiment_dir 字段到 training_runs 表
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sqlalchemy import text
from app.database import engine, SessionLocal
from loguru import logger


def add_experiment_dir_field():
    """添加 experiment_dir 字段"""
    with engine.connect() as conn:
        try:
            # 检查字段是否已存在
            result = conn.execute(text("PRAGMA table_info(training_runs)"))
            columns = [row[1] for row in result.fetchall()]

            if 'experiment_dir' in columns:
                logger.info("experiment_dir 字段已存在，无需迁移")
                return

            # 添加字段
            conn.execute(text(
                "ALTER TABLE training_runs "
                "ADD COLUMN experiment_dir VARCHAR(500)"
            ))
            conn.commit()

            logger.info("成功添加 experiment_dir 字段到 training_runs 表")

        except Exception as e:
            logger.error(f"添加字段失败: {e}")
            conn.rollback()
            raise


if __name__ == "__main__":
    add_experiment_dir_field()
    print("数据库迁移完成!")
