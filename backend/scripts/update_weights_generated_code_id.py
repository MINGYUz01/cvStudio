"""
数据更新脚本：更新现有权重记录的 generated_code_id

通过训练任务的 model.code_path 找到对应的 GeneratedCode，
然后更新权重记录的 generated_code_id 字段。
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models.weight_library import WeightLibrary
from app.models.training import TrainingRun
from app.models.model import Model
from app.models.generated_code import GeneratedCode
from loguru import logger


def update_weights():
    """更新现有权重记录的 generated_code_id"""
    print("开始更新权重记录的 generated_code_id...")

    db = SessionLocal()

    try:
        # 获取所有权重记录
        weights = db.query(WeightLibrary).all()
        print(f"找到 {len(weights)} 条权重记录")

        updated_count = 0
        skipped_count = 0

        for weight in weights:
            # 如果已经有 generated_code_id，跳过
            if weight.generated_code_id is not None:
                skipped_count += 1
                continue

            # 如果权重来自训练任务
            if weight.source_training_id:
                # 获取训练任务
                training_run = db.query(TrainingRun).filter(
                    TrainingRun.id == weight.source_training_id
                ).first()

                if training_run and training_run.model_id:
                    # 获取模型
                    model = db.query(Model).filter(
                        Model.id == training_run.model_id
                    ).first()

                    if model and model.code_path:
                        # 找到对应的 GeneratedCode
                        generated_code = db.query(GeneratedCode).filter(
                            GeneratedCode.file_path == model.code_path,
                            GeneratedCode.is_active == "active"
                        ).first()

                        if generated_code:
                            # 更新权重记录
                            weight.generated_code_id = generated_code.id
                            updated_count += 1
                            print(f"✅ 更新权重: {weight.name} -> GeneratedCode.id={generated_code.id}")
                        else:
                            print(f"⚠️  未找到 GeneratedCode: weight={weight.name}, code_path={model.code_path}")
                    else:
                        print(f"⚠️  模型不存在: weight={weight.name}, model_id={training_run.model_id}")
                else:
                    print(f"⚠️  训练任务不存在: weight={weight.name}, source_training_id={weight.source_training_id}")
            else:
                print(f"⚠️  权重无训练来源: {weight.name} (可能是手动上传)")

        # 提交更改
        db.commit()

        print(f"\n✅ 更新完成: {updated_count} 条记录已更新, {skipped_count} 条记录已跳过")

    except Exception as e:
        db.rollback()
        print(f"❌ 更新失败: {e}")
        logger.error(f"更新权重 generated_code_id 失败: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    update_weights()
