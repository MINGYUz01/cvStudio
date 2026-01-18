"""
预设模型数据初始化脚本

读取 preset_models.json 文件并将预设模型导入到数据库中。
"""

import json
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy.orm import Session
from app.database import engine, SessionLocal
from app.models.model import PresetModel


def load_preset_models():
    """加载预设模型数据到数据库"""

    # 读取预设模型JSON文件
    preset_json_path = project_root / "data" / "preset_models.json"

    if not preset_json_path.exists():
        print(f"错误：预设模型文件不存在: {preset_json_path}")
        return

    with open(preset_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    presets = data.get('presets', [])
    print(f"找到 {len(presets)} 个预设模型")

    # 创建数据库会话
    db: Session = SessionLocal()

    try:
        # 清空现有预设模型（可选）
        # db.query(PresetModel).delete()

        imported_count = 0
        updated_count = 0

        for preset_data in presets:
            # 检查是否已存在
            existing = db.query(PresetModel).filter(PresetModel.name == preset_data['name']).first()

            if existing:
                # 更新现有记录
                existing.description = preset_data['description']
                existing.category = preset_data['category']
                existing.difficulty = preset_data['difficulty']
                existing.tags = preset_data['tags']
                existing.architecture_data = preset_data['architecture_data']
                existing.extra_metadata = preset_data.get('extra_metadata')
                existing.is_active = True
                updated_count += 1
                print(f"更新: {preset_data['name']}")
            else:
                # 创建新记录
                preset = PresetModel(
                    name=preset_data['name'],
                    description=preset_data['description'],
                    category=preset_data['category'],
                    difficulty=preset_data['difficulty'],
                    tags=preset_data['tags'],
                    architecture_data=preset_data['architecture_data'],
                    extra_metadata=preset_data.get('extra_metadata'),
                    is_active=True
                )
                db.add(preset)
                imported_count += 1
                print(f"导入: {preset_data['name']}")

        db.commit()
        print(f"\n完成！导入 {imported_count} 个，更新 {updated_count} 个预设模型")

    except Exception as e:
        db.rollback()
        print(f"错误: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    print("开始导入预设模型数据...")
    load_preset_models()
