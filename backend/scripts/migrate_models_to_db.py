"""
模型管理数据迁移脚本

将文件系统中的模型架构和生成代码迁移到数据库。
同时创建新的数据库表。
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import engine, Base, SessionLocal
from app.models.model_architecture import ModelArchitecture
from app.models.generated_code import GeneratedCode
from app.models.user import User


# 存储路径
ARCHITECTURE_DIR = Path("data/architectures")
MODEL_DIR = Path("data/models")


def create_tables():
    """创建新的数据库表"""
    print("正在创建数据库表...")
    Base.metadata.create_all(bind=engine)
    print("数据库表创建完成！")


def migrate_architectures(db) -> int:
    """
    迁移模型架构文件到数据库

    读取 data/architectures/ 目录中的所有JSON文件，导入数据库。
    """
    print("\n=== 迁移模型架构 ===")
    count = 0

    if not ARCHITECTURE_DIR.exists():
        print(f"目录不存在: {ARCHITECTURE_DIR}")
        return 0

    for filepath in ARCHITECTURE_DIR.glob("*.json"):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 检查是否已存在同名记录
            existing = db.query(ModelArchitecture).filter(
                ModelArchitecture.file_name == filepath.name
            ).first()

            if existing:
                print(f"跳过已存在: {filepath.name}")
                continue

            # 创建数据库记录
            nodes = data.get("nodes", [])
            connections = data.get("connections", [])

            architecture = ModelArchitecture(
                name=data.get("name", filepath.stem),
                description=data.get("description", ""),
                version=data.get("version", "v1.0"),
                type=data.get("type", "Custom"),
                file_path=str(filepath),
                file_name=filepath.name,
                node_count=len(nodes),
                connection_count=len(connections),
                meta={
                    "nodes": nodes,
                    "connections": connections,
                    "thumbnail": data.get("thumbnail"),
                    "created": data.get("created"),
                    "updated": data.get("updated")
                }
            )

            db.add(architecture)
            count += 1
            print(f"已导入: {filepath.name}")

        except Exception as e:
            print(f"导入失败 {filepath.name}: {e}")

    db.commit()
    print(f"\n模型架构迁移完成！共导入 {count} 条记录")
    return count


def migrate_generated_codes(db) -> int:
    """
    迁移生成的代码文件到数据库

    读取 data/models/ 目录中的所有Python文件，导入数据库。
    """
    print("\n=== 迁移生成的代码 ===")
    count = 0

    if not MODEL_DIR.exists():
        print(f"目录不存在: {MODEL_DIR}")
        return 0

    for filepath in MODEL_DIR.glob("*.py"):
        try:
            # 读取代码内容
            with open(filepath, "r", encoding="utf-8") as f:
                code = f.read()

            # 从文件名提取模型名
            model_name = filepath.stem

            # 检查是否已存在
            existing = db.query(GeneratedCode).filter(
                GeneratedCode.file_name == filepath.name
            ).first()

            if existing:
                print(f"跳过已存在: {filepath.name}")
                continue

            # 创建数据库记录
            generated_code = GeneratedCode(
                name=model_name,
                file_path=str(filepath),
                file_name=filepath.name,
                code_size=len(code.encode("utf-8")),
                template_tag=None,
                meta={}
            )

            db.add(generated_code)
            count += 1
            print(f"已导入: {filepath.name}")

        except Exception as e:
            print(f"导入失败 {filepath.name}: {e}")

    db.commit()
    print(f"\n生成的代码迁移完成！共导入 {count} 条记录")
    return count


def verify_migration(db):
    """验证迁移结果"""
    print("\n=== 验证迁移结果 ===")

    arch_count = db.query(ModelArchitecture).filter(
        ModelArchitecture.is_active == "active"
    ).count()

    code_count = db.query(GeneratedCode).filter(
        GeneratedCode.is_active == "active"
    ).count()

    print(f"模型架构记录: {arch_count} 条")
    print(f"生成代码记录: {code_count} 条")


def main():
    """主函数"""
    print("=" * 50)
    print("模型管理数据迁移脚本")
    print("=" * 50)

    # 创建数据库会话
    db = SessionLocal()

    try:
        # 1. 创建表
        create_tables()

        # 2. 迁移架构
        migrate_architectures(db)

        # 3. 迁移代码
        migrate_generated_codes(db)

        # 4. 验证
        verify_migration(db)

        print("\n" + "=" * 50)
        print("迁移完成！")
        print("=" * 50)

    except Exception as e:
        print(f"\n迁移失败: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
