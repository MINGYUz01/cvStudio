"""
数据库初始化脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database import create_tables, drop_tables, SessionLocal
from app.core.security import get_password_hash

# 确保所有模型都被导入
from app.models import User, Dataset, Model, TrainingRun, Checkpoint, InferenceJob

def init_database():
    """初始化数据库"""
    print("🔧 正在初始化数据库...")

    # 创建所有表
    try:
        create_tables()
        print("✅ 数据库表创建成功")
    except Exception as e:
        print(f"❌ 创建数据库表失败: {e}")
        return False

    # 创建默认管理员用户
    try:
        db = SessionLocal()

        # 检查是否已存在管理员用户
        admin_user = db.query(User).filter(User.username == "admin").first()
        if not admin_user:
            admin_user = User(
                username="admin",
                email="admin@cvstudio.com",
                password_hash=get_password_hash("admin123"),
                is_superuser=True,
                is_active=True
            )
            db.add(admin_user)
            db.commit()
            print("✅ 默认管理员用户创建成功 (用户名: admin, 密码: admin123)")
        else:
            print("ℹ️ 管理员用户已存在")

        # 创建演示用户
        demo_user = db.query(User).filter(User.username == "demo").first()
        if not demo_user:
            demo_user = User(
                username="demo",
                email="demo@cvstudio.com",
                password_hash=get_password_hash("demo123"),
                is_superuser=False,
                is_active=True
            )
            db.add(demo_user)
            db.commit()
            print("✅ 演示用户创建成功 (用户名: demo, 密码: demo123)")
        else:
            print("ℹ️ 演示用户已存在")

        db.close()

    except Exception as e:
        print(f"❌ 创建默认用户失败: {e}")
        return False

    print("🎉 数据库初始化完成")
    return True

def reset_database():
    """重置数据库（删除所有表并重新创建）"""
    print("⚠️ 警告：这将删除所有数据！")
    confirm = input("确定要继续吗？(y/N): ")

    if confirm.lower() != 'y':
        print("❌ 操作已取消")
        return False

    try:
        drop_tables()
        print("✅ 数据库表删除成功")

        create_tables()
        print("✅ 数据库表创建成功")

        # 重新创建默认用户
        init_database()

        print("🎉 数据库重置完成")
        return True

    except Exception as e:
        print(f"❌ 数据库重置失败: {e}")
        return False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="数据库管理工具")
    parser.add_argument("--init", action="store_true", help="初始化数据库")
    parser.add_argument("--reset", action="store_true", help="重置数据库")

    args = parser.parse_args()

    if args.reset:
        reset_database()
    elif args.init:
        init_database()
    else:
        print("请指定操作: --init 或 --reset")
        print("示例: python init_db.py --init")