"""
修复数据库表结构脚本
"""

import sqlite3
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.security import get_password_hash


def fix_database():
    """修复数据库表结构"""
    
    # 数据库连接
    db_path = "cvstudio.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("正在修复数据库表结构...")
    
    # 检查表是否存在
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
    table_exists = cursor.fetchone()
    
    if table_exists:
        # 检查列是否存在
        cursor.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in cursor.fetchall()]
        
        print(f"当前表列: {columns}")
        
        # 添加缺失的列
        if 'is_active' not in columns:
            cursor.execute("ALTER TABLE users ADD COLUMN is_active BOOLEAN DEFAULT 1")
            print("✅ 添加了 is_active 列")
        
        if 'is_superuser' not in columns:
            cursor.execute("ALTER TABLE users ADD COLUMN is_superuser BOOLEAN DEFAULT 0")
            print("✅ 添加了 is_superuser 列")
        
        # 简化处理，先不添加updated_at列
        # SQLite不支持DEFAULT CURRENT_TIMESTAMP在ALTER TABLE中
            
    else:
        print("❌ 用户表不存在，请先运行 create_test_user.py")
        return
    
    conn.commit()
    conn.close()
    print("🎉 数据库表结构修复完成！")


if __name__ == "__main__":
    fix_database()