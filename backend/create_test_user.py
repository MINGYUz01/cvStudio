"""
创建测试用户的脚本
"""

import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.security import get_password_hash
import sqlite3
import json


def create_test_users():
    """创建测试用户"""
    
    # 数据库连接
    db_path = "cvstudio.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 创建用户表（如果不存在）
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # 测试用户数据
    test_users = [
        {
            "username": "admin",
            "password": "admin123",
            "email": "admin@cvstudio.com"
        },
        {
            "username": "demo",
            "password": "demo123",
            "email": "demo@cvstudio.com"
        },
        {
            "username": "test",
            "password": "test123",
            "email": "test@cvstudio.com"
        }
    ]
    
    # 插入测试用户
    for user in test_users:
        try:
            password_hash = get_password_hash(user["password"])
            cursor.execute("""
                INSERT INTO users (username, password_hash, email)
                VALUES (?, ?, ?)
            """, (user["username"], password_hash, user["email"]))
            print(f"✅ 用户 {user['username']} 创建成功")
        except sqlite3.IntegrityError:
            print(f"⚠️  用户 {user['username']} 已存在")
    
    conn.commit()
    conn.close()
    
    print("\n🎉 测试用户创建完成！")
    print("\n📋 可用的测试账号：")
    for user in test_users:
        print(f"👤 用户名: {user['username']}")
        print(f"🔑 密码: {user['password']}")
        print(f"📧 邮箱: {user['email']}")
        print("-" * 30)


if __name__ == "__main__":
    create_test_users()