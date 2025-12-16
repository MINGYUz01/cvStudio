"""
用户服务（扩展版本）
包含CRUD操作和业务逻辑
"""

from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from app.models.user import User
from app.schemas.user import UserCreate, UserUpdate, UserResponse
from app.services.base_service import CRUDBase
from app.core.security import get_password_hash, verify_password


class CRUDUser(CRUDBase[User, UserCreate, UserUpdate]):
    """
    用户CRUD服务类
    """

    def get_by_username(self, db: Session, *, username: str) -> Optional[User]:
        """
        根据用户名获取用户

        Args:
            db: 数据库会话
            username: 用户名

        Returns:
            用户对象或None
        """
        return db.query(User).filter(User.username == username).first()

    def get_by_email(self, db: Session, *, email: str) -> Optional[User]:
        """
        根据邮箱获取用户

        Args:
            db: 数据库会话
            email: 邮箱

        Returns:
            用户对象或None
        """
        return db.query(User).filter(User.email == email).first()

    def create(self, db: Session, *, obj_in: UserCreate) -> User:
        """
        创建新用户（包含密码加密）

        Args:
            db: 数据库会话
            obj_in: 用户创建数据

        Returns:
            创建的用户对象

        Raises:
            HTTPException: 用户名或邮箱已存在时
        """
        # 检查用户名是否已存在
        if self.get_by_username(db, username=obj_in.username):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="用户名已存在"
            )

        # 检查邮箱是否已存在
        if self.get_by_email(db, email=obj_in.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="邮箱已存在"
            )

        # 创建用户
        db_obj = User(
            username=obj_in.username,
            email=obj_in.email,
            password_hash=get_password_hash(obj_in.password),
            is_active=obj_in.is_active if hasattr(obj_in, 'is_active') else True,
            is_superuser=obj_in.is_superuser if hasattr(obj_in, 'is_superuser') else False
        )

        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update(
        self,
        db: Session,
        *,
        db_obj: User,
        obj_in: UserUpdate
    ) -> User:
        """
        更新用户信息

        Args:
            db: 数据库会话
            db_obj: 要更新的用户对象
            obj_in: 更新数据

        Returns:
            更新后的用户对象

        Raises:
            HTTPException: 用户名或邮箱已存在时
        """
        update_data = obj_in.dict(exclude_unset=True)

        # 如果更新用户名，检查是否已存在
        if "username" in update_data:
            existing_user = self.get_by_username(db, username=update_data["username"])
            if existing_user and existing_user.id != db_obj.id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="用户名已存在"
                )

        # 如果更新邮箱，检查是否已存在
        if "email" in update_data:
            existing_user = self.get_by_email(db, email=update_data["email"])
            if existing_user and existing_user.id != db_obj.id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="邮箱已存在"
                )

        # 如果更新密码，需要加密
        if "password" in update_data:
            update_data["password_hash"] = get_password_hash(update_data.pop("password"))

        return super().update(db, db_obj=db_obj, obj_in=update_data)

    def authenticate(self, db: Session, *, username: str, password: str) -> Optional[User]:
        """
        验证用户身份

        Args:
            db: 数据库会话
            username: 用户名
            password: 密码

        Returns:
            用户对象或None
        """
        user = self.get_by_username(db, username=username)
        if not user:
            return None
        if not verify_password(password, user.password_hash):
            return None
        return user

    def is_active(self, user: User) -> bool:
        """
        检查用户是否激活

        Args:
            user: 用户对象

        Returns:
            用户是否激活
        """
        return user.is_active

    def is_superuser(self, user: User) -> bool:
        """
        检查用户是否为超级用户

        Args:
            user: 用户对象

        Returns:
            用户是否为超级用户
        """
        return user.is_superuser

    def get_active_users(
        self,
        db: Session,
        *,
        skip: int = 0,
        limit: int = 100
    ) -> List[User]:
        """
        获取激活用户列表

        Args:
            db: 数据库会话
            skip: 跳过的记录数
            limit: 返回的记录数限制

        Returns:
            激活用户列表
        """
        return db.query(User).filter(User.is_active == True).offset(skip).limit(limit).all()

    def get_superusers(
        self,
        db: Session,
        *,
        skip: int = 0,
        limit: int = 100
    ) -> List[User]:
        """
        获取超级用户列表

        Args:
            db: 数据库会话
            skip: 跳过的记录数
            limit: 返回的记录数限制

        Returns:
            超级用户列表
        """
        return db.query(User).filter(User.is_superuser == True).offset(skip).limit(limit).all()

    def count_active_users(self, db: Session) -> int:
        """
        统计激活用户数量

        Args:
            db: 数据库会话

        Returns:
            激活用户数量
        """
        return db.query(User).filter(User.is_active == True).count()

    def count_superusers(self, db: Session) -> int:
        """
        统计超级用户数量

        Args:
            db: 数据库会话

        Returns:
            超级用户数量
        """
        return db.query(User).filter(User.is_superuser == True).count()

    def activate_user(self, db: Session, *, user_id: int) -> User:
        """
        激活用户

        Args:
            db: 数据库会话
            user_id: 用户ID

        Returns:
            更新后的用户对象
        """
        user = self.get(db, id=user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="用户不存在"
            )

        user.is_active = True
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    def deactivate_user(self, db: Session, *, user_id: int) -> User:
        """
        禁用用户

        Args:
            db: 数据库会话
            user_id: 用户ID

        Returns:
            更新后的用户对象
        """
        user = self.get(db, id=user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="用户不存在"
            )

        user.is_active = False
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    def change_password(
        self,
        db: Session,
        *,
        user_id: int,
        old_password: str,
        new_password: str
    ) -> User:
        """
        修改用户密码

        Args:
            db: 数据库会话
            user_id: 用户ID
            old_password: 旧密码
            new_password: 新密码

        Returns:
            更新后的用户对象

        Raises:
            HTTPException: 旧密码错误时
        """
        user = self.get(db, id=user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="用户不存在"
            )

        if not verify_password(old_password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="旧密码错误"
            )

        user.password_hash = get_password_hash(new_password)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user


# 创建用户服务实例
crud_user = CRUDUser(User)