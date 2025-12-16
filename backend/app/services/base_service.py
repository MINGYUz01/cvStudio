"""
通用CRUD服务基类
"""

from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import or_

from app.utils.responses import success_response, error_response, paginated_response

ModelType = TypeVar("ModelType")
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """
    通用CRUD操作基类
    """

    def __init__(self, model: Type[ModelType]):
        """
        初始化CRUD基类

        Args:
            model: SQLAlchemy模型类
        """
        self.model = model

    def get(self, db: Session, id: Any) -> Optional[ModelType]:
        """
        根据ID获取单条记录

        Args:
            db: 数据库会话
            id: 记录ID

        Returns:
            模型实例或None
        """
        return db.query(self.model).filter(self.model.id == id).first()

    def get_multi(
        self,
        db: Session,
        *,
        skip: int = 0,
        limit: int = 100,
        sort_by: str = "created_at",
        sort_order: str = "desc"
    ) -> List[ModelType]:
        """
        获取多条记录

        Args:
            db: 数据库会话
            skip: 跳过的记录数
            limit: 返回的记录数限制
            sort_by: 排序字段
            sort_order: 排序方向

        Returns:
            模型实例列表
        """
        query = db.query(self.model)

        # 排序
        if hasattr(self.model, sort_by):
            order_column = getattr(self.model, sort_by)
            if sort_order.lower() == "desc":
                query = query.order_by(order_column.desc())
            else:
                query = query.order_by(order_column.asc())

        return query.offset(skip).limit(limit).all()

    def get_multi_with_filters(
        self,
        db: Session,
        *,
        filters: Dict[str, Any] = None,
        skip: int = 0,
        limit: int = 100,
        sort_by: str = "created_at",
        sort_order: str = "desc"
    ) -> List[ModelType]:
        """
        根据过滤条件获取多条记录

        Args:
            db: 数据库会话
            filters: 过滤条件字典
            skip: 跳过的记录数
            limit: 返回的记录数限制
            sort_by: 排序字段
            sort_order: 排序方向

        Returns:
            模型实例列表
        """
        query = db.query(self.model)

        # 应用过滤条件
        if filters:
            for key, value in filters.items():
                if hasattr(self.model, key):
                    if isinstance(value, str) and "%" in value:
                        # 模糊搜索
                        query = query.filter(getattr(self.model, key).like(value))
                    else:
                        # 精确匹配
                        query = query.filter(getattr(self.model, key) == value)

        # 排序
        if hasattr(self.model, sort_by):
            order_column = getattr(self.model, sort_by)
            if sort_order.lower() == "desc":
                query = query.order_by(order_column.desc())
            else:
                query = query.order_by(order_column.asc())

        return query.offset(skip).limit(limit).all()

    def count(self, db: Session, filters: Dict[str, Any] = None) -> int:
        """
        统计记录数量

        Args:
            db: 数据库会话
            filters: 过滤条件字典

        Returns:
            记录数量
        """
        query = db.query(self.model)

        # 应用过滤条件
        if filters:
            for key, value in filters.items():
                if hasattr(self.model, key):
                    if isinstance(value, str) and "%" in value:
                        query = query.filter(getattr(self.model, key).like(value))
                    else:
                        query = query.filter(getattr(self.model, key) == value)

        return query.count()

    def create(self, db: Session, *, obj_in: CreateSchemaType) -> ModelType:
        """
        创建新记录

        Args:
            db: 数据库会话
            obj_in: 创建数据模式

        Returns:
            创建的模型实例
        """
        obj_in_data = jsonable_encoder(obj_in)
        db_obj = self.model(**obj_in_data)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update(
        self,
        db: Session,
        *,
        db_obj: ModelType,
        obj_in: Union[UpdateSchemaType, Dict[str, Any]]
    ) -> ModelType:
        """
        更新记录

        Args:
            db: 数据库会话
            db_obj: 要更新的模型实例
            obj_in: 更新数据

        Returns:
            更新后的模型实例
        """
        obj_data = jsonable_encoder(db_obj)

        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.dict(exclude_unset=True)

        for field in obj_data:
            if field in update_data:
                setattr(db_obj, field, update_data[field])

        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def remove(self, db: Session, *, id: int) -> ModelType:
        """
        删除记录

        Args:
            db: 数据库会话
            id: 记录ID

        Returns:
            删除的模型实例
        """
        obj = db.query(self.model).get(id)
        if obj:
            db.delete(obj)
            db.commit()
        return obj

    def exists(self, db: Session, *, id: int) -> bool:
        """
        检查记录是否存在

        Args:
            db: 数据库会话
            id: 记录ID

        Returns:
            记录是否存在
        """
        return db.query(self.model).filter(self.model.id == id).first() is not None

    def search(
        self,
        db: Session,
        *,
        keyword: str,
        search_fields: List[str],
        skip: int = 0,
        limit: int = 100
    ) -> List[ModelType]:
        """
        搜索记录

        Args:
            db: 数据库会话
            keyword: 搜索关键词
            search_fields: 搜索字段列表
            skip: 跳过的记录数
            limit: 返回的记录数限制

        Returns:
            模型实例列表
        """
        query = db.query(self.model)

        # 构建搜索条件
        search_conditions = []
        for field in search_fields:
            if hasattr(self.model, field):
                attr = getattr(self.model, field)
                search_conditions.append(attr.like(f"%{keyword}%"))

        if search_conditions:
            query = query.filter(or_(*search_conditions))

        return query.offset(skip).limit(limit).all()