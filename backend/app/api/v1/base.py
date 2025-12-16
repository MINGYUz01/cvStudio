"""
基础API路由模板
提供通用的CRUD操作接口
"""

from typing import Any, Dict, List, Type, TypeVar, Union, Generic
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from sqlalchemy.orm import Session
import json

from app.database import get_db
from app.services.base_service import CRUDBase
from app.utils.responses import success_response, error_response, paginated_response
from app.dependencies import CommonQueryParams

# 泛型类型变量
ModelType = TypeVar("ModelType")
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class BaseRouter(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """
    基础API路由类
    提供标准的CRUD操作接口
    """

    def __init__(
        self,
        crud_service: CRUDBase,
        prefix: str,
        tags: List[str] = None,
        create_schema: Type[CreateSchemaType] = None,
        update_schema: Type[UpdateSchemaType] = None,
        response_schema: Type[BaseModel] = None,
        search_fields: List[str] = None
    ):
        """
        初始化基础路由

        Args:
            crud_service: CRUD服务实例
            prefix: 路由前缀
            tags: API标签
            create_schema: 创建数据模式
            update_schema: 更新数据模式
            response_schema: 响应数据模式
            search_fields: 可搜索字段列表
        """
        self.crud_service = crud_service
        self.router = APIRouter(prefix=prefix, tags=tags or [])
        self.create_schema = create_schema
        self.update_schema = update_schema
        self.response_schema = response_schema
        self.search_fields = search_fields or []

        self._setup_routes()

    def _setup_routes(self):
        """设置路由"""

        @self.router.post("/", response_model=Dict[str, Any])
        def create(
            *,
            db: Session = Depends(get_db),
            obj_in: self.create_schema
        ):
            """
            创建新记录
            """
            try:
                db_obj = self.crud_service.create(db=db, obj_in=obj_in)
                if self.response_schema:
                    response_data = self.response_schema.from_orm(db_obj)
                else:
                    response_data = db_obj

                return success_response(
                    message="创建成功",
                    data=jsonable_encoder(response_data),
                    code=status.HTTP_201_CREATED
                )
            except Exception as e:
                return error_response(
                    message="创建失败",
                    error_detail=str(e),
                    code=status.HTTP_400_BAD_REQUEST
                )

        @self.router.get("/{id}", response_model=Dict[str, Any])
        def get_by_id(
            id: int,
            db: Session = Depends(get_db)
        ):
            """
            根据ID获取记录
            """
            db_obj = self.crud_service.get(db=db, id=id)
            if not db_obj:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="记录不存在"
                )

            if self.response_schema:
                response_data = self.response_schema.from_orm(db_obj)
            else:
                response_data = db_obj

            return success_response(
                message="获取成功",
                data=jsonable_encoder(response_data)
            )

        @self.router.get("/", response_model=Dict[str, Any])
        def get_multi(
            db: Session = Depends(get_db),
            commons: CommonQueryParams = Depends(),
            keyword: str = Query(None, description="搜索关键词"),
            filters: str = Query(None, description="过滤条件(JSON格式)")
        ):
            """
            获取记录列表（支持分页、搜索、过滤）
            """
            try:
                # 处理过滤条件
                filter_dict = {}
                if filters:
                    import json
                    try:
                        filter_dict = json.loads(filters)
                    except json.JSONDecodeError:
                        pass

                # 处理搜索
                if keyword and self.search_fields:
                    db_objs = self.crud_service.search(
                        db=db,
                        keyword=keyword,
                        search_fields=self.search_fields,
                        skip=(commons.page - 1) * commons.page_size,
                        limit=commons.page_size
                    )
                    total = self.crud_service.count(db, filter_dict)
                else:
                    # 普通查询
                    db_objs = self.crud_service.get_multi_with_filters(
                        db=db,
                        filters=filter_dict,
                        skip=(commons.page - 1) * commons.page_size,
                        limit=commons.page_size,
                        sort_by=commons.sort_by,
                        sort_order=commons.sort_order
                    )
                    total = self.crud_service.count(db, filter_dict)

                # 转换响应数据
                if self.response_schema:
                    response_data = [
                        self.response_schema.from_orm(obj)
                        for obj in db_objs
                    ]
                else:
                    response_data = db_objs

                return paginated_response(
                    message="获取成功",
                    data=jsonable_encoder(response_data),
                    page=commons.page,
                    page_size=commons.page_size,
                    total=total
                )

            except Exception as e:
                return error_response(
                    message="获取失败",
                    error_detail=str(e),
                    code=status.HTTP_400_BAD_REQUEST
                )

        @self.router.put("/{id}", response_model=Dict[str, Any])
        def update(
            *,
            db: Session = Depends(get_db),
            id: int,
            obj_in: Union[self.update_schema, Dict[str, Any]]
        ):
            """
            更新记录
            """
            db_obj = self.crud_service.get(db=db, id=id)
            if not db_obj:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="记录不存在"
                )

            try:
                updated_obj = self.crud_service.update(
                    db=db, db_obj=db_obj, obj_in=obj_in
                )
                if self.response_schema:
                    response_data = self.response_schema.from_orm(updated_obj)
                else:
                    response_data = updated_obj

                return success_response(
                    message="更新成功",
                    data=jsonable_encoder(response_data)
                )
            except Exception as e:
                return error_response(
                    message="更新失败",
                    error_detail=str(e),
                    code=status.HTTP_400_BAD_REQUEST
                )

        @self.router.delete("/{id}", response_model=Dict[str, Any])
        def delete(
            *,
            db: Session = Depends(get_db),
            id: int
        ):
            """
            删除记录
            """
            db_obj = self.crud_service.get(db=db, id=id)
            if not db_obj:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="记录不存在"
                )

            try:
                deleted_obj = self.crud_service.remove(db=db, id=id)
                return success_response(
                    message="删除成功",
                    data=jsonable_encoder(deleted_obj)
                )
            except Exception as e:
                return error_response(
                    message="删除失败",
                    error_detail=str(e),
                    code=status.HTTP_400_BAD_REQUEST
                )

        @self.router.get("/{id}/exists", response_model=Dict[str, Any])
        def check_exists(
            id: int,
            db: Session = Depends(get_db)
        ):
            """
            检查记录是否存在
            """
            exists = self.crud_service.exists(db=db, id=id)
            return success_response(
                message="检查完成",
                data={"exists": exists, "id": id}
            )

        # 如果指定了搜索字段，添加搜索接口
        if self.search_fields:

            @self.router.get("/search/{keyword}", response_model=Dict[str, Any])
            def search(
                keyword: str,
                db: Session = Depends(get_db),
                commons: CommonQueryParams = Depends()
            ):
                """
                搜索记录
                """
                try:
                    db_objs = self.crud_service.search(
                        db=db,
                        keyword=keyword,
                        search_fields=self.search_fields,
                        skip=(commons.page - 1) * commons.page_size,
                        limit=commons.page_size
                    )

                    if self.response_schema:
                        response_data = [
                            self.response_schema.from_orm(obj)
                            for obj in db_objs
                        ]
                    else:
                        response_data = db_objs

                    return paginated_response(
                        message="搜索成功",
                        data=jsonable_encoder(response_data),
                        page=commons.page,
                        page_size=commons.page_size,
                        total=len(response_data)
                    )

                except Exception as e:
                    return error_response(
                        message="搜索失败",
                        error_detail=str(e),
                        code=status.HTTP_400_BAD_REQUEST
                    )