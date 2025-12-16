"""
用户管理API路由
使用通用CRUD模板实现
"""

from typing import Any, Dict, List
import json
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User
from app.schemas.user import UserCreate, UserUpdate, UserResponse
from app.services.user_service import crud_user
from app.api.v1.base import BaseRouter
from app.dependencies import get_current_active_user, get_current_superuser, CommonQueryParams
from app.utils.responses import success_response, error_response, paginated_response

# 创建路由实例
router = APIRouter(prefix="/users", tags=["用户管理"])

# 先添加用户特定的API接口（避免路由冲突）
@router.get("/active", response_model=Dict[str, Any])
def get_active_users(
    commons: CommonQueryParams = Depends(),
    db: Session = Depends(get_db)
):
    """
    获取激活用户列表
    """
    try:
        users = crud_user.get_active_users(
            db=db,
            skip=(commons.page - 1) * commons.page_size,
            limit=commons.page_size
        )
        total = crud_user.count_active_users(db=db)

        response_data = [
            UserResponse.from_orm(user).dict()
            for user in users
        ]

        total_pages = (total + commons.page_size - 1) // commons.page_size if total > 0 else 0

        return {
            "success": True,
            "message": "获取激活用户成功",
            "data": response_data,
            "pagination": {
                "page": commons.page,
                "page_size": commons.page_size,
                "total": total,
                "total_pages": total_pages,
                "has_next": commons.page < total_pages,
                "has_prev": commons.page > 1
            },
            "code": 200
        }
    except Exception as e:
        return {
            "success": False,
            "message": "获取激活用户失败",
            "error_detail": str(e),
            "code": 400
        }


@router.get("/superusers", response_model=Dict[str, Any])
def get_superusers(
    commons: CommonQueryParams = Depends(),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_superuser)
):
    """
    获取超级用户列表（仅超级用户可访问）
    """
    try:
        users = crud_user.get_superusers(
            db=db,
            skip=(commons.page - 1) * commons.page_size,
            limit=commons.page_size
        )
        total = crud_user.count_superusers(db=db)

        response_data = [
            UserResponse.from_orm(user)
            for user in users
        ]

        return paginated_response(
            message="获取超级用户成功",
            data=response_data,
            page=commons.page,
            page_size=commons.page_size,
            total=total
        )
    except Exception as e:
        return error_response(
            message="获取超级用户失败",
            error_detail=str(e)
        )


@router.post("/{user_id}/activate", response_model=Dict[str, Any])
def activate_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_superuser)
):
    """
    激活用户（仅超级用户可访问）
    """
    try:
        user = crud_user.activate_user(db=db, user_id=user_id)
        response_data = UserResponse.from_orm(user)

        return success_response(
            message="用户激活成功",
            data=response_data
        )
    except HTTPException:
        raise
    except Exception as e:
        return error_response(
            message="用户激活失败",
            error_detail=str(e)
        )


@router.post("/{user_id}/deactivate", response_model=Dict[str, Any])
def deactivate_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_superuser)
):
    """
    禁用用户（仅超级用户可访问）
    """
    try:
        # 防止禁用自己
        if user_id == current_user.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="不能禁用自己的账户"
            )

        user = crud_user.deactivate_user(db=db, user_id=user_id)
        response_data = UserResponse.from_orm(user)

        return success_response(
            message="用户禁用成功",
            data=response_data
        )
    except HTTPException:
        raise
    except Exception as e:
        return error_response(
            message="用户禁用失败",
            error_detail=str(e)
        )


@router.post("/{user_id}/change-password", response_model=Dict[str, Any])
def change_user_password(
    user_id: int,
    old_password: str = Query(..., description="旧密码"),
    new_password: str = Query(..., description="新密码"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    修改用户密码
    """
    # 只能修改自己的密码，除非是超级用户
    if user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只能修改自己的密码"
        )

    try:
        user = crud_user.change_password(
            db=db,
            user_id=user_id,
            old_password=old_password,
            new_password=new_password
        )
        response_data = UserResponse.from_orm(user)

        return success_response(
            message="密码修改成功",
            data=response_data
        )
    except HTTPException:
        raise
    except Exception as e:
        return error_response(
            message="密码修改失败",
            error_detail=str(e)
        )


@router.get("/me/stats", response_model=Dict[str, Any])
def get_current_user_stats(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    获取当前用户统计信息
    """
    try:
        stats = {
            "user_id": current_user.id,
            "username": current_user.username,
            "is_active": current_user.is_active,
            "is_superuser": current_user.is_superuser,
            "created_at": str(current_user.created_at),
            "total_users": crud_user.count(db),
            "active_users": crud_user.count_active_users(db),
            "superusers": crud_user.count_superusers(db)
        }

        return success_response(
            message="获取用户统计信息成功",
            data=stats
        )
    except Exception as e:
        return error_response(
            message="获取用户统计信息失败",
            error_detail=str(e)
        )

# 手动添加CRUD路由，避免路由冲突
# 注意：路由顺序很重要，具体路径必须在参数化路径之前

@router.post("/", response_model=Dict[str, Any])
def create_user(
    *,
    db: Session = Depends(get_db),
    obj_in: UserCreate
):
    """创建新用户"""
    try:
        db_obj = crud_user.create(db=db, obj_in=obj_in)
        response_data = UserResponse.from_orm(db_obj)
        return success_response(
            message="创建成功",
            data=response_data.dict(),
            code=status.HTTP_201_CREATED
        )
    except Exception as e:
        return error_response(
            message="创建失败",
            error_detail=str(e),
            code=status.HTTP_400_BAD_REQUEST
        )

@router.get("/", response_model=Dict[str, Any])
def get_users(
    db: Session = Depends(get_db),
    commons: CommonQueryParams = Depends(),
    keyword: str = Query(None, description="搜索关键词"),
    filters: str = Query(None, description="过滤条件(JSON格式)")
):
    """获取用户列表（支持分页、搜索、过滤）"""
    try:
        # 处理过滤条件
        filter_dict = {}
        if filters:
            try:
                filter_dict = json.loads(filters)
            except json.JSONDecodeError:
                pass

        # 处理搜索
        if keyword and ["username", "email"]:
            db_objs = crud_user.search(
                db=db,
                keyword=keyword,
                search_fields=["username", "email"],
                skip=(commons.page - 1) * commons.page_size,
                limit=commons.page_size
            )
            total = crud_user.count(db, filter_dict)
        else:
            # 普通查询
            db_objs = crud_user.get_multi_with_filters(
                db=db,
                filters=filter_dict,
                skip=(commons.page - 1) * commons.page_size,
                limit=commons.page_size,
                sort_by=commons.sort_by,
                sort_order=commons.sort_order
            )
            total = crud_user.count(db, filter_dict)

        # 转换响应数据
        response_data = [
            UserResponse.from_orm(obj).dict()
            for obj in db_objs
        ]

        return paginated_response(
            message="获取成功",
            data=response_data,
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

@router.put("/{user_id}", response_model=Dict[str, Any])
def update_user(
    *,
    db: Session = Depends(get_db),
    user_id: int,
    obj_in: UserUpdate
):
    """更新用户"""
    db_obj = crud_user.get(db=db, id=user_id)
    if not db_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="记录不存在"
        )

    try:
        updated_obj = crud_user.update(db=db, db_obj=db_obj, obj_in=obj_in)
        response_data = UserResponse.from_orm(updated_obj)
        return success_response(
            message="更新成功",
            data=response_data.dict()
        )
    except Exception as e:
        return error_response(
            message="更新失败",
            error_detail=str(e),
            code=status.HTTP_400_BAD_REQUEST
        )

@router.delete("/{user_id}", response_model=Dict[str, Any])
def delete_user(
    *,
    db: Session = Depends(get_db),
    user_id: int
):
    """删除用户"""
    db_obj = crud_user.get(db=db, id=user_id)
    if not db_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="记录不存在"
        )

    try:
        deleted_obj = crud_user.remove(db=db, id=user_id)
        return success_response(
            message="删除成功",
            data=UserResponse.from_orm(deleted_obj).dict()
        )
    except Exception as e:
        return error_response(
            message="删除失败",
            error_detail=str(e),
            code=status.HTTP_400_BAD_REQUEST
        )

@router.get("/{user_id}/exists", response_model=Dict[str, Any])
def check_user_exists(
    user_id: int,
    db: Session = Depends(get_db)
):
    """检查用户是否存在"""
    exists = crud_user.exists(db=db, id=user_id)
    return success_response(
        message="检查完成",
        data={"exists": exists, "id": user_id}
    )

@router.get("/search/{keyword}", response_model=Dict[str, Any])
def search_users(
    keyword: str,
    db: Session = Depends(get_db),
    commons: CommonQueryParams = Depends()
):
    """搜索用户"""
    try:
        db_objs = crud_user.search(
            db=db,
            keyword=keyword,
            search_fields=["username", "email"],
            skip=(commons.page - 1) * commons.page_size,
            limit=commons.page_size
        )

        response_data = [
            UserResponse.from_orm(obj).dict()
            for obj in db_objs
        ]

        return paginated_response(
            message="搜索成功",
            data=response_data,
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

# 参数化路由必须放在最后
@router.get("/{user_id}", response_model=Dict[str, Any])
def get_user_by_id(
    user_id: int,
    db: Session = Depends(get_db)
):
    """根据ID获取用户"""
    db_obj = crud_user.get(db=db, id=user_id)
    if not db_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="记录不存在"
        )

    response_data = UserResponse.from_orm(db_obj)
    return success_response(
        message="获取成功",
        data=response_data.dict()
    )