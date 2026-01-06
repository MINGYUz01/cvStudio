"""
Celery任务模块
包含所有异步任务定义
"""

from app.tasks import training_tasks

__all__ = ['training_tasks']
