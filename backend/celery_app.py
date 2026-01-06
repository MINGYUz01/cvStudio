"""
Celery应用配置
用于训练任务的异步处理和调度
"""

from celery import Celery
from app.core.config import settings

# 创建Celery应用实例
celery_app = Celery(
    "cvstudio_training",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.tasks.training_tasks"]
)

# Celery配置
celery_app.conf.update(
    # 任务序列化配置
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',

    # 时区配置
    timezone='Asia/Shanghai',
    enable_utc=True,

    # 任务跟踪配置
    task_track_started=True,
    task_time_limit=3600 * 24,  # 24小时硬限制
    task_soft_time_limit=3600 * 12,  # 12小时软限制

    # Worker配置
    worker_prefetch_multiplier=1,  # 每次只预取一个任务
    worker_max_tasks_per_child=50,  # Worker执行50个任务后重启

    # 任务结果配置
    result_expires=3600,  # 结果保存1小时
    result_extended=True,  # 扩展结果保存

    # 任务路由配置
    task_routes={
        'app.tasks.training_tasks.start_training': {'queue': 'training'},
        'app.tasks.training_tasks.control_training': {'queue': 'control'},
    },

    # 任务重试配置
    task_autoretry_for=(Exception,),  # 所有异常自动重试
    task_retry_max_delay=300,  # 最大重试延迟5分钟
    task_retry_delay=60,  # 重试延迟60秒
    task_retry_backoff=True,  # 启用指数退避
    task_retry_backoff_max=600,  # 最大退避时间10分钟
    task_max_retries=3,  # 最大重试次数

    # Worker并发配置
    worker_concurrency=2,  # 最多2个并发任务
)

# 可选：配置任务优先级
# celery_app.conf.task_queues = [
#     Queue('training', routing_key='training'),
#     Queue('control', routing_key='control'),
# ]


# 定义任务
@celery_app.task(bind=True)
def debug_task(self):
    """调试任务"""
    print(f'Request: {self.request!r}')
