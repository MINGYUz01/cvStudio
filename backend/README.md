# CV Studio 后端

> 计算机视觉任务管理平台后端服务 — 基于 FastAPI 构建

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange)](https://pytorch.org)

---

## 项目简介

CV Studio 后端是整个平台的核心服务，提供数据集管理、模型构建、训练调度、推理执行、权重库管理等完整的RESTful API和WebSocket实时通信支持。

### 核心功能

| 模块 | 功能 | 状态 |
|------|------|------|
| **认证系统** | JWT认证、Token刷新、用户配置管理 | ✅ |
| **数据集服务** | 格式识别、上传注册、预览增强、统计分析 | ✅ |
| **模型服务** | 图验证、形状推断、代码生成 | ✅ |
| **训练服务** | 任务调度、日志收集、Checkpoint管理 | ✅ |
| **推理服务** | 单图/批量推理、分类与检测支持 | ✅ |
| **权重库** | 权重文件版本管理、关联训练任务 | ✅ |
| **WebSocket** | 系统状态流、训练日志流 | ✅ |

---

## 技术栈

| 分类 | 技术 |
|------|------|
| **Web框架** | FastAPI + Uvicorn |
| **数据库** | SQLite + SQLAlchemy 2.0 |
| **认证** | JWT (python-jose) + Passlib |
| **任务队列** | Celery + Redis |
| **图像处理** | OpenCV + Pillow |
| **数据增强** | Albumentations |
| **深度学习** | PyTorch 2.1 + torchvision |
| **模板引擎** | Jinja2 |
| **日志** | Loguru |
| **测试** | pytest |

---

## 快速开始

### 环境要求

- **Python**：3.10 或 3.11
- **Redis**：5.0+（用于Celery消息队列）
- **操作系统**：Windows 10/11、macOS、Linux

### 安装步骤

```bash
# 1. 进入后端目录
cd backend

# 2. 创建conda环境
conda create -n cvstudio python=3.10
conda activate cvstudio

# 3. 安装依赖
pip install -r requirements.txt

# 4. 复制环境变量文件
cp .env.example .env

# 5. 启动Redis（Docker方式，在另一个终端）
docker run -d -p 6379:6379 redis:latest

# 6. 启动Celery Worker（在另一个终端）
cd backend
python -m celery -A celery_app worker --loglevel=info --pool=solo

# 7. 启动后端服务
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 访问服务

| 服务 | 地址 |
|------|------|
| **API服务** | http://localhost:8000 |
| **Swagger文档** | http://localhost:8000/api/v1/docs |
| **ReDoc文档** | http://localhost:8000/api/v1/redoc |
| **健康检查** | http://localhost:8000/health |

---

## 项目结构

```
backend/
├── app/                              # 主应用目录
│   ├── __init__.py
│   ├── main.py                       # FastAPI应用入口
│   ├── config.py                     # 配置管理（使用 core/config.py）
│   ├── database.py                   # 数据库连接
│   ├── dependencies.py               # 依赖注入
│   │
│   ├── api/                          # API路由
│   │   ├── websocket.py              # WebSocket连接管理器
│   │   └── v1/                       # API版本1
│   │       ├── auth.py               # 认证相关API
│   │       ├── datasets.py           # 数据集管理API
│   │       ├── models.py             # 模型构建API
│   │       ├── training.py           # 训练管理API
│   │       ├── inference.py          # 推理API
│   │       ├── weights.py            # 权重库API
│   │       └── websocket.py          # WebSocket端点
│   │
│   ├── core/                         # 核心功能
│   │   ├── config.py                 # 核心配置（Pydantic Settings）
│   │   └── security.py               # 安全相关（JWT、密码哈希）
│   │
│   ├── models/                       # 数据模型（SQLAlchemy）
│   │   ├── user.py                   # 用户模型
│   │   ├── dataset.py                # 数据集模型
│   │   ├── model_architecture.py     # 模型架构模型
│   │   ├── generated_code.py         # 生成的代码模型
│   │   ├── training.py               # 训练任务模型
│   │   ├── inference.py              # 推理任务模型
│   │   └── weight_library.py         # 权重库模型
│   │
│   ├── schemas/                      # Pydantic请求/响应模式
│   │   ├── user.py                   # 用户模式
│   │   ├── dataset.py                # 数据集模式
│   │   ├── model.py                  # 模型模式
│   │   ├── training.py               # 训练模式
│   │   ├── inference.py              # 推理模式
│   │   └── augmentation.py           # 增强模式
│   │
│   ├── services/                     # 业务逻辑服务
│   │   ├── auth_service.py           # 认证服务
│   │   ├── dataset_service.py        # 数据集服务
│   │   ├── model_service.py          # 模型服务
│   │   ├── training_service.py       # 训练服务
│   │   ├── inference_service.py      # 推理服务
│   │   ├── code_generator_service.py # 代码生成服务
│   │   ├── weight_library_service.py # 权重库服务
│   │   └── augmentation_service.py   # 数据增强服务
│   │
│   ├── utils/                        # 工具函数
│   │   ├── format_recognizers/       # 格式识别器
│   │   │   ├── yolo.py               # YOLO格式识别
│   │   │   ├── coco.py               # COCO格式识别
│   │   │   ├── voc.py                # VOC格式识别
│   │   │   └── classification.py     # 分类格式识别
│   │   ├── code_generator/           # 代码生成器
│   │   ├── models/                   # 模型相关
│   │   │   └── factory.py            # 模型工厂
│   │   ├── data_loaders/            # 数据加载器
│   │   ├── losses/                   # 损失函数
│   │   ├── metrics/                  # 评估指标
│   │   ├── graph_traversal.py        # 图遍历算法
│   │   ├── shape_inference.py        # 形状推断引擎
│   │   ├── image_processor.py        # 图像处理
│   │   ├── augmentation.py           # 数据增强
│   │   ├── model_loader.py           # 模型加载器
│   │   ├── inference_executor.py     # 推理执行器
│   │   ├── experiment_manager.py     # 实验管理
│   │   ├── checkpoint_manager.py    # 检查点管理
│   │   ├── config_parser.py          # 配置解析器
│   │   └── trainer.py                # 训练执行器
│   │
│   ├── templates/                    # Jinja2模板
│   │   └── model_template.py.j2     # 模型代码模板
│   │
│   └── tasks/                        # Celery任务
│       └── training_tasks.py         # 训练相关任务
│
├── data/                             # 数据目录（统一）
│   ├── datasets/                     # 数据集存储
│   ├── models/                       # 生成的模型代码
│   ├── checkpoints/                  # 训练检查点
│   ├── experiments/                  # 实验数据
│   ├── weights/                      # 权重库
│   ├── uploads/                      # 上传文件
│   ├── architectures/                # 模型架构JSON
│   ├── thumbnails/                   # 缩略图
│   └── temp/                         # 临时文件
│
├── tests/                            # 测试目录
│   ├── conftest.py                   # pytest配置
│   ├── test_api/                     # API测试
│   ├── test_services/                # 服务测试
│   ├── test_utils/                   # 工具测试
│   └── temp/                         # 临时测试脚本
│
├── alembic/                          # 数据库迁移
│   └── versions/                     # 迁移版本
│
├── scripts/                          # 工具脚本
│   ├── migrate_models_to_db.py       # 模型迁移
│   └── ...
│
├── logs/                             # 日志目录
├── celery_app.py                     # Celery应用配置
├── init_db.py                        # 数据库初始化脚本
├── run.py                            # 启动脚本
├── requirements.txt                  # Python依赖
├── .env.example                      # 环境变量模板
└── README.md                         # 本文件
```

---

## API文档

### 认证模块

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/auth/register` | POST | 用户注册 |
| `/api/v1/auth/login` | POST | 用户登录 |
| `/api/v1/auth/refresh` | POST | 刷新Token |
| `/api/v1/auth/me` | GET | 获取当前用户信息 |
| `/api/v1/auth/config` | GET/PUT | 用户配置管理 |

### 数据集模块

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/datasets` | GET | 获取数据集列表 |
| `/api/v1/datasets` | POST | 注册数据集 |
| `/api/v1/datasets/{id}` | GET | 获取数据集详情 |
| `/api/v1/datasets/{id}/images` | GET | 获取图像列表 |
| `/api/v1/datasets/{id}/augment` | POST | 数据增强预览 |
| `/api/v1/datasets/{id}/detailed-stats` | GET | 详细统计分析 |
| `/api/v1/datasets/upload` | POST | 上传数据集文件 |

### 模型模块

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/models` | GET/POST | 获取/创建模型架构 |
| `/api/v1/models/{id}` | GET/PUT/DELETE | 模型详情/更新/删除 |
| `/api/v1/models/validate` | POST | 验证模型图 |
| `/api/v1/models/analyze` | POST | 分析模型结构 |
| `/api/v1/models/infer-shapes` | POST | 推断张量形状 |
| `/api/v1/models/generate` | POST | 生成PyTorch代码 |

### 训练模块

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/training` | GET/POST | 获取/创建训练任务 |
| `/api/v1/training/{id}` | GET | 获取训练详情 |
| `/api/v1/training/{id}/control` | POST | 控制训练（暂停/恢复/停止） |
| `/api/v1/training/{id}/metrics` | GET | 获取训练指标 |
| `/api/v1/training/{id}/logs` | GET | 获取训练日志 |
| `/api/v1/training/{id}/checkpoint` | POST | 创建检查点 |
| `/api/v1/training/{id}/checkpoints` | GET | 获取检查点列表 |

### 推理模块

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/inference/predict` | POST | 执行推理（JSON格式） |
| `/api/v1/inference/predict-image` | POST | 图片上传+推理一体化 |
| `/api/v1/inference/batch` | POST | 批量推理 |

### 权重库模块

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/weights` | GET | 获取权重列表 |
| `/api/v1/weights/tree` | GET | 获取权重树形结构 |
| `/api/v1/weights/upload` | POST | 上传权重 |
| `/api/v1/weights/{id}` | GET/DELETE | 权重详情/删除 |
| `/api/v1/weights/{id}/download` | GET | 下载权重 |

### WebSocket端点

| 端点 | 说明 |
|------|------|
| `ws://localhost:8000/api/v1/ws/system` | 系统状态流（CPU、内存、GPU） |
| `ws://localhost:8000/api/v1/ws/training/{id}` | 训练日志流（日志、指标、状态） |

---

## 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `DATABASE_URL` | `sqlite:///./cvstudio.db` | 数据库连接URL |
| `SECRET_KEY` | - | JWT密钥（必填） |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | Access Token过期时间（分钟） |
| `UPLOAD_DIR` | `./data/uploads` | 上传文件目录 |
| `DATASETS_DIR` | `./data/datasets` | 数据集目录 |
| `MODELS_DIR` | `./data/models` | 模型目录 |
| `CHECKPOINTS_DIR` | `./data/checkpoints` | 检查点目录 |
| `TEMP_DIR` | `./data/temp` | 临时文件目录 |
| `MAX_UPLOAD_SIZE` | `1073741824` | 最大上传大小（1GB） |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis连接URL |
| `LOG_LEVEL` | `DEBUG` | 日志级别 |

---

## 开发指南

### 代码风格

- 使用 **Google风格** 的docstring
- 使用 **类型提示**（Type Hints）
- 遵循 **PEP 8** 代码规范

### 测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试文件
pytest tests/test_services/test_dataset_service.py

# 生成覆盖率报告
pytest tests/ --cov=app --cov-report=html

# 查看覆盖率报告
open htmlcov/index.html
```

### 日志

使用 **loguru** 进行日志记录：

```python
from loguru import logger

logger.info("信息日志")
logger.error("错误日志")
logger.debug("调试日志")
```

### 添加新的API端点

1. 在 `app/schemas/` 中定义请求/响应模式
2. 在 `app/services/` 中实现业务逻辑
3. 在 `app/api/v1/` 中创建路由
4. 在 `tests/` 中添加测试

---

## 数据库模型

### 核心表结构

| 表名 | 说明 |
|------|------|
| `users` | 用户表 |
| `datasets` | 数据集表 |
| `model_architectures` | 模型架构表 |
| `generated_codes` | 生成的代码表 |
| `training_runs` | 训练任务表 |
| `inference_jobs` | 推理任务表 |
| `weight_libraries` | 权重库表 |
| `augmentation_configs` | 数据增强配置表 |

---

## 常见问题

### 1. Celery Worker无法启动？

确保Redis服务正在运行：
```bash
docker run -d -p 6379:6379 redis:latest
```

### 2. 训练任务没有日志输出？

检查WebSocket连接是否正常，查看 `logs/training.log` 文件。

### 3. 数据集识别失败？

确保数据集目录结构符合YOLO/COCO/VOC/分类格式规范。

### 4. 推理加载模型失败？

检查权重文件是否正确上传，模型架构是否与权重匹配。

---

## 维护者

- **MINGYUz01** - 项目维护者

---

## 许可证

MIT License

---

**最后更新时间**：2026-02-23
