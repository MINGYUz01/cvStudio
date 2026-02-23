# CV Studio

> 计算机视觉任务管理平台 — 一站式深度学习计算机视觉任务管理解决方案

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-blue)](https://react.dev)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 项目简介

CV Studio 是一个面向深度学习计算机视觉任务的一站式管理平台，提供从**数据集管理 → 模型构建 → 训练管理 → 推理预览**的完整工作流。平台采用现代化的技术栈，具有易扩展、易演示的特点。

### 核心功能

| 模块 | 功能描述 | 状态 |
|------|----------|------|
| **数据集管理** | 自动识别YOLO/COCO/VOC/分类格式，在线数据增强与预览 | ✅ 完成 |
| **模型构建器** | 可视化拖拽构建模型，自动生成PyTorch代码 | ✅ 完成 |
| **训练管理** | 训练任务调度，实时监控面板，Checkpoint管理 | ✅ 完成 |
| **推理模块** | 单图/批量推理，支持分类与检测任务 | ✅ 完成 |
| **权重库** | 权重文件版本管理，关联训练任务 | ✅ 完成 |
| **用户系统** | JWT认证，配置管理 | ✅ 完成 |

---

## 技术栈

### 后端
- **框架**：FastAPI
- **数据库**：SQLite + SQLAlchemy 2.0
- **认证**：JWT (python-jose)
- **任务队列**：Celery + Redis
- **图像处理**：OpenCV + Pillow + Albumentations
- **深度学习**：PyTorch 2.1 + torchvision

### 前端
- **框架**：React 19 + TypeScript
- **构建工具**：Vite 6
- **UI组件**：Lucide Icons + Recharts
- **拖拽系统**：@dnd-kit
- **代码高亮**：React Syntax Highlighter

---

## 快速开始

### 环境要求

- **Python**：3.10+
- **Node.js**：18+
- **Redis**：5.0+（用于Celery任务队列）
- **操作系统**：Windows 10/11、macOS、Linux

### 后端安装

```bash
# 进入后端目录
cd backend

# 创建conda环境（可选）
conda create -n cvstudio python=3.10
conda activate cvstudio

# 安装依赖
pip install -r requirements.txt

# 复制环境变量文件
cp .env.example .env

# 启动Redis（Docker方式，在另一个终端）
docker run -d -p 6379:6379 redis:latest

# 启动Celery Worker（在另一个终端）
cd backend
python -m celery -A celery_app worker --loglevel=info --pool=solo

# 启动后端服务
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 前端安装

```bash
# 新开一个终端，进入前端目录
cd frontend

# 安装依赖
npm install

# 复制环境变量文件
cp .env.example .env.local

# 启动前端开发服务器
npm run dev
```

### 访问应用

| 服务 | 地址 |
|------|------|
| **前端应用** | http://localhost:3000 |
| **后端API** | http://localhost:8000 |
| **API文档（Swagger）** | http://localhost:8000/api/v1/docs |
| **API文档（ReDoc）** | http://localhost:8000/api/v1/redoc |

### 默认账户

```
用户名：admin
密码：admin123
```

---

## 项目结构

```
cvStudio/
├── backend/                          # 后端代码
│   ├── app/
│   │   ├── api/                      # API路由
│   │   │   ├── v1/                   # API版本1
│   │   │   │   ├── auth.py           # 认证API
│   │   │   │   ├── datasets.py       # 数据集API
│   │   │   │   ├── models.py         # 模型API
│   │   │   │   ├── training.py       # 训练API
│   │   │   │   ├── inference.py      # 推理API
│   │   │   │   ├── weights.py        # 权重库API
│   │   │   │   └── websocket.py      # WebSocket
│   │   ├── core/                     # 核心功能
│   │   ├── models/                   # 数据模型（SQLAlchemy）
│   │   ├── schemas/                  # Pydantic模式
│   │   ├── services/                 # 业务逻辑
│   │   ├── utils/                    # 工具函数
│   │   ├── tasks/                    # Celery任务
│   │   └── main.py                   # 应用入口
│   ├── data/                         # 数据目录（统一）
│   │   ├── datasets/                 # 数据集存储
│   │   ├── models/                   # 生成的模型
│   │   ├── checkpoints/              # 训练检查点
│   │   ├── experiments/              # 实验数据
│   │   ├── weights/                  # 权重库
│   │   ├── uploads/                  # 上传文件
│   │   ├── architectures/            # 模型架构
│   │   ├── thumbnails/               # 缩略图
│   │   └── temp/                     # 临时文件
│   ├── tests/                        # 测试目录
│   ├── alembic/                      # 数据库迁移
│   ├── requirements.txt              # Python依赖
│   ├── run.py                        # 启动脚本
│   └── .env.example                  # 环境变量模板
│
├── frontend/                         # 前端代码
│   ├── src/
│   │   ├── components/               # React组件
│   │   │   ├── pages/                # 页面组件
│   │   │   │   ├── Dashboard.tsx     # 仪表盘
│   │   │   │   ├── DatasetManager.tsx # 数据集管理
│   │   │   │   ├── ModelBuilder.tsx  # 模型构建器
│   │   │   │   ├── TrainingMonitor.tsx # 训练监控
│   │   │   │   ├── InferenceView.tsx # 推理界面
│   │   │   │   ├── Settings.tsx      # 设置页面
│   │   │   │   └── Login.tsx         # 登录页面
│   │   │   ├── layout/               # 布局组件
│   │   │   │   ├── Sidebar.tsx       # 侧边栏
│   │   │   │   └── CommandPalette.tsx # 命令面板
│   │   │   └── shared/               # 共享组件
│   │   │       ├── GlobalStatusBar.tsx    # 状态栏
│   │   │       ├── WeightTreeSelect.tsx  # 权重选择器
│   │   │       ├── DataAugmentation.tsx  # 数据增强
│   │   │       ├── CodePreviewModal.tsx  # 代码预览
│   │   │       ├── TrainingConfigView.tsx # 训练配置
│   │   │       └── ...
│   │   ├── hooks/                    # 自定义Hooks
│   │   │   ├── useAuth.ts
│   │   │   ├── useDataset.ts
│   │   │   ├── useTraining.ts
│   │   │   └── useWebSocket.ts
│   │   ├── services/                 # API服务
│   │   │   ├── api.ts
│   │   │   ├── auth.ts
│   │   │   ├── datasets.ts
│   │   │   ├── models.ts
│   │   │   ├── training.ts
│   │   │   ├── inference.ts
│   │   │   └── weights.ts
│   │   ├── App.tsx                   # 根组件
│   │   ├── types.ts                  # TypeScript类型
│   │   └── index.css                 # 全局样式
│   ├── index.html                    # HTML模板
│   ├── index.tsx                     # Vite入口
│   ├── package.json                  # Node依赖
│   ├── vite.config.ts                # Vite配置
│   └── .env.example                  # 环境变量模板
│
├── docs/                             # 文档目录
│   ├── 开发周期.md                    # 开发计划
│   ├── api/                           # API文档
│   └── design/                        # 设计文档
│
├── logs/                             # 日志目录
├── CLAUDE.md                         # 开发者配置
└── README.md                         # 本文件
```

---

## 功能展示

### 数据集管理
- 支持4种主流格式自动识别：YOLO、COCO、VOC、文件夹分类
- 在线数据增强：翻转、旋转、缩放、颜色调整等11种操作
- 图像预览：缩略图、原图、标注叠加显示
- 数据统计分析：类别分布、图像尺寸、质量评估

### 模型构建器
- 20+ 原子算子库：Conv2d、Linear、BN、ReLU、Pooling等
- 可视化拖拽：节点拖拽、贝塞尔曲线连线
- 自动布局：拓扑排序、碰撞检测
- 代码生成：自动生成可执行的PyTorch代码
- 自定义算子：保存复用常用模块

### 训练管理
- 动态配置schema：根据任务类型生成配置表单
- 实时监控：WebSocket推送训练日志和指标
- 可视化图表：Loss、Accuracy、学习率曲线
- Checkpoint管理：自动保存、断点续训
- 训练控制：开始、暂停、停止、重命名

### 推理模块
- 单图/批量推理
- 权重树形选择（支持版本管理）
- 结果可视化：边界框、标签、置信度
- 性能测试：FPS测试、推理时间分析

### 权重库
- 权重文件管理
- 版本管理（父子关系）
- 关联训练任务
- 导入/导出权重

---

## API文档

### 主要API端点

| 模块 | 端点 | 说明 |
|------|------|------|
| 认证 | `POST /api/v1/auth/login` | 用户登录 |
| 认证 | `POST /api/v1/auth/register` | 用户注册 |
| 数据集 | `GET /api/v1/datasets` | 获取数据集列表 |
| 数据集 | `POST /api/v1/datasets/upload` | 上传数据集 |
| 模型 | `POST /api/v1/models/validate` | 验证模型图 |
| 模型 | `POST /api/v1/models/generate` | 生成PyTorch代码 |
| 训练 | `POST /api/v1/training/start` | 启动训练 |
| 训练 | `POST /api/v1/training/{id}/control` | 控制训练 |
| 推理 | `POST /api/v1/inference/predict-image` | 图片上传+推理 |
| 权重 | `GET /api/v1/weights/tree` | 获取权重树 |
| WebSocket | `ws://localhost:8000/api/v1/ws/training/{id}` | 训练日志流 |

---

## 开发指南

### 代码风格

- **后端**：遵循Google风格的docstring，使用类型提示
- **前端**：使用TypeScript，函数式组件，Hooks

### 测试

```bash
# 后端测试
cd backend
pytest tests/

# 前端测试（待完善）
cd frontend
npm run test
```

### 提交规范

```
feat: 新功能
fix: 修复bug
docs: 文档更新
style: 代码格式调整
refactor: 重构
test: 测试相关
chore: 构建/工具变动
```

---

## 常见问题

### 1. 训练任务无法启动？
确保Redis服务正在运行，Celery Worker已启动。

### 2. WebSocket连接失败？
检查后端服务是否正常运行，防火墙是否允许WebSocket连接。

### 3. 数据集上传失败？
检查文件大小限制，修改后端 `.env` 中的 `MAX_UPLOAD_SIZE` 配置。

### 4. 推理失败？
确保权重文件正确上传，模型架构与权重匹配。

---

## 更新日志

### v1.0.0（2026-02）
- ✅ 数据集管理模块
- ✅ 模型构建器与代码生成
- ✅ 训练任务调度系统
- ✅ WebSocket实时通信
- ✅ 用户认证与配置管理
- ✅ 权重库管理
- ✅ 推理模块
- ✅ 目录结构重构

---

## 贡献指南

欢迎提交Issue和Pull Request！

1. Fork本项目
2. 创建功能分支：`git checkout -b feature/xxx`
3. 提交更改：`git commit -m 'feat: xxx'`
4. 推送分支：`git push origin feature/xxx`
5. 创建Pull Request

---

## 许可证

本项目采用 [MIT](LICENSE) 许可证。

---

## 联系方式

- **项目维护者**：MINGYUz01
- **GitHub**：https://github.com/MINGYUz01/cvStudio

---

## 致谢

感谢以下开源项目的支持：

- [FastAPI](https://fastapi.tiangolo.com) - 现代化的Python Web框架
- [React](https://react.dev) - 用户界面JavaScript库
- [PyTorch](https://pytorch.org) - 深度学习框架
- [Vite](https://vitejs.dev) - 下一代前端构建工具

---

**最后更新时间**：2026-02-23
**文档版本**：v1.0
