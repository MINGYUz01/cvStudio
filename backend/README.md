# CV Studio 后端

计算机视觉任务管理平台后端服务，基于FastAPI构建。

## 🚀 快速开始

### 环境要求
- Python 3.10+
- Miniconda/Conda
- 已创建cvstudio环境

### 安装依赖

```bash
# 激活conda环境
conda activate cvstudio

# 安装依赖
pip install -r requirements.txt
```

### 配置环境变量

1. 复制环境变量模板：
```bash
cp .env.example .env
```

2. 根据需要修改 `.env` 文件中的配置

### 启动服务

```bash
# 方式1: 使用启动脚本
python run.py

# 方式2: 直接使用uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 访问服务

- **API服务**: http://localhost:8000
- **API文档**: http://localhost:8000/api/v1/docs
- **ReDoc文档**: http://localhost:8000/api/v1/redoc
- **健康检查**: http://localhost:8000/health

## 📁 项目结构

```
backend/
├── app/
│   ├── api/                    # API路由
│   │   └── v1/                # API版本1
│   │       ├── auth.py        # 认证API
│   │       ├── datasets.py    # 数据集API
│   │       ├── models.py      # 模型API
│   │       ├── training.py    # 训练API
│   │       └── inference.py   # 推理API
│   ├── core/                  # 核心功能
│   │   ├── config.py          # 配置管理
│   │   ├── security.py        # 安全功能
│   │   └── exceptions.py      # 异常处理
│   ├── models/                # 数据模型
│   ├── schemas/               # Pydantic模式
│   ├── services/              # 业务逻辑服务
│   ├── utils/                 # 工具函数
│   │   └── format_recognizers/ # 格式识别器
│   ├── templates/             # 代码模板
│   ├── database.py            # 数据库连接
│   ├── dependencies.py        # 依赖注入
│   └── main.py                # 应用入口
├── tests/                     # 测试目录
├── requirements.txt           # 依赖列表
├── .env.example              # 环境变量模板
├── .env                      # 环境变量配置
└── run.py                    # 启动脚本
```

## 🔧 主要功能

### 已实现功能
- ✅ FastAPI应用框架
- ✅ 配置管理系统
- ✅ 异常处理机制
- ✅ 认证安全模块
- ✅ API路由结构
- ✅ 数据库连接配置
- ✅ 依赖注入系统
- ✅ 日志记录配置

### 待开发功能
- 🔄 数据模型定义
- 🔄 用户认证系统
- 🔄 数据集管理
- 🔄 模型构建器
- 🔄 训练管理
- 🔄 推理服务

## 🛠️ 开发说明

### 代码风格
- 使用Python大众化风格
- 遵循Google风格的docstring
- 使用类型提示

### 环境配置
- 开发环境：debug模式，热重载
- 生产环境：优化配置，日志记录

### 安全措施
- JWT认证
- 密码哈希
- CORS配置
- 输入验证

## 📝 API文档

启动服务后，可以通过以下地址查看API文档：
- Swagger UI: http://localhost:8000/api/v1/docs
- ReDoc: http://localhost:8000/api/v1/redoc

## 🧪 测试

```bash
# 运行测试
pytest tests/

# 运行测试并生成覆盖率报告
pytest tests/ --cov=app --cov-report=html
```

## 📦 部署

### Docker部署（计划中）
```bash
# 构建镜像
docker build -t cvstudio-backend .

# 运行容器
docker run -p 8000:8000 cvstudio-backend
```

### 使用Docker Compose（计划中）
```bash
docker-compose up -d
```

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证。

## 📞 联系方式

- 项目维护者：MINGYUz01
- 项目地址：https://github.com/MINGYUz01/cvStudio

---

**开发状态**: 第1天完成 ✅