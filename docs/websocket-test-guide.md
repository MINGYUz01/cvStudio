# WebSocket实时通信功能测试指南

## 📋 功能概述

第9天已完成WebSocket实时通信系统的开发，主要包含以下功能：

### 1. 系统状态流
- 实时收集系统资源使用情况（CPU、内存、磁盘、GPU、网络）
- 1Hz频率推送
- 自动推送给所有订阅者

### 2. 训练日志流
- 实时收集训练日志（支持DEBUG、INFO、WARNING、ERROR、CRITICAL级别）
- 实时推送训练指标（Loss、Accuracy、mAP等）
- 训练状态变化通知（queued、running、paused、completed、failed、stopped）

### 3. 前端WebSocket客户端
- 自动连接和断线处理
- 自动重连机制
- 消息类型分发和回调处理

---

## 🚀 测试步骤

### 步骤1: 安装依赖

```bash
# 进入后端目录
cd backend

# 安装依赖（包括pynvml用于GPU监控）
pip install -r requirements.txt
```

### 步骤2: 启动后端服务器

```bash
# 进入backend目录
cd F:\claude_projects\cvStudio\backend

# 启动FastAPI服务器
python -m app.main
```

你应该看到类似的输出：
```
🚀 CV Studio 正在启动...
📍 环境: development
🌐 服务地址: http://localhost:8000
📊 系统指标收集器已启动
```

### 步骤3: 运行WebSocket测试脚本

在新的终端窗口中运行测试脚本：

```bash
# 进入backend目录
cd F:\claude_projects\cvStudio\backend

# 运行测试脚本
python tests/temp/test_websocket_functionality.py
```

### 步骤4: 查看测试结果

测试脚本会执行3个测试：

1. **系统状态WebSocket流测试**
   - 连接到 ws://localhost:8000/api/v1/ws/system
   - 接收5条系统状态更新
   - 显示CPU、内存、GPU等实时数据

2. **训练日志WebSocket流测试**
   - 创建测试训练会话
   - 连接到 ws://localhost:8000/api/v1/ws/training/{experiment_id}
   - 发送测试日志和指标
   - 验证WebSocket实时推送

3. **WebSocket统计信息测试**
   - 查看当前活跃连接数
   - 查看订阅者统计

---

## 🧪 手动测试

### 测试系统状态流

使用在线WebSocket测试工具（如 http://www.websocket.org/echo.html）或编写简单脚本：

```python
import asyncio
import websockets
import json

async def test_system_stats():
    uri = "ws://localhost:8000/api/v1/ws/system?client_id=manual_test"
    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            print(f"收到消息: {data['type']}")
            if data['type'] == 'system_stats':
                print(f"  CPU: {data['data']['cpu']['cpu_util']}%")
                print(f"  内存: {data['data']['memory']['ram_percent']}%")

asyncio.run(test_system_stats())
```

### 测试训练日志流

1. 创建训练会话：
```bash
curl -X POST http://localhost:8000/api/v1/training/logs/session \
  -H "Content-Type: application/json" \
  -d '{
    "experiment_id": "test_exp_123",
    "config": {"model": "yolov8", "dataset": "coco"},
    "total_epochs": 100
  }'
```

2. 连接WebSocket：
```python
import asyncio
import websockets
import json

async def test_training_logs():
    uri = "ws://localhost:8000/api/v1/ws/training/test_exp_123?client_id=test"
    async with websockets.connect(uri) as websocket:
        # 接收连接确认
        message = await websocket.recv()
        print(json.loads(message))

        # 保持连接接收日志
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            print(f"收到: {data['type']}")

asyncio.run(test_training_logs())
```

3. 发送测试日志：
```bash
curl -X POST http://localhost:8000/api/v1/training/logs/test_exp_123/log \
  -H "Content-Type: application/json" \
  -d '{
    "level": "INFO",
    "message": "Epoch 1/100 - Loss: 0.543",
    "source": "trainer"
  }'
```

---

## 📊 测试验证点

### 系统状态流验证
- [x] WebSocket连接成功建立
- [x] 每秒接收1条系统状态更新
- [x] CPU、内存、磁盘数据正确
- [x] 如果有GPU，GPU数据正确显示
- [x] 连接断开后自动清理订阅

### 训练日志流验证
- [x] 能够创建训练会话
- [x] WebSocket连接成功并订阅训练日志
- [x] 发送日志后实时推送到客户端
- [x] 发送指标后实时推送到客户端
- [x] 状态变化后实时推送到客户端
- [x] 支持多个客户端同时订阅同一训练任务

### 前端集成验证
- [x] TrainingMonitor组件能连接到WebSocket
- [x] 实时日志显示在日志查看器中
- [x] 实时指标更新图表显示
- [x] 训练状态变化自动更新UI
- [x] WebSocket断线后自动重连

---

## 🐛 故障排查

### 问题1: 无法连接到WebSocket

**可能原因：**
- 后端服务器未启动
- 防火墙阻止连接
- 端口被占用

**解决方案：**
```bash
# 检查后端是否运行
curl http://localhost:8000/health

# 检查端口占用
netstat -ano | findstr :8000  # Windows
lsof -i :8000                 # Linux/Mac
```

### 问题2: GPU监控不可用

**可能原因：**
- 没有安装pynvml
- 没有NVIDIA GPU
- NVIDIA驱动未安装

**解决方案：**
```bash
# 安装pynvml
pip install pynvml

# 检查GPU
nvidia-smi
```

### 问题3: 前端WebSocket连接失败

**可能原因：**
- 前端URL配置错误
- CORS配置问题
- 实验ID不正确

**解决方案：**
1. 检查浏览器控制台错误信息
2. 确认后端CORS配置正确
3. 验证实验ID是否存在

---

## 📝 API端点列表

### WebSocket端点

| 端点 | 用途 |
|------|------|
| `ws://localhost:8000/api/v1/ws/system` | 系统状态流 |
| `ws://localhost:8000/api/v1/ws/training/{experiment_id}` | 训练日志流 |

### REST API端点

| 方法 | 端点 | 用途 |
|------|------|------|
| POST | `/api/v1/training/logs/session` | 创建训练会话 |
| PUT | `/api/v1/training/logs/{id}/status` | 更新训练状态 |
| POST | `/api/v1/training/logs/{id}/log` | 添加训练日志 |
| POST | `/api/v1/training/logs/{id}/metrics` | 添加训练指标 |
| GET | `/api/v1/training/logs/{id}` | 获取训练日志 |
| GET | `/api/v1/training/logs/{id}/metrics` | 获取训练指标 |
| GET | `/api/v1/training/logs/{id}/info` | 获取会话信息 |
| DELETE | `/api/v1/training/logs/{id}` | 删除训练会话 |
| GET | `/api/v1/training/logs/sessions` | 列出所有会话 |
| GET | `/api/v1/ws/stats` | WebSocket统计信息 |

---

## ✅ 第9天完成清单

- [x] WebSocket服务器基础搭建（连接管理、消息广播、异常处理）
- [x] 全局状态流推送（GPU、内存、CPU监控，1Hz频率）
- [x] 训练日志流（日志收集、实时推送、级别过滤）
- [x] 训练指标推送（Loss/Accuracy实时更新、Epoch进度、状态变化）
- [x] 前端WebSocket客户端封装（连接建立、自动重连、消息分发）
- [x] TrainingMonitor集成WebSocket（接收实时数据、更新图表）
- [x] 测试WebSocket通信功能

---

**最后更新时间：** 2025-12-25
**负责人：** MINGYUz01
