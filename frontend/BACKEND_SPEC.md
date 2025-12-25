# NeuroCore Studio 后端开发白皮书 v2.0 (完整版)

## 1. 基础规范 (Basics)

*   **API Base URL**: `/api/v1`
*   **通信协议**: HTTP/1.1 (RESTful) & WebSocket (Real-time)
*   **数据格式**: JSON
*   **时间格式**: ISO 8601 (`YYYY-MM-DDTHH:mm:ssZ`)
*   **认证方式**: Bearer Token (JWT)

### 状态码标准
| Code | 含义 | 说明 |
| :--- | :--- | :--- |
| `200 OK` | 成功 | 请求成功处理 |
| `201 Created` | 已创建 | 资源创建成功 (如新建实验、上传文件) |
| `204 No Content` | 无内容 | 删除或更新成功，无需返回 Body |
| `400 Bad Request` | 请求错误 | 参数校验失败 |
| `401 Unauthorized`| 未授权 | Token 无效或过期 |
| `404 Not Found` | 未找到 | 资源不存在 |
| `422 Unprocessable`| 语义错误 | 如名称重复、业务规则冲突 |
| `500 Server Error`| 服务器错误 | 后端异常 |

---

## 2. 模块 API 详解

### 2.1 认证与系统 (Auth & System)

| 方法 | 路径 | 功能 | 请求参数 (Body/Query) | 响应数据 (Example) |
| :--- | :--- | :--- | :--- | :--- |
| `POST` | `/auth/login` | 用户登录 | `{email, password}` | `{token: "jwt...", user: {...}}` |
| `POST` | `/auth/logout` | 退出登录 | - | - |
| `GET` | `/system/stats` | 获取系统状态 | - | `{gpu: {util: 45, temp: 60}, cpu: {...}}` |
| `GET` | `/system/logs` | 系统日志 | `?type=backend&limit=100` | `[{time, level, msg}, ...]` |

### 2.2 数据集管理 (Datasets)

前端支持自动识别格式，后端需在接收上传后进行文件结构分析。

| 方法 | 路径 | 功能 | 请求参数 | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/datasets` | 获取列表 | `?page=1&size=20&q=search` | 返回列表及统计信息 |
| `POST` | `/datasets` | **导入/上传** | `Multipart/form-data` | 接收 `.zip` 包或服务器本地路径 |
| `GET` | `/datasets/{id}` | 获取详情 | - | 返回类别分布、分辨率统计、空标注率 |
| `GET` | `/datasets/{id}/preview`| 获取样本图 | `?page=1&limit=24` | 返回图片 URL 列表 |
| `PUT` | `/datasets/{id}` | 更新信息 | `{name, description}` | 重命名或修改备注 |
| `DELETE`| `/datasets/{id}` | **删除数据集**| - | **物理删除磁盘文件**及数据库记录 |

### 2.3 数据增强 (Augmentation)

前端生成的 JSON 流水线，后端直接存储，训练时解析。

| 方法 | 路径 | 功能 | 请求参数 | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/augmentation` | 获取策略列表 | - | |
| `POST` | `/augmentation` | 创建策略 | `{name, desc, pipeline: [...]}` | `pipeline` 字段存为 JSONB |
| `PUT` | `/augmentation/{id}` | **更新策略** | `{name, desc, pipeline: [...]}` | 修改现有策略 |
| `DELETE`| `/augmentation/{id}` | **删除策略** | - | |

### 2.4 模型工作台 (Model Builder) - **核心重点**

#### A. 架构设计 (Architectures)
前端 ReactFlow 绘制的图，后端需保存结构并在训练时**动态编译**为 PyTorch 代码。

| 方法 | 路径 | 功能 | 请求参数 | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/models/architectures` | 获取架构列表 | - | |
| `POST` | `/models/architectures` | 创建/另存为 | `{name, nodes: [], edges: []}` | 创建新记录 |
| `PUT` | `/models/architectures/{id}`| **保存/更新** | `{name, nodes: [], edges: []}` | 覆盖更新现有架构 |
| `DELETE`| `/models/architectures/{id}`| **删除架构** | - | |

**数据结构说明 (nodes JSON 示例)**:
```json
[
  {
    "id": "n1",
    "type": "Conv2d",
    "data": { "in": 3, "out": 64, "k": 3, "s": 1 },
    "inputs": [],
    "outputs": ["n2"]
  }
]
```

#### B. 权重库 (Weights / Checkpoints)

| 方法 | 路径 | 功能 | 请求参数 | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/models/weights` | 获取权重列表 | - | 返回文件大小、精度指标(mAP)、关联架构 |
| `POST` | `/models/weights` | **导入权重** | `Multipart` 或 `{path}` | 支持上传 .pt/.onnx 文件 |
| `DELETE`| `/models/weights/{id}` | **删除权重** | - | **物理删除磁盘上的权重文件** |
| `GET` | `/models/weights/{id}/download`| **下载权重** | - | `Content-Disposition: attachment` |
| `PUT` | `/models/weights/{id}` | 编辑元数据 | `{tags: ["best", "deploy"]}` | 修改标签或备注 |

### 2.5 训练监控 (Training)

| 方法 | 路径 | 功能 | 请求参数 | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/experiments` | 获取实验列表 | `?status=running` | |
| `POST` | `/experiments` | **启动训练** | `{name, task, dataset_id, model_arch_id, aug_id, hyperparams: {...}}` | 触发 Celery 异步任务 |
| `PUT` | `/experiments/{id}` | **重命名** | `{name: "New Name"}` | 仅修改名称/备注 |
| `DELETE`| `/experiments/{id}` | **删除实验** | - | 删除数据库记录、日志及关联的临时 Checkpoints |
| `POST` | `/experiments/{id}/control`| **控制任务** | `{action: "stop" | "pause" | "resume"}` | 发送信号给训练进程 |
| `GET` | `/experiments/{id}/metrics` | 获取图表数据 | - | 返回 Loss/Accuracy 历史数据点数组 |
| `POST` | `/experiments/{id}/save` | **保存到权重库** | - | 将当前/最佳 checkpoint 复制到权重库并注册 |

### 2.6 推理实验室 (Inference)

| 方法 | 路径 | 功能 | 请求参数 | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/inference/devices` | 获取设备列表 | - | `cv2.getBuildInformation()` 或枚举 `/dev/video*` |
| `POST` | `/inference/predict` | 图片/批量推理 | `Multipart` (Images) + `model_id` | 返回 JSON (Box, Class, Conf) |
| `POST` | `/inference/stream/start` | 启动流推理 | `{device_id, model_id}` | 启动后台推理进程 |
| `POST` | `/inference/stream/stop` | 停止流推理 | - | 释放摄像头资源 |

---

## 3. 实时通信协议 (WebSocket)

为了保证界面的数据跳动感和实时性，必须实现 WebSocket。

### 3.1 全局状态流
*   **URL**: `ws://api.neurocore.ai/ws/system`
*   **发送频率**: 1Hz (每秒一次)
*   **数据包结构**:
    ```json
    {
      "type": "system_stats",
      "data": {
        "gpu_util": 45,       // %
        "gpu_temp": 62,       // °C
        "vram_used": 12.4,    // GB
        "vram_total": 24.0,   // GB
        "cpu_load": 24,       // %
        "disk_usage": 65,     // %
        "ping": 24            // ms (optional)
      }
    }
    ```

### 3.2 训练日志与监控流
*   **URL**: `ws://api.neurocore.ai/ws/training/{experiment_id}`
*   **触发场景**: 用户进入“训练详情”页时连接。
*   **数据包 (日志)**:
    ```json
    {
      "type": "log",
      "data": "[INFO] Epoch 10/100: box_loss: 0.042, cls_loss: 0.012"
    }
    ```
*   **数据包 (指标更新)**:
    ```json
    {
      "type": "metrics_update",
      "data": { "epoch": 10, "train_loss": 0.054, "val_loss": 0.062, "val_metric": 0.88 }
    }
    ```
*   **数据包 (状态变更)**:
    ```json
    {
      "type": "status_change",
      "data": { "status": "completed", "end_time": "..." }
    }
    ```

### 3.3 推理视频流 (WebRTC 或 MJPEG over WS)
*   **URL**: `ws://api.neurocore.ai/ws/inference/stream`
*   **机制**:
    *   客户端发送指令：`{"command": "start", "model": "w1", "camera": "0"}`
    *   服务端推送：二进制流 (JPEG Frame) 或 Base64 字符串。
    *   *建议*: 对于低延迟需求，直接推 JPEG Base64 比较简单；如果追求高性能，建议走 WebRTC 信令通道。

---

## 4. 后端开发注意事项 (Developer Tips)

1.  **模型解析器 (The Parser)**:
    *   你需要编写一个能够遍历前端发来的 `nodes` 和 `edges` 数组的算法。
    *   **难点**: 处理 Skip Connection (跳跃连接，如 ResNet/YOLO 中的 Concat/Add)。前端已经在 `edges` 里定义了 `source` 和 `target`，你需要通过拓扑排序确定执行顺序。

2.  **异步训练**:
    *   `POST /experiments` 接口必须**立即返回**，不能阻塞。
    *   请使用 **Celery + Redis** 来在后台跑 PyTorch 训练脚本。
    *   训练脚本需要重定向 `stdout/stderr`，将其写入 Redis Pub/Sub，这样 WebSocket 才能实时推送到前端。

3.  **文件路径管理**:
    *   `DATA_DIR`: `/app/data/datasets`
    *   `WEIGHTS_DIR`: `/app/data/weights`
    *   `RUNS_DIR`: `/app/data/runs` (建议兼容 YOLO 的 runs 目录结构)

4.  **Mock 数据替换**:
    *   前端代码中包含 `INITIAL_...` 或 `MOCK_...` 的常量，对接时请确保 API 返回的数据结构与这些常量保持一致（字段名、嵌套结构等）。

---
*文档生成时间: 2023-10-27 | Author: NeuroCore AI Assistant*
