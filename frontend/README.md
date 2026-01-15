# CV Studio 前端

> 计算机视觉任务管理平台前端应用 — 基于 React 19 + TypeScript 构建

[![React](https://img.shields.io/badge/React-19-blue)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.8-blue)](https://typescriptlang.org)
[![Vite](https://img.shields.io/badge/Vite-6.2-purple)](https://vitejs.dev)

---

## 项目简介

CV Studio 前端是整个平台的用户界面，提供数据集管理、模型构建、训练监控、推理执行等功能的可视化操作界面。

### 核心功能

| 组件 | 功能 | 状态 |
|------|------|------|
| **Dashboard** | 系统概览、资源监控、快速访问 | ✅ |
| **DatasetManager** | 数据集列表、图像预览、统计展示 | ✅ |
| **DataAugmentation** | 增强配置、效果预览 | ✅ |
| **ModelBuilder** | 可视化模型构建、代码生成 | ✅ |
| **TrainingMonitor** | 训练配置、实时监控、日志查看 | ✅ |
| **InferenceView** | 推理执行、结果展示 | ✅ |
| **Settings** | 用户配置、系统设置 | ✅ |
| **Login** | 用户登录、注册 | ✅ |

---

## 技术栈

| 分类 | 技术 |
|------|------|
| **框架** | React 19 + TypeScript 5.8 |
| **构建工具** | Vite 6.2 |
| **路由** | React Router v7 |
| **拖拽系统** | @dnd-kit (core, sortable, utilities) |
| **图表库** | Recharts 3.6 |
| **图标** | Lucide React 0.562 |
| **代码高亮** | React Syntax Highlighter 16.1 |
| **HTTP客户端** | Fetch API |

---

## 快速开始

### 环境要求

- **Node.js**：18+
- **npm**：9+

### 安装步骤

```bash
# 1. 进入前端目录
cd frontend

# 2. 安装依赖
npm install

# 3. 复制环境变量文件
cp .env.example .env.local

# 4. 启动开发服务器
npm run dev
```

### 访问应用

- **开发服务器**：http://localhost:3000
- **后端API**：http://localhost:8000（需单独启动）

---

## 项目结构

```
frontend/
├── src/                              # 源代码目录
│   ├── main.tsx                      # 应用入口
│   ├── App.tsx                       # 根组件
│   ├── index.css                     # 全局样式
│   ├── types.ts                      # TypeScript类型定义
│   │
│   ├── components/                   # 组件目录
│   │   ├── Dashboard.tsx             # 仪表盘
│   │   ├── DatasetManager.tsx        # 数据集管理
│   │   ├── DataAugmentation.tsx      # 数据增强
│   │   ├── ModelBuilder.tsx          # 模型构建器
│   │   ├── TrainingMonitor.tsx       # 训练监控
│   │   ├── InferenceView.tsx         # 推理界面
│   │   ├── Settings.tsx              # 设置页面
│   │   ├── Login.tsx                 # 登录页面
│   │   ├── Sidebar.tsx               # 侧边栏
│   │   ├── CommandPalette.tsx        # 命令面板
│   │   ├── GlobalStatusBar.tsx       # 状态栏
│   │   ├── CodePreviewModal.tsx      # 代码预览
│   │   └── AuthGuard.tsx             # 路由保护
│   │
│   ├── services/                     # API服务层
│   │   ├── api.ts                    # API客户端封装
│   │   ├── auth.ts                   # 认证服务
│   │   ├── datasets.ts               # 数据集服务
│   │   ├── models.ts                 # 模型服务
│   │   ├── training.ts               # 训练服务
│   │   ├── inference.ts              # 推理服务
│   │   ├── weights.ts                # 权重库服务
│   │   └── augmentation.ts           # 数据增强服务
│   │
│   ├── hooks/                        # 自定义Hooks
│   │   ├── useAuth.ts                # 认证Hook
│   │   ├── useDataset.ts             # 数据集Hook
│   │   ├── useTraining.ts            # 训练Hook
│   │   └── useWebSocket.ts           # WebSocket Hook
│   │
│   ├── lib/                          # 工具库
│   │   ├── constants.ts              # 常量定义
│   │   └── utils.ts                  # 工具函数
│   │
│   └── styles/                       # 样式文件
│       ├── variables.css             # CSS变量
│       └── global.css                # 全局样式
│
├── public/                           # 静态资源
│   ├── favicon.ico                   # 网站图标
│   └── assets/                       # 资源文件
│
├── index.html                        # HTML模板
├── vite.config.ts                    # Vite配置
├── tsconfig.json                     # TypeScript配置
├── package.json                      # 依赖配置
├── .env.example                      # 环境变量模板
└── README.md                         # 本文件
```

---

## 主要组件

### Dashboard（仪表盘）

- 系统概览卡片：显示训练任务数、数据集数、模型数
- 资源监控图表：CPU、内存、GPU使用率
- 日志查看器：后端/前端日志
- 快速访问卡片

### DatasetManager（数据集管理）

- 数据集列表：搜索、过滤、分页
- 图像画廊：缩略图预览
- Lightbox：大图查看、缩放、拖拽
- 数据集详情：样本数、类别、格式、大小

### DataAugmentation（数据增强）

- 增强操作选择：11种增强操作
- 参数调整：实时预览
- 对比视图：原图 vs 增强图
- 批量处理

### ModelBuilder（模型构建器）

- 算子库：20+ 原子算子
- 画布操作：拖拽、缩放、平移
- 连线系统：贝塞尔曲线
- 自动布局：拓扑排序
- 代码生成：PyTorch代码导出
- 权重库：权重文件管理

### TrainingMonitor（训练监控）

- 实验列表：状态、搜索、过滤
- 创建实验：动态配置schema
- 训练控制：开始、暂停、停止
- 实时图表：Loss、Accuracy、学习率
- 日志查看：实时训练日志

### InferenceView（推理界面）

- 图像上传：单图/批量
- 模型选择
- 结果展示：边界框、标签、置信度
- 性能测试：FPS测试

---

## 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `VITE_API_BASE_URL` | `http://localhost:8000/api/v1` | 后端API地址 |
| `VITE_WS_BASE_URL` | `ws://localhost:8000/api/v1/ws` | WebSocket地址 |

---

## 开发指南

### 代码风格

- 使用 **TypeScript** 进行类型检查
- 使用 **函数式组件** + Hooks
- 使用 **CSS模块** 或 **CSS-in-JS**
- 遵循 **单一职责原则**

### 创建新组件

```tsx
// src/components/MyComponent.tsx
import React from 'react';

interface MyComponentProps {
  title: string;
  onClick?: () => void;
}

export const MyComponent: React.FC<MyComponentProps> = ({ title, onClick }) => {
  return (
    <div className="my-component" onClick={onClick}>
      <h1>{title}</h1>
    </div>
  );
};
```

### 添加新的服务

```typescript
// src/services/myService.ts
import { apiClient } from './api';

export const myService = {
  async getData() {
    return apiClient.get<DataType>('/my-endpoint');
  },

  async createData(data: CreateDataType) {
    return apiClient.post<DataType>('/my-endpoint', data);
  },
};
```

### 自定义Hook示例

```typescript
// src/hooks/useMyHook.ts
import { useState, useEffect } from 'react';

export const useMyHook = (id: string) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // 获取数据逻辑
  }, [id]);

  return { data, loading };
};
```

---

## 构建与部署

### 开发环境

```bash
npm run dev
```

### 生产构建

```bash
npm run build
```

构建产物输出到 `dist/` 目录。

### 预览生产构建

```bash
npm run preview
```

### Docker部署

```bash
# 构建镜像
docker build -t cvstudio-frontend .

# 运行容器
docker run -d -p 3000:80 cvstudio-frontend
```

---

## 路由配置

| 路径 | 组件 | 说明 |
|------|------|------|
| `/login` | Login | 登录页面 |
| `/` | Dashboard | 仪表盘 |
| `/datasets` | DatasetManager | 数据集管理 |
| `/augmentation` | DataAugmentation | 数据增强 |
| `/models` | ModelBuilder | 模型构建器 |
| `/training` | TrainingMonitor | 训练监控 |
| `/inference` | InferenceView | 推理界面 |
| `/settings` | Settings | 设置页面 |

---

## 常见问题

### 1. 如何连接到后端API？

确保 `.env.local` 文件中的 `VITE_API_BASE_URL` 配置正确。

### 2. WebSocket连接失败？

检查后端服务是否正常运行，防火墙是否允许WebSocket连接。

### 3. 如何添加新的页面组件？

1. 在 `src/components/` 中创建组件
2. 在 `App.tsx` 中添加路由
3. 在 `Sidebar.tsx` 中添加导航项

### 4. 样式没有生效？

确保CSS文件正确导入，检查CSS模块配置。

---

## UI设计规范

### 颜色系统

```css
--color-primary: #3b82f6;
--color-success: #10b981;
--color-warning: #f59e0b;
--color-danger: #ef4444;
--color-bg: #0f172a;
--color-bg-elevated: #1e293b;
--color-border: #334155;
```

### 间距系统

```css
--spacing-xs: 0.25rem;
--spacing-sm: 0.5rem;
--spacing-md: 1rem;
--spacing-lg: 1.5rem;
--spacing-xl: 2rem;
```

### 圆角系统

```css
--radius-sm: 0.25rem;
--radius-md: 0.5rem;
--radius-lg: 0.75rem;
--radius-full: 9999px;
```

---

## 快捷键

| 快捷键 | 功能 |
|--------|------|
| `Ctrl + K` | 打开命令面板 |
| `Ctrl + B` | 切换侧边栏 |

---

## 浏览器支持

| 浏览器 | 版本 |
|--------|------|
| Chrome | 90+ |
| Firefox | 88+ |
| Safari | 14+ |
| Edge | 90+ |

---

## 维护者

- **MINGYUz01** - 项目维护者

---

## 许可证

MIT License

---

**最后更新时间**：2026-01-15
