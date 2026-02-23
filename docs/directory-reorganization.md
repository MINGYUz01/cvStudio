# CV Studio 目录结构整理方案

> 创建日期：2026-02-23
> 目标：创建整洁、模块化的目录结构

---

## 一、当前问题分析

### 1.1 前端目录结构混乱

#### 问题1：双 components 目录
```
frontend/
├── components/          # 主要组件（18个文件）
│   ├── Dashboard.tsx
│   ├── ModelBuilder.tsx
│   ├── TrainingMonitor.tsx
│   ├── InferenceView.tsx
│   ├── Login.tsx
│   └── ...
└── src/
    └── components/      # 工具组件（3个文件）
        ├── AuthGuard.tsx
        ├── ErrorBoundary.tsx
        └── Loading.tsx
```

#### 问题2：hooks 目录分散
```
frontend/
├── hooks/
│   └── useWebSocket.ts
└── src/hooks/
    ├── useAuth.ts
    ├── useDataset.ts
    └── useTraining.ts
```

#### 问题3：services 目录重复
```
frontend/
├── services/             # 空目录
└── src/services/         # 实际的服务文件
```

#### 问题4：根目录文件位置不当
```
frontend/
├── types.ts              # 应该在 src/
├── App.tsx               # 入口文件在根目录
└── components/           # 主要组件在根目录
```

#### 问题5：混合文件扩展名
- `components/DatasetStatusLegend.jsx` - 唯一的 JSX 文件（其余都是 TSX）

### 1.2 后端目录结构

#### 问题1：__pycache__ 缓存文件
- 171 个 `__pycache__` 目录
- 大量 `.pyc` 文件

#### 问题2：根目录脚本文件
```
backend/
├── celery_app.py        # 可能应该移到 app/ 下
├── init_db.py
└── run.py
```

---

## 二、整理方案

### 2.1 前端目标结构

```
frontend/
├── public/                   # 静态资源
│   ├── index.html
│   └── assets/
├── src/
│   ├── components/         # 所有组件统一放这里
│   │   ├── pages/          # 页面级组件
│   │   │   ├── Dashboard.tsx
│   │   │   ├── DatasetManager.tsx
│   │   │   ├── ModelBuilder.tsx
│   │   │   ├── TrainingMonitor.tsx
│   │   │   ├── InferenceView.tsx
│   │   │   ├── Settings.tsx
│   │   │   └── Login.tsx
│   │   ├── layout/        # 布局组件
│   │   │   ├── Sidebar.tsx
│   │   │   └── CommandPalette.tsx
│   │   ├── shared/         # 共享/通用组件
│   │   │   ├── GlobalStatusBar.tsx
│   │   │   ├── WeightTreeSelect.tsx
│   │   │   ├── DataAugmentation.tsx
│   │   │   ├── PaginationControls.tsx
│   │   │   ├── AnnotationOverlay.tsx
│   │   │   ├── CodePreviewModal.tsx
│   │   │   ├── TrainingConfigView.tsx
│   │   │   ├── TrainingConfigDiffView.tsx
│   │   │   ├── AuthGuard.tsx
│   │   │   ├── ErrorBoundary.tsx
│   │   │   └── Loading.tsx
│   │   └── dataset/       # 数据集相关子组件
│   │       └── (dataset 子目录目前为空，可删除)
│   ├── hooks/            # 所有 hooks
│   │   ├── useAuth.ts
│   │   ├── useDataset.ts
│   │   ├── useTraining.ts
│   │   └── useWebSocket.ts
│   ├── services/         # API 服务
│   │   ├── api.ts
│   │   ├── auth.ts
│   │   ├── augmentation.ts
│   │   ├── datasetAdapter.ts
│   │   ├── datasets.ts
│   │   ├── inference.ts
│   │   ├── models.ts
│   │   ├── training.ts
│   │   └── weights.ts
│   ├── types.ts          # 类型定义
│   ├── App.tsx            # 应用入口
│   ├── main.tsx          # Vite 入口
│   ├── index.css         # 全局样式
│   └── router/           # 路由配置（如果有的话）
├── hooks/                 # DELETE - 合并到 src/hooks/
├── services/              # DELETE - 合并到 src/services/
├── components/            # DELETE - 合并到 src/components/
├── types.ts               # DELETE - 移到 src/types.ts
├── App.tsx                # DELETE - 移到 src/App.tsx
├── metadata.json         # 保留
├── package.json           # 保留
├── tsconfig.json         # 保留
└── vite.config.ts        # 保留
```

### 2.2 后端目标结构

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI 入口
│   ├── config.py
│   ├── database.py
│   ├── dependencies.py
│   ├── api/               # API 路由
│   │   ├── websocket.py
│   │   └── v1/
│   │       ├── auth.py
│   │       ├── datasets.py
│   │       ├── models.py
│   │       ├── training.py
│   │       └── inference.py
│   ├── core/              # 核心功能
│   │   └── security.py
│   ├── models/            # 数据模型
│   │   ├── user.py
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── training.py
│   │   ├── inference.py
│   │   ├── generated_code.py
│   │   └── weight_library.py
│   ├── schemas/           # Pydantic 模式
│   ├── services/          # 业务逻辑
│   │   ├── auth_service.py
│   │   ├── dataset_service.py
│   │   ├── model_service.py
│   │   ├── training_service.py
│   │   ├── code_generator_service.py
│   │   └── inference_service.py
│   ├── tasks/             # Celery 任务
│   │   └── training_tasks.py
│   └── utils/             # 工具函数
│       ├── augmentation.py
│       ├── format_recognizers/
│       ├── code_generator/
│       ├── data_loaders/
│       ├── losses/
│       ├── metrics/
│       ├── models/
│       ├── model_loader.py
│       ├── inference_executor.py
│       └── templates/
├── tests/                # 测试目录
│   ├── conftest.py
│   ├── test_api/
│   ├── test_services/
│   ├── test_utils/
│   └── temp/              # 临时测试
├── alembic/              # 数据库迁移
│   └── versions/
├── data/                 # 数据目录
│   ├── datasets/
│   ├── architectures/
│   └── checkpoints/
├── logs/                 # 日志目录
├── scripts/              # 脚本目录
├── requirements.txt       # 依赖
├── .env                   # 环境变量
└── run.py                # 启动脚本（保持）
```

---

## 三、文件移动清单

### 3.1 前端文件移动

| 源路径 | 目标路径 | 操作 |
|--------|---------|------|
| `components/Dashboard.tsx` | `src/components/pages/Dashboard.tsx` | 移动 |
| `components/DatasetManager.tsx` | `src/components/pages/DatasetManager.tsx` | 移动 |
| `components/ModelBuilder.tsx` | `src/components/pages/ModelBuilder.tsx` | 移动 |
| `components/TrainingMonitor.tsx` | `src/components/pages/TrainingMonitor.tsx` | 移动 |
| `components/InferenceView.tsx` | `src/components/pages/InferenceView.tsx` | 移动 |
| `components/Settings.tsx` | `src/components/pages/Settings.tsx` | 移动 |
| `components/Login.tsx` | `src/components/pages/Login.tsx` | 移动 |
| `components/Sidebar.tsx` | `src/components/layout/Sidebar.tsx` | 移动 |
| `components/CommandPalette.tsx` | `src/components/layout/CommandPalette.tsx` | 移动 |
| `components/GlobalStatusBar.tsx` | `src/components/shared/GlobalStatusBar.tsx` | 移动 |
| `components/WeightTreeSelect.tsx` | `src/components/shared/WeightTreeSelect.tsx` | 移动 |
| `components/DataAugmentation.tsx` | `src/components/shared/DataAugmentation.tsx` | 移动 |
| `components/PaginationControls.tsx` | `src/components/shared/PaginationControls.tsx` | 移动 |
| `components/AnnotationOverlay.tsx` | `src/components/shared/AnnotationOverlay.tsx` | 移动 |
| `components/CodePreviewModal.tsx` | `src/components/shared/CodePreviewModal.tsx` | 移动 |
| `components/TrainingConfigView.tsx` | `src/components/shared/TrainingConfigView.tsx` | 移动 |
| `components/TrainingConfigDiffView.tsx` | `src/components/shared/TrainingConfigDiffView.tsx` | 移动 |
| `src/components/AuthGuard.tsx` | `src/components/shared/AuthGuard.tsx` | 移动 |
| `src/components/ErrorBoundary.tsx` | `src/components/shared/ErrorBoundary.tsx` | 移动 |
| `src/components/Loading.tsx` | `src/components/shared/Loading.tsx` | 移动 |
| `components/DatasetStatusLegend.jsx` | `src/components/shared/DatasetStatusLegend.tsx` | 移动+重命名 |
| `hooks/useWebSocket.ts` | `src/hooks/useWebSocket.ts` | 移动 |
| `App.tsx` | `src/App.tsx` | 移动 |
| `types.ts` | `src/types.ts` | 移动 |
| `hooks/` | (删除) | 删除空目录 |
| `services/` | (删除) | 删除空目录 |
| `components/dataset/` | (删除) | 删除空目录 |

### 3.2 后端文件移动

| 源路径 | 目标路径 | 操作 |
|--------|---------|------|
| `celery_app.py` | `app/celery_app.py` | 移动（可选） |

### 3.3 文件删除

**前端删除：**
- `frontend/components/` (移动完成后删除)
- `frontend/hooks/` (移动完成后删除)
- `frontend/services/` (移动完成后删除)
- `frontend/components/dataset/` (空目录)

**后端删除：**
- 所有 `__pycache__/` 目录
- 所有 `.pyc` 文件
- `*.log` 文件（可选保留用于调试）

---

## 四、import 路径更新

### 4.1 需要更新 import 的文件

| 文件 | 更新内容 |
|------|---------|
| `src/main.tsx` | 更新入口文件引用 |
| `src/App.tsx` | 更新所有组件 import 路径 |
| `src/components/pages/*.tsx` | 更新相互引用 |
| `src/components/layout/*.tsx` | 更新组件引用 |
| `src/components/shared/*.tsx` | 更新组件引用 |
| `src/hooks/*.ts` | 更新 import 路径 |
| `vite.config.ts` | 可能需要更新 alias 配置 |

### 4.2 import 路径更新示例

```typescript
// 更新前
import Dashboard from './components/Dashboard';
import Sidebar from './components/Sidebar';
import { useAuth } from './src/hooks/useAuth';

// 更新后
import Dashboard from './components/pages/Dashboard';
import Sidebar from './components/layout/Sidebar';
import { useAuth } from './hooks/useAuth';
```

---

## 五、整理步骤

### 步骤 1：创建新目录结构
```bash
mkdir -p frontend/src/components/{pages,layout,shared}
mkdir -p frontend/src/components/shared/dataset
```

### 步骤 2：移动组件文件
```bash
# 按分类移动文件
mv frontend/components/{Dashboard,DatasetManager,ModelBuilder,TrainingMonitor,InferenceView,Settings,Login}.tsx frontend/src/components/pages/
mv frontend/components/{Sidebar,CommandPalette}.tsx frontend/src/components/layout/
mv frontend/components/{GlobalStatusBar,WeightTreeSelect,DataAugmentation,PaginationControls,AnnotationOverlay,CodePreviewModal,TrainingConfigView,TrainingConfigDiffView}.tsx frontend/src/components/shared/
mv frontend/src/components/{AuthGuard,ErrorBoundary,Loading}.tsx frontend/src/components/shared/
mv frontend/components/DatasetStatusLegend.jsx frontend/src/components/shared/DatasetStatusLegend.tsx
```

### 步骤 3：移动 hooks 和 services
```bash
mv frontend/hooks/useWebSocket.ts frontend/src/hooks/
rm -rf frontend/hooks/
rm -rf frontend/services/
```

### 步骤 4：移动根目录文件
```bash
mv frontend/App.tsx frontend/src/App.tsx
mv frontend/types.ts frontend/src/types.ts
```

### 步骤 5：更新 main.tsx
```bash
# 更新入口文件引用从 './App' 到 './src/App'
```

### 步骤 6：删除旧目录
```bash
rm -rf frontend/components/
```

### 步骤 7：后端清理
```bash
# 清理 Python 缓存
find backend -type d -name __pycache__ -exec rm -rf {} +
find backend -name "*.pyc" -delete

# 移动 celery_app.py（可选）
mv backend/celery_app.py backend/app/celery_app.py
```

---

## 六、验证检查清单

- [ ] 所有组件文件已移动
- [ ] 所有 import 路径已更新
- [ ] 应用能正常启动
- [ ] 所有页面功能正常
- [ ] 无 TypeScript 类型错误
- [ ] __pycache__ 已清理
- [ ] .pyc 文件已清理

---

## 七、备注

1. **备份建议**：执行整理前请先创建 git commit 或备份项目
2. **测试优先**：整理完成后需要全面测试所有功能
3. **渐进式**：可以分阶段执行，先整理前端再整理后端
4. **gitignore 更新**：建议添加 `__pycache__/` 和 `*.pyc` 到 .gitignore

---

**负责人**：MINGYUz01
**创建日期**：2026-02-23
**状态**：待确认
