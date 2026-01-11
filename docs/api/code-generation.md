# 模型代码生成功能使用文档

## 功能概述

代码自动生成功能可以将前端 ModelBuilder 构建的可视化模型图直接转换为可执行的 PyTorch 代码，无需手动编写。

## 支持的层类型

### 卷积层
- `Conv2d` - 二维卷积
- `ConvTranspose2d` - 转置卷积（反卷积）

### 池化层
- `MaxPool2d` - 最大池化
- `AvgPool2d` - 平均池化
- `AdaptiveAvgPool2d` - 自适应平均池化

### 归一化层
- `BatchNorm2d` - 批归一化
- `LayerNorm` - 层归一化
- `GroupNorm` - 组归一化
- `InstanceNorm2d` - 实例归一化

### 激活函数
- `ReLU`, `ReLU6` - ReLU激活
- `LeakyReLU` - 泄漏ReLU
- `SiLU` - Swish激活
- `GELU` - 高斯误差线性单元
- `Tanh` - 双曲正切
- `Sigmoid` - Sigmoid激活
- `Softmax` - Softmax激活

### 全连接层
- `Linear` - 全连接层

### 其他层
- `Dropout` - Dropout正则化
- `DropPath` - 随机路径深度（Stochastic Depth）
- `Flatten` - 展平层
- `Upsample` - 上采样
- `Identity` - 恒等映射

### 特殊操作
- `Concat` - 张量拼接
- `Add` - 张量相加（残差连接）
- `MultiheadAttention` - 多头注意力机制

## 使用流程

### 1. 构建模型图

在 ModelBuilder 中：
1. 从左侧算子库拖拽节点到画布
2. 连接节点构建模型结构
3. 在右侧属性面板调整节点参数

### 2. 生成代码

1. 点击顶部工具栏的 **"生成代码"** 按钮
2. 等待代码生成（会显示"生成中..."状态）
3. 生成成功后会自动弹出代码预览窗口

### 3. 预览与导出

在代码预览窗口中：
- **查看元数据**：显示层数、参数量、网络深度等信息
- **验证状态**：显示代码是否通过验证
- **复制代码**：点击"复制代码"按钮复制到剪贴板
- **下载文件**：点击"下载 .py 文件"保存为本地文件
- **查看文档**：点击"PyTorch 文档"链接查看官方文档

## API 端点

### 生成 PyTorch 代码

**端点**: `POST /api/v1/models/generate`

**请求参数**:
```json
{
  "nodes": [
    {
      "id": "n1",
      "type": "Input",
      "data": {"c": 3, "h": 224, "w": 224}
    },
    {
      "id": "n2",
      "type": "Conv2d",
      "data": {"in": 3, "out": 64, "k": 3, "s": 1, "p": 1}
    }
  ],
  "connections": [
    {"source": "n1", "target": "n2"}
  ]
}
```

**查询参数**:
- `model_name` (可选): 指定模型类名，默认为 "GeneratedModel"

**响应**:
```json
{
  "code": "import torch\nimport torch.nn as nn\n...",
  "model_name": "MyModel",
  "validation": {
    "valid": true,
    "syntax_valid": true,
    "executable": true,
    "forward_pass_success": true,
    "errors": [],
    "warnings": []
  },
  "metadata": {
    "layer_count": 5,
    "num_parameters": 12345,
    "input_shape": ["B", 3, 224, 224],
    "output_shape": ["B", 10],
    "depth": 5,
    "validation_passed": true
  }
}
```

### 其他相关端点

- `POST /api/v1/models/validate` - 验证模型图
- `POST /api/v1/models/analyze` - 分析模型结构
- `POST /api/v1/models/infer-shapes` - 推断张量形状
- `POST /api/v1/models/analyze-and-infer` - 组合分析
- `POST /api/v1/models/validate-code` - 验证生成的代码
- `GET /api/v1/models/templates` - 获取可用模板列表

## 代码生成特点

### 自动优化

生成的代码会经过以下优化：
- 合并连续的 nn.Sequential 块（如 Conv2d + BatchNorm2d + ReLU）
- 内联简单的激活函数
- 添加类型注解
- 优化变量命名

### 语义化命名

层名称自动生成，符合常见命名规范：
- `conv1`, `conv2`, ... - 卷积层
- `bn1`, `bn2`, ... - 批归一化层
- `fc1`, `fc2`, ... - 全连接层
- `pool1`, `pool2`, ... - 池化层

### 完整验证

生成的代码会经过：
1. **AST语法检查** - 确保代码语法正确
2. **动态导入验证** - 测试代码能否成功导入
3. **参数完整性验证** - 检查层参数是否完整
4. **前向传播测试** - 使用随机输入测试模型

## 注意事项

1. **输入节点**: 模型必须以 Input 节点开始，指定输入张量形状（channels, height, width）
2. **参数映射**: 前端使用简化的参数名（如 `in`, `out`, `k`），会自动转换为 PyTorch 参数名
3. **形状推断**: 系统会自动推断每个节点的输入输出张量形状
4. **错误处理**: 如果模型结构有问题（如循环依赖、孤立节点），会在生成前提示错误

## 示例

### 简单 CNN 示例

构建一个简单的分类网络：

```
Input(c=3, h=224, w=224)
  → Conv2d(in=3, out=32, k=3, s=1, p=1)
  → BatchNorm2d(num_f=32)
  → ReLU()
  → MaxPool2d(k=2, s=2)
  → Conv2d(in=32, out=64, k=3, s=1, p=1)
  → BatchNorm2d(num_f=64)
  → ReLU()
  → AdaptiveAvgPool2d(out=1)
  → Flatten()
  → Linear(in=64, out=10)
```

生成的代码包含：
- 完整的模型类定义
- __init__ 方法中的层定义
- forward 方法中的前向传播
- 模型元数据信息

## 故障排除

### 常见错误

1. **"图验证失败"** - 检查是否有循环依赖或孤立节点
2. **"形状推断失败"** - 检查层之间的通道数是否匹配
3. **"代码生成失败"** - 查看具体错误信息，通常是由于参数缺失或类型不支持

### 参数不匹配

如果出现参数不匹配的错误：
1. 检查 Conv2d 的 `in` 参数是否等于上一层输出的通道数
2. 检查 Linear 的 `in` 参数是否等于 Flatten 后的特征数
3. 使用形状推断功能查看每个节点的输入输出形状

## 技术细节

### 后端实现

- **图遍历**: `app/utils/graph_traversal.py` - 拓扑排序、循环检测
- **形状推断**: `app/utils/shape_inference.py` - 张量形状计算
- **代码生成**: `app/utils/code_generator/generator.py` - 主生成引擎
- **层构建器**: `app/utils/code_generator/layer_builder.py` - 层代码构建
- **代码验证**: `app/utils/code_generator/validator.py` - 代码验证
- **代码优化**: `app/utils/code_generator/optimizer.py` - 代码优化

### 前端实现

- **API 服务**: `frontend/src/services/models.ts` - API 调用封装
- **UI 组件**: `frontend/components/ModelBuilder.tsx` - 模型构建器

---

**最后更新**: 2025-12-25
**作者**: CV Studio 开发团队
