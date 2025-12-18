# 第四天API文档：数据集管理模块（预览与增强）

## 概述

第四天完成了数据集管理模块的预览与增强功能，包括图像浏览、数据增强、统计分析等核心API。

## 新增API端点

### 1. 数据集图像浏览

#### 获取图像列表（分页）
```http
GET /api/v1/datasets/{dataset_id}/images?page=1&page_size=20&sort_by=filename&sort_order=asc
```

**参数：**
- `dataset_id` (int): 数据集ID
- `page` (int): 页码，默认1
- `page_size` (int): 每页大小，默认20，最大100
- `sort_by` (str): 排序字段，默认filename
- `sort_order` (str): 排序顺序，asc或desc，默认asc

**响应：**
```json
{
  "success": true,
  "message": "获取图像列表成功",
  "data": {
    "images": [
      {
        "path": "string",
        "filename": "string",
        "width": 224,
        "height": 224,
        "channels": 3,
        "format": "jpeg",
        "size_bytes": 12345,
        "annotations": [...]
      }
    ],
    "total": 100,
    "page": 1,
    "page_size": 20,
    "total_pages": 5
  }
}
```

#### 获取单个图像详情
```http
GET /api/v1/datasets/{dataset_id}/images/{image_path}
```

**响应：**
```json
{
  "success": true,
  "message": "获取图像详情成功",
  "data": {
    "image": {...},
    "annotation_data": {
      "format": "yolo",
      "annotations": [...],
      "total_annotations": 5
    },
    "thumbnail_url": "/api/v1/datasets/{dataset_id}/images/{image_path}/thumbnail",
    "preview_url": "/api/v1/datasets/{dataset_id}/images/{image_path}/preview"
  }
}
```

#### 获取图像缩略图
```http
GET /api/v1/datasets/{dataset_id}/images/{image_path}/thumbnail?size=256
```

**参数：**
- `size` (int): 缩略图尺寸，默认256，范围64-512

**响应：** JPEG图像二进制数据

#### 获取图像预览
```http
GET /api/v1/datasets/{dataset_id}/images/{image_path}/preview?max_size=1024
```

**参数：**
- `max_size` (int): 预览图最大尺寸，默认1024，范围512-2048

**响应：** JPEG图像二进制数据

### 2. 数据增强功能

#### 单图增强预览
```http
POST /api/v1/datasets/{dataset_id}/augment?image_path=images/test.jpg
```

**请求体：**
```json
{
  "augmentation_configs": [
    {
      "flip_horizontal": true,
      "flip_vertical": false,
      "rotation_angle": 45.0,
      "brightness_factor": 1.2,
      "contrast_factor": 1.1,
      "saturation_factor": 1.0,
      "crop_params": null,
      "scale_factor": 1.0,
      "hue_shift": 0.0,
      "gaussian_blur": 0.0,
      "noise_std": 0.0
    }
  ]
}
```

**响应：**
```json
{
  "success": true,
  "message": "图像增强预览生成成功",
  "data": {
    "original_image": "base64_encoded_image",
    "augmented_images": [
      {
        "id": 1,
        "original_path": "string",
        "augmented_data": "base64_encoded_augmented_image",
        "augmentation_config": {...},
        "applied_operations": ["水平翻转", "旋转45.0度"],
        "created_at": "2025-12-18T19:45:00"
      }
    ],
    "augmentation_summary": {
      "total_configs": 1,
      "successful_augmentations": 1,
      "operations_used": ["水平翻转", "旋转"],
      "original_image_info": {...},
      "generated_at": "2025-12-18T19:45:00"
    }
  }
}
```

#### 批量数据增强
```http
POST /api/v1/datasets/{dataset_id}/batch-augment?image_paths=image1.jpg&image_paths=image2.jpg
```

**请求体：**
```json
{
  "flip_horizontal": true,
  "brightness_factor": 1.2,
  "rotation_angle": 15.0
}
```

**注意：** 最多支持10张图像的批量处理

### 3. 数据统计分析

#### 获取详细统计信息
```http
GET /api/v1/datasets/{dataset_id}/detailed-stats
```

**响应：**
```json
{
  "success": true,
  "message": "获取详细统计信息成功",
  "data": {
    "basic_stats": {
      "dataset_id": 1,
      "num_images": 100,
      "num_classes": 3,
      "class_distribution": {...},
      "image_size_distribution": {...},
      "format_details": {...},
      "quality_metrics": {...}
    },
    "image_quality_analysis": {
      "average_score": 3.5,
      "quality_distribution": {...},
      "format_distribution": {...},
      "size_distribution": {...},
      "total_analyzed": 100
    },
    "annotation_quality_analysis": {
      "total_annotations": 500,
      "images_with_annotations": 80,
      "coverage_rate": 0.8,
      "avg_annotations_per_image": 5.0,
      "bbox_size_stats": {...}
    },
    "class_balance_analysis": {
      "distribution": {...},
      "total_classes": 3,
      "total_annotations": 500,
      "balance_ratio": 0.75,
      "is_balanced": true,
      "format_type": "yolo"
    },
    "size_distribution_analysis": {
      "distribution": {...},
      "aspect_distribution": {...},
      "min_resolution": 65536,
      "max_resolution": 2073600,
      "avg_resolution": 500000,
      "total_analyzed": 100
    },
    "recommendations": [
      "建议提升图像分辨率，当前平均质量较低",
      "数据集类别不平衡，建议进行数据增强或重采样"
    ]
  }
}
```

#### 获取过滤选项
```http
GET /api/v1/datasets/{dataset_id}/filter-options
```

**响应：**
```json
{
  "success": true,
  "message": "获取过滤选项成功",
  "data": {
    "formats": ["jpeg", "png"],
    "size_ranges": ["small", "medium", "large"],
    "classes": ["cat", "dog", "bird"],
    "has_annotations": true,
    "total_images_sample": 100
  }
}
```

## 数据模型

### AugmentationConfig
数据增强配置模型：
```json
{
  "flip_horizontal": false,
  "flip_vertical": false,
  "rotation_angle": 0.0,
  "brightness_factor": 1.0,
  "contrast_factor": 1.0,
  "saturation_factor": 1.0,
  "crop_params": {"x": 0, "y": 0, "width": 224, "height": 224},
  "scale_factor": 1.0,
  "hue_shift": 0.0,
  "gaussian_blur": 0.0,
  "noise_std": 0.0
}
```

### ImageInfo
图像信息模型：
```json
{
  "path": "string",
  "filename": "string",
  "width": 224,
  "height": 224,
  "channels": 3,
  "format": "jpeg",
  "size_bytes": 12345,
  "annotations": []
}
```

### AugmentedImage
增强图像模型：
```json
{
  "id": 1,
  "original_path": "string",
  "augmented_data": "base64_encoded_image",
  "augmentation_config": {...},
  "applied_operations": ["水平翻转"],
  "created_at": "2025-12-18T19:45:00"
}
```

## 支持的数据增强操作

### 几何变换
- **翻转**：水平翻转（`flip_horizontal`）、垂直翻转（`flip_vertical`）
- **旋转**：任意角度旋转（`rotation_angle`）
- **裁剪**：指定区域裁剪（`crop_params`）
- **缩放**：比例缩放（`scale_factor`）

### 颜色变换
- **亮度调整**：亮度因子调整（`brightness_factor`）
- **对比度调整**：对比度因子调整（`contrast_factor`）
- **饱和度调整**：饱和度因子调整（`saturation_factor`）
- **色调调整**：色调偏移（`hue_shift`）

### 高级操作
- **高斯模糊**：指定标准差的高斯模糊（`gaussian_blur`）
- **添加噪声**：高斯噪声添加（`noise_std`）

## 安全特性

1. **路径验证**：所有图像路径都经过严格验证，确保在数据集范围内
2. **文件大小限制**：批量处理限制在10张图像以内
3. **参数验证**：所有输入参数都有合理的范围限制
4. **错误处理**：完善的异常处理和错误信息返回

## 性能优化

1. **分页加载**：避免一次性加载大量图像
2. **缩略图缓存**：生成的缩略图可以缓存复用
3. **异步处理**：数据增强等耗时操作支持异步处理
4. **内存管理**：合理的内存使用和清理机制

## 支持的数据集格式

- **YOLO格式**：支持obj.names和标注文件解析
- **COCO格式**：支持JSON标注文件解析
- **VOC格式**：支持XML标注文件解析
- **分类格式**：支持文件夹结构的分类数据集

## 测试验证

第四天的所有功能都通过了综合测试：

- ✅ 图像处理工具测试
- ✅ YOLO数据集图像浏览测试
- ✅ 数据增强预览测试
- ✅ 数据集统计分析测试
- ✅ 批量增强功能测试
- ✅ 过滤选项功能测试
- ✅ 分类格式数据集测试

## 使用示例

### 1. 浏览数据集图像
```bash
# 获取第一页图像
curl "http://localhost:8000/api/v1/datasets/1/images?page=1&page_size=20"

# 获取单张图像详情
curl "http://localhost:8000/api/v1/datasets/1/images/images/test.jpg"
```

### 2. 数据增强
```bash
# 增强单张图像
curl -X POST "http://localhost:8000/api/v1/datasets/1/augment?image_path=images/test.jpg" \
  -H "Content-Type: application/json" \
  -d '{
    "augmentation_configs": [{
      "flip_horizontal": true,
      "brightness_factor": 1.2
    }]
  }'
```

### 3. 获取统计信息
```bash
# 获取详细统计
curl "http://localhost:8000/api/v1/datasets/1/detailed-stats"

# 获取过滤选项
curl "http://localhost:8000/api/v1/datasets/1/filter-options"
```

---

**最后更新：** 2025-12-18
**版本：** v1.0
**状态：** 已完成并通过测试