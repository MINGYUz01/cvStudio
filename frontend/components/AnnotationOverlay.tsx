/**
 * 标注叠加层组件
 * 用于在图片上绘制边界框和标签
 */

import React from 'react';

/**
 * 标注数据接口
 */
export interface Annotation {
  class_id?: number;
  class_name?: string;
  category_name?: string;
  bbox: number[];  // YOLO: [x_center, y_center, width, height] (normalized)
                    // COCO/VOC: [xmin, ymin, width, height] (absolute)
  type: string;
  format: 'yolo_normalized' | 'coco_absolute' | 'voc_absolute';
}

/**
 * 组件属性
 */
export interface AnnotationOverlayProps {
  annotations: Annotation[];
  imageWidth: number;
  imageHeight: number;
  displayWidth: number;
  displayHeight: number;
  className?: string;
  showLabels?: boolean;
}

// 为不同类别生成颜色
const CLASS_COLORS = [
  '#ef4444', // red-500
  '#f97316', // orange-500
  '#eab308', // yellow-500
  '#22c55e', // green-500
  '#06b6d4', // cyan-500
  '#3b82f6', // blue-500
  '#8b5cf6', // violet-500
  '#ec4899', // pink-500
  '#f43f5e', // rose-500
  '#84cc16', // lime-500
];

/**
 * 根据类别名称生成颜色
 */
function getColorForClass(className: string): string {
  let hash = 0;
  for (let i = 0; i < className.length; i++) {
    hash = className.charCodeAt(i) + ((hash << 5) - hash);
  }
  return CLASS_COLORS[Math.abs(hash) % CLASS_COLORS.length];
}

/**
 * 坐标转换函数
 * 将不同格式的边界框坐标转换为像素坐标
 */
function convertBBoxToPixels(
  annotation: Annotation,
  imgWidth: number,
  imgHeight: number
): { x: number; y: number; width: number; height: number } {
  const bbox = annotation.bbox;

  if (annotation.format === 'yolo_normalized') {
    // YOLO格式：[x_center, y_center, width, height] (归一化 0-1)
    const xCenter = bbox[0] * imgWidth;
    const yCenter = bbox[1] * imgHeight;
    const width = bbox[2] * imgWidth;
    const height = bbox[3] * imgHeight;
    return {
      x: xCenter - width / 2,
      y: yCenter - height / 2,
      width,
      height
    };
  } else {
    // COCO/VOC格式：[xmin, ymin, width, height] (绝对坐标)
    return {
      x: bbox[0],
      y: bbox[1],
      width: bbox[2],
      height: bbox[3]
    };
  }
}

/**
 * 标注叠加层组件
 * 使用SVG绘制边界框和标签
 */
export const AnnotationOverlay: React.FC<AnnotationOverlayProps> = ({
  annotations,
  imageWidth,
  imageHeight,
  displayWidth,
  displayHeight,
  className = '',
  showLabels = true
}) => {
  // 如果displayWidth为0，使用百分比模式（自动适应容器）
  const usePercentageMode = displayWidth === 0 || displayHeight === 0;

  if (annotations.length === 0) {
    return null;
  }

  // 当使用百分比模式时，viewBox使用图片原始尺寸
  const viewBox = usePercentageMode
    ? `0 0 ${imageWidth} ${imageHeight}`
    : `0 0 ${displayWidth} ${displayHeight}`;

  return (
    <svg
      className={`absolute inset-0 pointer-events-none ${className}`}
      width={usePercentageMode ? '100%' : displayWidth}
      height={usePercentageMode ? '100%' : displayHeight}
      viewBox={viewBox}
      preserveAspectRatio="none"
      style={{ overflow: 'visible' }}
    >
      {annotations.map((ann, index) => {
        // 在百分比模式下直接使用原始坐标，否则进行缩放
        let x, y, width, height;

        if (usePercentageMode) {
          // 直接使用图片坐标，SVG会自动缩放
          const bbox = ann.bbox;
          if (ann.format === 'yolo_normalized') {
            const xCenter = bbox[0] * imageWidth;
            const yCenter = bbox[1] * imageHeight;
            const w = bbox[2] * imageWidth;
            const h = bbox[3] * imageHeight;
            x = xCenter - w / 2;
            y = yCenter - h / 2;
            width = w;
            height = h;
          } else {
            x = bbox[0];
            y = bbox[1];
            width = bbox[2];
            height = bbox[3];
          }
        } else {
          // 需要手动缩放
          const scaleX = displayWidth / imageWidth;
          const scaleY = displayHeight / imageHeight;
          const { x: px, y: py, width: pw, height: ph } = convertBBoxToPixels(ann, imageWidth, imageHeight);
          x = px * scaleX;
          y = py * scaleY;
          width = pw * scaleX;
          height = ph * scaleY;
        }

        const className = ann.class_name || ann.category_name || `class_${ann.class_id}`;
        const color = getColorForClass(className);

        // 计算线宽和字体大小
        const strokeWidth = usePercentageMode
          ? Math.max(1, imageWidth * 0.005)
          : Math.max(2, Math.min(displayWidth, displayHeight) * 0.01);
        const fontSize = usePercentageMode
          ? Math.max(10, imageWidth * 0.025)
          : Math.max(9, Math.min(displayWidth, displayHeight) * 0.025);

        return (
          <g key={index}>
            {/* 边界框 */}
            <rect
              x={x}
              y={y}
              width={width}
              height={height}
              fill="none"
              stroke={color}
              strokeWidth={strokeWidth}
              rx={2}
              strokeLinecap="round"
            />
            {/* 标签背景 */}
            {showLabels && height >= 14 && (
              <>
                <rect
                  x={x}
                  y={Math.max(0, y - Math.max(12, height * 0.1))}
                  width={Math.max(25, width * 0.8)}
                  height={Math.max(12, height * 0.1)}
                  fill={color}
                  opacity={0.85}
                  rx={2}
                />
                {/* 标签文字 */}
                <text
                  x={x + 3}
                  y={Math.max(8, y - 3)}
                  fill="white"
                  fontSize={fontSize}
                  fontWeight="600"
                  fontFamily="system-ui, -apple-system, sans-serif"
                  dominantBaseline="middle"
                >
                  {className}
                </text>
              </>
            )}
          </g>
        );
      })}
    </svg>
  );
};

export default AnnotationOverlay;
