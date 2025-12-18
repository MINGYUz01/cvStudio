/**
 * 通用加载组件
 * 提供多种加载样式和状态显示
 */

import React from 'react'
import { Loader2, Loader, Hash } from 'lucide-react'

/**
 * 加载状态枚举
 */
export const LoadingState = {
  IDLE: 'idle',           // 空闲状态
  LOADING: 'loading',       // 加载中
  SUCCESS: 'success',      // 成功
  ERROR: 'error',          // 错误
  WARNING: 'warning'      // 警告
}

/**
 * 加载类型枚举
 */
export const LoadingType = {
  SPINNER: 'spinner',     // 旋转器
  DOTS: 'dots',        // 点状
  PULSE: 'pulse',       // 脉冲
  BAR: 'bar',          // 进度条
  SKELETON: 'skeleton',    // 骨架屏
  OVERLAY: 'overlay'     // 遮罩
  INLINE: 'inline'       // 内联
  MODAL: 'modal'        // 模态框
}

/**
 * 通用加载组件
 * @param {Object} props - 组件属性
 * @param {string} props.type - 加载类型
 * @param {string} props.size - 尺寸大小
 * @param {string} props.color - 颜色
 * @param {string} props.message - 加载消息
 * @param {React.ReactNode} props.children - 子组件
 * @param {boolean} props.fullscreen - 是否全屏显示
 * @param {boolean} props.overlay - 是否显示遮罩
 * @param {boolean} props.centered - 是否居中
 * @param {boolean} props.showProgress - 是否显示进度条
 * @param {number} props.progress - 进度值（0-100）
 */
const Loading = ({
  type = LoadingType.SPINNER,
  size = 'medium',
  color = 'primary',
  message = '加载中...',
  children,
  fullscreen = false,
  overlay = false,
  centered = true,
  showProgress = false,
  progress = 0
}) => {
  // 颜色映射
  const colorMap = {
    primary: 'bg-blue-500',
    secondary: 'bg-purple-500',
    success: 'bg-emerald-500',
    error: 'bg-red-500',
    warning: 'bg-amber-500',
    info: 'bg-cyan-500',
    slate: 'bg-slate-600',
    gray: 'bg-gray-500',
    white: 'bg-white'
  }

  // 尺寸映射
  const sizeMap = {
    sm: { w: 5, h: 5, border: 2 },
    md: { w: 8, h: 8, border: 3 },
    lg: { w: 12, h: 12, border: 4 },
    xl: { w: 16, h: 16, border: 5 }
  }

  const currentColor = colorMap[color] || colorMap.primary

  // 渲染不同类型的加载效果
  const renderLoadingContent = () => {
    switch (type) {
      case LoadingType.SPINNER:
        return (
          <Loader2
            className={`animate-spin ${currentColorColor.replace('bg-', 'text-')}`}
            size={sizeMap[size]?.w}
            height={sizeMap[size]?.h}
          />
        )

      case LoadingType.DOTS:
        return (
        <div className="flex gap-1">
          {[1, 2, 3, 4, 5, 6].map((_, i) => (
            <div
              key={i}
              className={`w-2 h-2 ${currentColorColor} rounded-full animate-pulse`}
              style={{ animationDelay: `${i * 0.1}s` }}
            />
          ))}
        </div>
      )

      case LoadingType.PULSE:
        return (
        <div className={`w-8 h-8 ${currentColorColor} rounded-full animate-pulse`}></div>
      )

      case LoadingType.BAR:
        return (
        <div className="w-full bg-slate-700 rounded-full h-2">
          <div
            className={`h-full ${currentColor} transition-all duration-300 ease-out`}
            style={{ width: `${progress}%` }}
          ></div>
        </div>
      )

      case LoadingType.SKELETON:
        return (
        <div className="space-y-2">
          {[1, 2, 3, 4].map((i) => (
            <div
              key={i}
              className={`h-2 ${currentColorColor} rounded animate-pulse`}
              style={{
                width: '100%',
                animationDelay: `${i * 0.1}s`,
                animationDuration: '1.5s'
              }}
            />
          ))}
        </div>
      )

      case LoadingType.OVERLAY:
        return (
        <div className="fixed inset-0 bg-slate-900/80 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="text-center">
            <div className={`w-16 h-16 ${currentColor} rounded-lg animate-spin mb-4 mx-auto`}>
              <Loader2 size={48} />
            </div>
            <p className="text-white mt-4">{message}</p>
          </div>
        </div>
      )

      case LoadingType.INLINE:
        return (
        <div className="inline-flex items-center gap-2">
          <Loader2
            className={`animate-spin ${currentColorColor.replace('bg-', 'text-')}`}
            size={sizeMap[size]?.w}
            height={sizeMap[size]?.h}
          />
          <span className={`${currentColor.replace('bg-', 'text-')} ml-2`}>
            {message}
          </span>
        </div>
      )

      case LoadingType.MODAL:
        return (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-slate-800 rounded-xl p-6 max-w-md">
            <div className="text-center mb-4">
              <div className="w-16 h-16 bg-blue-500 rounded-lg animate-spin mb-4 mx-auto">
                <Loader2 size={48} />
              </div>
              <p className="text-white text-lg font-medium">{message}</p>
            </div>
            <div className="text-slate-300 text-sm text-center mt-4">
                <p>请稍候...</p>
              </div>
            </div>
          </div>
        </div>
      )

      default:
        return (
        <div className={`flex items-center gap-2`}>
          <Loader2 size={sizeMap[size]?.w} height={sizeMap[size]?.h} />
          <span className="text-white ml-2">{message}</span>
        </div>
      )
    }
  }

  // 全屏遮罩样式
  const overlayClass = overlay ? 'fixed inset-0 bg-black/50 backdrop-blur-sm z-50' : ''

  // 居中样式
  const centeredClass = centered ? 'items-center justify-center' : ''

  return (
    <div className={`loading-component ${overlayClass} ${centeredClass}`}>
      {renderLoadingContent()}
      {children}
    </div>
  )
}

export default Loading