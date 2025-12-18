/**
 * 确认对话框组件
 * 提供各种类型的确认和提示对话框
 */

import React, { useState, useEffect } from 'react'
import { createPortal } from 'react-dom'
import {
  AlertTriangle,
  CheckCircle,
  XCircle,
  Info,
  X,
  Loader2
} from 'lucide-react'

/**
 * 对话框类型枚举
 */
export const DialogType = {
  CONFIRM: 'confirm',       // 确认对话框
  WARNING: 'warning',       // 警告对话框
  ERROR: 'error',          // 错误对话框
  INFO: 'info',            // 信息对话框
  SUCCESS: 'success',      // 成功对话框
  DANGER: 'danger'         // 危险确认对话框
}

/**
 * 对话框大小枚举
 */
export const DialogSize = {
  SMALL: 'small',
  MEDIUM: 'medium',
  LARGE: 'large',
  FULLSCREEN: 'fullscreen'
}

/**
 * 确认对话框组件
 * @param {Object} props - 组件属性
 * @param {boolean} props.open - 是否打开
 * @param {string} props.type - 对话框类型
 * @param {string} props.size - 对话框大小
 * @param {string} props.title - 对话框标题
 * @param {string} props.message - 对话框消息
 * @param {string} props.description - 详细描述
 * @param {string} props.confirmText - 确认按钮文本
 * @param {string} props.cancelText - 取消按钮文本
 * @param {Function} props.onConfirm - 确认回调
 * @param {Function} props.onCancel - 取消回调
 * @param {Function} props.onClose - 关闭回调
 * @param {boolean} props.loading - 是否加载中
 * @param {boolean} props.showCancel - 是否显示取消按钮
 * @param {boolean} props.closeOnOverlay - 点击遮罩是否关闭
 * @param {boolean} props.closeOnEscape - 按ESC是否关闭
 * @param {string} props.className - 自定义类名
 * @param {React.ReactNode} props.children - 自定义内容
 * @param {React.ReactNode} props.footer - 自定义底部
 * @param {boolean} props.draggable - 是否可拖拽
 */
const ConfirmDialog = ({
  open = false,
  type = DialogType.CONFIRM,
  size = DialogSize.MEDIUM,
  title = '确认操作',
  message = '您确定要执行此操作吗？',
  description,
  confirmText = '确认',
  cancelText = '取消',
  onConfirm,
  onCancel,
  onClose,
  loading = false,
  showCancel = true,
  closeOnOverlay = true,
  closeOnEscape = true,
  className = '',
  children,
  footer,
  draggable = false,
  ...restProps
}) => {
  const [isDragging, setIsDragging] = useState(false)
  const [position, setPosition] = useState({ x: 0, y: 0 })
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 })

  // 获取对话框配置
  const getDialogConfig = () => {
    const configs = {
      [DialogType.CONFIRM]: {
        icon: AlertTriangle,
        iconColor: 'text-amber-500',
        iconBg: 'bg-amber-500/10',
        confirmVariant: 'primary'
      },
      [DialogType.WARNING]: {
        icon: AlertTriangle,
        iconColor: 'text-amber-500',
        iconBg: 'bg-amber-500/10',
        confirmVariant: 'warning'
      },
      [DialogType.ERROR]: {
        icon: XCircle,
        iconColor: 'text-red-500',
        iconBg: 'bg-red-500/10',
        confirmVariant: 'danger'
      },
      [DialogType.INFO]: {
        icon: Info,
        iconColor: 'text-blue-500',
        iconBg: 'bg-blue-500/10',
        confirmVariant: 'info'
      },
      [DialogType.SUCCESS]: {
        icon: CheckCircle,
        iconColor: 'text-emerald-500',
        iconBg: 'bg-emerald-500/10',
        confirmVariant: 'success'
      },
      [DialogType.DANGER]: {
        icon: XCircle,
        iconColor: 'text-red-500',
        iconBg: 'bg-red-500/10',
        confirmVariant: 'danger'
      }
    }
    return configs[type] || configs[DialogType.CONFIRM]
  }

  // 获取大小样式
  const getSizeClasses = () => {
    const sizes = {
      [DialogSize.SMALL]: 'max-w-md',
      [DialogSize.MEDIUM]: 'max-w-lg',
      [DialogSize.LARGE]: 'max-w-2xl',
      [DialogSize.FULLSCREEN]: 'max-w-[95vw] max-h-[95vh]'
    }
    return sizes[size] || sizes[DialogSize.MEDIUM]
  }

  const dialogConfig = getDialogConfig()
  const IconComponent = dialogConfig.icon

  // 处理键盘事件
  useEffect(() => {
    const handleEscape = (e) => {
      if (e.key === 'Escape' && closeOnEscape && open) {
        handleClose()
      }
    }

    if (open) {
      document.addEventListener('keydown', handleEscape)
      document.body.style.overflow = 'hidden'
    }

    return () => {
      document.removeEventListener('keydown', handleEscape)
      document.body.style.overflow = 'unset'
    }
  }, [open, closeOnEscape])

  // 处理拖拽
  const handleMouseDown = (e) => {
    if (draggable && e.target.closest('.dialog-header')) {
      setIsDragging(true)
      setDragStart({
        x: e.clientX - position.x,
        y: e.clientY - position.y
      })
    }
  }

  const handleMouseMove = (e) => {
    if (isDragging) {
      setPosition({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y
      })
    }
  }

  const handleMouseUp = () => {
    setIsDragging(false)
  }

  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)
      return () => {
        document.removeEventListener('mousemove', handleMouseMove)
        document.removeEventListener('mouseup', handleMouseUp)
      }
    }
  }, [isDragging, dragStart])

  // 处理关闭
  const handleClose = () => {
    if (!loading) {
      onCancel?.()
      onClose?.()
    }
  }

  // 处理确认
  const handleConfirm = async () => {
    if (!loading && onConfirm) {
      await onConfirm()
    }
  }

  // 处理遮罩点击
  const handleOverlayClick = (e) => {
    if (e.target === e.currentTarget && closeOnOverlay) {
      handleClose()
    }
  }

  // 如果不打开，返回null
  if (!open) return null

  // 渲染对话框内容
  const dialogContent = (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* 遮罩层 */}
      <div
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={handleOverlayClick}
      />

      {/* 对话框 */}
      <div
        className={`relative ${getSizeClasses()} w-full bg-slate-800 border border-slate-700 rounded-xl shadow-2xl transform transition-all duration-200 ${isDragging ? 'cursor-grabbing' : ''} ${className}`}
        style={{
          transform: `translate(${position.x}px, ${position.y}px)`,
          cursor: isDragging ? 'grabbing' : draggable ? 'grab' : 'default'
        }}
        onMouseDown={handleMouseDown}
        {...restProps}
      >
        {/* 头部 */}
        <div className="dialog-header flex items-center justify-between p-6 border-b border-slate-700">
          <div className="flex items-center space-x-3">
            <div className={`p-2 rounded-lg ${dialogConfig.iconBg}`}>
              <IconComponent size={20} className={dialogConfig.iconColor} />
            </div>
            <h3 className="text-lg font-semibold text-white">
              {title}
            </h3>
          </div>
          {!draggable && (
            <button
              onClick={handleClose}
              className="text-slate-400 hover:text-white transition-colors p-1 rounded-md hover:bg-slate-700"
            >
              <X size={20} />
            </button>
          )}
        </div>

        {/* 内容 */}
        <div className="p-6">
          <div className="text-slate-200 mb-4">
            {message}
          </div>

          {description && (
            <div className="text-slate-400 text-sm">
              {description}
            </div>
          )}

          {children && (
            <div className="mt-4">
              {children}
            </div>
          )}
        </div>

        {/* 底部 */}
        {footer || (
          <div className="flex items-center justify-end space-x-3 p-6 border-t border-slate-700">
            {showCancel && (
              <button
                onClick={handleClose}
                disabled={loading}
                className="px-4 py-2 text-sm font-medium text-slate-300 bg-slate-700 border border-slate-600 rounded-lg hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {cancelText}
              </button>
            )}
            <button
              onClick={handleConfirm}
              disabled={loading}
              className="px-4 py-2 text-sm font-medium text-white bg-blue-500 border border-blue-500 rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center space-x-2"
            >
              {loading && <Loader2 size={16} className="animate-spin" />}
              <span>{confirmText}</span>
            </button>
          </div>
        )}
      </div>
    </div>
  )

  // 使用Portal渲染到body
  return createPortal(dialogContent, document.body)
}

/**
 * 对话框Hook
 */
export const useConfirmDialog = () => {
  const [dialog, setDialog] = useState({
    open: false,
    type: DialogType.CONFIRM,
    title: '',
    message: '',
    description: '',
    confirmText: '确认',
    cancelText: '取消',
    onConfirm: null,
    onCancel: null
  })

  const showConfirm = ({
    title = '确认操作',
    message = '您确定要执行此操作吗？',
    description,
    confirmText = '确认',
    cancelText = '取消',
    onConfirm,
    onCancel
  }) => {
    return new Promise((resolve) => {
      setDialog({
        open: true,
        type: DialogType.CONFIRM,
        title,
        message,
        description,
        confirmText,
        cancelText,
        onConfirm: async () => {
          if (onConfirm) await onConfirm()
          resolve(true)
          setDialog(prev => ({ ...prev, open: false }))
        },
        onCancel: () => {
          if (onCancel) onCancel()
          resolve(false)
          setDialog(prev => ({ ...prev, open: false }))
        }
      })
    })
  }

  const showWarning = ({
    title = '警告',
    message = '此操作可能会产生不可预料的后果，请谨慎操作。',
    description,
    confirmText = '继续',
    cancelText = '取消',
    onConfirm,
    onCancel
  }) => {
    return new Promise((resolve) => {
      setDialog({
        open: true,
        type: DialogType.WARNING,
        title,
        message,
        description,
        confirmText,
        cancelText,
        onConfirm: async () => {
          if (onConfirm) await onConfirm()
          resolve(true)
          setDialog(prev => ({ ...prev, open: false }))
        },
        onCancel: () => {
          if (onCancel) onCancel()
          resolve(false)
          setDialog(prev => ({ ...prev, open: false }))
        }
      })
    })
  }

  const showDanger = ({
    title = '危险操作',
    message = '此操作将永久删除数据，且无法恢复。',
    description,
    confirmText = '删除',
    cancelText = '取消',
    onConfirm,
    onCancel
  }) => {
    return new Promise((resolve) => {
      setDialog({
        open: true,
        type: DialogType.DANGER,
        title,
        message,
        description,
        confirmText,
        cancelText,
        onConfirm: async () => {
          if (onConfirm) await onConfirm()
          resolve(true)
          setDialog(prev => ({ ...prev, open: false }))
        },
        onCancel: () => {
          if (onCancel) onCancel()
          resolve(false)
          setDialog(prev => ({ ...prev, open: false }))
        }
      })
    })
  }

  const showInfo = ({
    title = '提示',
    message,
    description,
    confirmText = '知道了',
    showCancel = false,
    onConfirm
  }) => {
    return new Promise((resolve) => {
      setDialog({
        open: true,
        type: DialogType.INFO,
        title,
        message,
        description,
        confirmText,
        showCancel,
        onConfirm: async () => {
          if (onConfirm) await onConfirm()
          resolve(true)
          setDialog(prev => ({ ...prev, open: false }))
        },
        onCancel: () => {
          resolve(false)
          setDialog(prev => ({ ...prev, open: false }))
        }
      })
    })
  }

  const close = () => {
    setDialog(prev => ({ ...prev, open: false }))
  }

  const DialogComponent = () => (
    <ConfirmDialog
      open={dialog.open}
      type={dialog.type}
      title={dialog.title}
      message={dialog.message}
      description={dialog.description}
      confirmText={dialog.confirmText}
      cancelText={dialog.cancelText}
      onConfirm={dialog.onConfirm}
      onCancel={dialog.onCancel}
      onClose={() => setDialog(prev => ({ ...prev, open: false }))}
      showCancel={dialog.showCancel !== false}
    />
  )

  return {
    showConfirm,
    showWarning,
    showDanger,
    showInfo,
    close,
    DialogComponent
  }
}

/**
 * 快速确认组件
 */
export const QuickConfirm = ({
  children,
  onConfirm,
  title = '确认操作',
  message = '您确定要执行此操作吗？',
  type = DialogType.CONFIRM,
  ...dialogProps
}) => {
  const [showDialog, setShowDialog] = useState(false)

  const handleConfirm = async () => {
    if (onConfirm) {
      await onConfirm()
    }
    setShowDialog(false)
  }

  const handleClick = () => {
    setShowDialog(true)
  }

  return (
    <>
      <div onClick={handleClick}>
        {children}
      </div>
      <ConfirmDialog
        open={showDialog}
        type={type}
        title={title}
        message={message}
        onConfirm={handleConfirm}
        onCancel={() => setShowDialog(false)}
        onClose={() => setShowDialog(false)}
        {...dialogProps}
      />
    </>
  )
}

export {
  DialogType,
  DialogSize
}

export default ConfirmDialog