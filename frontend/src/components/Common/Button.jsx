/**
 * 通用按钮组件
 * 提供多种样式、状态和交互效果
 */

import React from 'react'
import { Loader2 } from 'lucide-react'

/**
 * 按钮类型枚举
 */
export const ButtonType = {
  PRIMARY: 'primary',
  SECONDARY: 'secondary',
  OUTLINE: 'outline',
  GHOST: 'ghost',
  DANGER: 'danger',
  WARNING: 'warning',
  SUCCESS: 'success',
  INFO: 'info'
}

/**
 * 按钮大小枚举
 */
export const ButtonSize = {
  SMALL: 'small',
  MEDIUM: 'medium',
  LARGE: 'large',
  XLARGE: 'xlarge'
}

/**
 * 按钮变体枚举
 */
export const ButtonVariant = {
  SOLID: 'solid',
  OUTLINE: 'outline',
  GHOST: 'ghost',
  LINK: 'link'
}

/**
 * 通用按钮组件
 * @param {Object} props - 组件属性
 * @param {string} props.type - 按钮类型
 * @param {string} props.variant - 按钮变体
 * @param {string} props.size - 按钮大小
 * @param {boolean} props.disabled - 是否禁用
 * @param {boolean} props.loading - 是否加载中
 * @param {boolean} props.fullWidth - 是否全宽
 * @param {boolean} props.rounded - 是否圆角
 * @param {string} props.className - 自定义类名
 * @param {React.ReactNode} props.children - 子元素
 * @param {React.ReactNode} props.icon - 图标
 * @param {string} props.iconPosition - 图标位置
 * @param {Function} props.onClick - 点击事件
 * @param {string} props.href - 链接地址
 * @param {string} props.target - 链接目标
 * @param {string} props.title - 提示文本
 * @param {boolean} props.autoFocus - 是否自动聚焦
 * @param {string} props.form - 关联的表单ID
 * @param {string} props.typeAttr - HTML类型属性
 * @param {Object} props.loadingText - 加载文本配置
 */
const Button = React.forwardRef(({
  type = ButtonType.PRIMARY,
  variant = ButtonVariant.SOLID,
  size = ButtonSize.MEDIUM,
  disabled = false,
  loading = false,
  fullWidth = false,
  rounded = false,
  className = '',
  children,
  icon,
  iconPosition = 'left',
  onClick,
  href,
  target,
  title,
  autoFocus = false,
  form,
  typeAttr = 'button',
  loadingText = '加载中...',
  ...restProps
}, ref) => {
  // 获取按钮配置
  const getButtonConfig = () => {
    const configs = {
      [ButtonType.PRIMARY]: {
        bg: 'bg-blue-500',
        hover: 'hover:bg-blue-600',
        active: 'active:bg-blue-700',
        text: 'text-white',
        border: 'border-blue-500',
        outline: 'outline-blue-500'
      },
      [ButtonType.SECONDARY]: {
        bg: 'bg-slate-600',
        hover: 'hover:bg-slate-700',
        active: 'active:bg-slate-800',
        text: 'text-white',
        border: 'border-slate-600',
        outline: 'outline-slate-500'
      },
      [ButtonType.OUTLINE]: {
        bg: 'bg-transparent',
        hover: 'hover:bg-slate-700',
        active: 'active:bg-slate-800',
        text: 'text-slate-200',
        border: 'border-slate-600',
        outline: 'outline-slate-500'
      },
      [ButtonType.GHOST]: {
        bg: 'bg-transparent',
        hover: 'hover:bg-slate-700',
        active: 'active:bg-slate-800',
        text: 'text-slate-200',
        border: 'border-transparent',
        outline: 'outline-transparent'
      },
      [ButtonType.DANGER]: {
        bg: 'bg-red-500',
        hover: 'hover:bg-red-600',
        active: 'active:bg-red-700',
        text: 'text-white',
        border: 'border-red-500',
        outline: 'outline-red-500'
      },
      [ButtonType.WARNING]: {
        bg: 'bg-amber-500',
        hover: 'hover:bg-amber-600',
        active: 'active:bg-amber-700',
        text: 'text-white',
        border: 'border-amber-500',
        outline: 'outline-amber-500'
      },
      [ButtonType.SUCCESS]: {
        bg: 'bg-emerald-500',
        hover: 'hover:bg-emerald-600',
        active: 'active:bg-emerald-700',
        text: 'text-white',
        border: 'border-emerald-500',
        outline: 'outline-emerald-500'
      },
      [ButtonType.INFO]: {
        bg: 'bg-cyan-500',
        hover: 'hover:bg-cyan-600',
        active: 'active:bg-cyan-700',
        text: 'text-white',
        border: 'border-cyan-500',
        outline: 'outline-cyan-500'
      }
    }
    return configs[type] || configs[ButtonType.PRIMARY]
  }

  // 获取大小配置
  const getSizeConfig = () => {
    const configs = {
      [ButtonSize.SMALL]: {
        padding: 'px-3 py-1.5',
        fontSize: 'text-sm',
        height: 'h-8',
        iconSize: 14,
        rounded: rounded ? 'rounded-full' : 'rounded-md'
      },
      [ButtonSize.MEDIUM]: {
        padding: 'px-4 py-2',
        fontSize: 'text-base',
        height: 'h-10',
        iconSize: 16,
        rounded: rounded ? 'rounded-full' : 'rounded-lg'
      },
      [ButtonSize.LARGE]: {
        padding: 'px-6 py-3',
        fontSize: 'text-lg',
        height: 'h-12',
        iconSize: 20,
        rounded: rounded ? 'rounded-full' : 'rounded-xl'
      },
      [ButtonSize.XLARGE]: {
        padding: 'px-8 py-4',
        fontSize: 'text-xl',
        height: 'h-14',
        iconSize: 24,
        rounded: rounded ? 'rounded-full' : 'rounded-2xl'
      }
    }
    return configs[size] || configs[ButtonType.MEDIUM]
  }

  // 获取变体样式
  const getVariantStyles = () => {
    const config = getButtonConfig()

    switch (variant) {
      case ButtonVariant.OUTLINE:
        return `bg-transparent ${config.hover} ${config.text} border ${config.border}`
      case ButtonVariant.GHOST:
        return `bg-transparent ${config.hover} ${config.text} border border-transparent`
      case ButtonVariant.LINK:
        return `bg-transparent hover:underline ${config.text} border border-transparent p-0 h-auto`
      default:
        return `${config.bg} ${config.hover} ${config.text} border ${config.border}`
    }
  }

  const buttonConfig = getButtonConfig()
  const sizeConfig = getSizeConfig()
  const variantStyles = getVariantStyles()

  // 基础样式类
  const baseClasses = [
    'inline-flex',
    'items-center',
    'justify-center',
    'font-medium',
    'transition-all',
    'duration-200',
    'ease-in-out',
    'focus:outline-none',
    'focus:ring-2',
    'focus:ring-offset-2',
    'focus:ring-offset-slate-900',
    'disabled:opacity-50',
    'disabled:cursor-not-allowed',
    'disabled:pointer-events-none'
  ]

  // 条件样式类
  const conditionalClasses = [
    sizeConfig.padding,
    sizeConfig.fontSize,
    sizeConfig.height,
    sizeConfig.rounded,
    variantStyles,
    fullWidth ? 'w-full' : '',
    `focus:ring-${buttonConfig.outline.replace('outline-', '')}`,
    className
  ].filter(Boolean).join(' ')

  // 处理点击事件
  const handleClick = (e) => {
    if (disabled || loading) {
      e.preventDefault()
      return
    }
    onClick?.(e)
  }

  // 渲染图标
  const renderIcon = () => {
    if (!icon && !loading) return null

    const IconComponent = loading ? Loader2 : icon
    const iconSize = sizeConfig.iconSize

    return (
      <IconComponent
        size={iconSize}
        className={loading ? 'animate-spin' : ''}
      />
    )
  }

  // 渲染内容
  const renderContent = () => {
    if (loading) {
      return (
        <>
          {renderIcon()}
          <span>{loadingText}</span>
        </>
      )
    }

    if (icon && iconPosition === 'left') {
      return (
        <>
          {renderIcon()}
          <span className="ml-2">{children}</span>
        </>
      )
    }

    if (icon && iconPosition === 'right') {
      return (
        <>
          <span className="mr-2">{children}</span>
          {renderIcon()}
        </>
      )
    }

    return children
  }

  // 如果是链接类型
  if (href) {
    return (
      <a
        ref={ref}
        href={disabled || loading ? undefined : href}
        target={target}
        title={title}
        className={`${baseClasses.join(' ')} ${conditionalClasses}`}
        onClick={handleClick}
        {...restProps}
      >
        {renderContent()}
      </a>
    )
  }

  // 按钮类型
  return (
    <button
      ref={ref}
      type={typeAttr}
      disabled={disabled || loading}
      autoFocus={autoFocus}
      form={form}
      title={title}
      className={`${baseClasses.join(' ')} ${conditionalClasses}`}
      onClick={handleClick}
      {...restProps}
    >
      {renderContent()}
    </button>
  )
})

Button.displayName = 'Button'

/**
 * 按钮组组件
 */
export const ButtonGroup = ({
  children,
  direction = 'horizontal',
  spacing = 'compact',
  className = ''
}) => {
  const getDirectionClasses = () => {
    switch (direction) {
      case 'vertical':
        return 'flex-col'
      default:
        return 'flex-row'
    }
  }

  const getSpacingClasses = () => {
    switch (spacing) {
      case 'compact':
        return direction === 'vertical' ? '-mt-px' : '-ml-px'
      case 'normal':
        return direction === 'vertical' ? 'space-y-2' : 'space-x-2'
      case 'loose':
        return direction === 'vertical' ? 'space-y-4' : 'space-x-4'
      default:
        return 'space-x-2'
    }
  }

  return (
    <div className={`flex ${getDirectionClasses()} ${getSpacingClasses()} ${className}`}>
      {React.Children.map(children, (child, index) => {
        if (!React.isValidElement(child)) return child

        // 为按钮组中的按钮添加特殊样式
        if (child.type === Button || child.type?.displayName === 'Button') {
          const isFirstChild = index === 0
          const isLastChild = index === React.Children.count(children) - 1

          let roundedClass = ''
          if (spacing === 'compact') {
            if (direction === 'vertical') {
              if (isFirstChild) roundedClass = 'rounded-t-lg rounded-b-none'
              else if (isLastChild) roundedClass = 'rounded-b-lg rounded-t-none'
              else roundedClass = 'rounded-none'
            } else {
              if (isFirstChild) roundedClass = 'rounded-l-lg rounded-r-none'
              else if (isLastChild) roundedClass = 'rounded-r-lg rounded-l-none'
              else roundedClass = 'rounded-none'
            }
          }

          return React.cloneElement(child, {
            className: `${child.props.className || ''} ${roundedClass}`
          })
        }

        return child
      })}
    </div>
  )
}

/**
 * 浮动操作按钮 (FAB)
 */
export const FloatingActionButton = ({
  icon,
  onClick,
  position = 'bottom-right',
  color = ButtonType.PRIMARY,
  size = ButtonSize.LARGE,
  className = '',
  ...restProps
}) => {
  const getPositionClasses = () => {
    switch (position) {
      case 'top-left':
        return 'top-4 left-4'
      case 'top-right':
        return 'top-4 right-4'
      case 'bottom-left':
        return 'bottom-4 left-4'
      default:
        return 'bottom-4 right-4'
    }
  }

  return (
    <button
      className={`fixed ${getPositionClasses()} z-50 ${className}`}
      onClick={onClick}
      {...restProps}
    >
      <Button
        type={color}
        size={size}
        rounded
        className="shadow-lg hover:shadow-xl transform hover:scale-105"
      >
        {icon}
      </Button>
    </button>
  )
}

export {
  ButtonType,
  ButtonSize,
  ButtonVariant
}

export default Button