/**
 * 通用卡片组件
 * 提供灵活的卡片布局和样式选项
 */

import React from 'react'
import { MoreVertical, ExternalLink, Heart, Share2, Bookmark } from 'lucide-react'

/**
 * 卡片变体枚举
 */
export const CardVariant = {
  DEFAULT: 'default',
  ELEVATED: 'elevated',
  OUTLINED: 'outlined',
  FLAT: 'flat'
}

/**
 * 卡片大小枚举
 */
export const CardSize = {
  SMALL: 'small',
  MEDIUM: 'medium',
  LARGE: 'large'
}

/**
 * 卡片阴影级别
 */
export const CardShadow = {
  NONE: 'none',
  SMALL: 'small',
  MEDIUM: 'medium',
  LARGE: 'large',
  XLARGE: 'xlarge'
}

/**
 * 通用卡片组件
 * @param {Object} props - 组件属性
 * @param {string} props.variant - 卡片变体
 * @param {string} props.size - 卡片大小
 * @param {string} props.shadow - 阴影级别
 * @param {boolean} props.hoverable - 是否可悬停
 * @param {boolean} props.clickable - 是否可点击
 * @param {boolean} props.loading - 是否加载中
 * @param {string} props.className - 自定义类名
 * @param {React.ReactNode} props.children - 子元素
 * @param {React.ReactNode} props.header - 卡片头部
 * @param {React.ReactNode} props.title - 卡片标题
 * @param {React.ReactNode} props.subtitle - 卡片副标题
 * @param {React.ReactNode} props.extra - 额外内容
 * @param {React.ReactNode} props.cover - 封面图片
 * @param {React.ReactNode} props.actions - 操作按钮
 * @param {Function} props.onClick - 点击事件
 * @param {Object} props.style - 自定义样式
 */
const Card = React.forwardRef(({
  variant = CardVariant.DEFAULT,
  size = CardSize.MEDIUM,
  shadow = CardShadow.MEDIUM,
  hoverable = false,
  clickable = false,
  loading = false,
  className = '',
  children,
  header,
  title,
  subtitle,
  extra,
  cover,
  actions,
  onClick,
  style,
  ...restProps
}, ref) => {
  // 获取变体样式
  const getVariantStyles = () => {
    const variants = {
      [CardVariant.DEFAULT]: 'bg-slate-800 border-slate-700',
      [CardVariant.ELEVATED]: 'bg-slate-800 border-transparent',
      [CardVariant.OUTLINED]: 'bg-slate-900 border-slate-600',
      [CardVariant.FLAT]: 'bg-slate-900 border-transparent'
    }
    return variants[variant] || variants[CardVariant.DEFAULT]
  }

  // 获取大小样式
  const getSizeStyles = () => {
    const sizes = {
      [CardSize.SMALL]: 'p-4',
      [CardSize.MEDIUM]: 'p-6',
      [CardSize.LARGE]: 'p-8'
    }
    return sizes[size] || sizes[CardSize.MEDIUM]
  }

  // 获取阴影样式
  const getShadowStyles = () => {
    const shadows = {
      [CardShadow.NONE]: 'shadow-none',
      [CardShadow.SMALL]: 'shadow-sm',
      [CardShadow.MEDIUM]: 'shadow-md',
      [CardShadow.LARGE]: 'shadow-lg',
      [CardShadow.XLARGE]: 'shadow-xl'
    }
    return shadows[shadow] || shadows[CardShadow.MEDIUM]
  }

  // 基础样式类
  const baseClasses = [
    'rounded-xl',
    'border',
    'transition-all',
    'duration-200',
    'ease-in-out',
    getVariantStyles(),
    getSizeStyles(),
    getShadowStyles()
  ]

  // 条件样式类
  const conditionalClasses = [
    hoverable ? 'hover:shadow-lg hover:scale-[1.02] cursor-pointer' : '',
    clickable ? 'cursor-pointer active:scale-[0.98]' : '',
    loading ? 'opacity-60 pointer-events-none' : '',
    className
  ].filter(Boolean).join(' ')

  // 渲染卡片头部
  const renderHeader = () => {
    if (header) {
      return <div className="mb-4">{header}</div>
    }

    if (title || subtitle || extra) {
      return (
        <div className="flex items-start justify-between mb-4">
          <div className="flex-1 min-w-0">
            {title && (
              <h3 className="text-lg font-semibold text-white truncate">
                {title}
              </h3>
            )}
            {subtitle && (
              <p className="text-sm text-slate-400 mt-1 truncate">
                {subtitle}
              </p>
            )}
          </div>
          {extra && (
            <div className="flex items-center ml-4 flex-shrink-0">
              {extra}
            </div>
          )}
        </div>
      )
    }

    return null
  }

  // 渲染封面
  const renderCover = () => {
    if (!cover) return null

    return (
      <div className="mb-4 -mx-6 -mt-6">
        {cover}
      </div>
    )
  }

  // 渲染操作按钮
  const renderActions = () => {
    if (!actions) return null

    return (
      <div className="flex items-center justify-between mt-4 pt-4 border-t border-slate-700">
        <div className="flex items-center space-x-2">
          {actions}
        </div>
      </div>
    )
  }

  // 处理点击事件
  const handleCardClick = (e) => {
    if (clickable && !loading) {
      onClick?.(e)
    }
  }

  return (
    <div
      ref={ref}
      className={`${baseClasses.join(' ')} ${conditionalClasses}`}
      onClick={handleCardClick}
      style={style}
      {...restProps}
    >
      {renderHeader()}
      {renderCover()}
      <div className="flex-1">
        {children}
      </div>
      {renderActions()}
    </div>
  )
})

Card.displayName = 'Card'

/**
 * 统计卡片组件
 */
export const StatCard = ({
  title,
  value,
  change,
  changeType = 'increase',
  icon,
  color = 'blue',
  size = CardSize.MEDIUM,
  loading = false,
  className = '',
  ...restProps
}) => {
  const getColorClasses = () => {
    const colors = {
      blue: 'text-blue-500 bg-blue-500/10',
      green: 'text-emerald-500 bg-emerald-500/10',
      red: 'text-red-500 bg-red-500/10',
      yellow: 'text-amber-500 bg-amber-500/10',
      purple: 'text-purple-500 bg-purple-500/10',
      cyan: 'text-cyan-500 bg-cyan-500/10'
    }
    return colors[color] || colors.blue
  }

  const getChangeColor = () => {
    return changeType === 'increase' ? 'text-emerald-500' : 'text-red-500'
  }

  return (
    <Card
      size={size}
      hoverable
      loading={loading}
      className={`${className}`}
      {...restProps}
    >
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-slate-400">{title}</p>
          <p className="text-2xl font-bold text-white mt-1">
            {loading ? '--' : value}
          </p>
          {change && (
            <p className={`text-sm font-medium mt-2 ${getChangeColor()}`}>
              {changeType === 'increase' ? '↑' : '↓'} {change}
            </p>
          )}
        </div>
        {icon && (
          <div className={`p-3 rounded-lg ${getColorClasses()}`}>
            {icon}
          </div>
        )}
      </div>
    </Card>
  )
}

/**
 * 项目卡片组件
 */
export const ProjectCard = ({
  title,
  description,
  image,
  tags = [],
  stats = {},
  actions = [],
  href,
  loading = false,
  className = '',
  ...restProps
}) => {
  return (
    <Card
      hoverable
      loading={loading}
      className={`${className} group`}
      {...restProps}
    >
      {image && (
        <div className="aspect-video bg-slate-700 rounded-lg mb-4 overflow-hidden">
          {typeof image === 'string' ? (
            <img
              src={image}
              alt={title}
              className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-200"
            />
          ) : (
            image
          )}
        </div>
      )}

      <div className="mb-3">
        <h3 className="text-lg font-semibold text-white mb-2 group-hover:text-blue-400 transition-colors">
          {title}
        </h3>
        <p className="text-sm text-slate-400 line-clamp-2">
          {description}
        </p>
      </div>

      {tags.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-3">
          {tags.map((tag, index) => (
            <span
              key={index}
              className="px-2 py-1 text-xs font-medium bg-slate-700 text-slate-300 rounded-md"
            >
              {tag}
            </span>
          ))}
        </div>
      )}

      {Object.keys(stats).length > 0 && (
        <div className="flex items-center space-x-4 text-sm text-slate-400 mb-3">
          {Object.entries(stats).map(([key, value]) => (
            <div key={key} className="flex items-center">
              <span className="font-medium">{value}</span>
              <span className="ml-1">{key}</span>
            </div>
          ))}
        </div>
      )}

      {actions.length > 0 && (
        <div className="flex items-center justify-between pt-3 border-t border-slate-700">
          <div className="flex items-center space-x-2">
            {actions}
          </div>
          {href && (
            <ExternalLink size={16} className="text-slate-400 hover:text-white transition-colors" />
          )}
        </div>
      )}
    </Card>
  )
}

/**
 * 用户卡片组件
 */
export const UserCard = ({
  name,
  role,
  avatar,
  email,
  status,
  stats = {},
  actions = [],
  loading = false,
  className = '',
  ...restProps
}) => {
  const getStatusColor = () => {
    switch (status) {
      case 'online':
        return 'bg-emerald-500'
      case 'busy':
        return 'bg-amber-500'
      case 'offline':
        return 'bg-slate-500'
      default:
        return 'bg-slate-500'
    }
  }

  return (
    <Card
      loading={loading}
      className={`${className}`}
      {...restProps}
    >
      <div className="flex items-center space-x-4 mb-4">
        <div className="relative">
          {avatar && (
            <div className="w-16 h-16 rounded-full overflow-hidden bg-slate-700">
              {typeof avatar === 'string' ? (
                <img
                  src={avatar}
                  alt={name}
                  className="w-full h-full object-cover"
                />
              ) : (
                avatar
              )}
            </div>
          )}
          {status && (
            <div className={`absolute bottom-0 right-0 w-4 h-4 rounded-full border-2 border-slate-800 ${getStatusColor()}`} />
          )}
        </div>
        <div className="flex-1 min-w-0">
          <h3 className="text-lg font-semibold text-white truncate">
            {name}
          </h3>
          <p className="text-sm text-slate-400">{role}</p>
          {email && (
            <p className="text-sm text-slate-500 truncate">{email}</p>
          )}
        </div>
      </div>

      {Object.keys(stats).length > 0 && (
        <div className="grid grid-cols-3 gap-4 mb-4">
          {Object.entries(stats).map(([key, value]) => (
            <div key={key} className="text-center">
              <div className="text-lg font-semibold text-white">{value}</div>
              <div className="text-xs text-slate-400">{key}</div>
            </div>
          ))}
        </div>
      )}

      {actions.length > 0 && (
        <div className="flex items-center space-x-2">
          {actions}
        </div>
      )}
    </Card>
  )
}

/**
 * 卡片网格容器
 */
export const CardGrid = ({
  children,
  columns = 3,
  gap = 4,
  className = ''
}) => {
  const getGridClasses = () => {
    const gridCols = {
      1: 'grid-cols-1',
      2: 'grid-cols-2',
      3: 'grid-cols-3',
      4: 'grid-cols-4',
      5: 'grid-cols-5',
      6: 'grid-cols-6'
    }
    return gridCols[columns] || gridCols[3]
  }

  const getGapClasses = () => {
    const gaps = {
      1: 'gap-1',
      2: 'gap-2',
      3: 'gap-3',
      4: 'gap-4',
      6: 'gap-6',
      8: 'gap-8'
    }
    return gaps[gap] || gaps[4]
  }

  return (
    <div className={`grid ${getGridClasses()} ${getGapClasses()} ${className}`}>
      {children}
    </div>
  )
}

export {
  CardVariant,
  CardSize,
  CardShadow
}

export default Card