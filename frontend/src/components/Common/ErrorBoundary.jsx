/**
 * 错误边界组件
 * 捕获并处理React组件中的错误
 */

import React, { Component } from 'react'
import { AlertTriangle, RefreshCw, Home, MessageSquare } from 'lucide-react'

/**
 * 错误类型枚举
 */
export const ErrorType = {
  NETWORK: 'network',           // 网络错误
  PERMISSION: 'permission',     // 权限错误
  VALIDATION: 'validation',     // 验证错误
  RUNTIME: 'runtime',          // 运行时错误
  COMPONENT: 'component',       // 组件错误
  SYSTEM: 'system',            // 系统错误
  TIMEOUT: 'timeout',          // 超时错误
  UNKNOWN: 'unknown'           // 未知错误
}

/**
 * 错误严重级别
 */
export const ErrorSeverity = {
  LOW: 'low',                   // 低级错误
  MEDIUM: 'medium',             // 中级错误
  HIGH: 'high',                 // 高级错误
  CRITICAL: 'critical'          // 严重错误
}

/**
 * 错误边界状态组件
 */
class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      errorId: null,
      errorType: ErrorType.UNKNOWN,
      severity: ErrorSeverity.MEDIUM,
      retryCount: 0
    }
  }

  static getDerivedStateFromError(error) {
    // 更新state，下一次渲染将显示fallback UI
    return {
      hasError: true,
      errorId: `error_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    }
  }

  componentDidCatch(error, errorInfo) {
    // 记录错误信息
    console.error('ErrorBoundary caught an error:', error, errorInfo)

    // 分析错误类型
    const errorType = this.analyzeError(error)
    const severity = this.analyzeSeverity(error, errorInfo)

    this.setState({
      error,
      errorInfo,
      errorType,
      severity
    })

    // 调用错误回调（如果提供）
    if (this.props.onError) {
      this.props.onError({
        error,
        errorInfo,
        errorId: this.state.errorId,
        errorType,
        severity
      })
    }

    // 发送错误报告（如果启用）
    if (this.props.reportErrors && !this.props.isDevelopment) {
      this.reportError(error, errorInfo)
    }
  }

  /**
   * 分析错误类型
   */
  analyzeError = (error) => {
    const message = error.message.toLowerCase()
    const stack = error.stack?.toLowerCase() || ''

    if (message.includes('network') || message.includes('fetch') || message.includes('axios')) {
      return ErrorType.NETWORK
    }
    if (message.includes('permission') || message.includes('unauthorized') || message.includes('forbidden')) {
      return ErrorType.PERMISSION
    }
    if (message.includes('validation') || message.includes('invalid') || message.includes('required')) {
      return ErrorType.VALIDATION
    }
    if (message.includes('timeout') || message.includes('aborted')) {
      return ErrorType.TIMEOUT
    }
    if (stack.includes('react') || stack.includes('component')) {
      return ErrorType.COMPONENT
    }
    if (message.includes('system') || message.includes('memory') || message.includes('disk')) {
      return ErrorType.SYSTEM
    }

    return ErrorType.RUNTIME
  }

  /**
   * 分析错误严重程度
   */
  analyzeSeverity = (error, errorInfo) => {
    const message = error.message.toLowerCase()

    if (message.includes('critical') || message.includes('fatal') || message.includes('crash')) {
      return ErrorSeverity.CRITICAL
    }
    if (message.includes('error') || message.includes('failed') || message.includes('exception')) {
      return ErrorSeverity.HIGH
    }
    if (message.includes('warning') || message.includes('deprecated')) {
      return ErrorSeverity.LOW
    }

    return ErrorSeverity.MEDIUM
  }

  /**
   * 发送错误报告
   */
  reportError = async (error, errorInfo) => {
    try {
      const reportData = {
        errorId: this.state.errorId,
        message: error.message,
        stack: error.stack,
        componentStack: errorInfo.componentStack,
        errorType: this.state.errorType,
        severity: this.state.severity,
        userAgent: navigator.userAgent,
        url: window.location.href,
        timestamp: new Date().toISOString(),
        userInfo: this.props.userInfo || {}
      }

      // 这里可以集成错误监控服务
      // await sendErrorReport(reportData)
      console.log('Error report:', reportData)
    } catch (reportError) {
      console.error('Failed to report error:', reportError)
    }
  }

  /**
   * 重试功能
   */
  handleRetry = () => {
    const { maxRetries = 3 } = this.props

    if (this.state.retryCount < maxRetries) {
      this.setState(prevState => ({
        hasError: false,
        error: null,
        errorInfo: null,
        errorType: ErrorType.UNKNOWN,
        severity: ErrorSeverity.MEDIUM,
        retryCount: prevState.retryCount + 1
      }))
    }
  }

  /**
   * 重置错误状态
   */
  handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      errorType: ErrorType.UNKNOWN,
      severity: ErrorSeverity.MEDIUM,
      retryCount: 0
    })
  }

  /**
   * 返回首页
   */
  goHome = () => {
    window.location.href = '/'
  }

  render() {
    const {
      children,
      fallback,
      showErrorDetails = false,
      maxRetries = 3,
      isDevelopment = process.env.NODE_ENV === 'development'
    } = this props

    if (this.state.hasError) {
      // 自定义fallback UI
      if (fallback) {
        return fallback({
          error: this.state.error,
          errorInfo: this.state.errorInfo,
          retry: this.handleRetry,
          reset: this.handleReset
        })
      }

      // 默认错误UI
      return (
        <div className="error-boundary">
          <div className="error-container">
            <div className="error-icon">
              <AlertTriangle size={64} />
            </div>

            <div className="error-content">
              <h1 className="error-title">
                {this.getErrorTitle()}
              </h1>

              <p className="error-description">
                {this.getErrorDescription()}
              </p>

              {this.state.errorId && (
                <p className="error-id">
                  错误ID: {this.state.errorId}
                </p>
              )}

              {this.state.severity === ErrorSeverity.CRITICAL && (
                <div className="error-severity critical">
                  <AlertTriangle size={16} />
                  <span>严重错误 - 请立即联系管理员</span>
                </div>
              )}
            </div>

            <div className="error-actions">
              {this.state.retryCount < maxRetries && (
                <button
                  className="error-button primary"
                  onClick={this.handleRetry}
                >
                  <RefreshCw size={16} />
                  重试 ({this.state.retryCount}/{maxRetries})
                </button>
              )}

              <button
                className="error-button secondary"
                onClick={this.handleReset}
              >
                重置
              </button>

              <button
                className="error-button secondary"
                onClick={this.goHome}
              >
                <Home size={16} />
                返回首页
              </button>
            </div>

            {(showErrorDetails || isDevelopment) && this.state.error && (
              <details className="error-details">
                <summary>
                  <MessageSquare size={16} />
                  错误详情
                </summary>

                <div className="error-details-content">
                  <div className="error-section">
                    <h4>错误信息</h4>
                    <pre className="error-message">
                      {this.state.error.toString()}
                    </pre>
                  </div>

                  <div className="error-section">
                    <h4>错误堆栈</h4>
                    <pre className="error-stack">
                      {this.state.error.stack}
                    </pre>
                  </div>

                  {this.state.errorInfo && (
                    <div className="error-section">
                      <h4>组件堆栈</h4>
                      <pre className="component-stack">
                        {this.state.errorInfo.componentStack}
                      </pre>
                    </div>
                  )}

                  <div className="error-section">
                    <h4>诊断信息</h4>
                    <div className="error-diagnostics">
                      <p><strong>错误类型:</strong> {this.state.errorType}</p>
                      <p><strong>严重程度:</strong> {this.state.severity}</p>
                      <p><strong>重试次数:</strong> {this.state.retryCount}</p>
                      <p><strong>用户代理:</strong> {navigator.userAgent}</p>
                      <p><strong>页面URL:</strong> {window.location.href}</p>
                      <p><strong>时间戳:</strong> {new Date().toISOString()}</p>
                    </div>
                  </div>
                </div>
              </details>
            )}
          </div>

          <style jsx>{`
            .error-boundary {
              min-height: 100vh;
              display: flex;
              align-items: center;
              justify-content: center;
              background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
              padding: 20px;
            }

            .error-container {
              max-width: 600px;
              width: 100%;
              background: rgba(30, 41, 59, 0.95);
              backdrop-filter: blur(10px);
              border: 1px solid rgba(59, 130, 246, 0.2);
              border-radius: 16px;
              padding: 40px;
              text-align: center;
              box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1),
                         0 10px 10px -5px rgba(0, 0, 0, 0.04);
            }

            .error-icon {
              display: flex;
              justify-content: center;
              margin-bottom: 24px;
              color: #ef4444;
            }

            .error-content {
              margin-bottom: 32px;
            }

            .error-title {
              font-size: 24px;
              font-weight: 600;
              color: #f8fafc;
              margin-bottom: 12px;
            }

            .error-description {
              color: #cbd5e1;
              font-size: 16px;
              line-height: 1.6;
              margin-bottom: 8px;
            }

            .error-id {
              font-family: 'Monaco', 'Menlo', monospace;
              font-size: 12px;
              color: #64748b;
              background: rgba(15, 23, 42, 0.5);
              padding: 4px 8px;
              border-radius: 4px;
              display: inline-block;
            }

            .error-severity {
              display: flex;
              align-items: center;
              justify-content: center;
              gap: 8px;
              margin-top: 16px;
              padding: 8px 16px;
              border-radius: 8px;
              font-size: 14px;
              font-weight: 500;
            }

            .error-severity.critical {
              background: rgba(239, 68, 68, 0.1);
              color: #fca5a5;
              border: 1px solid rgba(239, 68, 68, 0.3);
            }

            .error-actions {
              display: flex;
              gap: 12px;
              justify-content: center;
              flex-wrap: wrap;
              margin-bottom: 24px;
            }

            .error-button {
              display: flex;
              align-items: center;
              gap: 8px;
              padding: 12px 20px;
              border: none;
              border-radius: 8px;
              font-size: 14px;
              font-weight: 500;
              cursor: pointer;
              transition: all 0.2s;
            }

            .error-button.primary {
              background: #3b82f6;
              color: white;
            }

            .error-button.primary:hover {
              background: #2563eb;
            }

            .error-button.secondary {
              background: rgba(100, 116, 139, 0.2);
              color: #cbd5e1;
              border: 1px solid rgba(100, 116, 139, 0.3);
            }

            .error-button.secondary:hover {
              background: rgba(100, 116, 139, 0.3);
            }

            .error-details {
              text-align: left;
              background: rgba(15, 23, 42, 0.5);
              border: 1px solid rgba(100, 116, 139, 0.3);
              border-radius: 8px;
              overflow: hidden;
            }

            .error-details summary {
              padding: 16px;
              cursor: pointer;
              display: flex;
              align-items: center;
              gap: 8px;
              color: #cbd5e1;
              font-weight: 500;
              background: rgba(15, 23, 42, 0.8);
            }

            .error-details-content {
              padding: 16px;
              max-height: 400px;
              overflow-y: auto;
            }

            .error-section {
              margin-bottom: 24px;
            }

            .error-section:last-child {
              margin-bottom: 0;
            }

            .error-section h4 {
              color: #f8fafc;
              margin-bottom: 8px;
              font-size: 14px;
              font-weight: 600;
            }

            .error-message,
            .error-stack,
            .component-stack {
              background: #0f172a;
              color: #e2e8f0;
              padding: 12px;
              border-radius: 6px;
              font-family: 'Monaco', 'Menlo', monospace;
              font-size: 12px;
              line-height: 1.4;
              overflow-x: auto;
              white-space: pre-wrap;
              word-break: break-all;
            }

            .error-diagnostics {
              background: rgba(15, 23, 42, 0.8);
              padding: 12px;
              border-radius: 6px;
            }

            .error-diagnostics p {
              margin: 4px 0;
              font-size: 13px;
              color: #cbd5e1;
            }

            .error-diagnostics strong {
              color: #94a3b8;
            }

            @media (max-width: 640px) {
              .error-container {
                padding: 24px;
                margin: 16px;
              }

              .error-actions {
                flex-direction: column;
              }

              .error-button {
                width: 100%;
                justify-content: center;
              }
            }
          `}</style>
        </div>
      )
    }

    return children
  }
}

/**
 * 错误消息组件
 */
export const ErrorMessage = ({
  type = ErrorType.RUNTIME,
  message,
  description,
  showIcon = true,
  className = '',
  action
}) => {
  const getErrorConfig = () => {
    const configs = {
      [ErrorType.NETWORK]: {
        color: '#f59e0b',
        icon: '🌐',
        title: '网络错误'
      },
      [ErrorType.PERMISSION]: {
        color: '#ef4444',
        icon: '🔒',
        title: '权限错误'
      },
      [ErrorType.VALIDATION]: {
        color: '#8b5cf6',
        icon: '⚠️',
        title: '验证错误'
      },
      [ErrorType.RUNTIME]: {
        color: '#ef4444',
        icon: '💥',
        title: '运行时错误'
      },
      [ErrorType.TIMEOUT]: {
        color: '#f59e0b',
        icon: '⏰',
        title: '超时错误'
      }
    }
    return configs[type] || configs[ErrorType.UNKNOWN]
  }

  const config = getErrorConfig()

  return (
    <div className={`error-message ${className}`}>
      {showIcon && (
        <span className="error-message-icon" style={{ color: config.color }}>
          {config.icon}
        </span>
      )}
      <div className="error-message-content">
        <h4 className="error-message-title" style={{ color: config.color }}>
          {config.title}
        </h4>
        {message && (
          <p className="error-message-text">{message}</p>
        )}
        {description && (
          <p className="error-message-description">{description}</p>
        )}
        {action && (
          <div className="error-message-action">
            {action}
          </div>
        )}
      </div>
    </div>
  )
}

export default ErrorBoundary