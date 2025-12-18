/**
 * 登录页面组件 - 现代化设计
 */

import React, { useState } from 'react'
import { useDispatch, useSelector } from 'react-redux'
import { useNavigate, Link } from 'react-router-dom'
import { loginStart, loginSuccess, loginFailure } from '@store/authSlice'
import { login } from '@services/auth'
import {
  User,
  Lock,
  Eye,
  EyeOff,
  Activity,
  AlertCircle,
  CheckCircle2,
  Sparkles,
  Zap,
  Shield,
  Cpu
} from 'lucide-react'
import './Login.css'

/**
 * 登录页面
 * 现代化设计的登录界面
 */
const Login = () => {
  const [formData, setFormData] = useState({
    username: '',
    password: ''
  })
  const [showPassword, setShowPassword] = useState(false)
  const [errors, setErrors] = useState({})
  const [loading, setLoading] = useState(false)
  const dispatch = useDispatch()
  const navigate = useNavigate()
  const { loading: authLoading, error } = useSelector(state => state.auth)

  /**
   * 处理输入变化
   * @param {Object} e - 事件对象
   */
  const handleInputChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value
    }))

    // 清除错误
    if (errors[name]) {
      setErrors(prev => ({
        ...prev,
        [name]: ''
      }))
    }
  }

  /**
   * 表单验证
   * @returns {boolean} 验证是否通过
   */
  const validateForm = () => {
    const newErrors = {}

    if (!formData.username.trim()) {
      newErrors.username = '请输入用户名'
    } else if (formData.username.length < 3) {
      newErrors.username = '用户名至少3个字符'
    }

    if (!formData.password.trim()) {
      newErrors.password = '请输入密码'
    } else if (formData.password.length < 6) {
      newErrors.password = '密码至少6个字符'
    }

    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  /**
   * 处理登录表单提交
   * @param {Object} e - 事件对象
   */
  const handleSubmit = async (e) => {
    e.preventDefault()

    if (!validateForm()) {
      return
    }

    setLoading(true)
    dispatch(loginStart())

    try {
      const response = await login(formData)

      // 保存token到localStorage
      localStorage.setItem('token', response.access_token)

      // 更新Redux状态
      dispatch(loginSuccess({
        user: response.user,
        token: response.access_token
      }))

      // 登录成功，跳转到仪表盘
      navigate('/dashboard')
    } catch (error) {
      const errorMessage = error.response?.data?.message || error.message || '登录失败，请检查用户名和密码'
      dispatch(loginFailure(errorMessage))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="login-container">
      {/* 背景效果 */}
      <div className="login-background">
        <div className="login-grid"></div>
        <div className="login-particles">
          {[...Array(6)].map((_, i) => (
            <div
              key={i}
              className="particle"
              style={{
                left: `${Math.random() * 100}%`,
                animationDelay: `${Math.random() * 5}s`,
                animationDuration: `${15 + Math.random() * 10}s`
              }}
            />
          ))}
        </div>
      </div>

      {/* 登录卡片 */}
      <div className="login-card">
        {/* 头部 */}
        <div className="login-header">
          <div className="login-logo">
            <div className="logo-icon">
              <Activity size={32} className="text-white" />
            </div>
            <h1 className="logo-text">CV Studio</h1>
          </div>
          <p className="login-subtitle">计算机视觉任务管理平台</p>
          <div className="login-features">
            <div className="feature-item">
              <Sparkles size={16} />
              <span>智能管理</span>
            </div>
            <div className="feature-item">
              <Zap size={16} />
              <span>高效训练</span>
            </div>
            <div className="feature-item">
              <Shield size={16} />
              <span>安全可靠</span>
            </div>
          </div>
        </div>

        {/* 登录表单 */}
        <form onSubmit={handleSubmit} className="login-form">
          {/* 用户名输入 */}
          <div className="form-group">
            <div className="input-wrapper">
              <User size={20} className="input-icon" />
              <input
                type="text"
                name="username"
                value={formData.username}
                onChange={handleInputChange}
                placeholder="用户名"
                className={`form-input ${errors.username ? 'error' : ''}`}
                autoComplete="username"
              />
            </div>
            {errors.username && (
              <div className="error-message">
                <AlertCircle size={14} />
                {errors.username}
              </div>
            )}
          </div>

          {/* 密码输入 */}
          <div className="form-group">
            <div className="input-wrapper">
              <Lock size={20} className="input-icon" />
              <input
                type={showPassword ? 'text' : 'password'}
                name="password"
                value={formData.password}
                onChange={handleInputChange}
                placeholder="密码"
                className={`form-input ${errors.password ? 'error' : ''}`}
                autoComplete="current-password"
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="password-toggle"
              >
                {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
              </button>
            </div>
            {errors.password && (
              <div className="error-message">
                <AlertCircle size={14} />
                {errors.password}
              </div>
            )}
          </div>

          {/* 全局错误信息 */}
          {error && (
            <div className="global-error">
              <AlertCircle size={16} />
              {error}
            </div>
          )}

          {/* 登录按钮 */}
          <button
            type="submit"
            disabled={loading || authLoading}
            className="login-button"
          >
            {loading || authLoading ? (
              <div className="loading-spinner">
                <div className="spinner"></div>
                <span>登录中...</span>
              </div>
            ) : (
              <>
                <span>登录</span>
                <Cpu size={18} />
              </>
            )}
          </button>

          {/* 其他选项 */}
          <div className="login-options">
            <Link to="/forgot-password" className="forgot-link">
              忘记密码？
            </Link>
          </div>
        </form>

        {/* 底部 */}
        <div className="login-footer">
          <div className="demo-info">
            <div className="demo-badge">
              <Sparkles size={14} />
              演示模式
            </div>
            <p className="demo-text">
              试用账号: <code>demo</code> / <code>123456</code>
            </p>
          </div>
          <p className="copyright">
            © 2024 CV Studio. All rights reserved.
          </p>
        </div>
      </div>
    </div>
  )
}

export default Login