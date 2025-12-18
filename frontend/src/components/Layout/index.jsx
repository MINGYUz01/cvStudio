/**
 * 主布局组件 - 采用现代化深色主题设计
 */

import React, { useState } from 'react'
import { Outlet, useNavigate, useLocation } from 'react-router-dom'
import { useSelector, useDispatch } from 'react-redux'
import { logout } from '@store/authSlice'
import {
  LayoutDashboard,
  Database,
  Box,
  Activity,
  PlayCircle,
  Settings,
  Menu,
  X,
  User,
  LogOut,
  ChevronRight
} from 'lucide-react'
import './index.css'

const Layout = ({ children }) => {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const navigate = useNavigate()
  const location = useLocation()
  const dispatch = useDispatch()
  const { user } = useSelector(state => state.auth)

  // 导航菜单配置
  const navigation = [
    {
      name: '仪表盘',
      href: '/dashboard',
      icon: LayoutDashboard,
      current: location.pathname === '/dashboard'
    },
    {
      name: '数据集管理',
      href: '/datasets',
      icon: Database,
      current: location.pathname === '/datasets'
    },
    {
      name: '模型构建',
      href: '/models',
      icon: Box,
      current: location.pathname === '/models'
    },
    {
      name: '训练管理',
      href: '/training',
      icon: Activity,
      current: location.pathname === '/training'
    },
    {
      name: '推理测试',
      href: '/inference',
      icon: PlayCircle,
      current: location.pathname === '/inference'
    },
    {
      name: '设置',
      href: '/settings',
      icon: Settings,
      current: location.pathname === '/settings'
    }
  ]

  // 处理导航点击
  const handleNavigate = (href) => {
    navigate(href)
    setMobileMenuOpen(false)
  }

  // 处理用户登出
  const handleLogout = () => {
    dispatch(logout())
    localStorage.removeItem('token')
    navigate('/login')
  }

  return (
    <div className="app-layout">
      {/* 侧边栏 */}
      <aside className={`sidebar ${sidebarOpen ? 'sidebar-open' : 'sidebar-closed'} ${mobileMenuOpen ? 'mobile-open' : ''}`}>
        {/* Logo区域 */}
        <div className="sidebar-header">
          <div className="logo">
            <div className="logo-icon">
              <Activity size={24} className="text-white" />
            </div>
            <h1 className={`logo-text ${sidebarOpen ? 'block' : 'hidden'}`}>
              CV Studio
            </h1>
          </div>
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="sidebar-toggle"
          >
            <Menu size={20} />
          </button>
        </div>

        {/* 导航菜单 */}
        <nav className="sidebar-nav">
          <div className="nav-section">
            <h3 className="nav-section-title">核心功能</h3>
            <div className="nav-items">
              {navigation.slice(0, 2).map((item) => (
                <button
                  key={item.name}
                  onClick={() => handleNavigate(item.href)}
                  className={`nav-item ${item.current ? 'nav-item-active' : ''}`}
                >
                  <item.icon size={20} className="nav-icon" />
                  <span className={`nav-text ${sidebarOpen ? 'block' : 'hidden'}`}>
                    {item.name}
                  </span>
                  {item.current && <ChevronRight size={16} className="nav-arrow" />}
                </button>
              ))}
            </div>
          </div>

          <div className="nav-section">
            <h3 className="nav-section-title">工作站</h3>
            <div className="nav-items">
              {navigation.slice(2, 5).map((item) => (
                <button
                  key={item.name}
                  onClick={() => handleNavigate(item.href)}
                  className={`nav-item ${item.current ? 'nav-item-active' : ''}`}
                >
                  <item.icon size={20} className="nav-icon" />
                  <span className={`nav-text ${sidebarOpen ? 'block' : 'hidden'}`}>
                    {item.name}
                  </span>
                  {item.current && <ChevronRight size={16} className="nav-arrow" />}
                </button>
              ))}
            </div>
          </div>
        </nav>

        {/* 用户区域 */}
        <div className="sidebar-footer">
          <div className="user-menu">
            <button
              onClick={() => handleNavigate('/settings')}
              className={`nav-item ${location.pathname === '/settings' ? 'nav-item-active' : ''}`}
            >
              <Settings size={20} className="nav-icon" />
              <span className={`nav-text ${sidebarOpen ? 'block' : 'hidden'}`}>
                设置
              </span>
            </button>
          </div>

          {sidebarOpen && (
            <div className="user-info">
              <div className="user-avatar">
                <User size={18} />
              </div>
              <div className="user-details">
                <p className="user-name">{user?.username || '用户'}</p>
                <p className="user-role">管理员</p>
              </div>
              <button
                onClick={handleLogout}
                className="logout-btn"
                title="退出登录"
              >
                <LogOut size={16} />
              </button>
            </div>
          )}
        </div>
      </aside>

      {/* 主内容区域 */}
      <div className="main-content">
        {/* 顶部导航栏 */}
        <header className="top-header">
          <div className="header-left">
            {/* 移动端菜单按钮 */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="mobile-menu-btn"
            >
              {mobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
            </button>

            {/* 页面标题 */}
            <h2 className="page-title">
              {navigation.find(item => item.current)?.name || 'CV Studio'}
            </h2>
          </div>

          <div className="header-right">
            {/* 系统状态指示器 */}
            <div className="status-indicator">
              <div className="status-dot status-online"></div>
              <span className="status-text">系统正常</span>
            </div>

            {/* GPU状态 */}
            <div className="gpu-status">
              <div className="gpu-indicator"></div>
              <span className="gpu-text">GPU: 就绪</span>
            </div>
          </div>
        </header>

        {/* 内容区域 */}
        <main className="content-area">
          <div className="content-wrapper">
            {children || <Outlet />}
          </div>
        </main>
      </div>

      {/* 移动端遮罩 */}
      {mobileMenuOpen && (
        <div
          className="mobile-overlay"
          onClick={() => setMobileMenuOpen(false)}
        ></div>
      )}
    </div>
  )
}

export default Layout