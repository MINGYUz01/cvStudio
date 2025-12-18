import React, { useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom'
import { Provider } from 'react-redux'
import { ConfigProvider, theme } from 'antd'
import { store } from './store'
import { getRouteConfig } from './router'
import './App.css'

/**
 * 应用主题配置
 * 配置Ant Design的深色主题
 */
const antdTheme = {
  algorithm: theme.darkAlgorithm,
  token: {
    colorPrimary: '#3b82f6',
    colorSuccess: '#10b981',
    colorWarning: '#f59e0b',
    colorError: '#ef4444',
    colorInfo: '#6366f1',
    borderRadius: 8,
    wireframe: false
  },
  components: {
    Layout: {
      headerBg: 'transparent',
      siderBg: 'transparent'
    },
    Menu: {
      darkItemBg: 'transparent',
      darkSubMenuItemBg: 'transparent'
    }
  }
}

/**
 * 页面标题管理组件
 */
const PageTitleManager = () => {
  const location = useLocation()

  useEffect(() => {
    // 根据路由设置页面标题
    const routePath = location.pathname
    let title = 'CV Studio'

    // 简单的标题映射
    const titleMap = {
      '/login': '登录 - CV Studio',
      '/dashboard': '仪表盘 - CV Studio',
      '/datasets': '数据集管理 - CV Studio',
      '/models': '模型构建 - CV Studio',
      '/training': '训练管理 - CV Studio',
      '/inference': '推理测试 - CV Studio',
      '/settings': '设置 - CV Studio'
    }

    if (titleMap[routePath]) {
      title = titleMap[routePath]
    }

    document.title = title
  }, [location.pathname])

  return null
}

/**
 * CV Studio 主应用组件
 * 包含路由配置和全局状态管理
 */
const App = () => {
  // 获取路由配置
  const routeConfig = getRouteConfig()

  return (
    <Provider store={store}>
      <ConfigProvider theme={antdTheme}>
        <Router>
          <PageTitleManager />
          <div className="App">
            <Routes>
              {routeConfig.map((route, index) => (
                <Route
                  key={index}
                  path={route.path}
                  element={route.element}
                />
              ))}
            </Routes>
          </div>
        </Router>
      </ConfigProvider>
    </Provider>
  )
}

export default App
