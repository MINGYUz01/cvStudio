import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { Provider } from 'react-redux'
import { ConfigProvider } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import { store } from './store'
import Layout from '@components/Layout'
import Login from '@pages/Login'
import Dashboard from '@pages/Dashboard'
import Datasets from '@pages/Datasets'
import Models from '@pages/Models'
import Training from '@pages/Training'
import Inference from '@pages/Inference'
import Settings from '@pages/Settings'
import ProtectedRoute from '@components/Common/ProtectedRoute'
import './App.css'

/**
 * CV Studio 主应用组件
 * 包含路由配置和全局状态管理
 */
function App() {
  return (
    <Provider store={store}>
      <ConfigProvider locale={zhCN}>
        <Router>
          <div className="App">
            <Routes>
              {/* 登录页面 */}
              <Route path="/login" element={<Login />} />

              {/* 受保护的路由 */}
              <Route path="/" element={
                <ProtectedRoute>
                  <Layout />
                </ProtectedRoute>
              }>
                <Route index element={<Dashboard />} />
                <Route path="dashboard" element={<Dashboard />} />
                <Route path="datasets" element={<Datasets />} />
                <Route path="models" element={<Models />} />
                <Route path="training" element={<Training />} />
                <Route path="inference" element={<Inference />} />
                <Route path="settings" element={<Settings />} />
              </Route>
            </Routes>
          </div>
        </Router>
      </ConfigProvider>
    </Provider>
  )
}

export default App
