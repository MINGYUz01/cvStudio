/**
 * 路由配置文件
 * 集中管理所有路由配置
 */

import React from 'react'
import { Navigate } from 'react-router-dom'
import ProtectedRoute from '@components/Common/ProtectedRoute'
import Layout from '@components/Layout'

// 懒加载页面组件
import Login from '@pages/Login'
import Dashboard from '@pages/Dashboard'
import Datasets from '@pages/Datasets'
import Models from '@pages/Models'
import Training from '@pages/Training'
import Inference from '@pages/Inference'
import Settings from '@pages/Settings'

/**
 * 公共路由配置（不需要登录就可以访问）
 */
export const publicRoutes = [
  {
    path: '/login',
    element: <Login />,
    meta: {
      title: '登录 - CV Studio',
      requireAuth: false
    }
  }
]

/**
 * 受保护的路由配置（需要登录才能访问）
 */
export const protectedRoutes = [
  {
    path: '/',
    element: <Navigate to="/dashboard" replace />,
    meta: {
      title: 'CV Studio',
      requireAuth: true
    }
  },
  {
    path: '/dashboard',
    element: <Dashboard />,
    meta: {
      title: '仪表盘 - CV Studio',
      requireAuth: true
    }
  },
  {
    path: '/datasets',
    element: <Datasets />,
    meta: {
      title: '数据集管理 - CV Studio',
      requireAuth: true
    }
  },
  {
    path: '/models',
    element: <Models />,
    meta: {
      title: '模型构建 - CV Studio',
      requireAuth: true
    }
  },
  {
    path: '/training',
    element: <Training />,
    meta: {
      title: '训练管理 - CV Studio',
      requireAuth: true
    }
  },
  {
    path: '/inference',
    element: <Inference />,
    meta: {
      title: '推理测试 - CV Studio',
      requireAuth: true
    }
  },
  {
    path: '/settings',
    element: <Settings />,
    meta: {
      title: '设置 - CV Studio',
      requireAuth: true
    }
  }
]

/**
 * 路由配置合并
 */
export const allRoutes = [
  ...publicRoutes,
  ...protectedRoutes
]

/**
 * 包装受保护的路由
 * @param {Object} route - 路由对象
 * @returns {Object} 包装后的路由对象
 */
const wrapProtectedRoute = (route) => ({
  ...route,
  element: (
    <ProtectedRoute>
      <Layout>
        {route.element}
      </Layout>
    </ProtectedRoute>
  )
})

/**
 * 获取最终的路由配置
 */
export const getRouteConfig = () => {
  // 公共路由保持原样
  const finalPublicRoutes = publicRoutes

  // 受保护的路由需要包装
  const finalProtectedRoutes = protectedRoutes.map(wrapProtectedRoute)

  return [
    ...finalPublicRoutes,
    ...finalProtectedRoutes
  ]
}

/**
 * 根据路径获取路由信息
 * @param {string} pathname - 当前路径
 * @returns {Object|null} 路由信息或null
 */
export const getRouteByPath = (pathname) => {
  return allRoutes.find(route => {
    // 精确匹配
    if (route.path === pathname) return true

    // 处理动态路由（将来扩展）
    const routeSegments = route.path.split('/')
    const pathSegments = pathname.split('/')

    if (routeSegments.length !== pathSegments.length) return false

    return routeSegments.every((segment, index) => {
      // 动态参数检查
      if (segment.startsWith(':')) return true
      return segment === pathSegments[index]
    })
  }) || null
}

/**
 * 获取导航菜单配置
 * @returns {Array} 导航菜单配置
 */
export const getNavigationConfig = () => {
  return [
    {
      name: '仪表盘',
      path: '/dashboard',
      icon: 'LayoutDashboard',
      order: 1
    },
    {
      name: '数据集管理',
      path: '/datasets',
      icon: 'Database',
      order: 2
    },
    {
      name: '模型构建',
      path: '/models',
      icon: 'Box',
      order: 3
    },
    {
      name: '训练管理',
      path: '/training',
      icon: 'Activity',
      order: 4
    },
    {
      name: '推理测试',
      path: '/inference',
      icon: 'PlayCircle',
      order: 5
    },
    {
      name: '设置',
      path: '/settings',
      icon: 'Settings',
      order: 6
    }
  ]
}

export default {
  publicRoutes,
  protectedRoutes,
  allRoutes,
  getRouteConfig,
  getRouteByPath,
  getNavigationConfig
}