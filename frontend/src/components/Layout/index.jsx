/**
 * 主布局组件
 */

import React, { useState } from 'react'
import { Layout, Menu, Button, Avatar, Dropdown, Space } from 'antd'
import {
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  DashboardOutlined,
  DatabaseOutlined,
  NodeIndexOutlined,
  PlayCircleOutlined,
  ExperimentOutlined,
  SettingOutlined,
  UserOutlined,
  LogoutOutlined
} from '@ant-design/icons'
import { Outlet, useNavigate, useLocation } from 'react-router-dom'
import { useSelector, useDispatch } from 'react-redux'
import { logout } from '@store/authSlice'
import './index.css'

const { Header, Sider, Content } = Layout

/**
 * 主布局组件
 * 包含侧边栏、顶部导航和内容区域
 */
const LayoutComponent = () => {
  const [collapsed, setCollapsed] = useState(false)
  const navigate = useNavigate()
  const location = useLocation()
  const dispatch = useDispatch()
  const { user } = useSelector(state => state.auth)

  // 菜单项配置
  const menuItems = [
    {
      key: '/dashboard',
      icon: <DashboardOutlined />,
      label: '仪表盘'
    },
    {
      key: '/datasets',
      icon: <DatabaseOutlined />,
      label: '数据集管理'
    },
    {
      key: '/models',
      icon: <NodeIndexOutlined />,
      label: '模型构建'
    },
    {
      key: '/training',
      icon: <PlayCircleOutlined />,
      label: '训练管理'
    },
    {
      key: '/inference',
      icon: <ExperimentOutlined />,
      label: '推理测试'
    },
    {
      key: '/settings',
      icon: <SettingOutlined />,
      label: '设置'
    }
  ]

  // 用户下拉菜单
  const userMenuItems = [
    {
      key: 'profile',
      icon: <UserOutlined />,
      label: '个人资料'
    },
    {
      type: 'divider'
    },
    {
      key: 'logout',
      icon: <LogoutOutlined />,
      label: '退出登录'
    }
  ]

  /**
   * 处理菜单点击
   * @param {Object} item - 菜单项
   */
  const handleMenuClick = ({ key }) => {
    navigate(key)
  }

  /**
   * 处理用户菜单点击
   * @param {Object} item - 菜单项
   */
  const handleUserMenuClick = ({ key }) => {
    if (key === 'logout') {
      dispatch(logout())
      localStorage.removeItem('token')
      navigate('/login')
    } else if (key === 'profile') {
      navigate('/settings/profile')
    }
  }

  return (
    <Layout className="app-layout">
      <Sider
        trigger={null}
        collapsible
        collapsed={collapsed}
        className="app-sider"
        width={240}
      >
        <div className="app-logo">
          <h1 className={collapsed ? 'logo-collapsed' : 'logo-expanded'}>
            {collapsed ? 'CV' : 'CV Studio'}
          </h1>
        </div>
        
        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[location.pathname]}
          items={menuItems}
          onClick={handleMenuClick}
        />
      </Sider>
      
      <Layout className="site-layout">
        <Header className="app-header">
          <div className="header-left">
            <Button
              type="text"
              icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
              onClick={() => setCollapsed(!collapsed)}
              className="trigger"
            />
          </div>
          
          <div className="header-right">
            <Dropdown
              menu={{
                items: userMenuItems,
                onClick: handleUserMenuClick
              }}
              placement="bottomRight"
            >
              <Space className="user-info">
                <Avatar icon={<UserOutlined />} />
                <span className="username">{user?.username || '用户'}</span>
              </Space>
            </Dropdown>
          </div>
        </Header>
        
        <Content className="app-content">
          <div className="content-wrapper">
            <Outlet />
          </div>
        </Content>
      </Layout>
    </Layout>
  )
}

export default LayoutComponent