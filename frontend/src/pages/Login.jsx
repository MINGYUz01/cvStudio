/**
 * 登录页面组件
 */

import React, { useState } from 'react'
import { Form, Input, Button, Card, message, Typography, Space } from 'antd'
import { UserOutlined, LockOutlined } from '@ant-design/icons'
import { useDispatch, useSelector } from 'react-redux'
import { useNavigate } from 'react-router-dom'
import { loginStart, loginSuccess, loginFailure } from '@store/authSlice'
import { login } from '@services/auth'
import './Login.css'

const { Title, Text } = Typography

/**
 * 登录页面
 * 包含用户名和密码输入框，以及登录按钮
 */
const Login = () => {
  const [form] = Form.useForm()
  const [loading, setLoading] = useState(false)
  const dispatch = useDispatch()
  const navigate = useNavigate()
  const { loading: authLoading } = useSelector(state => state.auth)

  /**
   * 处理登录表单提交
   * @param {Object} values - 表单数据
   */
  const handleSubmit = async (values) => {
    setLoading(true)
    dispatch(loginStart())
    
    try {
      const response = await login(values)
      
      // 保存token到localStorage
      localStorage.setItem('token', response.access_token)
      
      // 更新Redux状态
      dispatch(loginSuccess({
        user: response.user,
        token: response.access_token
      }))
      
      message.success('登录成功')
      navigate('/dashboard')
    } catch (error) {
      const errorMessage = error.message || '登录失败，请检查用户名和密码'
      dispatch(loginFailure(errorMessage))
      message.error(errorMessage)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="login-container">
      <Card className="login-card">
        <Space direction="vertical" size="large" style={{ width: '100%' }}>
          <div className="login-header">
            <Title level={2} className="login-title">
              CV Studio
            </Title>
            <Text type="secondary">
              计算机视觉任务管理平台
            </Text>
          </div>
          
          <Form
            form={form}
            name="login"
            onFinish={handleSubmit}
            layout="vertical"
            size="large"
          >
            <Form.Item
              name="username"
              rules={[
                { required: true, message: '请输入用户名' },
                { min: 3, message: '用户名至少3个字符' }
              ]}
            >
              <Input
                prefix={<UserOutlined />}
                placeholder="用户名"
                autoComplete="username"
              />
            </Form.Item>
            
            <Form.Item
              name="password"
              rules={[
                { required: true, message: '请输入密码' },
                { min: 6, message: '密码至少6个字符' }
              ]}
            >
              <Input.Password
                prefix={<LockOutlined />}
                placeholder="密码"
                autoComplete="current-password"
              />
            </Form.Item>
            
            <Form.Item>
              <Button
                type="primary"
                htmlType="submit"
                loading={loading || authLoading}
                block
              >
                登录
              </Button>
            </Form.Item>
          </Form>
        </Space>
      </Card>
    </div>
  )
}

export default Login