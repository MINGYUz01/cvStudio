/**
 * 仪表盘页面组件
 */

import React from 'react'
import { Card, Row, Col, Statistic, Typography, Space } from 'antd'
import {
  DatabaseOutlined,
  NodeIndexOutlined,
  PlayCircleOutlined,
  ExperimentOutlined
} from '@ant-design/icons'

const { Title } = Typography

/**
 * 仪表盘页面
 * 显示系统概览和统计数据
 */
const Dashboard = () => {
  return (
    <div className="dashboard">
      <Title level={2}>仪表盘</Title>
      
      <Row gutter={[16, 16]}>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="数据集总数"
              value={0}
              prefix={<DatabaseOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="模型总数"
              value={0}
              prefix={<NodeIndexOutlined />}
              valueStyle={{ color: '#cf1322' }}
            />
          </Card>
        </Col>
        
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="训练任务"
              value={0}
              prefix={<PlayCircleOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="推理任务"
              value={0}
              prefix={<ExperimentOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
      </Row>
      
      <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
        <Col xs={24} lg={12}>
          <Card title="最近的数据集" extra={<a href="/datasets">查看全部</a>}>
            <div style={{ textAlign: 'center', padding: '40px 0', color: '#999' }}>
              暂无数据集
            </div>
          </Card>
        </Col>
        
        <Col xs={24} lg={12}>
          <Card title="最近的模型" extra={<a href="/models">查看全部</a>}>
            <div style={{ textAlign: 'center', padding: '40px 0', color: '#999' }}>
              暂无模型
            </div>
          </Card>
        </Col>
      </Row>
      
      <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
        <Col span={24}>
          <Card title="系统状态">
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>后端服务：运行中</div>
              <div>数据库：连接正常</div>
              <div>存储空间：充足</div>
            </Space>
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default Dashboard