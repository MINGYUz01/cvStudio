/**
 * 训练管理页面组件
 */

import React from 'react'
import { Typography, Empty } from 'antd'

const { Title } = Typography

/**
 * 训练管理页面
 * 占位符组件，后续会实现完整功能
 */
const Training = () => {
  return (
    <div className="training">
      <Title level={2}>训练管理</Title>
      <Empty 
        description="训练管理功能正在开发中..."
        style={{ marginTop: 60 }}
      />
    </div>
  )
}

export default Training