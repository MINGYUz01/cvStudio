/**
 * 模型构建页面组件
 */

import React from 'react'
import { Typography, Empty } from 'antd'

const { Title } = Typography

/**
 * 模型构建页面
 * 占位符组件，后续会实现完整功能
 */
const Models = () => {
  return (
    <div className="models">
      <Title level={2}>模型构建</Title>
      <Empty 
        description="模型构建功能正在开发中..."
        style={{ marginTop: 60 }}
      />
    </div>
  )
}

export default Models