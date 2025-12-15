/**
 * 数据集管理页面组件
 */

import React from 'react'
import { Typography, Empty } from 'antd'

const { Title } = Typography

/**
 * 数据集管理页面
 * 占位符组件，后续会实现完整功能
 */
const Datasets = () => {
  return (
    <div className="datasets">
      <Title level={2}>数据集管理</Title>
      <Empty 
        description="数据集管理功能正在开发中..."
        style={{ marginTop: 60 }}
      />
    </div>
  )
}

export default Datasets