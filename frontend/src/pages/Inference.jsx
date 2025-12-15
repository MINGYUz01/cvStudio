/**
 * 推理测试页面组件
 */

import React from 'react'
import { Typography, Empty } from 'antd'

const { Title } = Typography

/**
 * 推理测试页面
 * 占位符组件，后续会实现完整功能
 */
const Inference = () => {
  return (
    <div className="inference">
      <Title level={2}>推理测试</Title>
      <Empty 
        description="推理测试功能正在开发中..."
        style={{ marginTop: 60 }}
      />
    </div>
  )
}

export default Inference