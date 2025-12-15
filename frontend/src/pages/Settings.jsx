/**
 * 设置页面组件
 */

import React from 'react'
import { Typography, Empty } from 'antd'

const { Title } = Typography

/**
 * 设置页面
 * 占位符组件，后续会实现完整功能
 */
const Settings = () => {
  return (
    <div className="settings">
      <Title level={2}>系统设置</Title>
      <Empty 
        description="系统设置功能正在开发中..."
        style={{ marginTop: 60 }}
      />
    </div>
  )
}

export default Settings