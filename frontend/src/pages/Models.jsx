/**
 * 模型构建页面组件 - 现代化设计
 */

import React from 'react'
import {
  Box,
  Plus,
  Play,
  Save,
  Upload,
  Download,
  GitBranch,
  Zap,
  Settings,
  Eye,
  Edit,
  Trash2,
  Copy,
  Layers,
  Cpu,
  BarChart3
} from 'lucide-react'

/**
 * 模型构建页面
 * 可视化模型构建器
 */
const Models = () => {
  // 模拟数据
  const models = [
    {
      id: 1,
      name: 'YOLOv8-Custom',
      description: '自定义目标检测模型，基于YOLOv8架构',
      type: 'Object Detection',
      accuracy: '92.3%',
      status: 'trained',
      nodes: 45,
      layers: 8,
      params: '11.2M',
      createdAt: '2024-01-15',
      lastModified: '2024-01-16'
    },
    {
      id: 2,
      name: 'ResNet50-Transfer',
      description: '基于ResNet50的迁移学习分类模型',
      type: 'Classification',
      accuracy: '87.1%',
      status: 'training',
      nodes: 68,
      layers: 50,
      params: '25.6M',
      createdAt: '2024-01-10',
      lastModified: '2024-01-14'
    },
    {
      id: 3,
      name: 'Custom-CNN',
      description: '自定义卷积神经网络',
      type: 'Classification',
      accuracy: '78.5%',
      status: 'draft',
      nodes: 32,
      layers: 6,
      params: '8.4M',
      createdAt: '2024-01-08',
      lastModified: '2024-01-12'
    }
  ]

  /**
   * 状态标签组件
   */
  const StatusBadge = ({ status }) => {
    const statusConfig = {
      trained: { color: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20', text: '已训练' },
      training: { color: 'bg-blue-500/10 text-blue-400 border-blue-500/20', text: '训练中' },
      draft: { color: 'bg-slate-500/10 text-slate-400 border-slate-500/20', text: '草稿' },
      error: { color: 'bg-red-500/10 text-red-400 border-red-500/20', text: '错误' }
    }

    const config = statusConfig[status] || statusConfig.draft

    return (
      <span className={`px-2 py-1 rounded-full text-xs font-medium border ${config.color}`}>
        {config.text}
      </span>
    )
  }

  /**
   * 模型卡片组件
   */
  const ModelCard = ({ model }) => (
    <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6 hover:shadow-lg transition-all duration-300 hover:border-slate-600 group">
      {/* 头部 */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
            <Box size={24} className="text-white" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-white">{model.name}</h3>
            <p className="text-slate-400 text-sm">{model.type}</p>
          </div>
        </div>
        <StatusBadge status={model.status} />
      </div>

      {/* 描述 */}
      <p className="text-slate-400 text-sm mb-4 line-clamp-2">
        {model.description}
      </p>

      {/* 统计信息 */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="flex items-center gap-2">
          <Layers size={16} className="text-slate-500" />
          <span className="text-slate-300 text-sm">{model.layers} 层</span>
        </div>
        <div className="flex items-center gap-2">
          <GitBranch size={16} className="text-slate-500" />
          <span className="text-slate-300 text-sm">{model.nodes} 节点</span>
        </div>
        <div className="flex items-center gap-2">
          <Cpu size={16} className="text-slate-500" />
          <span className="text-slate-300 text-sm">{model.params}</span>
        </div>
        <div className="flex items-center gap-2">
          <BarChart3 size={16} className="text-slate-500" />
          <span className="text-slate-300 text-sm">{model.accuracy}</span>
        </div>
      </div>

      {/* 操作按钮 */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <button className="p-2 bg-slate-700/50 hover:bg-slate-700 rounded-lg transition-colors">
            <Eye size={16} className="text-slate-400" />
          </button>
          <button className="p-2 bg-slate-700/50 hover:bg-slate-700 rounded-lg transition-colors">
            <Edit size={16} className="text-slate-400" />
          </button>
          <button className="p-2 bg-slate-700/50 hover:bg-slate-700 rounded-lg transition-colors">
            <Copy size={16} className="text-slate-400" />
          </button>
          <button className="p-2 bg-slate-700/50 hover:bg-slate-700 rounded-lg transition-colors">
            <Download size={16} className="text-slate-400" />
          </button>
        </div>
        <button className="p-2 bg-red-500/10 hover:bg-red-500/20 rounded-lg transition-colors">
          <Trash2 size={16} className="text-red-400" />
        </button>
      </div>
    </div>
  )

  return (
    <div className="space-y-6">
      {/* 页面标题和操作 */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white mb-2">模型构建</h1>
          <p className="text-slate-400">创建、编辑和管理您的深度学习模型</p>
        </div>
        <div className="flex items-center gap-3">
          <button className="px-4 py-2 bg-slate-700/50 hover:bg-slate-700 rounded-lg transition-colors flex items-center gap-2 text-slate-300">
            <Upload size={16} />
            导入模型
          </button>
          <button className="px-4 py-2 bg-blue-500 hover:bg-blue-600 rounded-lg transition-colors flex items-center gap-2 text-white">
            <Plus size={16} />
            新建模型
          </button>
        </div>
      </div>

      {/* 模型构建器画布占位 */}
      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-8 min-h-[400px] flex items-center justify-center">
        <div className="text-center">
          <div className="w-24 h-24 mx-auto mb-6 bg-slate-700/50 rounded-full flex items-center justify-center">
            <Layers size={32} className="text-slate-500" />
          </div>
          <h3 className="text-xl font-semibold text-white mb-2">可视化模型构建器</h3>
          <p className="text-slate-400 mb-6">拖拽节点来构建您的深度学习模型</p>
          <button className="px-6 py-3 bg-blue-500 hover:bg-blue-600 rounded-lg transition-colors flex items-center gap-2 text-white mx-auto">
            <Plus size={20} />
            开始构建
          </button>
        </div>
      </div>

      {/* 模型列表 */}
      <div>
        <h3 className="text-lg font-semibold text-white mb-4">最近的项目</h3>
        {models.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {models.map(model => (
              <ModelCard key={model.id} model={model} />
            ))}
          </div>
        ) : (
          <div className="text-center py-16">
            <div className="w-24 h-24 mx-auto mb-6 bg-slate-700/50 rounded-full flex items-center justify-center">
              <Box size={32} className="text-slate-500" />
            </div>
            <h3 className="text-xl font-semibold text-white mb-2">暂无模型</h3>
            <p className="text-slate-400 mb-6">开始创建您的第一个深度学习模型</p>
            <button className="px-6 py-3 bg-blue-500 hover:bg-blue-600 rounded-lg transition-colors flex items-center gap-2 text-white mx-auto">
              <Plus size={20} />
              创建模型
            </button>
          </div>
        )}
      </div>

      {/* 统计信息 */}
      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <BarChart3 size={20} />
          模型统计
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="text-center p-4 bg-slate-700/30 rounded-lg">
            <p className="text-2xl font-bold text-white">{models.length}</p>
            <p className="text-slate-400 text-sm">模型总数</p>
          </div>
          <div className="text-center p-4 bg-slate-700/30 rounded-lg">
            <p className="text-2xl font-bold text-white">{models.filter(m => m.status === 'trained').length}</p>
            <p className="text-slate-400 text-sm">已训练</p>
          </div>
          <div className="text-center p-4 bg-slate-700/30 rounded-lg">
            <p className="text-2xl font-bold text-white">{models.filter(m => m.status === 'training').length}</p>
            <p className="text-slate-400 text-sm">训练中</p>
          </div>
          <div className="text-center p-4 bg-slate-700/30 rounded-lg">
            <p className="text-2xl font-bold text-white">
              {models.reduce((sum, m) => sum + parseInt(m.params), 0).toLocaleString()}M
            </p>
            <p className="text-slate-400 text-sm">总参数</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Models