/**
 * 数据集管理页面组件 - 现代化设计
 */

import React from 'react'
import {
  Database,
  Upload,
  Search,
  Filter,
  Grid,
  List,
  Plus,
  FolderOpen,
  FileText,
  Image,
  BarChart3,
  Download,
  Trash2,
  Eye,
  Edit
} from 'lucide-react'

/**
 * 数据集管理页面
 * 管理和查看所有数据集
 */
const Datasets = () => {
  // 模拟数据
  const datasets = [
    {
      id: 1,
      name: 'COCO 2017 Dataset',
      description: '目标检测数据集，包含80个类别',
      format: 'COCO',
      size: '25GB',
      images: 118287,
      classes: 80,
      status: 'ready',
      createdAt: '2024-01-15',
      thumbnail: '/api/placeholder/300/200'
    },
    {
      id: 2,
      name: 'ImageNet Subset',
      description: '图像分类数据集，1000个类别子集',
      format: 'Classification',
      size: '45GB',
      images: 50000,
      classes: 1000,
      status: 'processing',
      createdAt: '2024-01-10',
      thumbnail: '/api/placeholder/300/200'
    },
    {
      id: 3,
      name: 'Custom Detection Dataset',
      description: '自定义目标检测数据集',
      format: 'YOLO',
      size: '8GB',
      images: 12500,
      classes: 15,
      status: 'ready',
      createdAt: '2024-01-08',
      thumbnail: '/api/placeholder/300/200'
    }
  ]

  /**
   * 状态标签组件
   */
  const StatusBadge = ({ status }) => {
    const statusConfig = {
      ready: { color: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20', text: '就绪' },
      processing: { color: 'bg-blue-500/10 text-blue-400 border-blue-500/20', text: '处理中' },
      error: { color: 'bg-red-500/10 text-red-400 border-red-500/20', text: '错误' }
    }

    const config = statusConfig[status] || statusConfig.ready

    return (
      <span className={`px-2 py-1 rounded-full text-xs font-medium border ${config.color}`}>
        {config.text}
      </span>
    )
  }

  /**
   * 数据集卡片组件
   */
  const DatasetCard = ({ dataset }) => (
    <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl overflow-hidden hover:shadow-lg transition-all duration-300 hover:border-slate-600 group">
      {/* 缩略图 */}
      <div className="h-48 bg-slate-700/50 relative overflow-hidden">
        <img
          src={dataset.thumbnail}
          alt={dataset.name}
          className="w-full h-full object-cover opacity-70 group-hover:opacity-100 transition-opacity"
          onError={(e) => {
            e.target.src = `https://via.placeholder.com/400x300/1e293b/f8fafc?text=${encodeURIComponent(dataset.name)}`
          }}
        />
        <div className="absolute top-2 right-2">
          <StatusBadge status={dataset.status} />
        </div>
      </div>

      {/* 内容 */}
      <div className="p-4">
        <h3 className="text-lg font-semibold text-white mb-2 line-clamp-1">
          {dataset.name}
        </h3>
        <p className="text-slate-400 text-sm mb-4 line-clamp-2">
          {dataset.description}
        </p>

        {/* 统计信息 */}
        <div className="grid grid-cols-2 gap-3 mb-4">
          <div className="flex items-center gap-2">
            <Image size={16} className="text-slate-500" />
            <span className="text-slate-300 text-sm">{dataset.images.toLocaleString()} 张</span>
          </div>
          <div className="flex items-center gap-2">
            <FileText size={16} className="text-slate-500" />
            <span className="text-slate-300 text-sm">{dataset.classes} 类别</span>
          </div>
          <div className="flex items-center gap-2">
            <FolderOpen size={16} className="text-slate-500" />
            <span className="text-slate-300 text-sm">{dataset.format}</span>
          </div>
          <div className="flex items-center gap-2">
            <BarChart3 size={16} className="text-slate-500" />
            <span className="text-slate-300 text-sm">{dataset.size}</span>
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
              <Download size={16} className="text-slate-400" />
            </button>
          </div>
          <button className="p-2 bg-red-500/10 hover:bg-red-500/20 rounded-lg transition-colors">
            <Trash2 size={16} className="text-red-400" />
          </button>
        </div>
      </div>
    </div>
  )

  return (
    <div className="space-y-6">
      {/* 页面标题和操作 */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white mb-2">数据集管理</h1>
          <p className="text-slate-400">管理和查看所有数据集，支持多种格式</p>
        </div>
        <div className="flex items-center gap-3">
          <button className="px-4 py-2 bg-slate-700/50 hover:bg-slate-700 rounded-lg transition-colors flex items-center gap-2 text-slate-300">
            <Upload size={16} />
            上传数据集
          </button>
          <button className="px-4 py-2 bg-blue-500 hover:bg-blue-600 rounded-lg transition-colors flex items-center gap-2 text-white">
            <Plus size={16} />
            新建数据集
          </button>
        </div>
      </div>

      {/* 搜索和过滤 */}
      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-4">
        <div className="flex flex-col lg:flex-row gap-4">
          <div className="flex-1 relative">
            <Search size={20} className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-500" />
            <input
              type="text"
              placeholder="搜索数据集..."
              className="w-full pl-10 pr-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-blue-500"
            />
          </div>
          <div className="flex items-center gap-2">
            <button className="px-4 py-2 bg-slate-700/50 hover:bg-slate-700 rounded-lg transition-colors flex items-center gap-2 text-slate-300">
              <Filter size={16} />
              格式
            </button>
            <button className="px-4 py-2 bg-slate-700/50 hover:bg-slate-700 rounded-lg transition-colors flex items-center gap-2 text-slate-300">
              <BarChart3 size={16} />
              状态
            </button>
            <div className="flex items-center bg-slate-700/50 rounded-lg p-1">
              <button className="p-2 bg-blue-500 text-white rounded-md">
                <Grid size={16} />
              </button>
              <button className="p-2 text-slate-400 hover:text-white transition-colors">
                <List size={16} />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* 数据集网格 */}
      {datasets.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {datasets.map(dataset => (
            <DatasetCard key={dataset.id} dataset={dataset} />
          ))}
        </div>
      ) : (
        <div className="text-center py-16">
          <div className="w-24 h-24 mx-auto mb-6 bg-slate-700/50 rounded-full flex items-center justify-center">
            <Database size={32} className="text-slate-500" />
          </div>
          <h3 className="text-xl font-semibold text-white mb-2">暂无数据集</h3>
          <p className="text-slate-400 mb-6">开始上传您的第一个数据集</p>
          <button className="px-6 py-3 bg-blue-500 hover:bg-blue-600 rounded-lg transition-colors flex items-center gap-2 text-white mx-auto">
            <Plus size={20} />
            创建数据集
          </button>
        </div>
      )}

      {/* 统计信息 */}
      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <BarChart3 size={20} />
          数据集统计
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="text-center p-4 bg-slate-700/30 rounded-lg">
            <p className="text-2xl font-bold text-white">{datasets.length}</p>
            <p className="text-slate-400 text-sm">数据集总数</p>
          </div>
          <div className="text-center p-4 bg-slate-700/30 rounded-lg">
            <p className="text-2xl font-bold text-white">
              {datasets.reduce((sum, d) => sum + d.images, 0).toLocaleString()}
            </p>
            <p className="text-slate-400 text-sm">图像总数</p>
          </div>
          <div className="text-center p-4 bg-slate-700/30 rounded-lg">
            <p className="text-2xl font-bold text-white">
              {datasets.reduce((sum, d) => sum + d.classes, 0).toLocaleString()}
            </p>
            <p className="text-slate-400 text-sm">类别总数</p>
          </div>
          <div className="text-center p-4 bg-slate-700/30 rounded-lg">
            <p className="text-2xl font-bold text-white">78GB</p>
            <p className="text-slate-400 text-sm">总大小</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Datasets