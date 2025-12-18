/**
 * 仪表盘页面组件 - 现代化设计
 */

import React from 'react'
import {
  Database,
  Box,
  Activity,
  PlayCircle,
  TrendingUp,
  Clock,
  Zap,
  AlertCircle,
  CheckCircle2,
  Server,
  HardDrive,
  Cpu
} from 'lucide-react'

/**
 * 仪表盘页面
 * 显示系统概览和关键统计数据
 */
const Dashboard = () => {
  // 模拟数据 - 实际使用时从API获取
  const stats = {
    datasets: 12,
    models: 8,
    trainingTasks: 3,
    inferenceTasks: 156
  }

  const recentDatasets = [
    { id: 1, name: 'COCO 2017', size: '25GB', format: 'COCO', status: 'ready' },
    { id: 2, name: 'ImageNet Subset', size: '45GB', format: 'Classification', status: 'processing' }
  ]

  const recentModels = [
    { id: 1, name: 'YOLOv8-Custom', accuracy: '92.3%', status: 'trained' },
    { id: 2, name: 'ResNet50-Transfer', accuracy: '87.1%', status: 'training' }
  ]

  const systemStatus = [
    { label: '后端服务', status: 'online', icon: Server },
    { label: '数据库', status: 'online', icon: HardDrive },
    { label: 'GPU', status: 'online', icon: Cpu },
    { label: '存储空间', status: 'warning', icon: HardDrive }
  ]

  /**
   * 状态指示器组件
   */
  const StatusIndicator = ({ status }) => {
    const statusConfig = {
      online: { color: 'bg-emerald-500', text: '正常' },
      warning: { color: 'bg-amber-500', text: '警告' },
      offline: { color: 'bg-red-500', text: '离线' },
      processing: { color: 'bg-blue-500', text: '处理中' }
    }

    const config = statusConfig[status] || statusConfig.offline

    return (
      <div className="flex items-center gap-2">
        <div className={`w-2 h-2 rounded-full ${config.color} animate-pulse`}></div>
        <span className="text-sm text-slate-400">{config.text}</span>
      </div>
    )
  }

  /**
   * 统计卡片组件
   */
  const StatCard = ({ icon: Icon, title, value, trend, color }) => (
    <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6 hover:shadow-lg transition-all duration-300 hover:border-slate-600">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-slate-400 text-sm font-medium mb-1">{title}</p>
          <p className="text-3xl font-bold text-white">{value}</p>
          {trend && (
            <div className="flex items-center gap-1 mt-2">
              <TrendingUp size={16} className={trend > 0 ? 'text-emerald-500' : 'text-red-500'} />
              <span className={`text-sm ${trend > 0 ? 'text-emerald-500' : 'text-red-500'}`}>
                {Math.abs(trend)}%
              </span>
            </div>
          )}
        </div>
        <div className={`w-12 h-12 rounded-lg flex items-center justify-center ${color}`}>
          <Icon size={24} className="text-white" />
        </div>
      </div>
    </div>
  )

  /**
   * 项目列表组件
   */
  const ProjectList = ({ items, type, icon: Icon }) => (
    <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white flex items-center gap-2">
          <Icon size={20} />
          最近的{type}
        </h3>
        <button className="text-sm text-blue-400 hover:text-blue-300 transition-colors">
          查看全部
        </button>
      </div>

      {items.length > 0 ? (
        <div className="space-y-3">
          {items.map(item => (
            <div
              key={item.id}
              className="flex items-center justify-between p-3 bg-slate-700/50 rounded-lg hover:bg-slate-700/70 transition-colors cursor-pointer"
            >
              <div className="flex-1">
                <p className="text-white font-medium">{item.name}</p>
                <p className="text-slate-400 text-sm">
                  {item.size} · {item.format}
                </p>
              </div>
              <StatusIndicator status={item.status} />
            </div>
          ))}
        </div>
      ) : (
        <div className="text-center py-8">
          <div className="w-16 h-16 mx-auto mb-4 bg-slate-700/50 rounded-full flex items-center justify-center">
            <Icon size={24} className="text-slate-500" />
          </div>
          <p className="text-slate-400">暂无{type}</p>
        </div>
      )}
    </div>
  )

  return (
    <div className="space-y-6">
      {/* 页面标题 */}
      <div>
        <h1 className="text-2xl font-bold text-white mb-2">仪表盘</h1>
        <p className="text-slate-400">欢迎使用 CV Studio，这里是您的工作概览</p>
      </div>

      {/* 统计卡片网格 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          icon={Database}
          title="数据集总数"
          value={stats.datasets}
          trend={12}
          color="bg-blue-500"
        />
        <StatCard
          icon={Box}
          title="模型总数"
          value={stats.models}
          trend={8}
          color="bg-purple-500"
        />
        <StatCard
          icon={Activity}
          title="训练任务"
          value={stats.trainingTasks}
          trend={-5}
          color="bg-green-500"
        />
        <StatCard
          icon={PlayCircle}
          title="推理任务"
          value={stats.inferenceTasks}
          trend={23}
          color="bg-orange-500"
        />
      </div>

      {/* 最近项目和系统状态 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ProjectList items={recentDatasets} type="数据集" icon={Database} />
        <ProjectList items={recentModels} type="模型" icon={Box} />
      </div>

      {/* 系统状态 */}
      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Zap size={20} />
          系统状态
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {systemStatus.map((item, index) => (
            <div
              key={index}
              className="flex items-center justify-between p-4 bg-slate-700/30 rounded-lg"
            >
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-slate-600/50 rounded-lg flex items-center justify-center">
                  <item.icon size={18} className="text-slate-300" />
                </div>
                <span className="text-white font-medium">{item.label}</span>
              </div>
              <StatusIndicator status={item.status} />
            </div>
          ))}
        </div>
      </div>

      {/* 快速操作 */}
      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Clock size={20} />
          快速操作
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <button className="p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg hover:bg-blue-500/20 transition-colors text-blue-400 hover:text-blue-300">
            <Database size={20} className="mb-2 mx-auto" />
            <p className="text-sm font-medium">上传数据集</p>
          </button>
          <button className="p-4 bg-purple-500/10 border border-purple-500/30 rounded-lg hover:bg-purple-500/20 transition-colors text-purple-400 hover:text-purple-300">
            <Box size={20} className="mb-2 mx-auto" />
            <p className="text-sm font-medium">创建模型</p>
          </button>
          <button className="p-4 bg-green-500/10 border border-green-500/30 rounded-lg hover:bg-green-500/20 transition-colors text-green-400 hover:text-green-300">
            <Activity size={20} className="mb-2 mx-auto" />
            <p className="text-sm font-medium">开始训练</p>
          </button>
          <button className="p-4 bg-orange-500/10 border border-orange-500/30 rounded-lg hover:bg-orange-500/20 transition-colors text-orange-400 hover:text-orange-300">
            <PlayCircle size={20} className="mb-2 mx-auto" />
            <p className="text-sm font-medium">运行推理</p>
          </button>
        </div>
      </div>
    </div>
  )
}

export default Dashboard