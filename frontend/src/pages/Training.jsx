/**
 * 训练管理页面组件 - 现代化设计
 */

import React from 'react'
import {
  Activity,
  Play,
  Pause,
  Square,
  Settings,
  Clock,
  TrendingUp,
  TrendingDown,
  BarChart3,
  Cpu,
  HardDrive,
  Zap,
  CheckCircle2,
  AlertCircle,
  XCircle,
  Eye,
  Download,
  RefreshCw,
  Plus,
  Filter,
  Search
} from 'lucide-react'

/**
 * 训练管理页面
 * 显示训练任务、进度监控和性能指标
 */
const Training = () => {
  // 模拟训练任务数据
  const trainingTasks = [
    {
      id: 1,
      name: 'YOLOv8-Custom-Object-Detection',
      model: 'YOLOv8n',
      dataset: 'COCO 2017 Subset',
      status: 'training',
      progress: 67,
      epochs: 100,
      currentEpoch: 67,
      startTime: '2024-01-15 14:30:00',
      estimatedTime: '2小时15分钟',
      metrics: {
        loss: 0.245,
        accuracy: 0.892,
        mAP: 0.876,
        learningRate: 0.001
      },
      gpu: 'RTX 4090',
      resources: {
        cpu: 45,
        memory: 68,
        gpu: 92
      }
    },
    {
      id: 2,
      name: 'ResNet50-Image-Classification',
      model: 'ResNet50',
      dataset: 'ImageNet Subset',
      status: 'completed',
      progress: 100,
      epochs: 50,
      currentEpoch: 50,
      startTime: '2024-01-14 09:15:00',
      endTime: '2024-01-14 16:42:00',
      duration: '7小时27分钟',
      metrics: {
        loss: 0.156,
        accuracy: 0.943,
        top5Accuracy: 0.998,
        learningRate: 0.0001
      },
      gpu: 'RTX 4090',
      resources: {
        cpu: 38,
        memory: 72,
        gpu: 88
      }
    },
    {
      id: 3,
      name: 'Custom-SegmentAnything-FineTune',
      model: 'SAM-B',
      dataset: 'Medical Images v2',
      status: 'paused',
      progress: 23,
      epochs: 200,
      currentEpoch: 46,
      startTime: '2024-01-13 11:00:00',
      pausedTime: '2024-01-15 10:30:00',
      metrics: {
        loss: 0.412,
        accuracy: 0.756,
        diceScore: 0.823,
        learningRate: 0.0005
      },
      gpu: 'A100',
      resources: {
        cpu: 52,
        memory: 85,
        gpu: 78
      }
    },
    {
      id: 4,
      name: 'StableDiffusion-Finetune-ArtStyle',
      model: 'SD 1.5',
      dataset: 'Art Portfolio Dataset',
      status: 'failed',
      progress: 15,
      epochs: 300,
      currentEpoch: 45,
      startTime: '2024-01-12 15:20:00',
      errorTime: '2024-01-12 18:45:00',
      error: 'CUDA out of memory',
      metrics: {
        loss: 0.678,
        accuracy: 0.634,
        fid: 145.6,
        learningRate: 0.0002
      },
      gpu: 'RTX 3080',
      resources: {
        cpu: 41,
        memory: 93,
        gpu: 95
      }
    }
  ]

  /**
   * 状态指示器组件
   */
  const StatusIndicator = ({ status }) => {
    const statusConfig = {
      training: {
        color: 'bg-blue-500',
        text: '训练中',
        icon: Activity,
        animation: 'animate-pulse'
      },
      completed: {
        color: 'bg-emerald-500',
        text: '已完成',
        icon: CheckCircle2,
        animation: ''
      },
      paused: {
        color: 'bg-amber-500',
        text: '已暂停',
        icon: Pause,
        animation: ''
      },
      failed: {
        color: 'bg-red-500',
        text: '训练失败',
        icon: XCircle,
        animation: ''
      },
      queued: {
        color: 'bg-slate-500',
        text: '队列中',
        icon: Clock,
        animation: 'animate-pulse'
      }
    }

    const config = statusConfig[status] || statusConfig.failed
    const Icon = config.icon

    return (
      <div className={`flex items-center gap-2 ${config.animation}`}>
        <div className={`w-3 h-3 rounded-full ${config.color}`}></div>
        <Icon size={16} className={config.color.replace('bg-', 'text-')} />
        <span className="text-sm font-medium text-slate-300">{config.text}</span>
      </div>
    )
  }

  /**
   * 进度条组件
   */
  const ProgressBar = ({ progress, status }) => {
    const progressColor = {
      training: 'bg-blue-500',
      completed: 'bg-emerald-500',
      paused: 'bg-amber-500',
      failed: 'bg-red-500',
      queued: 'bg-slate-500'
    }[status] || 'bg-slate-500'

    return (
      <div className="w-full">
        <div className="flex items-center justify-between mb-1">
          <span className="text-sm text-slate-400">进度</span>
          <span className="text-sm font-medium text-white">{progress}%</span>
        </div>
        <div className="w-full bg-slate-700 rounded-full h-2">
          <div
            className={`h-2 rounded-full ${progressColor} transition-all duration-300`}
            style={{ width: `${progress}%` }}
          ></div>
        </div>
      </div>
    )
  }

  /**
   * 资源使用率组件
   */
  const ResourceUsage = ({ resources }) => {
    const resourceConfig = [
      { name: 'CPU', value: resources.cpu, color: 'text-blue-400', bgColor: 'bg-blue-500/20' },
      { name: 'Memory', value: resources.memory, color: 'text-green-400', bgColor: 'bg-green-500/20' },
      { name: 'GPU', value: resources.gpu, color: 'text-purple-400', bgColor: 'bg-purple-500/20' }
    ]

    return (
      <div className="grid grid-cols-3 gap-3">
        {resourceConfig.map((resource) => (
          <div key={resource.name} className="text-center">
            <div className="text-xs text-slate-400 mb-1">{resource.name}</div>
            <div className="relative w-full h-2 bg-slate-700 rounded-full overflow-hidden">
              <div
                className={`h-full ${resource.bgColor} transition-all duration-300`}
                style={{ width: `${resource.value}%` }}
              ></div>
            </div>
            <div className={`text-xs font-medium ${resource.color} mt-1`}>{resource.value}%</div>
          </div>
        ))}
      </div>
    )
  }

  /**
   * 训练任务卡片组件
   */
  const TrainingTaskCard = ({ task }) => (
    <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6 hover:shadow-lg transition-all duration-300 hover:border-slate-600">
      {/* 任务头部 */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <h3 className="text-lg font-semibold text-white mb-1 line-clamp-1">{task.name}</h3>
          <div className="flex items-center gap-4 text-sm text-slate-400">
            <span className="flex items-center gap-1">
              <BarChart3 size={14} />
              {task.model}
            </span>
            <span className="flex items-center gap-1">
              <HardDrive size={14} />
              {task.dataset}
            </span>
            <span className="flex items-center gap-1">
              <Cpu size={14} />
              {task.gpu}
            </span>
          </div>
        </div>
        <StatusIndicator status={task.status} />
      </div>

      {/* 进度条 */}
      <ProgressBar progress={task.progress} status={task.status} />

      {/* 训练信息 */}
      <div className="grid grid-cols-2 gap-4 mt-4">
        <div>
          <div className="text-xs text-slate-400 mb-1">训练轮次</div>
          <div className="text-sm font-medium text-white">
            {task.currentEpoch} / {task.epochs}
          </div>
        </div>
        <div>
          <div className="text-xs text-slate-400 mb-1">开始时间</div>
          <div className="text-sm font-medium text-white">
            {new Date(task.startTime).toLocaleString('zh-CN', {
              month: '2-digit',
              day: '2-digit',
              hour: '2-digit',
              minute: '2-digit'
            })}
          </div>
        </div>
      </div>

      {/* 性能指标 */}
      {task.metrics && (
        <div className="mt-4 p-3 bg-slate-700/30 rounded-lg">
          <div className="grid grid-cols-2 gap-3 text-xs">
            {Object.entries(task.metrics).slice(0, 4).map(([key, value]) => (
              <div key={key} className="flex justify-between">
                <span className="text-slate-400 capitalize">{key}:</span>
                <span className="text-white font-medium">{typeof value === 'number' ? value.toFixed(3) : value}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 资源使用率 */}
      {task.resources && (
        <div className="mt-4">
          <div className="text-xs text-slate-400 mb-2">资源使用率</div>
          <ResourceUsage resources={task.resources} />
        </div>
      )}

      {/* 操作按钮 */}
      <div className="flex items-center gap-2 mt-4 pt-4 border-t border-slate-700">
        <button className={`flex-1 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
          task.status === 'training'
            ? 'bg-amber-500/20 text-amber-400 hover:bg-amber-500/30 border border-amber-500/30'
            : task.status === 'paused'
            ? 'bg-blue-500/20 text-blue-400 hover:bg-blue-500/30 border border-blue-500/30'
            : task.status === 'completed'
            ? 'bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30 border border-emerald-500/30'
            : 'bg-slate-700/50 text-slate-400 hover:bg-slate-700/70 border border-slate-600/50'
        }`}>
          {task.status === 'training' && <><Pause size={14} className="inline mr-1" /> 暂停</>}
          {task.status === 'paused' && <><Play size={14} className="inline mr-1" /> 继续</>}
          {task.status === 'completed' && <><Download size={14} className="inline mr-1" /> 下载</>}
          {task.status === 'failed' && <><RefreshCw size={14} className="inline mr-1" /> 重试</>}
          {task.status === 'queued' && <><Clock size={14} className="inline mr-1" /> 等待中</>}
        </button>
        <button className="px-3 py-2 bg-slate-700/50 text-slate-300 rounded-lg hover:bg-slate-700/70 transition-colors border border-slate-600/50">
          <Settings size={14} />
        </button>
        <button className="px-3 py-2 bg-slate-700/50 text-slate-300 rounded-lg hover:bg-slate-700/70 transition-colors border border-slate-600/50">
          <Eye size={14} />
        </button>
      </div>
    </div>
  )

  /**
   * 统计卡片组件
   */
  const StatCard = ({ icon: Icon, title, value, change, color }) => (
    <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6 hover:shadow-lg transition-all duration-300 hover:border-slate-600">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-slate-400 text-sm font-medium mb-1">{title}</p>
          <p className="text-2xl font-bold text-white">{value}</p>
          {change && (
            <div className="flex items-center gap-1 mt-2">
              {change > 0 ? (
                <TrendingUp size={16} className="text-emerald-500" />
              ) : (
                <TrendingDown size={16} className="text-red-500" />
              )}
              <span className={`text-sm ${change > 0 ? 'text-emerald-500' : 'text-red-500'}`}>
                {Math.abs(change)}%
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

  return (
    <div className="space-y-6">
      {/* 页面标题 */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white mb-2">训练管理</h1>
          <p className="text-slate-400">监控和管理您的模型训练任务</p>
        </div>
        <div className="flex items-center gap-3">
          <div className="relative">
            <Search size={18} className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400" />
            <input
              type="text"
              placeholder="搜索训练任务..."
              className="pl-10 pr-4 py-2 bg-slate-800/50 border border-slate-700 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:border-blue-500 w-64"
            />
          </div>
          <button className="px-4 py-2 bg-slate-700/50 border border-slate-600/50 text-slate-300 rounded-lg hover:bg-slate-700/70 transition-colors flex items-center gap-2">
            <Filter size={16} />
            筛选
          </button>
          <button className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors flex items-center gap-2">
            <Plus size={16} />
            新建训练
          </button>
        </div>
      </div>

      {/* 统计卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          icon={Activity}
          title="活跃训练"
          value="3"
          change={25}
          color="bg-blue-500"
        />
        <StatCard
          icon={CheckCircle2}
          title="今日完成"
          value="12"
          change={8}
          color="bg-emerald-500"
        />
        <StatCard
          icon={Clock}
          title="队列等待"
          value="5"
          change={-10}
          color="bg-amber-500"
        />
        <StatCard
          icon={Cpu}
          title="GPU使用率"
          value="78%"
          change={5}
          color="bg-purple-500"
        />
      </div>

      {/* 训练任务列表 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {trainingTasks.map((task) => (
          <TrainingTaskCard key={task.id} task={task} />
        ))}
      </div>

      {/* 系统资源监控 */}
      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Zap size={20} />
          系统资源监控
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-slate-400">CPU使用率</span>
              <span className="text-sm font-medium text-white">45%</span>
            </div>
            <div className="w-full bg-slate-700 rounded-full h-3">
              <div className="h-3 bg-blue-500 rounded-full" style={{ width: '45%' }}></div>
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-slate-400">内存使用</span>
              <span className="text-sm font-medium text-white">68%</span>
            </div>
            <div className="w-full bg-slate-700 rounded-full h-3">
              <div className="h-3 bg-green-500 rounded-full" style={{ width: '68%' }}></div>
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-slate-400">GPU使用率</span>
              <span className="text-sm font-medium text-white">92%</span>
            </div>
            <div className="w-full bg-slate-700 rounded-full h-3">
              <div className="h-3 bg-purple-500 rounded-full" style={{ width: '92%' }}></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Training