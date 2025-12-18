/**
 * 推理测试页面组件 - 现代化设计
 */

import React from 'react'
import {
  Play,
  Upload,
  Download,
  Image,
  Video,
  FileText,
  BarChart3,
  Clock,
  CheckCircle2,
  AlertCircle,
  XCircle,
  Settings,
  Eye,
  Zap,
  Camera,
  Monitor,
  Cpu,
  HardDrive,
  TrendingUp,
  Filter,
  Search,
  Plus,
  Trash2,
  Copy,
  Share2,
  RefreshCw,
  Pause,
  Square
} from 'lucide-react'

/**
 * 推理测试页面
 * 显示模型推理任务、测试结果和性能分析
 */
const Inference = () => {
  // 模拟推理任务数据
  const inferenceTasks = [
    {
      id: 1,
      name: 'Batch-Object-Detection-Test',
      model: 'YOLOv8-Custom',
      inputType: 'images',
      status: 'completed',
      progress: 100,
      startTime: '2024-01-15 15:30:00',
      endTime: '2024-01-15 15:32:45',
      duration: '2分45秒',
      inputCount: 150,
      processedCount: 150,
      accuracy: 0.923,
      inferenceTime: 0.089,
      confidence: 0.85,
      results: {
        truePositives: 138,
        falsePositives: 8,
        falseNegatives: 4
      }
    },
    {
      id: 2,
      name: 'Real-Time-Segmentation-Demo',
      model: 'SAM-B-Finetuned',
      inputType: 'webcam',
      status: 'running',
      progress: 100,
      startTime: '2024-01-15 16:00:00',
      inputCount: 0,
      processedCount: 0,
      accuracy: 0.0,
      inferenceTime: 0.156,
      confidence: 0.75,
      fps: 24,
      resolution: '1920x1080'
    },
    {
      id: 3,
      name: 'Image-Classification-Batch',
      model: 'ResNet50-Transfer',
      inputType: 'images',
      status: 'processing',
      progress: 67,
      startTime: '2024-01-15 14:15:00',
      estimatedTime: '5分钟',
      inputCount: 500,
      processedCount: 335,
      accuracy: 0.876,
      inferenceTime: 0.023,
      confidence: 0.92,
      results: {
        correct: 293,
        incorrect: 42
      }
    },
    {
      id: 4,
      name: 'Video-Analysis-Test',
      model: 'YOLOv8-Custom',
      inputType: 'video',
      status: 'failed',
      progress: 35,
      startTime: '2024-01-15 13:20:00',
      errorTime: '2024-01-15 13:28:00',
      error: 'Video format not supported',
      inputCount: 1,
      processedCount: 0,
      accuracy: 0.0,
      inferenceTime: 0.0,
      confidence: 0.8
    }
  ]

  // 模拟可用模型数据
  const availableModels = [
    {
      id: 1,
      name: 'YOLOv8-Custom',
      type: '目标检测',
      accuracy: '92.3%',
      size: '45MB',
      inputFormats: ['图片', '视频', '摄像头'],
      status: 'ready'
    },
    {
      id: 2,
      name: 'ResNet50-Transfer',
      type: '图像分类',
      accuracy: '94.3%',
      size: '98MB',
      inputFormats: ['图片'],
      status: 'ready'
    },
    {
      id: 3,
      name: 'SAM-B-Finetuned',
      type: '图像分割',
      accuracy: '89.1%',
      size: '256MB',
      inputFormats: ['图片', '摄像头'],
      status: 'ready'
    }
  ]

  /**
   * 状态指示器组件
   */
  const StatusIndicator = ({ status }) => {
    const statusConfig = {
      running: {
        color: 'bg-blue-500',
        text: '运行中',
        icon: Play,
        animation: 'animate-pulse'
      },
      completed: {
        color: 'bg-emerald-500',
        text: '已完成',
        icon: CheckCircle2,
        animation: ''
      },
      processing: {
        color: 'bg-amber-500',
        text: '处理中',
        icon: RefreshCw,
        animation: 'animate-spin'
      },
      failed: {
        color: 'bg-red-500',
        text: '推理失败',
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
      running: 'bg-blue-500',
      completed: 'bg-emerald-500',
      processing: 'bg-amber-500',
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
   * 推理任务卡片组件
   */
  const InferenceTaskCard = ({ task }) => (
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
              {task.inputType === 'images' && <Image size={14} />}
              {task.inputType === 'video' && <Video size={14} />}
              {task.inputType === 'webcam' && <Camera size={14} />}
              {task.inputType === 'text' && <FileText size={14} />}
              {task.inputType === 'images' && '图片'}
              {task.inputType === 'video' && '视频'}
              {task.inputType === 'webcam' && '摄像头'}
              {task.inputType === 'text' && '文本'}
            </span>
          </div>
        </div>
        <StatusIndicator status={task.status} />
      </div>

      {/* 进度条 */}
      {task.status !== 'running' && task.status !== 'completed' && (
        <ProgressBar progress={task.progress} status={task.status} />
      )}

      {/* 推理信息 */}
      <div className="grid grid-cols-2 gap-4 mt-4">
        <div>
          <div className="text-xs text-slate-400 mb-1">处理进度</div>
          <div className="text-sm font-medium text-white">
            {task.processedCount} / {task.inputCount}
          </div>
        </div>
        <div>
          <div className="text-xs text-slate-400 mb-1">推理时间</div>
          <div className="text-sm font-medium text-white">
            {task.duration || '进行中'}
          </div>
        </div>
      </div>

      {/* 性能指标 */}
      {(task.accuracy > 0 || task.inferenceTime > 0) && (
        <div className="mt-4 p-3 bg-slate-700/30 rounded-lg">
          <div className="grid grid-cols-2 gap-3 text-xs">
            {task.accuracy > 0 && (
              <div className="flex justify-between">
                <span className="text-slate-400">准确率:</span>
                <span className="text-white font-medium">{(task.accuracy * 100).toFixed(1)}%</span>
              </div>
            )}
            {task.inferenceTime > 0 && (
              <div className="flex justify-between">
                <span className="text-slate-400">推理速度:</span>
                <span className="text-white font-medium">{task.inferenceTime}s</span>
              </div>
            )}
            {task.confidence && (
              <div className="flex justify-between">
                <span className="text-slate-400">置信度:</span>
                <span className="text-white font-medium">{task.confidence}</span>
              </div>
            )}
            {task.fps && (
              <div className="flex justify-between">
                <span className="text-slate-400">FPS:</span>
                <span className="text-white font-medium">{task.fps}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* 实时状态（仅运行中的任务） */}
      {task.status === 'running' && (
        <div className="mt-4 p-3 bg-blue-500/10 border border-blue-500/30 rounded-lg">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
              <span className="text-sm text-blue-400 font-medium">实时推理中</span>
            </div>
            <div className="text-xs text-slate-400">
              {task.fps && `${task.fps} FPS`}
              {task.resolution && ` · ${task.resolution}`}
            </div>
          </div>
        </div>
      )}

      {/* 错误信息（失败的任务） */}
      {task.status === 'failed' && task.error && (
        <div className="mt-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
          <div className="flex items-center gap-2">
            <XCircle size={16} className="text-red-400" />
            <span className="text-sm text-red-400">错误: {task.error}</span>
          </div>
        </div>
      )}

      {/* 操作按钮 */}
      <div className="flex items-center gap-2 mt-4 pt-4 border-t border-slate-700">
        <button className={`flex-1 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
          task.status === 'running'
            ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30 border border-red-500/30'
            : task.status === 'completed'
            ? 'bg-blue-500/20 text-blue-400 hover:bg-blue-500/30 border border-blue-500/30'
            : task.status === 'processing'
            ? 'bg-amber-500/20 text-amber-400 hover:bg-amber-500/30 border border-amber-500/30'
            : 'bg-slate-700/50 text-slate-400 hover:bg-slate-700/70 border border-slate-600/50'
        }`}>
          {task.status === 'running' && <><Square size={14} className="inline mr-1" /> 停止</>}
          {task.status === 'completed' && <><Eye size={14} className="inline mr-1" /> 查看结果</>}
          {task.status === 'processing' && <><Pause size={14} className="inline mr-1" /> 暂停</>}
          {task.status === 'failed' && <><RefreshCw size={14} className="inline mr-1" /> 重试</>}
          {task.status === 'queued' && <><Clock size={14} className="inline mr-1" /> 等待中</>}
        </button>
        <button className="px-3 py-2 bg-slate-700/50 text-slate-300 rounded-lg hover:bg-slate-700/70 transition-colors border border-slate-600/50">
          <Download size={14} />
        </button>
        <button className="px-3 py-2 bg-slate-700/50 text-slate-300 rounded-lg hover:bg-slate-700/70 transition-colors border border-slate-600/50">
          <Share2 size={14} />
        </button>
        <button className="px-3 py-2 bg-slate-700/50 text-slate-300 rounded-lg hover:bg-slate-700/70 transition-colors border border-slate-600/50">
          <Trash2 size={14} />
        </button>
      </div>
    </div>
  )

  /**
   * 模型卡片组件
   */
  const ModelCard = ({ model }) => (
    <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-4 hover:shadow-lg transition-all duration-300 hover:border-slate-600 cursor-pointer group">
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-white font-medium">{model.name}</h4>
        <StatusIndicator status={model.status} />
      </div>
      <div className="space-y-2 text-sm">
        <div className="flex justify-between">
          <span className="text-slate-400">类型</span>
          <span className="text-white">{model.type}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-slate-400">准确率</span>
          <span className="text-emerald-400 font-medium">{model.accuracy}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-slate-400">大小</span>
          <span className="text-white">{model.size}</span>
        </div>
      </div>
      <div className="mt-3 pt-3 border-t border-slate-700">
        <div className="text-xs text-slate-400 mb-2">支持输入</div>
        <div className="flex flex-wrap gap-1">
          {model.inputFormats.map((format, index) => (
            <span key={index} className="px-2 py-1 bg-slate-700/50 text-xs text-slate-300 rounded">
              {format}
            </span>
          ))}
        </div>
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
              <TrendingUp size={16} className={change > 0 ? 'text-emerald-500' : 'text-red-500'} />
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
          <h1 className="text-2xl font-bold text-white mb-2">推理测试</h1>
          <p className="text-slate-400">运行模型推理并分析性能结果</p>
        </div>
        <div className="flex items-center gap-3">
          <div className="relative">
            <Search size={18} className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400" />
            <input
              type="text"
              placeholder="搜索推理任务..."
              className="pl-10 pr-4 py-2 bg-slate-800/50 border border-slate-700 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:border-blue-500 w-64"
            />
          </div>
          <button className="px-4 py-2 bg-slate-700/50 border border-slate-600/50 text-slate-300 rounded-lg hover:bg-slate-700/70 transition-colors flex items-center gap-2">
            <Filter size={16} />
            筛选
          </button>
          <button className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors flex items-center gap-2">
            <Plus size={16} />
            新建推理
          </button>
        </div>
      </div>

      {/* 统计卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          icon={Play}
          title="活跃推理"
          value="2"
          change={15}
          color="bg-blue-500"
        />
        <StatCard
          icon={CheckCircle2}
          title="今日完成"
          value="28"
          change={12}
          color="bg-emerald-500"
        />
        <StatCard
          icon={Zap}
          title="平均推理时间"
          value="0.089s"
          change={-8}
          color="bg-purple-500"
        />
        <StatCard
          icon={TrendingUp}
          title="平均准确率"
          value="91.2%"
          change={5}
          color="bg-orange-500"
        />
      </div>

      {/* 主要内容区域 */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 推理任务列表 */}
        <div className="lg:col-span-2 space-y-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white">推理任务</h3>
            <button className="text-sm text-blue-400 hover:text-blue-300 transition-colors">
              查看全部
            </button>
          </div>
          <div className="space-y-4">
            {inferenceTasks.map((task) => (
              <InferenceTaskCard key={task.id} task={task} />
            ))}
          </div>
        </div>

        {/* 侧边栏 - 可用模型 */}
        <div className="space-y-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white">可用模型</h3>
            <Settings size={18} className="text-slate-400 cursor-pointer hover:text-slate-300 transition-colors" />
          </div>
          <div className="space-y-3">
            {availableModels.map((model) => (
              <ModelCard key={model.id} model={model} />
            ))}
          </div>

          {/* 快速操作 */}
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-4">
            <h4 className="text-white font-medium mb-3">快速操作</h4>
            <div className="space-y-2">
              <button className="w-full px-3 py-2 bg-blue-500/10 border border-blue-500/30 rounded-lg hover:bg-blue-500/20 transition-colors text-blue-400 hover:text-blue-300 text-sm font-medium flex items-center justify-center gap-2">
                <Upload size={16} />
                上传文件推理
              </button>
              <button className="w-full px-3 py-2 bg-purple-500/10 border border-purple-500/30 rounded-lg hover:bg-purple-500/20 transition-colors text-purple-400 hover:text-purple-300 text-sm font-medium flex items-center justify-center gap-2">
                <Camera size={16} />
                摄像头实时推理
              </button>
              <button className="w-full px-3 py-2 bg-emerald-500/10 border border-emerald-500/30 rounded-lg hover:bg-emerald-500/20 transition-colors text-emerald-400 hover:text-emerald-300 text-sm font-medium flex items-center justify-center gap-2">
                <BarChart3 size={16} />
                批量性能测试
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* 性能监控 */}
      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Monitor size={20} />
          系统性能监控
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-slate-400">GPU使用率</span>
              <span className="text-sm font-medium text-white">65%</span>
            </div>
            <div className="w-full bg-slate-700 rounded-full h-3">
              <div className="h-3 bg-purple-500 rounded-full" style={{ width: '65%' }}></div>
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-slate-400">内存使用</span>
              <span className="text-sm font-medium text-white">42%</span>
            </div>
            <div className="w-full bg-slate-700 rounded-full h-3">
              <div className="h-3 bg-green-500 rounded-full" style={{ width: '42%' }}></div>
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-slate-400">推理队列</span>
              <span className="text-sm font-medium text-white">3个任务</span>
            </div>
            <div className="w-full bg-slate-700 rounded-full h-3">
              <div className="h-3 bg-blue-500 rounded-full" style={{ width: '30%' }}></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Inference