/**
 * 训练状态管理
 */

import { createSlice } from '@reduxjs/toolkit'

const initialState = {
  // 训练任务列表
  trainingTasks: [],
  // 当前训练任务
  currentTask: null,
  // 训练历史记录
  trainingHistory: [],
  // 训练配置
  trainingConfig: {
    epochs: 100,
    batchSize: 32,
    learningRate: 0.001,
    optimizer: 'Adam',
    device: 'cuda',
    saveFrequency: 10
  },
  // 实时指标
  realtimeMetrics: {
    loss: [],
    accuracy: [],
    valLoss: [],
    valAccuracy: [],
    learningRate: []
  },
  // 训练状态
  trainingStatus: 'idle', // idle, running, paused, completed, failed
  // 加载状态
  loading: false,
  // 错误信息
  error: null,
  // GPU使用情况
  gpuUsage: {
    memoryUsed: 0,
    memoryTotal: 0,
    utilization: 0,
    temperature: 0
  }
}

const trainingSlice = createSlice({
  name: 'training',
  initialState,
  reducers: {
    // 设置训练任务列表
    setTrainingTasks: (state, action) => {
      state.trainingTasks = action.payload
    },
    // 添加训练任务
    addTrainingTask: (state, action) => {
      state.trainingTasks.unshift(action.payload)
    },
    // 设置当前训练任务
    setCurrentTask: (state, action) => {
      state.currentTask = action.payload
    },
    // 更新训练任务状态
    updateTaskStatus: (state, action) => {
      const { taskId, status, progress, metrics } = action.payload
      const task = state.trainingTasks.find(t => t.id === taskId)
      if (task) {
        task.status = status
        if (progress !== undefined) task.progress = progress
        if (metrics) task.metrics = metrics
      }
      if (state.currentTask && state.currentTask.id === taskId) {
        state.currentTask.status = status
        if (progress !== undefined) state.currentTask.progress = progress
        if (metrics) state.currentTask.metrics = metrics
      }
    },
    // 更新训练配置
    updateTrainingConfig: (state, action) => {
      state.trainingConfig = { ...state.trainingConfig, ...action.payload }
    },
    // 更新实时指标
    updateRealtimeMetrics: (state, action) => {
      const { epoch, loss, accuracy, valLoss, valAccuracy, learningRate } = action.payload
      state.realtimeMetrics.loss.push({ epoch, value: loss })
      state.realtimeMetrics.accuracy.push({ epoch, value: accuracy })
      if (valLoss !== undefined) state.realtimeMetrics.valLoss.push({ epoch, value: valLoss })
      if (valAccuracy !== undefined) state.realtimeMetrics.valAccuracy.push({ epoch, value: valAccuracy })
      if (learningRate !== undefined) state.realtimeMetrics.learningRate.push({ epoch, value: learningRate })
    },
    // 设置训练状态
    setTrainingStatus: (state, action) => {
      state.trainingStatus = action.payload
    },
    // 开始训练
    startTraining: (state, action) => {
      state.loading = false
      state.error = null
      state.trainingStatus = 'running'
    },
    // 暂停训练
    pauseTraining: (state) => {
      state.trainingStatus = 'paused'
    },
    // 恢复训练
    resumeTraining: (state) => {
      state.trainingStatus = 'running'
    },
    // 停止训练
    stopTraining: (state) => {
      state.trainingStatus = 'idle'
      state.currentTask = null
    },
    // 训练完成
    completeTraining: (state, action) => {
      state.trainingStatus = 'completed'
      if (action.payload) {
        state.trainingHistory.unshift(action.payload)
      }
    },
    // 训练失败
    failTraining: (state, action) => {
      state.trainingStatus = 'failed'
      state.error = action.payload
    },
    // 更新GPU使用情况
    updateGpuUsage: (state, action) => {
      state.gpuUsage = { ...state.gpuUsage, ...action.payload }
    },
    // 清除错误
    clearError: (state) => {
      state.error = null
    },
    // 重置实时指标
    resetRealtimeMetrics: (state) => {
      state.realtimeMetrics = {
        loss: [],
        accuracy: [],
        valLoss: [],
        valAccuracy: [],
        learningRate: []
      }
    },
    // 删除训练任务
    deleteTrainingTask: (state, action) => {
      const taskId = action.payload
      state.trainingTasks = state.trainingTasks.filter(t => t.id !== taskId)
      if (state.currentTask && state.currentTask.id === taskId) {
        state.currentTask = null
        state.trainingStatus = 'idle'
      }
    }
  }
})

export const {
  setTrainingTasks,
  addTrainingTask,
  setCurrentTask,
  updateTaskStatus,
  updateTrainingConfig,
  updateRealtimeMetrics,
  setTrainingStatus,
  startTraining,
  pauseTraining,
  resumeTraining,
  stopTraining,
  completeTraining,
  failTraining,
  updateGpuUsage,
  clearError,
  resetRealtimeMetrics,
  deleteTrainingTask
} = trainingSlice.actions

export default trainingSlice.reducer