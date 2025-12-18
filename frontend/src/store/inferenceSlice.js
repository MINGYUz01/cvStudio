/**
 * 推理状态管理
 */

import { createSlice } from '@reduxjs/toolkit'

const initialState = {
  // 推理任务列表
  inferenceTasks: [],
  // 当前推理任务
  currentTask: null,
  // 推理结果
  inferenceResults: [],
  // 推理配置
  inferenceConfig: {
    modelId: null,
    batchSize: 1,
    confidenceThreshold: 0.5,
    nmsThreshold: 0.4,
    inputSize: [640, 640],
    device: 'cuda'
  },
  // 推理状态
  inferenceStatus: 'idle', // idle, running, completed, failed
  // 性能指标
  performanceMetrics: {
    fps: 0,
    latency: 0,
    throughput: 0,
    gpuMemory: 0,
    cpuUsage: 0
  },
  // 加载状态
  loading: false,
  // 错误信息
  error: null,
  // 批量推理队列
  batchQueue: [],
  // 实时推理状态
  realtimeInference: {
    isProcessing: false,
    processedCount: 0,
    totalCount: 0,
    currentImage: null
  }
}

const inferenceSlice = createSlice({
  name: 'inference',
  initialState,
  reducers: {
    // 设置推理任务列表
    setInferenceTasks: (state, action) => {
      state.inferenceTasks = action.payload
    },
    // 添加推理任务
    addInferenceTask: (state, action) => {
      state.inferenceTasks.unshift(action.payload)
    },
    // 设置当前推理任务
    setCurrentTask: (state, action) => {
      state.currentTask = action.payload
    },
    // 更新推理任务状态
    updateTaskStatus: (state, action) => {
      const { taskId, status, progress, results } = action.payload
      const task = state.inferenceTasks.find(t => t.id === taskId)
      if (task) {
        task.status = status
        if (progress !== undefined) task.progress = progress
        if (results) task.results = results
      }
      if (state.currentTask && state.currentTask.id === taskId) {
        state.currentTask.status = status
        if (progress !== undefined) state.currentTask.progress = progress
        if (results) state.currentTask.results = results
      }
    },
    // 设置推理结果
    setInferenceResults: (state, action) => {
      state.inferenceResults = action.payload
    },
    // 添加推理结果
    addInferenceResult: (state, action) => {
      state.inferenceResults.unshift(action.payload)
    },
    // 更新推理配置
    updateInferenceConfig: (state, action) => {
      state.inferenceConfig = { ...state.inferenceConfig, ...action.payload }
    },
    // 设置推理状态
    setInferenceStatus: (state, action) => {
      state.inferenceStatus = action.payload
    },
    // 开始推理
    startInference: (state, action) => {
      state.loading = false
      state.error = null
      state.inferenceStatus = 'running'
      if (action.payload?.batchMode) {
        state.realtimeInference.isProcessing = true
        state.realtimeInference.totalCount = action.payload.totalCount || 0
        state.realtimeInference.processedCount = 0
      }
    },
    // 推理完成
    completeInference: (state, action) => {
      state.inferenceStatus = 'completed'
      state.realtimeInference.isProcessing = false
      if (action.payload) {
        // 更新结果
        if (Array.isArray(action.payload)) {
          state.inferenceResults = [...action.payload, ...state.inferenceResults]
        } else {
          state.inferenceResults.unshift(action.payload)
        }
      }
    },
    // 推理失败
    failInference: (state, action) => {
      state.inferenceStatus = 'failed'
      state.error = action.payload
      state.realtimeInference.isProcessing = false
    },
    // 更新性能指标
    updatePerformanceMetrics: (state, action) => {
      state.performanceMetrics = { ...state.performanceMetrics, ...action.payload }
    },
    // 更新实时推理状态
    updateRealtimeInference: (state, action) => {
      state.realtimeInference = { ...state.realtimeInference, ...action.payload }
    },
    // 增加处理计数
    incrementProcessedCount: (state) => {
      state.realtimeInference.processedCount += 1
    },
    // 重置推理状态
    resetInference: (state) => {
      state.inferenceStatus = 'idle'
      state.currentTask = null
      state.error = null
      state.realtimeInference = {
        isProcessing: false,
        processedCount: 0,
        totalCount: 0,
        currentImage: null
      }
    },
    // 添加到批量队列
    addToBatchQueue: (state, action) => {
      state.batchQueue.push(...action.payload)
    },
    // 从批量队列移除
    removeFromBatchQueue: (state, action) => {
      const index = state.batchQueue.findIndex(item => item.id === action.payload)
      if (index > -1) {
        state.batchQueue.splice(index, 1)
      }
    },
    // 清空批量队列
    clearBatchQueue: (state) => {
      state.batchQueue = []
    },
    // 删除推理任务
    deleteInferenceTask: (state, action) => {
      const taskId = action.payload
      state.inferenceTasks = state.inferenceTasks.filter(t => t.id !== taskId)
      if (state.currentTask && state.currentTask.id === taskId) {
        state.currentTask = null
        state.inferenceStatus = 'idle'
      }
    },
    // 清除错误
    clearError: (state) => {
      state.error = null
    },
    // 清空推理结果
    clearInferenceResults: (state) => {
      state.inferenceResults = []
    }
  }
})

export const {
  setInferenceTasks,
  addInferenceTask,
  setCurrentTask,
  updateTaskStatus,
  setInferenceResults,
  addInferenceResult,
  updateInferenceConfig,
  setInferenceStatus,
  startInference,
  completeInference,
  failInference,
  updatePerformanceMetrics,
  updateRealtimeInference,
  incrementProcessedCount,
  resetInference,
  addToBatchQueue,
  removeFromBatchQueue,
  clearBatchQueue,
  deleteInferenceTask,
  clearError,
  clearInferenceResults
} = inferenceSlice.actions

export default inferenceSlice.reducer