/**
 * Redux Hooks
 * 提供类型安全的Redux hooks
 */

import { useDispatch, useSelector } from 'react-redux'
// import { RootState, AppDispatch } from './index'

// 便捷的hooks（暂时移除类型注解，待后续升级到TypeScript）
export const useAppDispatch = () => useDispatch()
export const useAppSelector = useSelector

// 便捷的selectors
export const useAuth = () => useAppSelector(state => state.auth)
export const useDataset = () => useAppSelector(state => state.dataset)
export const useModel = () => useAppSelector(state => state.model)
export const useTraining = () => useAppSelector(state => state.training)
export const useInference = () => useAppSelector(state => state.inference)
export const useUI = () => useAppSelector(state => state.ui)

// 组合hooks
export const useAuthState = () => {
  const auth = useAuth()
  return {
    isAuthenticated: auth.isAuthenticated,
    user: auth.user,
    token: auth.token,
    loading: auth.loading,
    error: auth.error
  }
}

export const useTrainingState = () => {
  const training = useTraining()
  return {
    tasks: training.trainingTasks,
    currentTask: training.currentTask,
    status: training.trainingStatus,
    config: training.trainingConfig,
    metrics: training.realtimeMetrics,
    gpuUsage: training.gpuUsage,
    loading: training.loading,
    error: training.error
  }
}

export const useInferenceState = () => {
  const inference = useInference()
  return {
    tasks: inference.inferenceTasks,
    currentTask: inference.currentTask,
    results: inference.inferenceResults,
    status: inference.inferenceStatus,
    config: inference.inferenceConfig,
    performance: inference.performanceMetrics,
    batchQueue: inference.batchQueue,
    realtime: inference.realtimeInference,
    loading: inference.loading,
    error: inference.error
  }
}

export const useUIState = () => {
  const ui = useUI()
  return {
    sidebar: ui.sidebar,
    theme: ui.theme,
    notifications: ui.notifications,
    modals: ui.modals,
    globalLoading: ui.globalLoading,
    pageTitle: ui.pageTitle,
    preferences: ui.preferences,
    globalSearch: ui.globalSearch
  }
}