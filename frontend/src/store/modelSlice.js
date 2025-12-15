/**
 * 模型状态管理
 */

import { createSlice } from '@reduxjs/toolkit'

const initialState = {
  models: [],
  currentModel: null,
  templates: [],
  loading: false,
  saving: false,
  error: null,
  pagination: {
    page: 1,
    pageSize: 20,
    total: 0
  },
  canvasState: {
    nodes: [],
    edges: [],
    selectedNode: null,
    viewport: { x: 0, y: 0, zoom: 1 }
  }
}

const modelSlice = createSlice({
  name: 'model',
  initialState,
  reducers: {
    // 获取模型列表开始
    fetchModelsStart: (state) => {
      state.loading = true
      state.error = null
    },
    // 获取模型列表成功
    fetchModelsSuccess: (state, action) => {
      state.loading = false
      state.models = action.payload.models
      state.pagination = action.payload.pagination
    },
    // 获取模型列表失败
    fetchModelsFailure: (state, action) => {
      state.loading = false
      state.error = action.payload
    },
    // 设置当前模型
    setCurrentModel: (state, action) => {
      state.currentModel = action.payload
      state.canvasState = action.payload?.graph || state.canvasState
    },
    // 保存模型开始
    saveModelStart: (state) => {
      state.saving = true
      state.error = null
    },
    // 保存模型成功
    saveModelSuccess: (state, action) => {
      state.saving = false
      if (action.payload.id) {
        // 更新现有模型
        const index = state.models.findIndex(model => model.id === action.payload.id)
        if (index !== -1) {
          state.models[index] = action.payload
        }
        if (state.currentModel?.id === action.payload.id) {
          state.currentModel = action.payload
        }
      } else {
        // 新增模型
        state.models.unshift(action.payload)
      }
    },
    // 保存模型失败
    saveModelFailure: (state, action) => {
      state.saving = false
      state.error = action.payload
    },
    // 删除模型
    deleteModel: (state, action) => {
      state.models = state.models.filter(model => model.id !== action.payload)
      if (state.currentModel?.id === action.payload) {
        state.currentModel = null
      }
    },
    // 更新画布状态
    updateCanvasState: (state, action) => {
      state.canvasState = { ...state.canvasState, ...action.payload }
    },
    // 设置选中节点
    setSelectedNode: (state, action) => {
      state.canvasState.selectedNode = action.payload
    },
    // 添加节点
    addNode: (state, action) => {
      state.canvasState.nodes.push(action.payload)
    },
    // 更新节点
    updateNode: (state, action) => {
      const { id, data } = action.payload
      const node = state.canvasState.nodes.find(node => node.id === id)
      if (node) {
        node.data = { ...node.data, ...data }
      }
    },
    // 删除节点
    removeNode: (state, action) => {
      state.canvasState.nodes = state.canvasState.nodes.filter(node => node.id !== action.payload)
      // 删除相关的边
      state.canvasState.edges = state.canvasState.edges.filter(
        edge => edge.source !== action.payload && edge.target !== action.payload
      )
    },
    // 添加边
    addEdge: (state, action) => {
      state.canvasState.edges.push(action.payload)
    },
    // 删除边
    removeEdge: (state, action) => {
      state.canvasState.edges = state.canvasState.edges.filter(edge => edge.id !== action.payload)
    },
    // 清除错误
    clearError: (state) => {
      state.error = null
    }
  }
})

export const {
  fetchModelsStart,
  fetchModelsSuccess,
  fetchModelsFailure,
  setCurrentModel,
  saveModelStart,
  saveModelSuccess,
  saveModelFailure,
  deleteModel,
  updateCanvasState,
  setSelectedNode,
  addNode,
  updateNode,
  removeNode,
  addEdge,
  removeEdge,
  clearError
} = modelSlice.actions

export default modelSlice.reducer