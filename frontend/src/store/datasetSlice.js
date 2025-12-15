/**
 * 数据集状态管理
 */

import { createSlice } from '@reduxjs/toolkit'

const initialState = {
  datasets: [],
  currentDataset: null,
  loading: false,
  uploading: false,
  error: null,
  pagination: {
    page: 1,
    pageSize: 20,
    total: 0
  },
  filters: {
    format: '',
    search: ''
  }
}

const datasetSlice = createSlice({
  name: 'dataset',
  initialState,
  reducers: {
    // 获取数据集列表开始
    fetchDatasetsStart: (state) => {
      state.loading = true
      state.error = null
    },
    // 获取数据集列表成功
    fetchDatasetsSuccess: (state, action) => {
      state.loading = false
      state.datasets = action.payload.datasets
      state.pagination = action.payload.pagination
    },
    // 获取数据集列表失败
    fetchDatasetsFailure: (state, action) => {
      state.loading = false
      state.error = action.payload
    },
    // 设置当前数据集
    setCurrentDataset: (state, action) => {
      state.currentDataset = action.payload
    },
    // 上传数据集开始
    uploadDatasetStart: (state) => {
      state.uploading = true
      state.error = null
    },
    // 上传数据集成功
    uploadDatasetSuccess: (state, action) => {
      state.uploading = false
      state.datasets.unshift(action.payload)
    },
    // 上传数据集失败
    uploadDatasetFailure: (state, action) => {
      state.uploading = false
      state.error = action.payload
    },
    // 删除数据集
    deleteDataset: (state, action) => {
      state.datasets = state.datasets.filter(dataset => dataset.id !== action.payload)
      if (state.currentDataset?.id === action.payload) {
        state.currentDataset = null
      }
    },
    // 更新过滤器
    updateFilters: (state, action) => {
      state.filters = { ...state.filters, ...action.payload }
    },
    // 更新分页
    updatePagination: (state, action) => {
      state.pagination = { ...state.pagination, ...action.payload }
    },
    // 清除错误
    clearError: (state) => {
      state.error = null
    }
  }
})

export const {
  fetchDatasetsStart,
  fetchDatasetsSuccess,
  fetchDatasetsFailure,
  setCurrentDataset,
  uploadDatasetStart,
  uploadDatasetSuccess,
  uploadDatasetFailure,
  deleteDataset,
  updateFilters,
  updatePagination,
  clearError
} = datasetSlice.actions

export default datasetSlice.reducer