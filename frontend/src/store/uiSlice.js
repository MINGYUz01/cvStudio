/**
 * UI状态管理
 * 管理应用的全局UI状态
 */

import { createSlice } from '@reduxjs/toolkit'

const initialState = {
  // 侧边栏状态
  sidebar: {
    isOpen: true,
    isCollapsed: false
  },
  // 主题设置
  theme: {
    mode: 'dark', // dark, light
    primaryColor: '#3b82f6',
    accentColor: '#6366f1'
  },
  // 通知系统
  notifications: [],
  // 模态框状态
  modals: {
    datasetUpload: false,
    modelCreate: false,
    trainingConfig: false,
    inferenceSettings: false,
    userSettings: false,
    confirmDialog: null
  },
  // 加载状态
  globalLoading: false,
  loadingMessage: '',
  // 页面标题
  pageTitle: 'CV Studio',
  // 面包屑导航
  breadcrumbs: [],
  // 全局搜索
  globalSearch: {
    isOpen: false,
    query: '',
    results: []
  },
  // 用户偏好
  preferences: {
    language: 'zh-CN',
    autoSave: true,
    showNotifications: true,
    compactMode: false,
    gpuAcceleration: true
  },
  // 快捷键提示
  shortcutsVisible: false,
  // 系统消息
  systemMessage: null,
  // 版本信息
  version: {
    current: '1.0.0',
    updateAvailable: false,
    latestVersion: '1.0.0'
  }
}

const uiSlice = createSlice({
  name: 'ui',
  initialState,
  reducers: {
    // 切换侧边栏
    toggleSidebar: (state) => {
      state.sidebar.isOpen = !state.sidebar.isOpen
    },
    // 设置侧边栏状态
    setSidebarState: (state, action) => {
      state.sidebar = { ...state.sidebar, ...action.payload }
    },
    // 折叠/展开侧边栏
    toggleSidebarCollapse: (state) => {
      state.sidebar.isCollapsed = !state.sidebar.isCollapsed
    },
    // 设置主题
    setTheme: (state, action) => {
      state.theme = { ...state.theme, ...action.payload }
    },
    // 添加通知
    addNotification: (state, action) => {
      const notification = {
        id: Date.now() + Math.random(),
        timestamp: new Date().toISOString(),
        ...action.payload
      }
      state.notifications.unshift(notification)
      // 限制通知数量
      if (state.notifications.length > 50) {
        state.notifications = state.notifications.slice(0, 50)
      }
    },
    // 移除通知
    removeNotification: (state, action) => {
      const id = action.payload
      state.notifications = state.notifications.filter(n => n.id !== id)
    },
    // 清空通知
    clearNotifications: (state) => {
      state.notifications = []
    },
    // 标记通知为已读
    markNotificationAsRead: (state, action) => {
      const id = action.payload
      const notification = state.notifications.find(n => n.id === id)
      if (notification) {
        notification.read = true
      }
    },
    // 设置模态框状态
    setModalState: (state, action) => {
      const { modal, isOpen, data } = action.payload
      state.modals[modal] = isOpen
      if (data !== undefined) {
        state.modals[modal + 'Data'] = data
      }
    },
    // 关闭所有模态框
    closeAllModals: (state) => {
      Object.keys(state.modals).forEach(key => {
        if (typeof state.modals[key] === 'boolean') {
          state.modals[key] = false
        }
      })
      state.modals.confirmDialog = null
    },
    // 设置全局加载状态
    setGlobalLoading: (state, action) => {
      const { loading, message } = action.payload
      state.globalLoading = loading
      if (message !== undefined) {
        state.loadingMessage = message
      }
    },
    // 设置页面标题
    setPageTitle: (state, action) => {
      state.pageTitle = action.payload
      // 同时更新浏览器标题
      if (typeof document !== 'undefined') {
        document.title = action.payload
      }
    },
    // 设置面包屑
    setBreadcrumbs: (state, action) => {
      state.breadcrumbs = action.payload
    },
    // 添加面包屑项目
    addBreadcrumb: (state, action) => {
      state.breadcrumbs.push(action.payload)
    },
    // 打开全局搜索
    openGlobalSearch: (state) => {
      state.globalSearch.isOpen = true
      state.globalSearch.query = ''
      state.globalSearch.results = []
    },
    // 关闭全局搜索
    closeGlobalSearch: (state) => {
      state.globalSearch.isOpen = false
    },
    // 设置搜索查询
    setSearchQuery: (state, action) => {
      state.globalSearch.query = action.payload
    },
    // 设置搜索结果
    setSearchResults: (state, action) => {
      state.globalSearch.results = action.payload
    },
    // 更新用户偏好
    updatePreferences: (state, action) => {
      state.preferences = { ...state.preferences, ...action.payload }
    },
    // 切换快捷键提示
    toggleShortcutsVisible: (state) => {
      state.shortcutsVisible = !state.shortcutsVisible
    },
    // 设置系统消息
    setSystemMessage: (state, action) => {
      state.systemMessage = {
        id: Date.now(),
        timestamp: new Date().toISOString(),
        ...action.payload
      }
    },
    // 清除系统消息
    clearSystemMessage: (state) => {
      state.systemMessage = null
    },
    // 设置版本信息
    setVersionInfo: (state, action) => {
      state.version = { ...state.version, ...action.payload }
    },
    // 重置UI状态
    resetUIState: (state) => {
      // 保留一些重要设置
      const preservedSettings = {
        theme: state.theme,
        preferences: state.preferences
      }
      return {
        ...initialState,
        ...preservedSettings
      }
    }
  }
})

export const {
  toggleSidebar,
  setSidebarState,
  toggleSidebarCollapse,
  setTheme,
  addNotification,
  removeNotification,
  clearNotifications,
  markNotificationAsRead,
  setModalState,
  closeAllModals,
  setGlobalLoading,
  setPageTitle,
  setBreadcrumbs,
  addBreadcrumb,
  openGlobalSearch,
  closeGlobalSearch,
  setSearchQuery,
  setSearchResults,
  updatePreferences,
  toggleShortcutsVisible,
  setSystemMessage,
  clearSystemMessage,
  setVersionInfo,
  resetUIState
} = uiSlice.actions

export default uiSlice.reducer