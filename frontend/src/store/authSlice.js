/**
 * 认证状态管理
 */

import { createSlice } from '@reduxjs/toolkit'

const initialState = {
  isAuthenticated: false,
  user: null,
  token: null,
  loading: false,
  error: null
}

const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    // 登录开始
    loginStart: (state) => {
      state.loading = true
      state.error = null
    },
    // 登录成功
    loginSuccess: (state, action) => {
      state.loading = false
      state.isAuthenticated = true
      state.user = action.payload.user
      state.token = action.payload.token
    },
    // 登录失败
    loginFailure: (state, action) => {
      state.loading = false
      state.error = action.payload
    },
    // 登出
    logout: (state) => {
      state.isAuthenticated = false
      state.user = null
      state.token = null
      state.error = null
    },
    // 清除错误
    clearError: (state) => {
      state.error = null
    },
    // 更新用户信息
    updateUser: (state, action) => {
      state.user = { ...state.user, ...action.payload }
    }
  }
})

export const {
  loginStart,
  loginSuccess,
  loginFailure,
  logout,
  clearError,
  updateUser
} = authSlice.actions

export default authSlice.reducer