/**
 * 认证相关API服务
 */

import { apiMethods } from './api'

export const authService = {
  /**
   * 用户登录
   * @param {Object} credentials - 登录凭证
   * @param {string} credentials.username - 用户名
   * @param {string} credentials.password - 密码
   * @returns {Promise} API响应
   */
  login: (credentials) => {
    return apiMethods.post('/auth/login', credentials)
  },

  /**
   * 用户注册
   * @param {Object} userData - 用户数据
   * @returns {Promise} API响应
   */
  register: (userData) => {
    return apiMethods.post('/auth/register', userData)
  },

  /**
   * 用户登出
   * @returns {Promise} API响应
   */
  logout: () => {
    return apiMethods.post('/auth/logout')
  },

  /**
   * 刷新token
   * @returns {Promise} API响应
   */
  refreshToken: () => {
    return apiMethods.post('/auth/refresh')
  },

  /**
   * 获取当前用户信息
   * @returns {Promise} API响应
   */
  getCurrentUser: () => {
    return apiMethods.get('/auth/me')
  },

  /**
   * 更新用户信息
   * @param {Object} userData - 用户数据
   * @returns {Promise} API响应
   */
  updateProfile: (userData) => {
    return apiMethods.put('/auth/profile', userData)
  },

  /**
   * 修改密码
   * @param {Object} passwordData - 密码数据
   * @param {string} passwordData.oldPassword - 旧密码
   * @param {string} passwordData.newPassword - 新密码
   * @returns {Promise} API响应
   */
  changePassword: (passwordData) => {
    return apiMethods.post('/auth/change-password', passwordData)
  },

  /**
   * 忘记密码
   * @param {string} email - 邮箱地址
   * @returns {Promise} API响应
   */
  forgotPassword: (email) => {
    return apiMethods.post('/auth/forgot-password', { email })
  },

  /**
   * 重置密码
   * @param {Object} resetData - 重置数据
   * @param {string} resetData.token - 重置token
   * @param {string} resetData.password - 新密码
   * @returns {Promise} API响应
   */
  resetPassword: (resetData) => {
    return apiMethods.post('/auth/reset-password', resetData)
  }
}

// 导出各个方法
export const {
  login,
  register,
  logout,
  refreshToken,
  getCurrentUser,
  updateProfile,
  changePassword,
  forgotPassword,
  resetPassword
} = authService

export default authService