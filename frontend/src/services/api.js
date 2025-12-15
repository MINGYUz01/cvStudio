/**
 * API基础配置
 * 使用axios作为HTTP客户端
 */

import axios from 'axios'

// 创建axios实例
const api = axios.create({
  baseURL: '/api/v1',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  }
})

// 请求拦截器
api.interceptors.request.use(
  (config) => {
    // 从localStorage获取token
    const token = localStorage.getItem('token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// 响应拦截器
api.interceptors.response.use(
  (response) => {
    return response.data
  },
  (error) => {
    // 统一错误处理
    if (error.response) {
      // 服务器返回的错误
      const { status, data } = error.response
      
      switch (status) {
        case 401:
          // 未授权，跳转到登录页
          localStorage.removeItem('token')
          window.location.href = '/login'
          break
        case 403:
          // 权限不足
          console.error('权限不足', data.message)
          break
        case 404:
          // 资源未找到
          console.error('资源未找到', data.message)
          break
        case 500:
          // 服务器错误
          console.error('服务器错误', data.message)
          break
        default:
          // 其他错误
          console.error('API错误', data.message || error.message)
      }
      
      return Promise.reject(error.response.data)
    } else if (error.request) {
      // 网络错误
      console.error('网络错误', error.message)
      return Promise.reject({ message: '网络连接失败' })
    } else {
      // 其他错误
      console.error('请求错误', error.message)
      return Promise.reject({ message: error.message })
    }
  }
)

// API方法封装
export const apiMethods = {
  // GET请求
  get: (url, config = {}) => api.get(url, config),
  
  // POST请求
  post: (url, data = {}, config = {}) => api.post(url, data, config),
  
  // PUT请求
  put: (url, data = {}, config = {}) => api.put(url, data, config),
  
  // DELETE请求
  delete: (url, config = {}) => api.delete(url, config),
  
  // PATCH请求
  patch: (url, data = {}, config = {}) => api.patch(url, data, config),
  
  // 文件上传
  upload: (url, formData, config = {}) => {
    return api.post(url, formData, {
      ...config,
      headers: {
        'Content-Type': 'multipart/form-data',
        ...config.headers,
      },
    })
  }
}

export default api