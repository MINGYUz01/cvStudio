/**
 * API基础配置
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';

/**
 * API请求配置
 */
interface RequestConfig extends RequestInit {
  params?: Record<string, string | number>;
}

/**
 * API客户端类
 */
class ApiClient {
  private baseURL: string;
  private defaultHeaders: Record<string, string>;

  constructor(baseURL: string) {
    this.baseURL = baseURL;
    this.defaultHeaders = {
      'Content-Type': 'application/json',
    };
  }

  /**
   * 获取认证token
   */
  private getToken(): string | null {
    return localStorage.getItem('access_token');
  }

  /**
   * 获取刷新token
   */
  private getRefreshToken(): string | null {
    return localStorage.getItem('refresh_token');
  }

  /**
   * 设置认证token
   */
  setTokens(accessToken: string, refreshToken: string): void {
    localStorage.setItem('access_token', accessToken);
    localStorage.setItem('refresh_token', refreshToken);
  }

  /**
   * 清除认证token
   */
  clearTokens(): void {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('user');
  }

  /**
   * 刷新访问令牌
   */
  private async refreshAccessToken(): Promise<string> {
    const refreshToken = this.getRefreshToken();
    if (!refreshToken) {
      throw new Error('No refresh token available');
    }

    const response = await fetch(`${this.baseURL}/auth/refresh`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ refresh_token: refreshToken }),
    });

    if (!response.ok) {
      throw new Error('Failed to refresh token');
    }

    const data = await response.json();
    this.setTokens(data.access_token, data.refresh_token);

    // 更新用户信息
    localStorage.setItem('user', JSON.stringify(data.user));

    return data.access_token;
  }

  /**
   * 发送HTTP请求
   */
  private async request<T>(
    endpoint: string,
    config: RequestConfig = {}
  ): Promise<T> {
    const { params, headers = {}, ...restConfig } = config;

    // 构建URL
    let url = `${this.baseURL}${endpoint}`;
    if (params) {
      const searchParams = new URLSearchParams();
      Object.entries(params).forEach(([key, value]) => {
        searchParams.append(key, String(value));
      });
      url += `?${searchParams.toString()}`;
    }

    // 添加认证token
    const token = this.getToken();
    const requestHeaders: HeadersInit = {
      ...this.defaultHeaders,
      ...headers,
    };

    if (token) {
      requestHeaders['Authorization'] = `Bearer ${token}`;
    }

    let response = await fetch(url, {
      ...restConfig,
      headers: requestHeaders,
    });

    // 处理401未授权错误，尝试刷新token
    if (response.status === 401 && restConfig.method !== 'POST') {
      try {
        const newToken = await this.refreshAccessToken();
        requestHeaders['Authorization'] = `Bearer ${newToken}`;
        response = await fetch(url, {
          ...restConfig,
          headers: requestHeaders,
        });
      } catch (error) {
        // 刷新token失败，清除所有token
        this.clearTokens();
        window.location.href = '/login';
        throw error;
      }
    }

    // 处理错误响应
    if (!response.ok) {
      const error = await response.json().catch(() => ({
        detail: '请求失败',
      }));
      throw new Error(error.detail || '请求失败');
    }

    return response.json();
  }

  /**
   * GET请求
   */
  async get<T>(endpoint: string, config?: RequestConfig): Promise<T> {
    return this.request<T>(endpoint, { ...config, method: 'GET' });
  }

  /**
   * POST请求
   */
  async post<T>(endpoint: string, data?: unknown, config?: RequestConfig): Promise<T> {
    return this.request<T>(endpoint, {
      ...config,
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  /**
   * PUT请求
   */
  async put<T>(endpoint: string, data?: unknown, config?: RequestConfig): Promise<T> {
    return this.request<T>(endpoint, {
      ...config,
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  /**
   * PATCH请求
   */
  async patch<T>(endpoint: string, data?: unknown, config?: RequestConfig): Promise<T> {
    return this.request<T>(endpoint, {
      ...config,
      method: 'PATCH',
      body: JSON.stringify(data),
    });
  }

  /**
   * DELETE请求
   */
  async delete<T>(endpoint: string, config?: RequestConfig): Promise<T> {
    return this.request<T>(endpoint, { ...config, method: 'DELETE' });
  }
}

// 创建全局API客户端实例
export const apiClient = new ApiClient(API_BASE_URL);

export default apiClient;
