/**
 * 认证相关API服务
 */

import { apiClient } from './api';

/**
 * 用户信息接口
 */
export interface User {
  id: number;
  username: string;
  email: string;
  is_active: boolean;
  is_superuser: boolean;
  created_at: string;
}

/**
 * Token响应接口
 */
export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  user: User;
}

/**
 * 用户注册数据接口
 */
export interface RegisterData {
  username: string;
  email: string;
  password: string;
}

/**
 * 用户登录数据接口
 */
export interface LoginData {
  username: string;
  password: string;
}

/**
 * 用户配置接口
 */
export interface UserConfig {
  theme?: string;
  language?: string;
  default_dataset_id?: number;
  default_model_id?: number;
  notifications_enabled?: boolean;
  auto_save?: boolean;
}

/**
 * 修改密码数据接口
 */
export interface PasswordChangeData {
  old_password: string;
  new_password: string;
}

/**
 * 认证服务类
 */
class AuthService {
  /**
   * 用户注册
   */
  async register(data: RegisterData): Promise<User> {
    const response = await apiClient.post<User>('/auth/register', data);
    return response;
  }

  /**
   * 用户登录
   */
  async login(data: LoginData): Promise<TokenResponse> {
    const response = await apiClient.post<TokenResponse>('/auth/login', data);

    // 保存tokens和用户信息
    apiClient.setTokens(response.access_token, response.refresh_token);
    localStorage.setItem('user', JSON.stringify(response.user));

    return response;
  }

  /**
   * 刷新token
   */
  async refreshToken(refreshToken: string): Promise<TokenResponse> {
    const response = await apiClient.post<TokenResponse>('/auth/refresh', {
      refresh_token: refreshToken,
    });

    // 更新tokens和用户信息
    apiClient.setTokens(response.access_token, response.refresh_token);
    localStorage.setItem('user', JSON.stringify(response.user));

    return response;
  }

  /**
   * 获取当前用户信息
   */
  async getCurrentUser(): Promise<User> {
    const response = await apiClient.get<User>('/auth/me');
    return response;
  }

  /**
   * 获取用户配置
   */
  async getUserConfig(): Promise<UserConfig> {
    const response = await apiClient.get<UserConfig>('/auth/config');
    return response;
  }

  /**
   * 更新用户配置
   */
  async updateUserConfig(config: UserConfig): Promise<UserConfig> {
    const response = await apiClient.put<UserConfig>('/auth/config', config);
    return response;
  }

  /**
   * 修改密码
   */
  async changePassword(data: PasswordChangeData): Promise<{ message: string }> {
    const response = await apiClient.post<{ message: string }>('/auth/change-password', data);
    return response;
  }

  /**
   * 更新用户基本信息
   */
  async updateProfile(email: string): Promise<User> {
    const response = await apiClient.patch<User>('/auth/me', { email });
    return response;
  }

  /**
   * 登出
   */
  logout(): void {
    apiClient.clearTokens();
    localStorage.removeItem('user');
  }

  /**
   * 检查是否已登录
   */
  isAuthenticated(): boolean {
    return !!localStorage.getItem('access_token');
  }

  /**
   * 获取本地存储的用户信息
   */
  getLocalUser(): User | null {
    const userStr = localStorage.getItem('user');
    if (userStr) {
      try {
        return JSON.parse(userStr);
      } catch {
        return null;
      }
    }
    return null;
  }
}

// 创建全局认证服务实例
export const authService = new AuthService();

export default authService;
