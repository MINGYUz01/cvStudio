/**
 * 认证Hook
 */

import { useState, useEffect, useCallback } from 'react';
import { authService, User, LoginData, RegisterData, UserConfig } from '../services/auth';

interface UseAuthReturn {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  login: (data: LoginData) => Promise<void>;
  register: (data: RegisterData) => Promise<void>;
  logout: () => void;
  refreshUser: () => Promise<void>;
  updateUserConfig: (config: UserConfig) => Promise<void>;
}

/**
 * 认证Hook
 */
export function useAuth(): UseAuthReturn {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // 初始化：从本地存储加载用户信息
  useEffect(() => {
    const loadUser = () => {
      const localUser = authService.getLocalUser();
      if (localUser) {
        setUser(localUser);
      }
      setIsLoading(false);
    };

    loadUser();
  }, []);

  // 登录
  const login = useCallback(async (data: LoginData) => {
    setError(null);
    try {
      const response = await authService.login(data);
      setUser(response.user);
    } catch (err) {
      const message = err instanceof Error ? err.message : '登录失败';
      setError(message);
      throw err;
    }
  }, []);

  // 注册
  const register = useCallback(async (data: RegisterData) => {
    setError(null);
    try {
      const response = await authService.login(data);
      setUser(response.user);
    } catch (err) {
      const message = err instanceof Error ? err.message : '注册失败';
      setError(message);
      throw err;
    }
  }, []);

  // 登出
  const logout = useCallback(() => {
    authService.logout();
    setUser(null);
    setError(null);
  }, []);

  // 刷新用户信息
  const refreshUser = useCallback(async () => {
    setError(null);
    try {
      const currentUser = await authService.getCurrentUser();
      setUser(currentUser);
      localStorage.setItem('user', JSON.stringify(currentUser));
    } catch (err) {
      const message = err instanceof Error ? err.message : '获取用户信息失败';
      setError(message);
      throw err;
    }
  }, []);

  // 更新用户配置
  const updateUserConfig = useCallback(async (config: UserConfig) => {
    setError(null);
    try {
      await authService.updateUserConfig(config);
      // 刷新用户信息以获取最新配置
      await refreshUser();
    } catch (err) {
      const message = err instanceof Error ? err.message : '更新配置失败';
      setError(message);
      throw err;
    }
  }, [refreshUser]);

  return {
    user,
    isAuthenticated: !!user,
    isLoading,
    error,
    login,
    register,
    logout,
    refreshUser,
    updateUserConfig,
  };
}

export default useAuth;
