/**
 * 路由保护组件
 * 用于保护需要登录才能访问的页面
 */

import { useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';

interface AuthGuardProps {
  children: React.ReactNode;
}

/**
 * 路由保护组件
 */
export function AuthGuard({ children }: AuthGuardProps) {
  const { isAuthenticated, isLoading } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    // 如果未登录且不在登录页面，重定向到登录页
    if (!isLoading && !isAuthenticated && location.pathname !== '/login') {
      // 保存当前路径，登录后可以返回
      navigate('/login', { replace: true, state: { from: location.pathname } });
    }
  }, [isAuthenticated, isLoading, navigate, location]);

  // 显示加载状态
  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-900">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
          <p className="mt-4 text-gray-400">加载中...</p>
        </div>
      </div>
    );
  }

  // 未认证时不渲染子组件
  if (!isAuthenticated) {
    return null;
  }

  // 已认证，渲染子组件
  return <>{children}</>;
}

export default AuthGuard;
