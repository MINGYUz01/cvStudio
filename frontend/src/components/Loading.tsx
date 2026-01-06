/**
 * 全局加载状态组件
 */

import React from 'react';

export interface LoadingProps {
  size?: 'small' | 'medium' | 'large';
  text?: string;
  fullScreen?: boolean;
}

export const Loading: React.FC<LoadingProps> = ({
  size = 'medium',
  text = '加载中...',
  fullScreen = false
}) => {
  const sizeClasses = {
    small: 'h-4 w-4 border-2',
    medium: 'h-12 w-12 border-4',
    large: 'h-16 w-16 border-4'
  };

  const spinner = (
    <div className="flex flex-col items-center justify-center">
      <div className={`${sizeClasses[size]} rounded-full border-t-cyan-500 border-slate-700 animate-spin`}></div>
      {text && <p className="mt-4 text-slate-400 text-sm">{text}</p>}
    </div>
  );

  if (fullScreen) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm">
        {spinner}
      </div>
    );
  }

  return <div className="flex items-center justify-center p-6">{spinner}</div>;
};

export default Loading;
