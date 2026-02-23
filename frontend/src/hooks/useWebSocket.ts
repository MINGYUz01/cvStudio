/**
 * WebSocket Hook
 * 封装WebSocket连接，提供自动重连和消息分发功能
 */

import { useEffect, useRef, useState, useCallback, useMemo } from 'react';

// WebSocket消息类型定义
export type WSMessage =
  | { type: 'connection_established'; data: any }
  | { type: 'system_stats'; data: SystemStats }
  | { type: 'log'; data: LogEntry }
  | { type: 'metrics_update'; data: MetricsEntry }
  | { type: 'status_change'; data: StatusChange }
  | { type: 'error'; data: any };

// 系统状态数据类型
export interface SystemStats {
  timestamp: string;
  cpu: {
    cpu_util: number;
    cpu_count: number;
    cpu_freq_current?: number;
    cpu_freq_max?: number;
  };
  memory: {
    ram_used: number;
    ram_total: number;
    ram_percent: number;
    ram_available: number;
  };
  disk: {
    disk_used: number;
    disk_total: number;
    disk_percent: number;
  };
  gpu: Array<{
    gpu_id: number;
    gpu_name: string;
    gpu_util: number;
    gpu_temp: number;
    vram_used: number;
    vram_total: number;
    vram_percent: number;
    power_usage?: number;
  }>;
  network: {
    bytes_sent: number;
    bytes_recv: number;
    packets_sent: number;
    packets_recv: number;
  };
}

// 日志条目类型
export interface LogEntry {
  level: string;
  message: string;
  source: string;
  timestamp: string;
}

// 指标条目类型
export interface MetricsEntry {
  epoch: number;
  timestamp: string;
  train_loss?: number;
  train_acc?: number;
  val_loss?: number;
  val_acc?: number;
  best_metric?: number;  // 添加最佳指标字段
  [key: string]: any;
}

// 状态变化类型
export interface StatusChange {
  status: string;
  current_epoch: number;
  total_epochs: number;
  best_metric?: number;  // 添加最佳指标字段
  started_at?: string;
  ended_at?: string;
  message: string;
}

// WebSocket Hook配置
interface UseWebSocketOptions {
  onMessage?: (message: WSMessage) => void;
  onSystemStats?: (data: SystemStats) => void;
  onLog?: (data: LogEntry) => void;
  onMetrics?: (data: MetricsEntry) => void;
  onStatusChange?: (data: StatusChange) => void;
  onError?: (error: Event) => void;
  reconnectInterval?: number; // 重连间隔（毫秒）
  maxReconnectAttempts?: number; // 最大重连次数
  enabled?: boolean; // 是否启用连接，默认true
}

// WebSocket Hook返回值
interface UseWebSocketReturn {
  socket: WebSocket | null;
  connected: boolean;
  connecting: boolean;
  error: Error | null;
  sendMessage: (message: any) => void;
  disconnect: () => void;
  reconnect: () => void;
}

/**
 * WebSocket Hook
 */
export const useWebSocket = (
  url: string,
  options: UseWebSocketOptions = {}
): UseWebSocketReturn => {
  const {
    onMessage,
    onSystemStats,
    onLog,
    onMetrics,
    onStatusChange,
    onError,
    reconnectInterval = 3000,
    maxReconnectAttempts = 10,
    enabled = true,
  } = options;

  const [connected, setConnected] = useState(false);
  const [connecting, setConnecting] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const socketRef = useRef<WebSocket | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const manualCloseRef = useRef(false);

  // 清理重连定时器
  const clearReconnectTimeout = () => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
  };

  // 连接WebSocket
  const connect = useCallback(() => {
    // 如果未启用连接，直接返回
    if (!enabled) {
      console.log('🔌 [WS] 连接被禁用，跳过连接');
      return;
    }

    if (socketRef.current?.readyState === WebSocket.OPEN) {
      console.log('🔌 [WS] 已连接，跳过重复连接');
      return;
    }

    setConnecting(true);
    setError(null);

    console.log('🔌 [WS] 正在连接到:', url);
    console.log('🔌 [WS] 回调状态:', {
      hasOnMessage: !!onMessage,
      hasOnLog: !!onLog,
      hasOnMetrics: !!onMetrics,
      hasOnStatusChange: !!onStatusChange
    });

    try {
      const socket = new WebSocket(url);
      socketRef.current = socket;

      socket.onopen = () => {
        setConnected(true);
        setConnecting(false);
        setError(null);
        reconnectAttemptsRef.current = 0;

        console.log('✅ WebSocket连接已建立');
      };

      socket.onmessage = (event) => {
        try {
          const message: WSMessage = JSON.parse(event.data);

          // 调试：显示收到的所有消息
          console.log('📨 [WS] 收到消息:', message.type, message.data);

          // 调用通用回调
          if (onMessage) {
            onMessage(message);
          }

          // 根据消息类型分发到特定回调
          switch (message.type) {
            case 'system_stats':
              console.log('📨 [WS] 分发到 onSystemStats');
              onSystemStats?.(message.data);
              break;
            case 'log':
              console.log('📨 [WS] 分发到 onLog');
              onLog?.(message.data);
              break;
            case 'metrics_update':
              console.log('📨 [WS] 分发到 onMetrics, data=', message.data);
              onMetrics?.(message.data);
              break;
            case 'status_change':
              console.log('📨 [WS] 分发到 onStatusChange');
              onStatusChange?.(message.data);
              break;
            case 'connection_established':
              console.log('WebSocket连接已确认:', message.data);
              break;
            default:
              console.warn('未知的WebSocket消息类型:', message);
          }
        } catch (err) {
          console.error('解析WebSocket消息失败:', err);
        }
      };

      socket.onerror = (event) => {
        console.error('WebSocket错误:', event);
        setError(new Error('WebSocket连接错误'));
        onError?.(event);
      };

      socket.onclose = (event) => {
        setConnected(false);
        setConnecting(false);
        socketRef.current = null;

        // 如果不是手动关闭，尝试重连
        if (!manualCloseRef.current && reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current += 1;

          console.log(
            `WebSocket连接已关闭，${reconnectInterval}ms后进行第${reconnectAttemptsRef.current}次重连...`
          );

          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, reconnectInterval);
        } else if (reconnectAttemptsRef.current >= maxReconnectAttempts) {
          console.error('❌ WebSocket重连失败，已达到最大重连次数');
          setError(new Error('达到最大重连次数'));
        }
      };
    } catch (err) {
      setError(err as Error);
      setConnecting(false);
      console.error('创建WebSocket连接失败:', err);
    }
  }, [url, reconnectInterval, maxReconnectAttempts, onMessage, onSystemStats, onLog, onMetrics, onStatusChange, onError, enabled]);

  // 手动发送消息
  const sendMessage = useCallback((message: any) => {
    const socket = socketRef.current;
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket未连接，无法发送消息');
    }
  }, []);

  // 手动断开连接
  const disconnect = useCallback(() => {
    manualCloseRef.current = true;
    clearReconnectTimeout();

    if (socketRef.current) {
      socketRef.current.close();
      socketRef.current = null;
    }

    setConnected(false);
    setConnecting(false);
  }, []);

  // 手动重连
  const reconnect = useCallback(() => {
    manualCloseRef.current = false;
    reconnectAttemptsRef.current = 0;
    clearReconnectTimeout();

    if (socketRef.current) {
      socketRef.current.close();
    }

    connect();
  }, [connect]);

  // 组件挂载时自动连接
  useEffect(() => {
    manualCloseRef.current = false;
    connect();

    return () => {
      disconnect();
    };
  }, [url, connect, disconnect]); // 在URL或connect变化时重新连接

  return {
    socket: socketRef.current,
    connected,
    connecting,
    error,
    sendMessage,
    disconnect,
    reconnect,
  };
};

/**
 * 系统状态WebSocket Hook
 */
export const useSystemStatsWS = (options?: UseWebSocketOptions) => {
  // 使用useRef保持client_id稳定，避免频繁重连
  const clientIdRef = useRef<string | undefined>(undefined);
  if (!clientIdRef.current) {
    clientIdRef.current = `sys_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  const wsUrl = `ws://localhost:8000/api/v1/ws/system?client_id=${clientIdRef.current}`;

  // 直接传递options，不用useMemo避免创建新对象
  return useWebSocket(wsUrl, options || {});
};

/**
 * 训练日志WebSocket Hook
 * 当 experimentId 为空时不建立连接
 */
export const useTrainingLogsWS = (
  experimentId: string,
  options: UseWebSocketOptions = {}
) => {
  // 使用useRef保持client_id稳定，避免频繁重连
  const clientIdRef = useRef<string | undefined>(undefined);
  if (!clientIdRef.current) {
    clientIdRef.current = `train_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  const wsUrl = experimentId
    ? `ws://localhost:8000/api/v1/ws/training/${experimentId}?client_id=${clientIdRef.current}`
    : '';

  // 直接传递options，不用useMemo避免创建新对象
  // enabled 由 shouldConnectWS 控制（在调用方已经处理）
  return useWebSocket(wsUrl, experimentId ? options : { ...options, enabled: false });
};

export default useWebSocket;
