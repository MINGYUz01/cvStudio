/**
 * 训练Hook
 * 结合REST API和WebSocket实现实时训练监控
 */

import { useState, useEffect, useCallback } from 'react';
import {
  trainingService,
  TrainingRun,
  TrainingRunCreateData,
  TrainingRunUpdateData,
  ControlAction,
  MetricsEntry,
  LogEntry,
} from '../services/training';
import { useTrainingLogsWS, MetricsEntry as WSMetricsEntry, LogEntry as WSLogEntry } from './useWebSocket';

/**
 * Hook返回值接口
 */
interface UseTrainingReturn {
  experiments: TrainingRun[];
  loading: boolean;
  error: string | null;
  selectedExp: TrainingRun | null;
  metrics: MetricsEntry[];
  logs: LogEntry[];
  fetchExperiments: () => Promise<void>;
  selectExperiment: (id: number) => Promise<void>;
  createExperiment: (data: TrainingRunCreateData) => Promise<void>;
  updateExperiment: (id: number, data: TrainingRunUpdateData) => Promise<void>;
  deleteExperiment: (id: number) => Promise<void>;
  controlExperiment: (id: number, action: ControlAction) => Promise<void>;
}

/**
 * 训练Hook
 */
export function useTraining(): UseTrainingReturn {
  const [experiments, setExperiments] = useState<TrainingRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedExp, setSelectedExp] = useState<TrainingRun | null>(null);
  const [metrics, setMetrics] = useState<MetricsEntry[]>([]);
  const [logs, setLogs] = useState<LogEntry[]>([]);

  /**
   * 获取训练任务列表
   */
  const fetchExperiments = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await trainingService.getTrainingRuns({ limit: 50 });
      setExperiments(data);
    } catch (err) {
      const message = err instanceof Error ? err.message : '获取训练列表失败';
      setError(message);
      console.error('获取训练列表失败:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  /**
   * 选择训练任务并加载详细数据
   */
  const selectExperiment = useCallback(async (id: number) => {
    setError(null);
    try {
      const exp = await trainingService.getTrainingRun(id);
      setSelectedExp(exp);

      // 获取历史指标和日志
      const [metricsData, logsData] = await Promise.all([
        trainingService.getMetrics(id, 100),
        trainingService.getLogs(id, undefined, 100),
      ]);

      setMetrics(metricsData);
      setLogs(logsData);
    } catch (err) {
      const message = err instanceof Error ? err.message : '获取训练详情失败';
      setError(message);
      throw err;
    }
  }, []);

  /**
   * 创建训练任务
   */
  const createExperiment = useCallback(async (data: TrainingRunCreateData) => {
    setError(null);
    try {
      await trainingService.createTrainingRun(data);
      // 重新获取列表
      await fetchExperiments();
    } catch (err) {
      const message = err instanceof Error ? err.message : '创建训练任务失败';
      setError(message);
      throw err;
    }
  }, [fetchExperiments]);

  /**
   * 更新训练任务
   */
  const updateExperiment = useCallback(
    async (id: number, data: TrainingRunUpdateData) => {
      setError(null);
      try {
        await trainingService.updateTrainingRun(id, data);
        // 重新获取列表
        await fetchExperiments();
      } catch (err) {
        const message = err instanceof Error ? err.message : '更新训练任务失败';
        setError(message);
        throw err;
      }
    },
    [fetchExperiments]
  );

  /**
   * 删除训练任务
   */
  const deleteExperiment = useCallback(
    async (id: number) => {
      setError(null);
      try {
        await trainingService.deleteTrainingRun(id);
        // 如果删除的是当前选中的任务，清空选中状态
        if (selectedExp?.id === id) {
          setSelectedExp(null);
          setMetrics([]);
          setLogs([]);
        }
        // 重新获取列表
        await fetchExperiments();
      } catch (err) {
        const message = err instanceof Error ? err.message : '删除训练任务失败';
        setError(message);
        throw err;
      }
    },
    [fetchExperiments, selectedExp]
  );

  /**
   * 控制训练任务
   */
  const controlExperiment = useCallback(
    async (id: number, action: ControlAction) => {
      setError(null);
      try {
        await trainingService.controlTraining(id, action);
        // 重新获取列表以更新状态
        await fetchExperiments();

        // 如果控制的是当前选中的任务，也更新详情
        if (selectedExp?.id === id) {
          await selectExperiment(id);
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : '控制训练任务失败';
        setError(message);
        throw err;
      }
    },
    [fetchExperiments, selectedExp, selectExperiment]
  );

  /**
   * WebSocket实时更新（当有选中的训练任务时）
   */
  useTrainingLogsWS(
    selectedExp ? `exp_${selectedExp.id}` : '',
    {
      onMetrics: (data: WSMetricsEntry) => {
        // 更新或添加新指标
        setMetrics((prev) => {
          const index = prev.findIndex((m) => m.epoch === data.epoch);
          if (index >= 0) {
            const updated = [...prev];
            updated[index] = { ...updated[index], ...data };
            return updated;
          }
          return [...prev, data as MetricsEntry];
        });
      },
      onLog: (data: WSLogEntry) => {
        // 添加新日志
        setLogs((prev) => [...prev, data as LogEntry]);
      },
      onStatusChange: (data) => {
        // 更新训练任务状态
        setExperiments((prev) =>
          prev.map((exp) =>
            exp.id === selectedExp?.id
              ? {
                  ...exp,
                  status: data.status as TrainingStatus,
                  current_epoch: data.current_epoch,
                }
              : exp
          )
        );

        if (selectedExp) {
          setSelectedExp({
            ...selectedExp,
            status: data.status as TrainingStatus,
            current_epoch: data.current_epoch,
          });
        }
      },
    }
  );

  // 初始化时获取训练任务列表
  useEffect(() => {
    fetchExperiments();
  }, [fetchExperiments]);

  return {
    experiments,
    loading,
    error,
    selectedExp,
    metrics,
    logs,
    fetchExperiments,
    selectExperiment,
    createExperiment,
    updateExperiment,
    deleteExperiment,
    controlExperiment,
  };
}

export default useTraining;
