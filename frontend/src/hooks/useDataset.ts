/**
 * 数据集Hook
 * 提供数据集数据获取和状态管理
 */

import { useState, useEffect, useCallback } from 'react';
import {
  datasetService,
  Dataset,
  DatasetRegisterData,
  DatasetUploadData,
} from '../services/datasets';

/**
 * Hook返回值接口
 */
interface UseDatasetReturn {
  datasets: Dataset[];
  loading: boolean;
  error: string | null;
  fetchDatasets: () => Promise<void>;
  getDataset: (id: number) => Promise<Dataset>;
  uploadDataset: (data: DatasetUploadData) => Promise<void>;
  registerDataset: (data: DatasetRegisterData) => Promise<void>;
  updateDataset: (id: number, data: { name?: string; description?: string }) => Promise<void>;
  deleteDataset: (id: number) => Promise<void>;
  rescanDataset: (id: number) => Promise<void>;
}

/**
 * 数据集Hook
 */
export function useDataset(): UseDatasetReturn {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  /**
   * 获取数据集列表
   */
  const fetchDatasets = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await datasetService.getDatasets({ limit: 100 });
      setDatasets(data);
    } catch (err) {
      const message = err instanceof Error ? err.message : '获取数据集列表失败';
      setError(message);
      console.error('获取数据集列表失败:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  /**
   * 获取单个数据集
   */
  const getDataset = useCallback(async (id: number): Promise<Dataset> => {
    setError(null);
    try {
      const dataset = await datasetService.getDataset(id);
      return dataset;
    } catch (err) {
      const message = err instanceof Error ? err.message : '获取数据集详情失败';
      setError(message);
      throw err;
    }
  }, []);

  /**
   * 上传数据集
   */
  const uploadDataset = useCallback(async (data: DatasetUploadData) => {
    setError(null);
    try {
      await datasetService.uploadDataset(data);
      // 重新获取列表
      await fetchDatasets();
    } catch (err) {
      const message = err instanceof Error ? err.message : '上传数据集失败';
      setError(message);
      throw err;
    }
  }, [fetchDatasets]);

  /**
   * 注册现有数据集
   */
  const registerDataset = useCallback(async (data: DatasetRegisterData) => {
    setError(null);
    try {
      await datasetService.registerDataset(data);
      // 重新获取列表
      await fetchDatasets();
    } catch (err) {
      const message = err instanceof Error ? err.message : '注册数据集失败';
      setError(message);
      throw err;
    }
  }, [fetchDatasets]);

  /**
   * 更新数据集
   */
  const updateDataset = useCallback(
    async (id: number, data: { name?: string; description?: string }) => {
      setError(null);
      try {
        await datasetService.updateDataset(id, data);
        // 重新获取列表
        await fetchDatasets();
      } catch (err) {
        const message = err instanceof Error ? err.message : '更新数据集失败';
        setError(message);
        throw err;
      }
    },
    [fetchDatasets]
  );

  /**
   * 删除数据集
   */
  const deleteDataset = useCallback(
    async (id: number) => {
      setError(null);
      try {
        await datasetService.deleteDataset(id);
        // 从列表中移除
        setDatasets(prev => prev.filter(ds => ds.id !== id));
      } catch (err) {
        const message = err instanceof Error ? err.message : '删除数据集失败';
        setError(message);
        throw err;
      }
    },
    []
  );

  /**
   * 重新扫描数据集
   */
  const rescanDataset = useCallback(
    async (id: number) => {
      setError(null);
      try {
        await datasetService.rescanDataset(id);
        // 重新获取列表
        await fetchDatasets();
      } catch (err) {
        const message = err instanceof Error ? err.message : '重新扫描数据集失败';
        setError(message);
        throw err;
      }
    },
    [fetchDatasets]
  );

  // 初始化时获取数据集列表
  useEffect(() => {
    fetchDatasets();
  }, [fetchDatasets]);

  return {
    datasets,
    loading,
    error,
    fetchDatasets,
    getDataset,
    uploadDataset,
    registerDataset,
    updateDataset,
    deleteDataset,
    rescanDataset,
  };
}

export default useDataset;
