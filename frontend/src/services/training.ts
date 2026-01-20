/**
 * 训练相关API服务
 */

import { apiClient } from './api';

// ===== 类型定义 =====

/**
 * 任务类型
 */
export type TaskType = 'detection' | 'classification';

/**
 * 训练状态
 */
export type TrainingStatus =
  | 'pending'
  | 'queued'
  | 'running'
  | 'paused'
  | 'completed'
  | 'failed'
  | 'cancelled';

/**
 * 控制动作
 */
export type ControlAction = 'pause' | 'resume' | 'stop';

/**
 * 训练任务
 */
export interface TrainingRun {
  id: number;
  name: string;
  description?: string;
  model_id: number;
  dataset_id: number;
  hyperparams: Record<string, any>;
  status: TrainingStatus;
  progress: number;
  current_epoch: number;
  total_epochs: number;
  best_metric?: number;
  device: string;
  log_file?: string;
  error_message?: string;
  start_time?: string;
  end_time?: string;
  created_at: string;
  updated_at?: string;
}

/**
 * 训练任务创建数据
 */
export interface TrainingRunCreateData {
  name: string;
  description?: string;
  model_id: number;
  dataset_id: number;
  config: TrainingConfig;
  user_id: number;
}

/**
 * 训练配置
 */
export interface TrainingConfig {
  task_type: TaskType;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  optimizer: string;
  device: string;
  [key: string]: any;
}

/**
 * 训练任务更新数据
 */
export interface TrainingRunUpdateData {
  name?: string;
  description?: string;
}

/**
 * 训练控制响应
 */
export interface TrainingControlResponse {
  success: boolean;
  action: string;
  task_id: string;
  experiment_id: string;
  message?: string;
}

/**
 * 训练指标条目
 */
export interface MetricsEntry {
  epoch: number;
  timestamp: string;
  train_loss?: number;
  train_acc?: number;
  val_loss?: number;
  val_acc?: number;
  [key: string]: any;
}

/**
 * 日志条目
 */
export interface LogEntry {
  level: string;
  message: string;
  source: string;
  timestamp: string;
}

/**
 * Checkpoint信息
 */
export interface CheckpointInfo {
  id: number;
  epoch: number;
  metric_value?: number;
  metrics: Record<string, any>;
  path: string;
  is_best: boolean;
  file_size?: number;
  created_at?: string;
}

/**
 * 权重库条目
 */
export interface WeightLibraryItem {
  id: number;
  name: string;
  display_name: string;
  description?: string;
  task_type: TaskType;
  version: string;
  file_name: string;
  file_size_mb?: number;
  framework: string;
  created_at: string;
}

/**
 * 保存到权重库请求数据
 */
export interface SaveToWeightsRequest {
  name: string;
  description?: string;
  include_last?: boolean;
}

/**
 * 保存到权重库响应
 */
export interface SaveToWeightsResponse {
  success: boolean;
  message: string;
  best_weight?: WeightLibraryItem;
  last_weight?: WeightLibraryItem;
}

// ===== 服务类 =====

/**
 * 训练服务类
 */
class TrainingServiceClass {
  private baseUrl = '/training';

  /**
   * 获取训练任务列表
   * @param params 查询参数
   * @returns 训练任务列表
   */
  async getTrainingRuns(params?: {
    status?: TrainingStatus;
    skip?: number;
    limit?: number;
  }): Promise<TrainingRun[]> {
    return apiClient.get<TrainingRun[]>(`${this.baseUrl}/`, { params });
  }

  /**
   * 创建训练任务
   * @param data 训练任务数据
   * @returns 创建的训练任务
   */
  async createTrainingRun(data: TrainingRunCreateData): Promise<TrainingRun> {
    console.log('[DEBUG] 创建训练任务 - 请求数据:', JSON.stringify(data, null, 2));
    const response = await apiClient.post<TrainingRun>(`${this.baseUrl}/`, data);
    console.log('[DEBUG] 创建训练任务 - 响应数据:', JSON.stringify(response, null, 2));
    return response;
  }

  /**
   * 启动训练任务
   * @param id 训练任务ID
   * @returns 启动响应
   */
  async startTraining(id: number): Promise<{ success: boolean; message: string; task_id?: string }> {
    console.log(`[DEBUG] 启动训练任务 - ID: ${id}`);
    const response = await apiClient.post<{ success: boolean; message: string; task_id?: string }>(`${this.baseUrl}/${id}/start`);
    console.log(`[DEBUG] 启动训练任务 - 响应:`, JSON.stringify(response, null, 2));
    return response;
  }

  /**
   * 获取训练任务详情
   * @param id 训练任务ID
   * @returns 训练任务详情
   */
  async getTrainingRun(id: number): Promise<TrainingRun> {
    return apiClient.get<TrainingRun>(`${this.baseUrl}/${id}`);
  }

  /**
   * 更新训练任务
   * @param id 训练任务ID
   * @param data 更新数据
   * @returns 更新后的训练任务
   */
  async updateTrainingRun(
    id: number,
    data: TrainingRunUpdateData
  ): Promise<TrainingRun> {
    return apiClient.put<TrainingRun>(`${this.baseUrl}/${id}`, data);
  }

  /**
   * 删除训练任务
   * @param id 训练任务ID
   */
  async deleteTrainingRun(id: number): Promise<void> {
    await apiClient.delete<void>(`${this.baseUrl}/${id}`);
  }

  /**
   * 控制训练任务
   * @param id 训练任务ID
   * @param action 控制动作
   * @returns 控制响应
   */
  async controlTraining(
    id: number,
    action: ControlAction
  ): Promise<TrainingControlResponse> {
    return apiClient.post<TrainingControlResponse>(
      `${this.baseUrl}/${id}/control`,
      { action }
    );
  }

  /**
   * 获取训练指标
   * @param id 训练任务ID
   * @param limit 返回条数限制
   * @returns 训练指标列表
   */
  async getMetrics(id: number, limit: number = 100): Promise<MetricsEntry[]> {
    return apiClient.get<MetricsEntry[]>(`${this.baseUrl}/${id}/metrics`, {
      params: { limit },
    });
  }

  /**
   * 获取训练日志
   * @param id 训练任务ID
   * @param level 日志级别过滤
   * @param limit 返回条数限制
   * @returns 日志条目列表
   */
  async getLogs(
    id: number,
    level?: string,
    limit: number = 100
  ): Promise<LogEntry[]> {
    return apiClient.get<LogEntry[]>(`${this.baseUrl}/${id}/logs`, {
      params: { level, limit },
    });
  }

  /**
   * 获取检查点列表
   * @param id 训练任务ID
   * @returns 检查点列表
   */
  async getCheckpoints(id: number): Promise<CheckpointInfo[]> {
    return apiClient.get<CheckpointInfo[]>(`${this.baseUrl}/${id}/checkpoints`);
  }

  /**
   * 保存最佳模型到权重库
   * @param id 训练任务ID
   * @param weightsDir 权重库目录
   * @returns 保存响应
   */
  async saveCheckpoint(
    id: number,
    weightsDir: string = 'data/weights'
  ): Promise<{ success: boolean; message: string; path: string }> {
    return apiClient.post<{ success: boolean; message: string; path: string }>(
      `${this.baseUrl}/${id}/save`,
      { weights_dir: weightsDir }
    );
  }

  /**
   * 保存训练模型到权重库
   * @param id 训练任务ID
   * @param data 保存数据
   * @returns 保存响应
   */
  async saveToWeights(
    id: number,
    data: SaveToWeightsRequest
  ): Promise<SaveToWeightsResponse> {
    return apiClient.post<SaveToWeightsResponse>(
      `${this.baseUrl}/${id}/save-to-weights`,
      data
    );
  }
}

// 导出单例
export const trainingService = new TrainingServiceClass();

export default trainingService;
