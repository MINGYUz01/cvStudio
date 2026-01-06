/**
 * 数据集相关API服务
 */

import { apiClient } from './api';

// ===== 类型定义 =====

/**
 * 数据集信息接口
 */
export interface Dataset {
  id: number;
  name: string;
  description?: string;
  path: string;
  format: string;
  num_images: number;
  num_classes: number;
  classes: string[];
  meta: Record<string, any>;
  is_active: string;
  created_at: string;
  updated_at?: string;
}

/**
 * 数据集列表响应
 */
export interface DatasetListResponse {
  success: boolean;
  message: string;
  data: Dataset[];
  total: number;
  page: number;
  page_size: number;
}

/**
 * 单个数据集响应
 */
export interface DatasetResponse {
  success: boolean;
  message: string;
  data: Dataset;
}

/**
 * 数据集注册数据
 */
export interface DatasetRegisterData {
  name: string;
  description?: string;
  dataset_path: string;
}

/**
 * 数据集上传数据
 */
export interface DatasetUploadData {
  name: string;
  description?: string;
  files: File[];
}

/**
 * 图像信息
 */
export interface ImageInfo {
  filename: string;
  path: string;
  size: number;
  format: string;
  width: number;
  height: number;
  annotations?: any[];
}

/**
 * 图像列表响应
 */
export interface ImageListResponse {
  success: boolean;
  message: string;
  data: {
    images: ImageInfo[];
    total: number;
    page: number;
    page_size: number;
  };
}

/**
 * 数据集统计信息
 */
export interface DatasetStatistics {
  dataset_id: number;
  num_images: number;
  num_classes: number;
  class_distribution: Record<string, number>;
  image_size_distribution: Record<string, any>;
  format_details: Record<string, any>;
  quality_metrics: Record<string, any>;
}

/**
 * 统计信息响应
 */
export interface StatisticsResponse {
  success: boolean;
  message: string;
  data: DatasetStatistics;
}

/**
 * 分页参数
 */
export interface PaginationParams {
  skip?: number;
  limit?: number;
  format_filter?: string;
}

// ===== 服务类 =====

/**
 * 数据集服务类
 */
class DatasetServiceClass {
  private baseUrl = '/datasets';

  /**
   * 获取数据集列表
   * @param params 分页和过滤参数
   * @returns 数据集列表
   */
  async getDatasets(params?: PaginationParams): Promise<Dataset[]> {
    const response = await apiClient.get<DatasetListResponse>(
      `${this.baseUrl}/`,
      { params }
    );
    return response.data;
  }

  /**
   * 获取数据集详情
   * @param id 数据集ID
   * @returns 数据集详情
   */
  async getDataset(id: number): Promise<Dataset> {
    const response = await apiClient.get<DatasetResponse>(
      `${this.baseUrl}/${id}`
    );
    return response.data;
  }

  /**
   * 上传数据集
   * @param data 上传数据
   * @returns 上传的数据集
   */
  async uploadDataset(data: DatasetUploadData): Promise<Dataset> {
    const formData = new FormData();
    formData.append('name', data.name);
    if (data.description) {
      formData.append('description', data.description);
    }
    data.files.forEach(file => {
      formData.append('files', file);
    });

    // 使用原生fetch以支持文件上传进度
    const token = localStorage.getItem('access_token');
    const response = await fetch(
      `${apiClient['baseURL']}${this.baseUrl}/upload`,
      {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        },
        body: formData
      }
    );

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || '上传失败');
    }

    const result: DatasetResponse = await response.json();
    return result.data;
  }

  /**
   * 注册现有数据集
   * @param data 注册数据
   * @returns 注册的数据集
   */
  async registerDataset(data: DatasetRegisterData): Promise<Dataset> {
    const response = await apiClient.post<DatasetResponse>(
      `${this.baseUrl}/register`,
      data
    );
    return response.data;
  }

  /**
   * 更新数据集
   * @param id 数据集ID
   * @param data 更新数据
   * @returns 更新后的数据集
   */
  async updateDataset(
    id: number,
    data: { name?: string; description?: string }
  ): Promise<Dataset> {
    const response = await apiClient.put<DatasetResponse>(
      `${this.baseUrl}/${id}`,
      data
    );
    return response.data;
  }

  /**
   * 删除数据集
   * @param id 数据集ID
   */
  async deleteDataset(id: number): Promise<void> {
    await apiClient.delete<{ success: boolean; message: string }>(
      `${this.baseUrl}/${id}`
    );
  }

  /**
   * 重新扫描数据集
   * @param id 数据集ID
   * @returns 重新扫描后的数据集
   */
  async rescanDataset(id: number): Promise<Dataset> {
    const response = await apiClient.post<DatasetResponse>(
      `${this.baseUrl}/${id}/rescan`,
      {}
    );
    return response.data;
  }

  /**
   * 获取数据集统计信息
   * @param id 数据集ID
   * @returns 统计信息
   */
  async getStatistics(id: number): Promise<DatasetStatistics> {
    const response = await apiClient.get<StatisticsResponse>(
      `${this.baseUrl}/${id}/statistics`
    );
    return response.data;
  }

  /**
   * 获取数据集图像列表
   * @param id 数据集ID
   * @param params 分页和排序参数
   * @returns 图像列表
   */
  async getImages(
    id: number,
    params?: {
      page?: number;
      page_size?: number;
      sort_by?: string;
      sort_order?: 'asc' | 'desc';
    }
  ): Promise<ImageListResponse['data']> {
    const response = await apiClient.get<ImageListResponse>(
      `${this.baseUrl}/${id}/images`,
      { params }
    );
    return response.data;
  }

  /**
   * 获取数据集缩略图列表
   * @param id 数据集ID
   * @returns 缩略图路径列表
   */
  async getThumbnails(id: number): Promise<string[]> {
    const response = await apiClient.get<{
      success: boolean;
      message: string;
      data: string[];
    }>(`${this.baseUrl}/${id}/thumbnails`);
    return response.data;
  }

  /**
   * 生成数据集缩略图
   * @param id 数据集ID
   */
  async generateThumbnails(id: number): Promise<void> {
    await apiClient.post<{ success: boolean; message: string }>(
      `${this.baseUrl}/${id}/thumbnails/generate`,
      {}
    );
  }

  /**
   * 删除数据集缩略图
   * @param id 数据集ID
   */
  async deleteThumbnails(id: number): Promise<void> {
    await apiClient.delete<{ success: boolean; message: string }>(
      `${this.baseUrl}/${id}/thumbnails`
    );
  }
}

// 导出单例
export const datasetService = new DatasetServiceClass();

export default datasetService;
