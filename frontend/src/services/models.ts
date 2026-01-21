/**
 * 模型相关API服务
 *
 * 提供模型图验证、分析、形状推断、代码生成等功能
 */

import { apiClient } from './api';

// ==============================
// 类型定义
// ==============================

interface NodeData {
  [key: string]: any;
}

interface ModelNode {
  id: string;
  type: string;
  label?: string;
  x?: number;
  y?: number;
  inputs?: string[];
  outputs?: string[];
  data: NodeData;
}

interface ModelConnection {
  id?: string;
  source: string;
  target: string;
}

interface ModelGraph {
  nodes: ModelNode[];
  connections: ModelConnection[];
}

interface ValidationResult {
  valid: boolean;
  syntax_valid?: boolean;
  executable?: boolean;
  parameters_valid?: boolean;
  forward_pass_success?: boolean;
  errors: string[];
  warnings: string[];
}

interface CodeGenerationResult {
  code: string;
  model_name: string;
  validation: ValidationResult;
  metadata: {
    layer_count: number;
    num_parameters: number;
    input_shape?: any[];
    output_shape?: any[];
    depth: number;
    validation_passed: boolean;
  };
}

interface ShapeInfo {
  input?: any[];
  output: any[];
  input_str?: string;
  output_str: string;
}

interface AnalysisResult {
  execution_order: {
    forward: string[];
    backward: string[];
    layers: string[];
    inputs: string[];
    outputs: string[];
  };
  num_parameters: number;
  depth: number;
}

interface AnalyzeAndInferResult {
  validation: ValidationResult;
  analysis: AnalysisResult;
  shapes: Record<string, ShapeInfo>;
}


// ==============================
// API方法
// ==============================

/**
 * 验证模型图
 */
export async function validateModelGraph(graph: ModelGraph): Promise<ValidationResult> {
  return await apiClient.post<ValidationResult>('/models/validate', graph);
}

/**
 * 分析模型结构
 */
export async function analyzeModelGraph(graph: ModelGraph): Promise<AnalysisResult> {
  return await apiClient.post<AnalysisResult>('/models/analyze', graph);
}

/**
 * 推断张量形状
 */
export async function inferTensorShapes(graph: ModelGraph): Promise<{ valid: boolean; shapes: Record<string, ShapeInfo>; errors: string[]; warnings: string[] }> {
  return await apiClient.post('/models/infer-shapes', graph);
}

/**
 * 组合分析：验证、分析和形状推断
 */
export async function analyzeAndInfer(graph: ModelGraph): Promise<AnalyzeAndInferResult> {
  return await apiClient.post<AnalyzeAndInferResult>('/models/analyze-and-infer', graph);
}

/**
 * 生成PyTorch模型代码
 */
export async function generatePyTorchCode(
  graph: ModelGraph,
  modelName?: string
): Promise<CodeGenerationResult> {
  const params = modelName ? { model_name: modelName } : undefined;
  return await apiClient.post<CodeGenerationResult>('/models/generate', graph, { params });
}

/**
 * 验证生成的代码
 */
export async function validateGeneratedCode(code: string, modelName: string): Promise<{ validation: ValidationResult; test_results: any }> {
  return await apiClient.post('/models/validate-code', {
    code,
    model_name: modelName
  });
}

/**
 * 获取可用的代码模板列表
 */
export async function listTemplates(): Promise<any> {
  return await apiClient.get('/models/templates');
}

// ==============================
// 导出
// ==============================

export default {
  validateModelGraph,
  analyzeModelGraph,
  inferTensorShapes,
  analyzeAndInfer,
  generatePyTorchCode,
  validateGeneratedCode,
  listTemplates
};
