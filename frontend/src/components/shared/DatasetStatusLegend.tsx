/**
 * 数据集状态说明弹窗组件
 * 显示不同状态标签的含义
 */

import React from 'react';
import { Info, CheckCircle, AlertTriangle, XCircle, X } from 'lucide-react';

/**
 * 数据集状态说明弹窗
 * @param {Object} props - 组件属性
 * @param {boolean} props.isOpen - 是否显示弹窗
 * @param {Function} props.onClose - 关闭弹窗的回调函数
 */
export const DatasetStatusLegend = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[200] flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
      <div className="bg-slate-900 border border-slate-700 p-6 rounded-xl w-full max-w-md shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center">
            <div className="w-10 h-10 rounded-full flex items-center justify-center bg-cyan-900/30 text-cyan-500 mr-3">
              <Info size={20} />
            </div>
            <div>
              <h3 className="text-lg font-bold text-white">数据集状态说明</h3>
              <p className="text-xs text-slate-400">不同状态标签的含义</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-slate-800 text-slate-400 hover:text-white transition-colors"
            aria-label="关闭"
          >
            <X size={20} />
          </button>
        </div>

        {/* Status Items */}
        <div className="space-y-4">
          {/* Standard Format */}
          <div className="flex items-start p-3 bg-emerald-900/10 border border-emerald-800/30 rounded-lg">
            <CheckCircle size={20} className="text-emerald-500 mr-3 mt-0.5 flex-shrink-0" />
            <div>
              <h4 className="font-medium text-emerald-400 mb-1">标准格式</h4>
              <p className="text-sm text-slate-400">
                数据集格式识别置信度高（≥70%），结构完整，可直接用于模型训练。
                包含足够的图像和标注数据。
              </p>
            </div>
          </div>

          {/* Non-standard Format */}
          <div className="flex items-start p-3 bg-amber-900/10 border border-amber-800/30 rounded-lg">
            <AlertTriangle size={20} className="text-amber-500 mr-3 mt-0.5 flex-shrink-0" />
            <div>
              <h4 className="font-medium text-amber-400 mb-1">非标准格式</h4>
              <p className="text-sm text-slate-400">
                能识别数据集格式，但置信度较低（30%-70%）或标注数据不完整。
                可以预览内容，但不建议直接用于训练。
              </p>
            </div>
          </div>

          {/* Unrecognized Format */}
          <div className="flex items-start p-3 bg-red-900/10 border border-red-800/30 rounded-lg">
            <XCircle size={20} className="text-red-500 mr-3 mt-0.5 flex-shrink-0" />
            <div>
              <h4 className="font-medium text-red-400 mb-1">未识别格式</h4>
              <p className="text-sm text-slate-400">
                无法识别数据集格式（置信度&lt;30%）或格式不被支持。
                仅可预览文件结构，无法用于训练。
              </p>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-6 pt-4 border-t border-slate-800">
          <p className="text-xs text-slate-500 text-center">
            支持的标准格式：YOLO、COCO、VOC、Classification
          </p>
        </div>
      </div>
    </div>
  );
};

export default DatasetStatusLegend;
