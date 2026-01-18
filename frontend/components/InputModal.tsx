import React, { useState, useEffect, useRef } from 'react';
import { X, FolderOpen } from 'lucide-react';

interface InputModalProps {
  /** 是否显示模态框 */
  show: boolean;
  /** 标题 */
  title: string;
  /** 预设模型名称 */
  presetName: string;
  /** 预设模型描述 */
  presetDescription?: string;
  /** 输入框占位符 */
  placeholder?: string;
  /** 确认回调 */
  onConfirm: (name: string, description: string) => void;
  /** 关闭回调 */
  onClose: () => void;
  /** 是否正在处理 */
  loading?: boolean;
}

/**
 * 输入模态框组件
 *
 * 用于从预设模型创建新架构时输入名称和描述
 */
const InputModal: React.FC<InputModalProps> = ({
  show,
  title,
  presetName,
  presetDescription,
  placeholder,
  onConfirm,
  onClose,
  loading = false
}) => {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const nameInputRef = useRef<HTMLInputElement>(null);

  // 当模态框显示时，设置默认值并聚焦输入框
  useEffect(() => {
    if (show) {
      setName(`${presetName} - 副本`);
      setDescription(presetDescription || `基于预设模型「${presetName}」创建`);
      // 延迟聚焦，确保动画完成后
      setTimeout(() => {
        nameInputRef.current?.focus();
        nameInputRef.current?.select();
      }, 100);
    }
  }, [show, presetName, presetDescription]);

  // 处理确认
  const handleConfirm = () => {
    if (name.trim()) {
      onConfirm(name.trim(), description.trim());
    }
  };

  // 处理键盘事件
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleConfirm();
    } else if (e.key === 'Escape') {
      onClose();
    }
  };

  if (!show) return null;

  return (
    <div className="fixed inset-0 z-[200] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
      <div className="bg-slate-900 border border-slate-700 rounded-xl w-full max-w-md shadow-2xl animate-in fade-in zoom-in duration-200">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-cyan-900/30 rounded-lg">
              <FolderOpen className="text-cyan-400" size={20} />
            </div>
            <h3 className="text-lg font-bold text-white">{title}</h3>
          </div>
          <button
            onClick={onClose}
            disabled={loading}
            className="p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors disabled:opacity-50"
          >
            <X size={20} />
          </button>
        </div>

        {/* Body */}
        <div className="px-6 py-5">
          {/* 预设模型信息 */}
          <div className="mb-5 p-4 bg-slate-950 rounded-lg border border-slate-800">
            <div className="text-sm text-slate-400 mb-1">基于预设模型</div>
            <div className="text-base font-semibold text-white">{presetName}</div>
            {presetDescription && (
              <div className="text-xs text-slate-500 mt-2 line-clamp-2">{presetDescription}</div>
            )}
          </div>

          {/* 名称输入 */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-slate-300 mb-2">
              架构名称 <span className="text-rose-400">*</span>
            </label>
            <input
              ref={nameInputRef}
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={placeholder || "请输入新架构的名称"}
              className="w-full px-4 py-3 bg-slate-950 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 transition-colors"
              disabled={loading}
              maxLength={100}
            />
            <div className="flex justify-end mt-1">
              <span className="text-xs text-slate-500">{name.length}/100</span>
            </div>
          </div>

          {/* 描述输入 */}
          <div className="mb-2">
            <label className="block text-sm font-medium text-slate-300 mb-2">
              架构描述
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="请输入架构描述（可选）"
              className="w-full px-4 py-3 bg-slate-950 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 transition-colors resize-none"
              disabled={loading}
              rows={3}
              maxLength={500}
            />
            <div className="flex justify-end mt-1">
              <span className="text-xs text-slate-500">{description.length}/500</span>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-slate-800 bg-slate-900/30 rounded-b-xl">
          <button
            onClick={onClose}
            disabled={loading}
            className="px-5 py-2.5 text-slate-300 hover:text-white hover:bg-slate-800 rounded-lg transition-colors disabled:opacity-50"
          >
            取消
          </button>
          <button
            onClick={handleConfirm}
            disabled={loading || !name.trim()}
            className="px-5 py-2.5 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-medium rounded-lg shadow-lg shadow-cyan-900/20 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {loading ? (
              <>
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                创建中...
              </>
            ) : (
              <>
                <FolderOpen size={18} />
                创建架构
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default InputModal;
