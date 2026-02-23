import React from 'react';
import { X, Copy, Download, HardDrive, Trash2, AlertTriangle, FileText } from 'lucide-react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface CodePreviewModalProps {
  /** 是否显示模态框 */
  show: boolean;
  /** 要显示的代码 */
  code: string | null;
  /** 模型元数据 */
  metadata?: {
    layer_count?: number;
    depth?: number;
    num_parameters?: number;
    validation_passed?: boolean;
    layer_types?: Record<string, number>;
  };
  /** 预览来源 */
  source: 'builder' | 'library';
  /** 模型名称（用于下载文件名） */
  modelName: string;
  /** 文件名（用于library模式下的删除操作） */
  filename?: string;
  /** 错误信息 */
  error?: string | null;
  /** 关闭回调 */
  onClose: () => void;
  /** 保存到库回调（builder模式） */
  onSave?: () => void;
  /** 删除回调（library模式） */
  onDelete?: () => void;
  /** 显示通知的回调 */
  showNotification?: (message: string, type: 'success' | 'error' | 'info') => void;
}

/**
 * 代码预览模态框组件
 *
 * 支持两种模式：
 * - builder: 从模型构建器打开，显示"保存到库"按钮
 * - library: 从模型库打开，显示"删除"按钮
 */
const CodePreviewModal: React.FC<CodePreviewModalProps> = ({
  show,
  code,
  metadata,
  source,
  modelName,
  filename,
  error,
  onClose,
  onSave,
  onDelete,
  showNotification
}) => {
  if (!show) return null;

  /**
   * 复制代码到剪贴板
   */
  const handleCopyCode = async () => {
    if (!code) return;
    try {
      await navigator.clipboard.writeText(code);
      showNotification?.("代码已复制到剪贴板", "success");
    } catch (error) {
      showNotification?.("复制失败，请手动复制", "error");
    }
  };

  /**
   * 下载代码
   */
  const handleDownloadCode = () => {
    if (!code) return;

    const blob = new Blob([code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    // 根据来源使用不同的文件名
    const downloadName = source === 'library' && filename
      ? filename
      : `${modelName.replace(/\s+/g, '_')}.py`;
    a.download = downloadName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    showNotification?.("代码已下载", "success");
  };

  /**
   * 处理主操作按钮点击（保存/删除）
   */
  const handleMainAction = () => {
    if (source === 'builder') {
      onSave?.();
    } else {
      onDelete?.();
    }
  };

  const isLibraryMode = source === 'library';
  const actionDisabled = source === 'builder' && !metadata?.validation_passed;

  return (
    <div className="fixed inset-0 z-[200] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
      <div className="bg-slate-900 border border-slate-700 rounded-xl w-full max-w-6xl h-[85vh] flex flex-col shadow-2xl animate-in fade-in zoom-in duration-200">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-emerald-900/30 rounded-lg">
              <FileText className="text-emerald-400" size={20} />
            </div>
            <div>
              <h3 className="text-lg font-bold text-white">
                {isLibraryMode ? '已保存的 PyTorch 模型代码' : '生成的 PyTorch 模型代码'}
              </h3>
              {metadata && (
                <div className="flex items-center space-x-3 text-xs text-slate-400 mt-1">
                  <span>{metadata.layer_count} 层</span>
                  <span>•</span>
                  <span>{metadata.num_parameters?.toLocaleString()} 参数</span>
                  <span>•</span>
                  <span>深度 {metadata.depth}</span>
                  {metadata.validation_passed && (
                    <>
                      <span>•</span>
                      <span className="text-emerald-400">验证通过 ✓</span>
                    </>
                  )}
                </div>
              )}
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors"
          >
            <X size={20} />
          </button>
        </div>

        {/* 统计图表区域 - 仅在有元数据时显示 */}
        {metadata && metadata.layer_types && Object.keys(metadata.layer_types).length > 0 && (
          <div className="px-6 py-4 border-b border-slate-800 bg-slate-900/30">
            <div className="grid grid-cols-2 gap-6">
              {/* 层类型分布饼图 */}
              <div className="flex items-center space-x-4">
                <div className="flex-1">
                  <h4 className="text-xs text-slate-500 mb-2">层类型分布</h4>
                  <div className="flex items-center space-x-2 text-xs text-slate-400">
                    {Object.entries(metadata.layer_types).map(([name, value], index) => (
                      <div key={name} className="flex items-center">
                        <div
                          className="w-2 h-2 rounded-full mr-1"
                          style={{ backgroundColor: ['#22d3ee', '#a855f7', '#f43f5e', '#fbbf24', '#34d399', '#60a5fa'][index % 6] }}
                        />
                        <span>{name}: {value}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* 参数统计 */}
              <div className="bg-slate-950 rounded-lg p-4">
                <h4 className="text-xs text-slate-500 mb-3">模型统计</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-[10px] text-slate-500">总参数量</div>
                    <div className="text-lg font-bold text-cyan-400 font-mono">
                      {(metadata.num_parameters || 0).toLocaleString()}
                    </div>
                  </div>
                  <div>
                    <div className="text-[10px] text-slate-500">网络深度</div>
                    <div className="text-lg font-bold text-purple-400 font-mono">
                      {metadata.depth || 0}
                    </div>
                  </div>
                  <div>
                    <div className="text-[10px] text-slate-500">层数量</div>
                    <div className="text-lg font-bold text-emerald-400 font-mono">
                      {metadata.layer_count || 0}
                    </div>
                  </div>
                  <div>
                    <div className="text-[10px] text-slate-500">验证状态</div>
                    <div className={`text-lg font-bold font-mono ${metadata.validation_passed ? 'text-emerald-400' : 'text-amber-400'}`}>
                      {metadata.validation_passed ? '通过 ✓' : '未通过'}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Main Content Area: Code + Action Buttons */}
        <div className="flex-1 flex overflow-hidden">
          {/* Code Content Area */}
          <div className="flex-1 flex flex-col overflow-hidden">
            {/* Code with Syntax Highlighting */}
            <div className="flex-1 overflow-hidden">
              <SyntaxHighlighter
                language="python"
                style={vscDarkPlus}
                customStyle={{
                  background: 'transparent',
                  padding: '24px',
                  margin: 0,
                  height: '100%',
                  fontSize: '13px'
                }}
                className="h-full overflow-auto custom-scrollbar"
                showLineNumbers
                lineNumberStyle={{ color: '#475569', fontSize: '12px' }}
              >
                {code || ''}
              </SyntaxHighlighter>
            </div>

            {/* Warning Area - Full Width */}
            {error && (
              <div className="border-t border-slate-800 bg-slate-900/50 p-4">
                <div className="p-3 bg-amber-900/20 border border-amber-700/50 rounded-lg">
                  <div className="flex items-start gap-2">
                    <AlertTriangle size={16} className="text-amber-400 shrink-0 mt-0.5" />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium text-amber-400">代码生成警告</span>
                        <button
                          onClick={() => {
                            navigator.clipboard.writeText(error);
                            showNotification?.("警告信息已复制", "success");
                          }}
                          className="text-xs text-amber-300 hover:text-amber-100 flex items-center gap-1 transition-colors"
                        >
                          <Copy size={12} />
                          复制
                        </button>
                      </div>
                      <pre className="text-xs text-amber-200/80 whitespace-pre-wrap break-words font-mono bg-black/20 rounded p-2 max-h-32 overflow-auto">
                        {error}
                      </pre>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Right Action Sidebar */}
          <div className="w-16 shrink-0 border-l border-slate-800 bg-slate-900/50 flex flex-col items-center py-4 space-y-2">
            <button
              onClick={handleCopyCode}
              className="p-3 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg border border-slate-700 transition-colors group relative"
              title="复制代码"
            >
              <Copy size={18} />
              <span className="absolute right-full mr-2 px-2 py-1 bg-slate-800 text-xs text-white rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">
                复制代码
              </span>
            </button>
            <button
              onClick={handleDownloadCode}
              className="p-3 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg border border-slate-700 transition-colors group relative"
              title="下载"
            >
              <Download size={18} />
              <span className="absolute right-full mr-2 px-2 py-1 bg-slate-800 text-xs text-white rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">
                下载到本地
              </span>
            </button>

            {/* 主操作按钮 - 根据来源显示不同按钮 */}
            <button
              onClick={handleMainAction}
              disabled={actionDisabled}
              className={`p-3 text-white rounded-lg shadow-lg transition-colors group relative ${
                isLibraryMode
                  ? 'bg-rose-700 hover:bg-rose-600 shadow-rose-900/20'
                  : metadata?.validation_passed
                    ? 'bg-emerald-700 hover:bg-emerald-600 shadow-emerald-900/20'
                    : 'bg-rose-700 opacity-60 cursor-not-allowed shadow-rose-900/20'
              }`}
              title={isLibraryMode ? '删除' : (metadata?.validation_passed ? '保存到库' : '验证未通过')}
            >
              {isLibraryMode ? <Trash2 size={18} /> : <HardDrive size={18} />}
              <span className="absolute right-full mr-2 px-2 py-1 bg-slate-800 text-xs text-white rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">
                {isLibraryMode ? '删除' : (metadata?.validation_passed ? '保存到库' : '验证未通过')}
              </span>
            </button>

            <div className="flex-1" />
            <button
              onClick={onClose}
              className="p-3 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg border border-slate-600 transition-colors group relative"
              title="关闭"
            >
              <X size={18} />
              <span className="absolute right-full mr-2 px-2 py-1 bg-slate-800 text-xs text-white rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">
                关闭预览
              </span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CodePreviewModal;
