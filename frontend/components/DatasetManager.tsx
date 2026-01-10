import React, { useState, useEffect, useRef } from 'react';
import {
  Folder,
  FolderOpen,
  File,
  Search,
  Layout,
  X,
  Eye,
  Calendar,
  HardDrive,
  Upload,
  ChevronDown,
  ChevronRight,
  Plus,
  RefreshCw,
  AlertCircle,
  Archive,
  CheckCircle,
  Loader2,
  Trash2,
  FileImage,
  FileText,
  Filter,
  SlidersHorizontal
} from 'lucide-react';
import { DatasetItem } from '../types';
import { useDataset } from '../src/hooks/useDataset';
import { adaptDatasetList } from '../src/services/datasetAdapter';
import { datasetService } from '../src/services/datasets';
import { apiClient } from '../src/services/api';

// --- Import Dataset Dialog Component ---
interface ImportDatasetDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onUploadComplete: () => void;
}

const ImportDatasetDialog: React.FC<ImportDatasetDialogProps> = ({ isOpen, onClose, onUploadComplete }) => {
  const [datasetName, setDatasetName] = useState('');
  const [description, setDescription] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  // 支持的压缩格式
  const supportedFormats = ['.zip', '.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2', '.tar.xz', '.txz', '.7z'];

  // 重置表单
  const resetForm = () => {
    setDatasetName('');
    setDescription('');
    setSelectedFile(null);
    setUploadProgress(0);
    setError('');
  };

  // 处理文件选择
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // 验证文件格式
    const isValidFormat = supportedFormats.some(fmt => file.name.toLowerCase().endsWith(fmt));
    if (!isValidFormat) {
      setError(`不支持的文件格式。支持的格式: ${supportedFormats.join(', ')}`);
      setSelectedFile(null);
      return;
    }

    // 验证文件大小 (1GB)
    const maxSize = 1024 * 1024 * 1024;
    if (file.size > maxSize) {
      setError('文件大小超过1GB限制');
      setSelectedFile(null);
      return;
    }

    setError('');
    setSelectedFile(file);

    // 自动填充数据集名称（从文件名提取，去除扩展名）
    const nameWithoutExt = file.name.replace(/\.(zip|tar|gz|tgz|bz2|tbz2|xz|txz|7z)$/i, '');
    setDatasetName(nameWithoutExt);
  };

  // 处理上传
  const handleUpload = async () => {
    if (!selectedFile) {
      setError('请选择压缩包文件');
      return;
    }

    if (!datasetName.trim()) {
      setError('请输入数据集名称');
      return;
    }

    setIsUploading(true);
    setError('');
    setUploadProgress(0);

    try {
      await datasetService.uploadDatasetArchive(
        datasetName.trim(),
        description.trim() || undefined,
        selectedFile,
        (progress) => setUploadProgress(progress)
      );

      // 上传成功
      onUploadComplete();
      resetForm();
      onClose();
    } catch (err: any) {
      setError(err.message || '上传失败，请重试');
    } finally {
      setIsUploading(false);
    }
  };

  // 弹窗关闭时重置
  useEffect(() => {
    if (!isOpen) {
      resetForm();
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[150] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
      <div className="bg-slate-900 border border-slate-700 p-6 rounded-xl w-full max-w-md shadow-2xl animate-in fade-in zoom-in duration-200">
        {/* Header */}
        <div className="flex items-center mb-6">
          <div className="w-12 h-12 rounded-full flex items-center justify-center bg-cyan-900/30 text-cyan-500 mr-4">
            <Archive size={24} />
          </div>
          <div>
            <h3 className="text-xl font-bold text-white">导入数据集</h3>
            <p className="text-sm text-slate-400">上传压缩包创建新数据集</p>
          </div>
        </div>

        {/* Form */}
        <div className="space-y-4 mb-6">
          {/* 文件选择 */}
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              压缩包文件 <span className="text-red-500">*</span>
            </label>
            <div className="relative">
              <input
                ref={fileInputRef}
                type="file"
                accept={supportedFormats.join(',')}
                onChange={handleFileSelect}
                className="hidden"
                disabled={isUploading}
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={isUploading}
                className="w-full px-4 py-3 bg-slate-950 border border-slate-700 rounded-lg text-left text-slate-400 hover:border-cyan-500 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {selectedFile ? (
                  <div className="flex items-center">
                    <CheckCircle size={18} className="text-cyan-500 mr-2" />
                    <span className="text-white truncate">{selectedFile.name}</span>
                    <span className="text-slate-500 text-xs ml-auto">
                      {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                    </span>
                  </div>
                ) : (
                  <div className="flex items-center">
                    <Upload size={18} className="mr-2" />
                    <span>选择压缩包文件</span>
                  </div>
                )}
              </button>
            </div>
            <p className="text-xs text-slate-500 mt-1">
              支持格式: {supportedFormats.join(', ')} | 最大1GB
            </p>
          </div>

          {/* 数据集名称 */}
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              数据集名称 <span className="text-red-500">*</span>
            </label>
            <input
              type="text"
              value={datasetName}
              onChange={(e) => setDatasetName(e.target.value)}
              placeholder="输入数据集名称"
              disabled={isUploading}
              className="w-full bg-slate-950 border border-slate-700 rounded-lg px-4 py-2.5 text-white outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 transition-all disabled:opacity-50"
            />
          </div>

          {/* 描述 */}
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              描述 <span className="text-slate-500">(可选)</span>
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="输入数据集描述"
              disabled={isUploading}
              rows={3}
              className="w-full bg-slate-950 border border-slate-700 rounded-lg px-4 py-2.5 text-white outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 transition-all disabled:opacity-50 resize-none"
            />
          </div>

          {/* 上传进度 */}
          {isUploading && (
            <div className="bg-slate-950 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-slate-300">正在上传...</span>
                <span className="text-sm text-cyan-400">{uploadProgress}%</span>
              </div>
              <div className="w-full bg-slate-800 rounded-full h-2">
                <div
                  className="bg-gradient-to-r from-cyan-500 to-cyan-400 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
            </div>
          )}

          {/* 错误提示 */}
          {error && (
            <div className="bg-red-900/20 border border-red-800 rounded-lg p-3 flex items-center">
              <AlertCircle size={18} className="text-red-500 mr-2" />
              <span className="text-sm text-red-400">{error}</span>
            </div>
          )}
        </div>

        {/* Buttons */}
        <div className="flex gap-3">
          <button
            onClick={onClose}
            disabled={isUploading}
            className="flex-1 py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg font-medium transition-colors border border-slate-700 disabled:opacity-50"
          >
            取消
          </button>
          <button
            onClick={handleUpload}
            disabled={isUploading || !selectedFile}
            className="flex-1 py-2.5 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg font-bold transition-colors shadow-lg shadow-cyan-900/20 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
          >
            {isUploading ? (
              <>
                <Loader2 size={18} className="mr-2 animate-spin" />
                上传中
              </>
            ) : (
              <>
                <Upload size={18} className="mr-2" />
                开始上传
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

// --- Delete Dataset Dialog Component ---
interface DeleteDatasetDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  datasetName: string;
  isDeleting: boolean;
}

const DeleteDatasetDialog: React.FC<DeleteDatasetDialogProps> = ({
  isOpen,
  onClose,
  onConfirm,
  datasetName,
  isDeleting
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[150] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
      <div className="bg-slate-900 border border-slate-700 p-6 rounded-xl w-full max-w-md shadow-2xl animate-in fade-in zoom-in duration-200">
        {/* Header */}
        <div className="flex items-center mb-6">
          <div className="w-12 h-12 rounded-full flex items-center justify-center bg-red-900/30 text-red-500 mr-4">
            <Trash2 size={24} />
          </div>
          <div>
            <h3 className="text-xl font-bold text-white">删除数据集</h3>
            <p className="text-sm text-slate-400">此操作不可恢复</p>
          </div>
        </div>

        {/* Warning Message */}
        <div className="bg-red-900/20 border border-red-800 rounded-lg p-4 mb-6">
          <p className="text-sm text-red-300">
            确定要删除数据集 <span className="font-bold text-white">"{datasetName}"</span> 吗？
          </p>
          <p className="text-xs text-red-400 mt-2">
            此操作将同时删除数据库记录和数据集文件夹中的所有文件。
          </p>
        </div>

        {/* Buttons */}
        <div className="flex gap-3">
          <button
            onClick={onClose}
            disabled={isDeleting}
            className="flex-1 py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg font-medium transition-colors border border-slate-700 disabled:opacity-50"
          >
            取消
          </button>
          <button
            onClick={onConfirm}
            disabled={isDeleting}
            className="flex-1 py-2.5 bg-red-600 hover:bg-red-500 text-white rounded-lg font-bold transition-colors shadow-lg shadow-red-900/20 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
          >
            {isDeleting ? (
              <>
                <Loader2 size={18} className="mr-2 animate-spin" />
                删除中
              </>
            ) : (
              <>
                <Trash2 size={18} className="mr-2" />
                确认删除
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

// --- View Description Dialog Component ---
interface ViewDescriptionDialogProps {
  isOpen: boolean;
  onClose: () => void;
  datasetName: string;
  description?: string;
}

const ViewDescriptionDialog: React.FC<ViewDescriptionDialogProps> = ({
  isOpen,
  onClose,
  datasetName,
  description
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[150] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
      <div className="bg-slate-900 border border-slate-700 p-6 rounded-xl w-full max-w-lg shadow-2xl animate-in fade-in zoom-in duration-200">
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center">
            <div className="w-10 h-10 rounded-full flex items-center justify-center bg-cyan-900/30 text-cyan-500 mr-3">
              <FileText size={20} />
            </div>
            <div>
              <h3 className="text-lg font-bold text-white">{datasetName}</h3>
              <p className="text-xs text-slate-400">数据集描述</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-slate-800 text-slate-500 hover:text-white transition-colors"
          >
            <X size={20} />
          </button>
        </div>

        {/* Description Content */}
        <div className="bg-slate-950 border border-slate-800 rounded-lg p-4 max-h-96 overflow-y-auto custom-scrollbar">
          {description && description.trim() ? (
            <p className="text-slate-300 text-sm leading-relaxed whitespace-pre-wrap">{description}</p>
          ) : (
            <p className="text-slate-600 text-sm italic">暂无描述</p>
          )}
        </div>

        {/* Close Button */}
        <div className="mt-4 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-lg text-sm transition-colors border border-slate-700"
          >
            关闭
          </button>
        </div>
      </div>
    </div>
  );
};

// --- Directory Tree Component for Unknown Format ---
interface DirectoryTreeNode {
  name: string;
  type: 'folder' | 'file' | 'error';
  path?: string;
  extension?: string;
  is_image?: boolean;
  size?: number;
  children?: DirectoryTreeNode[];
  child_count?: number;
  truncated?: boolean;
  message?: string;
}

interface DirectoryTreeProps {
  data: DirectoryTreeNode;
  level?: number;
}

const DirectoryTree: React.FC<DirectoryTreeProps> = ({ data, level = 0 }) => {
  const [isExpanded, setIsExpanded] = useState(level < 2); // 默认展开前两层

  const handleClick = () => {
    if (data.type === 'folder') {
      setIsExpanded(!isExpanded);
    }
  };

  if (data.type === 'file') {
    const isImage = data.is_image || false;
    const size = data.size ? (data.size < 1024 ? `${data.size}B` : data.size < 1024 * 1024 ? `${(data.size / 1024).toFixed(1)}KB` : `${(data.size / 1024 / 1024).toFixed(1)}MB`) : '';
    return (
      <div
        className="flex items-center py-1 px-2 text-slate-400 text-sm hover:bg-slate-800 rounded"
        style={{ paddingLeft: `${level * 16 + 8}px` }}
      >
        {isImage ? (
          <FileImage size={14} className="mr-2 text-cyan-500" />
        ) : (
          <File size={14} className="mr-2 text-slate-500" />
        )}
        <span className="flex-1 truncate">{data.name}</span>
        {size && <span className="text-xs text-slate-600 ml-2">{size}</span>}
      </div>
    );
  }

  if (data.type === 'error') {
    return (
      <div
        className="flex items-center py-1 px-2 text-red-400 text-sm"
        style={{ paddingLeft: `${level * 16 + 8}px` }}
      >
        <AlertCircle size={14} className="mr-2" />
        <span>{data.message || '无法访问'}</span>
      </div>
    );
  }

  // folder
  return (
    <div>
      <div
        onClick={handleClick}
        className="flex items-center py-1 px-2 text-slate-300 text-sm hover:bg-slate-800 rounded cursor-pointer select-none"
        style={{ paddingLeft: `${level * 16 + 8}px` }}
      >
        {isExpanded ? (
          <ChevronDown size={14} className="mr-1 text-slate-500" />
        ) : (
          <ChevronRight size={14} className="mr-1 text-slate-500" />
        )}
        {isExpanded ? (
          <FolderOpen size={14} className="mr-2 text-cyan-500" />
        ) : (
          <Folder size={14} className="mr-2 text-cyan-500" />
        )}
        <span className="flex-1">{data.name}</span>
        {data.child_count !== undefined && (
          <span className="text-xs text-slate-600">{data.child_count} 项</span>
        )}
      </div>
      {isExpanded && data.children && (
        <div>
          {data.children.map((child, index) => (
            <DirectoryTree key={index} data={child} level={level + 1} />
          ))}
          {data.truncated && (
            <div className="py-1 px-2 text-slate-600 text-xs italic" style={{ paddingLeft: `${(level + 1) * 16 + 8}px` }}>
              ... (已达到最大深度)
            </div>
          )}
        </div>
      )}
    </div>
  );
};

interface UnknownFormatViewProps {
  datasetId: string;
  datasetName: string;
}

const UnknownFormatView: React.FC<UnknownFormatViewProps> = ({ datasetId, datasetName }) => {
  const [structure, setStructure] = useState<DirectoryTreeNode | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchStructure = async () => {
      setLoading(true);
      setError('');
      try {
        const data = await datasetService.getDirectoryStructure(parseInt(datasetId));
        setStructure(data);
      } catch (err: any) {
        setError(err.message || '获取目录结构失败');
      } finally {
        setLoading(false);
      }
    };
    fetchStructure();
  }, [datasetId]);

  return (
    <div className="bg-slate-900/50 border border-amber-900/50 rounded-lg p-6">
      {/* Header */}
      <div className="flex items-center mb-4">
        <div className="w-10 h-10 rounded-full flex items-center justify-center bg-amber-900/30 text-amber-500 mr-3">
          <AlertCircle size={20} />
        </div>
        <div>
          <h4 className="text-lg font-bold text-amber-400">数据集格式未识别</h4>
          <p className="text-xs text-slate-400">系统无法自动解析此数据集的结构</p>
        </div>
      </div>

      {/* Info Message */}
      <div className="bg-amber-900/20 border border-amber-800/50 rounded-lg p-3 mb-4 text-sm text-amber-300">
        <p>数据集 <strong>"{datasetName}"</strong> 的格式无法被自动识别。</p>
        <p className="mt-1 text-xs text-amber-400/70">以下是文件夹结构，您可以查看其中的内容。</p>
      </div>

      {/* Directory Tree */}
      {loading ? (
        <div className="flex items-center justify-center py-8">
          <Loader2 size={24} className="animate-spin text-cyan-500" />
          <span className="ml-3 text-slate-400">正在加载目录结构...</span>
        </div>
      ) : error ? (
        <div className="bg-red-900/20 border border-red-800 rounded-lg p-4 text-red-400">
          {error}
        </div>
      ) : structure ? (
        <div className="bg-slate-950 rounded-lg p-3 max-h-96 overflow-y-auto border border-slate-800">
          <DirectoryTree data={structure} />
        </div>
      ) : null}
    </div>
  );
};

// --- Modern Lightbox Component ---
const Lightbox: React.FC<{ src: string, onClose: () => void }> = ({ src, onClose }) => {
  const [scale, setScale] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const isDragging = useRef(false);
  const startPos = useRef({ x: 0, y: 0 });

  const handleWheel = (e: React.WheelEvent) => {
    e.stopPropagation();
    const delta = -e.deltaY * 0.001;
    setScale(s => Math.min(Math.max(0.5, s + delta), 5));
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    isDragging.current = true;
    startPos.current = { x: e.clientX - position.x, y: e.clientY - position.y };
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging.current) return;
    setPosition({
      x: e.clientX - startPos.current.x,
      y: e.clientY - startPos.current.y
    });
  };

  const handleMouseUp = () => {
    isDragging.current = false;
  };

  return (
    <div 
      className="fixed inset-0 z-[100] bg-black/95 backdrop-blur-sm flex items-center justify-center overflow-hidden" 
      onWheel={handleWheel}
      onClick={onClose}
    >
      <button 
        onClick={onClose} 
        className="absolute top-6 right-6 p-2 text-slate-400 hover:text-white transition-colors z-[101]"
      >
        <X size={32} />
      </button>

      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 text-slate-500 text-xs pointer-events-none bg-black/50 px-3 py-1 rounded-full">
        滚轮缩放 • 拖拽平移
      </div>

      <img 
        src={src} 
        className="max-w-none transition-transform duration-75 cursor-move" 
        style={{ 
          transform: `translate(${position.x}px, ${position.y}px) scale(${scale})`,
          maxHeight: '90vh',
          maxWidth: '90vw'
        }} 
        onClick={(e) => e.stopPropagation()}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onDragStart={(e) => e.preventDefault()}
      />
    </div>
  );
};

const DatasetManager: React.FC = () => {
  // 使用数据集Hook获取真实数据
  const { datasets: apiDatasets, loading, error, fetchDatasets } = useDataset();

  // 使用适配器转换API数据为组件格式
  const datasets: DatasetItem[] = adaptDatasetList(apiDatasets);

  // 搜索过滤状态
  const [searchQuery, setSearchQuery] = useState<string>('');

  // 筛选状态
  const [filterOpen, setFilterOpen] = useState(false);
  const [filterOptions, setFilterOptions] = useState({
    types: [] as string[],      // 数据集类型筛选
    minSize: 0,                 // 最小大小(MB)
    maxSize: 0,                 // 最大大小(MB)，0表示不限制
    sortBy: 'name' as 'name' | 'date' | 'size' | 'count',  // 排序方式
    sortOrder: 'asc' as 'asc' | 'desc'  // 排序顺序
  });

  // 获取所有可用的数据集类型
  const availableTypes = Array.from(new Set(datasets.map(ds => ds.type))).sort();

  // 应用筛选和排序
  const filteredDatasets = datasets.filter(ds => {
    // 搜索过滤
    const matchesSearch = ds.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         ds.type.toLowerCase().includes(searchQuery.toLowerCase());
    if (!matchesSearch) return false;

    // 类型筛选
    if (filterOptions.types.length > 0 && !filterOptions.types.includes(ds.type)) {
      return false;
    }

    // 大小筛选（将字符串转为MB比较）
    if (filterOptions.minSize > 0 || filterOptions.maxSize > 0) {
      const sizeInMB = parseSizeToMB(ds.size);
      if (filterOptions.minSize > 0 && sizeInMB < filterOptions.minSize) return false;
      if (filterOptions.maxSize > 0 && sizeInMB > filterOptions.maxSize) return false;
    }

    return true;
  }).sort((a, b) => {
    // 排序
    let compareValue = 0;
    switch (filterOptions.sortBy) {
      case 'name':
        compareValue = a.name.localeCompare(b.name);
        break;
      case 'date':
        compareValue = new Date(a.lastModified).getTime() - new Date(b.lastModified).getTime();
        break;
      case 'size':
        compareValue = parseSizeToMB(a.size) - parseSizeToMB(b.size);
        break;
      case 'count':
        compareValue = a.count - b.count;
        break;
    }
    return filterOptions.sortOrder === 'asc' ? compareValue : -compareValue;
  });

  // 辅助函数：解析大小字符串为MB
  function parseSizeToMB(sizeStr: string): number {
    const match = sizeStr.match(/^([\d.]+)\s*(B|KB|MB|GB|TB)?$/i);
    if (!match) return 0;
    const value = parseFloat(match[1]);
    const unit = (match[2] || 'B').toUpperCase();
    const multipliers: Record<string, number> = { 'B': 1 / 1024 / 1024, 'KB': 1 / 1024, 'MB': 1, 'GB': 1024, 'TB': 1024 * 1024 };
    return value * (multipliers[unit] || 1);
  }

  // 检查是否有激活的筛选
  const hasActiveFilters = filterOptions.types.length > 0 ||
                          filterOptions.minSize > 0 ||
                          filterOptions.maxSize > 0 ||
                          filterOptions.sortBy !== 'name' ||
                          filterOptions.sortOrder !== 'asc';

  const [selectedDsId, setSelectedDsId] = useState<string>('');
  const [lightboxImg, setLightboxImg] = useState<string | null>(null);
  const [imageUrls, setImageUrls] = useState<string[]>([]);

  // 导入弹窗状态
  const [importDialogOpen, setImportDialogOpen] = useState(false);

  // 删除弹窗状态
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [datasetToDelete, setDatasetToDelete] = useState<DatasetItem | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  // 查看描述弹窗状态
  const [viewDescriptionDialogOpen, setViewDescriptionDialogOpen] = useState(false);
  const [datasetToViewDescription, setDatasetToViewDescription] = useState<DatasetItem | null>(null);

  // 当数据集列表加载完成后，设置默认选中项
  useEffect(() => {
    if (filteredDatasets.length > 0 && !selectedDsId) {
      setSelectedDsId(filteredDatasets[0].id);
    }
  }, [filteredDatasets, selectedDsId]);

  // 当选中的数据集改变时，获取图片URL列表
  useEffect(() => {
    if (selectedDsId && apiDatasets.length > 0) {
      const selectedDataset = apiDatasets.find((ds: any) => ds.id === parseInt(selectedDsId));
      if (selectedDataset) {
        // 从元数据中获取图片路径，然后构建URL
        const imagePaths = selectedDataset.meta?.image_paths || [];
        const urls = imagePaths.map((_: string, index: number) =>
          `${apiClient['baseURL']}/datasets/${selectedDataset.id}/image-file?index=${index}`
        );
        setImageUrls(urls);
      }
    }
  }, [selectedDsId, apiDatasets]);

  // Pagination State for Samples
  const [visibleSamples, setVisibleSamples] = useState(24);
  const [isLoadingMore, setIsLoadingMore] = useState(false);

  // Reset pagination when dataset changes
  useEffect(() => {
    setVisibleSamples(24);
  }, [selectedDsId]);

  const handleLoadMore = () => {
    setIsLoadingMore(true);
    // Simulate network delay
    setTimeout(() => {
      setVisibleSamples(prev => prev + 24);
      setIsLoadingMore(false);
    }, 500);
  };

  // 打开导入弹窗
  const handleOpenImportDialog = () => {
    setImportDialogOpen(true);
  };

  // 上传完成后刷新列表
  const handleUploadComplete = () => {
    fetchDatasets();
  };

  // 打开删除确认弹窗
  const handleOpenDeleteDialog = (dataset: DatasetItem, e: React.MouseEvent) => {
    e.stopPropagation(); // 阻止事件冒泡，避免触发选中数据集
    setDatasetToDelete(dataset);
    setDeleteDialogOpen(true);
  };

  // 确认删除数据集
  const handleConfirmDelete = async () => {
    if (!datasetToDelete) return;

    setIsDeleting(true);
    try {
      await datasetService.deleteDataset(parseInt(datasetToDelete.id));

      // 如果删除的是当前选中的数据集，需要切换选中项
      if (selectedDsId === datasetToDelete.id) {
        const remainingDatasets = datasets.filter(ds => ds.id !== datasetToDelete.id);
        if (remainingDatasets.length > 0) {
          setSelectedDsId(remainingDatasets[0].id);
        } else {
          setSelectedDsId('');
        }
      }

      // 刷新列表
      fetchDatasets();
      setDeleteDialogOpen(false);
      setDatasetToDelete(null);
    } catch (err: any) {
      console.error('删除数据集失败:', err);
      alert(err.message || '删除数据集失败，请重试');
    } finally {
      setIsDeleting(false);
    }
  };

  // 加载状态
  if (loading && datasets.length === 0) {
    return (
      <div className="h-full flex flex-col items-center justify-center p-6 space-y-6">
        <div className="w-12 h-12 rounded-full border-4 border-cyan-500 border-t-transparent animate-spin"></div>
        <p className="text-slate-400 text-sm">正在加载数据集...</p>
      </div>
    );
  }

  // 错误状态
  if (error) {
    return (
      <div className="h-full flex flex-col items-center justify-center p-6 space-y-6">
        <AlertCircle size={48} className="text-red-500" />
        <p className="text-red-400 text-sm">{error}</p>
        <button
          onClick={fetchDatasets}
          className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg text-sm flex items-center"
        >
          <RefreshCw size={16} className="mr-2" />
          重试
        </button>
      </div>
    );
  }

  // 空状态
  if (datasets.length === 0) {
    return (
      <div className="h-full flex flex-col items-center justify-center p-6 space-y-6">
        <Folder size={48} className="text-slate-600" />
        <p className="text-slate-400 text-sm">暂无数据集</p>
        <p className="text-slate-500 text-xs">点击下方"导入数据集"按钮添加您的第一个数据集</p>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col p-6 space-y-6">
      <div className="flex justify-between items-center">
         <div>
           <h2 className="text-2xl font-bold text-white">数据集管理</h2>
           <p className="text-slate-400 text-sm">查看原始数据分布与元信息。</p>
         </div>
         <div className="flex gap-2">
           <button
             onClick={fetchDatasets}
             disabled={loading}
             className="flex items-center px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white font-bold rounded-xl border border-slate-700 transition-all disabled:opacity-50"
             title="刷新列表"
           >
              <RefreshCw size={18} className={`mr-2 ${loading ? 'animate-spin' : ''}`} />
              刷新
           </button>
           <button
             onClick={handleOpenImportDialog}
             className="flex items-center px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white font-bold rounded-xl shadow-lg shadow-cyan-900/20 transition-all"
           >
              <Upload size={18} className="mr-2" /> 导入数据
           </button>
         </div>
      </div>

      <div className="flex-1 flex gap-6 min-h-0">
        {/* Left List */}
        <div className="w-full md:w-1/4 flex flex-col gap-4 min-h-0">
           {/* Search and Filter Row */}
           <div className="flex gap-2">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={14} />
                <input
                  type="text"
                  placeholder="搜索数据集..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-9 pr-4 py-2 bg-slate-900 border border-slate-700 rounded-lg text-xs text-white focus:outline-none focus:border-cyan-500 placeholder:text-slate-600"
                />
                {searchQuery && (
                  <button
                    onClick={() => setSearchQuery('')}
                    className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
                  >
                    <X size={12} />
                  </button>
                )}
              </div>
              <button
                onClick={() => setFilterOpen(!filterOpen)}
                className={`px-3 py-2 rounded-lg border transition-all flex items-center justify-center ${
                  hasActiveFilters
                    ? 'bg-cyan-600/20 border-cyan-500 text-cyan-400'
                    : 'bg-slate-900 border-slate-700 text-slate-500 hover:border-slate-600'
                }`}
                title="筛选设置"
              >
                <SlidersHorizontal size={16} />
              </button>
           </div>

           {/* Filter Panel */}
           {filterOpen && (
             <div className="bg-slate-900 border border-slate-700 rounded-lg p-3 space-y-3">
                {/* Type Filter */}
                <div>
                  <label className="text-xs text-slate-400 mb-2 block">数据集类型</label>
                  <div className="flex flex-wrap gap-1.5">
                    {availableTypes.map(type => (
                      <button
                        key={type}
                        onClick={() => {
                          setFilterOptions(prev => ({
                            ...prev,
                            types: prev.types.includes(type)
                              ? prev.types.filter(t => t !== type)
                              : [...prev.types, type]
                          }));
                        }}
                        className={`px-2 py-1 rounded text-xs transition-all ${
                          filterOptions.types.includes(type)
                            ? 'bg-cyan-600 text-white'
                            : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
                        }`}
                      >
                        {type}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Size Filter */}
                <div>
                  <label className="text-xs text-slate-400 mb-2 block">数据集大小</label>
                  <div className="flex gap-2">
                    <select
                      value={filterOptions.minSize}
                      onChange={(e) => setFilterOptions(prev => ({ ...prev, minSize: Number(e.target.value) }))}
                      className="flex-1 bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs text-white focus:outline-none focus:border-cyan-500"
                    >
                      <option value={0}>最小: 不限</option>
                      <option value={1}>最小: 1MB</option>
                      <option value={10}>最小: 10MB</option>
                      <option value={100}>最小: 100MB</option>
                      <option value={1024}>最小: 1GB</option>
                    </select>
                    <select
                      value={filterOptions.maxSize}
                      onChange={(e) => setFilterOptions(prev => ({ ...prev, maxSize: Number(e.target.value) }))}
                      className="flex-1 bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs text-white focus:outline-none focus:border-cyan-500"
                    >
                      <option value={0}>最大: 不限</option>
                      <option value={100}>最大: 100MB</option>
                      <option value={1024}>最大: 1GB</option>
                      <option value={10240}>最大: 10GB</option>
                      <option value={102400}>最大: 100GB</option>
                    </select>
                  </div>
                </div>

                {/* Sort Options */}
                <div>
                  <label className="text-xs text-slate-400 mb-2 block">排序方式</label>
                  <div className="flex gap-2">
                    <select
                      value={filterOptions.sortBy}
                      onChange={(e) => setFilterOptions(prev => ({ ...prev, sortBy: e.target.value as any }))}
                      className="flex-1 bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs text-white focus:outline-none focus:border-cyan-500"
                    >
                      <option value="name">按名称</option>
                      <option value="date">按时间</option>
                      <option value="size">按大小</option>
                      <option value="count">按数量</option>
                    </select>
                    <button
                      onClick={() => setFilterOptions(prev => ({ ...prev, sortOrder: prev.sortOrder === 'asc' ? 'desc' : 'asc' }))}
                      className="px-3 bg-slate-800 border border-slate-700 rounded text-white hover:bg-slate-700 transition-all flex items-center"
                      title={filterOptions.sortOrder === 'asc' ? '升序' : '降序'}
                    >
                      {filterOptions.sortOrder === 'asc' ? '↑' : '↓'}
                    </button>
                  </div>
                </div>

                {/* Actions */}
                <div className="flex gap-2 pt-1">
                  <button
                    onClick={() => setFilterOptions({
                      types: [],
                      minSize: 0,
                      maxSize: 0,
                      sortBy: 'name',
                      sortOrder: 'asc'
                    })}
                    className="flex-1 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-400 rounded text-xs transition-all"
                  >
                    重置
                  </button>
                  <button
                    onClick={() => setFilterOpen(false)}
                    className="flex-1 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-white rounded text-xs transition-all"
                  >
                    应用 ({filteredDatasets.length})
                  </button>
                </div>
             </div>
           )}
           <div className="flex-1 overflow-y-auto space-y-2 pr-1 custom-scrollbar">
              {filteredDatasets.length > 0 ? (
                filteredDatasets.map((ds) => (
                <div
                  key={ds.id}
                  onClick={() => setSelectedDsId(ds.id)}
                  className={`p-3 rounded-lg border cursor-pointer transition-all relative overflow-hidden group ${
                    selectedDsId === ds.id
                      ? 'bg-cyan-950/20 border-cyan-500/50 shadow-[0_0_15px_rgba(6,182,212,0.15)]'
                      : 'bg-slate-900/40 border-slate-800 hover:border-cyan-500/50 hover:bg-slate-900/80'
                  }`}
                >
                  <div className="flex items-center mb-2">
                    <div className="group-hover:scale-110 transition-transform origin-left">
                       <Folder size={16} className={`${selectedDsId === ds.id ? 'text-cyan-400' : 'text-slate-500 group-hover:text-cyan-400 transition-colors'}`} />
                    </div>
                    <span className={`ml-2 text-sm font-medium truncate flex-1 transition-colors ${selectedDsId === ds.id ? 'text-white' : 'text-slate-200 group-hover:text-cyan-400'}`}>{ds.name}</span>
                    {/* 查看描述按钮 */}
                    {ds.description && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setDatasetToViewDescription(ds);
                          setViewDescriptionDialogOpen(true);
                        }}
                        className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-cyan-900/30 text-slate-500 hover:text-cyan-400 transition-all"
                        title="查看描述"
                      >
                        <FileText size={14} />
                      </button>
                    )}
                    {/* 删除按钮 */}
                    <button
                      onClick={(e) => handleOpenDeleteDialog(ds, e)}
                      className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-red-900/30 text-slate-500 hover:text-red-400 transition-all"
                      title="删除数据集"
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                  <div className="flex justify-between text-[10px] text-slate-500">
                    <span className="bg-slate-900 px-1.5 py-0.5 rounded border border-slate-800">{ds.type}</span>
                    <span>{ds.count.toLocaleString()} imgs</span>
                  </div>

                  {/* Gradient Glow Effect */}
                  <div className="absolute bottom-0 left-0 w-full h-0.5 bg-gradient-to-r from-cyan-500 to-purple-600 transform scale-x-0 group-hover:scale-x-100 transition-transform origin-left"></div>
                </div>
              ))): (
                <div className="flex flex-col items-center justify-center py-8 text-slate-500">
                  <Search size={24} className="mb-2 opacity-50" />
                  <p className="text-xs">未找到匹配的数据集</p>
                </div>
              )}

              {/* Import New Dataset Card */}
              <div
                onClick={handleOpenImportDialog}
                className="p-4 rounded-lg border-2 border-dashed border-slate-800 hover:border-cyan-500/50 hover:bg-slate-900/40 cursor-pointer transition-all flex flex-col items-center justify-center text-slate-500 hover:text-cyan-400 group"
              >
                  <Plus size={24} className="mb-2 group-hover:scale-110 transition-transform" />
                  <span className="text-xs font-medium">导入数据集</span>
              </div>
           </div>
        </div>

        {/* Right Content */}
        <div className="flex-1 glass-panel border border-slate-800 rounded-xl overflow-hidden flex flex-col">
           <div className="h-14 border-b border-slate-800 bg-slate-900/50 flex items-center px-6 justify-between">
              <h3 className="font-bold text-white flex items-center">
                <Layout size={18} className="mr-2 text-cyan-400" /> {datasets.find(d => d.id === selectedDsId)?.name}
              </h3>
              <div className="flex items-center space-x-4 text-xs text-slate-400 font-mono">
                 <span className="flex items-center"><HardDrive size={12} className="mr-1"/> {datasets.find(d => d.id === selectedDsId)?.size}</span>
                 <span className="flex items-center"><Calendar size={12} className="mr-1"/> {datasets.find(d => d.id === selectedDsId)?.lastModified}</span>
              </div>
           </div>
           
           <div className="p-6 overflow-y-auto custom-scrollbar">
               {/* 检查数据集格式，如果是UNKNOWN则显示文件夹树 */}
               {(() => {
                 const currentDataset = datasets.find(d => d.id === selectedDsId);
                 const isUnknownFormat = currentDataset?.type?.toUpperCase() === 'UNKNOWN';

                 if (isUnknownFormat) {
                   return (
                     <div className="py-4">
                       <UnknownFormatView
                         datasetId={selectedDsId}
                         datasetName={currentDataset?.name || ''}
                       />
                     </div>
                   );
                 }

                 return (
                   <>
                    {/* Metadata Cards */}
                    <div className="grid grid-cols-4 gap-4 mb-8">
                      <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-800 text-center">
                          <div className="text-2xl font-mono text-white">{currentDataset?.count.toLocaleString() || '-'}</div>
                          <div className="text-xs text-slate-500 uppercase mt-1">样本总数</div>
                      </div>
                      <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-800 text-center">
                          <div className="text-2xl font-mono text-emerald-400">
                            {currentDataset?.stats?.numClasses
                              ? `${currentDataset.stats.numClasses} 类`
                              : currentDataset?.count === 0
                                ? '-'
                                : '未知'
                            }
                          </div>
                          <div className="text-xs text-slate-500 uppercase mt-1">类别分布</div>
                      </div>
                      <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-800 text-center">
                          <div className="text-2xl font-mono text-cyan-400">
                            {currentDataset?.stats?.avgWidth && currentDataset?.stats?.avgHeight
                              ? `${Math.round(currentDataset.stats.avgWidth)}x${Math.round(currentDataset.stats.avgHeight)}`
                              : currentDataset?.count === 0
                                ? '-'
                                : '未知'
                            }
                          </div>
                          <div className="text-xs text-slate-500 uppercase mt-1">平均分辨率</div>
                      </div>
                      <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-800 text-center">
                          <div className="text-2xl font-mono text-purple-400">
                            {currentDataset?.stats?.annotationRate !== undefined
                              ? `${(currentDataset.stats.annotationRate * 100).toFixed(1)}%`
                              : currentDataset?.count === 0
                                ? '-'
                                : '未知'
                            }
                          </div>
                          <div className="text-xs text-slate-500 uppercase mt-1">标注率</div>
                      </div>
                    </div>

                    {/* Gallery */}
                    <div className="pb-4">
                       <h3 className="text-sm font-bold text-slate-300 mb-4 flex items-center">
                         <Eye size={16} className="mr-2" /> 样本概览
                       </h3>
                       <div className="grid grid-cols-4 md:grid-cols-5 lg:grid-cols-6 gap-3 mb-6">
                         {imageUrls.slice(0, visibleSamples).length > 0 ? (
                           imageUrls.slice(0, visibleSamples).map((imgUrl, i) => (
                             <div key={i} onClick={() => setLightboxImg(imgUrl)} className="aspect-square bg-slate-800 rounded border border-slate-800 overflow-hidden relative group cursor-pointer hover:border-cyan-500 transition-all duration-200">
                               <img
                                 src={imgUrl}
                                 className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity"
                                 onError={(e) => {
                                   // 如果图片加载失败，使用占位符
                                   (e.target as HTMLImageElement).src = `https://picsum.photos/600/600?random=${i}`;
                                 }}
                                 alt={`Sample ${i + 1}`}
                               />
                               <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors" />
                             </div>
                           ))
                         ) : (
                           // 如果没有图片，显示占位符
                           Array.from({length: Math.min(visibleSamples, 24)}).map((_, i) => (
                             <div key={i} className="aspect-square bg-slate-800 rounded border border-slate-800 overflow-hidden relative flex items-center justify-center">
                               <span className="text-slate-600 text-xs">无图片</span>
                             </div>
                           ))
                         )}
                       </div>

                       {/* Load More Button - 只在有更多图片时显示 */}
                       {imageUrls.length > visibleSamples && (
                         <div className="flex justify-center">
                           <button
                             onClick={handleLoadMore}
                             disabled={isLoadingMore}
                             className="px-6 py-2 bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-white rounded-full border border-slate-700 transition-all text-xs font-medium flex items-center disabled:opacity-50"
                           >
                             {isLoadingMore ? (
                                <span className="flex items-center"><div className="w-3 h-3 rounded-full border-2 border-slate-500 border-t-transparent animate-spin mr-2"></div> Loading...</span>
                             ) : (
                                <span className="flex items-center"><ChevronDown size={14} className="mr-1" /> 加载更多 ({imageUrls.length - visibleSamples})</span>
                             )}
                           </button>
                         </div>
                       )}
                    </div>
                   </>
                 );
               })()}
           </div>
        </div>
      </div>

      {lightboxImg && <Lightbox src={lightboxImg} onClose={() => setLightboxImg(null)} />}

      {/* Import Dataset Dialog */}
      <ImportDatasetDialog
        isOpen={importDialogOpen}
        onClose={() => setImportDialogOpen(false)}
        onUploadComplete={handleUploadComplete}
      />

      {/* Delete Dataset Dialog */}
      <DeleteDatasetDialog
        isOpen={deleteDialogOpen}
        onClose={() => setDeleteDialogOpen(false)}
        onConfirm={handleConfirmDelete}
        datasetName={datasetToDelete?.name || ''}
        isDeleting={isDeleting}
      />

      {/* View Description Dialog */}
      <ViewDescriptionDialog
        isOpen={viewDescriptionDialogOpen}
        onClose={() => setViewDescriptionDialogOpen(false)}
        datasetName={datasetToViewDescription?.name || ''}
        description={datasetToViewDescription?.description}
      />
    </div>
  );
};

export default DatasetManager;