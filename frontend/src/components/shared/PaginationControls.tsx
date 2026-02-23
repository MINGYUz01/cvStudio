/**
 * 分页控件组件
 * 支持翻页按钮、页码按钮、每页数量选择
 */

import React from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';

// 每页数量选项
export const PAGE_SIZE_OPTIONS = [12, 24, 48, 96, 'all'] as const;
export type PageSizeOption = typeof PAGE_SIZE_OPTIONS[number];

/**
 * 分页控件属性接口
 */
interface PaginationControlsProps {
  /** 当前页码（从1开始） */
  currentPage: number;
  /** 总页数 */
  totalPages: number;
  /** 每页显示数量 */
  pageSize: PageSizeOption;
  /** 总数量 */
  total: number;
  /** 页码变化回调 */
  onPageChange: (page: number) => void;
  /** 每页数量变化回调 */
  onPageSizeChange: (size: PageSizeOption) => void;
  /** 是否禁用 */
  disabled?: boolean;
  /** 当前加载的图片数量（用于 All 模式显示） */
  loadedCount?: number;
}

/**
 * 分页控件组件
 */
const PaginationControls: React.FC<PaginationControlsProps> = ({
  currentPage,
  totalPages,
  pageSize,
  total,
  onPageChange,
  onPageSizeChange,
  disabled = false,
  loadedCount = 0
}) => {
  /**
   * 生成页码按钮数组（处理省略号逻辑）
   */
  const getPageNumbers = (): (number | string)[] => {
    const pages: (number | string)[] = [];
    const maxVisible = 7; // 最多显示7个页码按钮

    if (totalPages <= maxVisible) {
      return Array.from({ length: totalPages }, (_, i) => i + 1);
    }

    // 始终显示第一页
    pages.push(1);

    // 计算中间范围
    let start = Math.max(2, currentPage - 2);
    let end = Math.min(totalPages - 1, currentPage + 2);

    // 如果前面需要省略号
    if (start > 2) {
      pages.push('ellipsis');
    }

    // 添加中间页码
    for (let i = start; i <= end; i++) {
      pages.push(i);
    }

    // 如果后面需要省略号
    if (end < totalPages - 1) {
      pages.push('ellipsis');
    }

    // 始终显示最后一页
    pages.push(totalPages);

    return pages;
  };

  // 如果是 All 模式，显示简化版控件
  if (pageSize === 'all') {
    return (
      <div className="flex items-center justify-center bg-slate-900/50 border border-slate-800 rounded-lg p-3">
        <div className="flex items-center gap-4">
          <span className="text-xs text-slate-400">
            已加载 {loadedCount.toLocaleString()} / {total.toLocaleString()} 张图片
          </span>
          <button
            onClick={() => onPageSizeChange(24)}
            disabled={disabled}
            className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-cyan-400 hover:text-cyan-300 rounded-lg text-xs border border-slate-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            切换回分页视图
          </button>
        </div>
      </div>
    );
  }

  // 单页数据不显示分页控件
  if (totalPages <= 1) {
    return (
      <div className="flex items-center justify-end bg-slate-900/50 border border-slate-800 rounded-lg p-3">
        <span className="text-xs text-slate-500">
          共 {total.toLocaleString()} 张图片
        </span>
      </div>
    );
  }

  const pageNumbers = getPageNumbers();

  return (
    <div className="flex items-center justify-between bg-slate-900/50 border border-slate-800 rounded-lg p-3 gap-4">
      {/* 左侧：每页数量选择 */}
      <div className="flex items-center gap-2 flex-shrink-0">
        <span className="text-xs text-slate-400 whitespace-nowrap">每页显示:</span>
        <select
          value={pageSize}
          onChange={(e) => onPageSizeChange(e.target.value as PageSizeOption)}
          disabled={disabled}
          className="bg-slate-800 border border-slate-700 rounded px-2 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {PAGE_SIZE_OPTIONS.map((size) => (
            <option key={size} value={size}>
              {size === 'all' ? '全部' : String(size)}
            </option>
          ))}
        </select>
        <span className="text-xs text-slate-500 whitespace-nowrap">
          共 {total.toLocaleString()} 张
        </span>
      </div>

      {/* 右侧：页码按钮 */}
      <div className="flex items-center gap-1">
        {/* 上一页按钮 */}
        <button
          onClick={() => onPageChange(currentPage - 1)}
          disabled={disabled || currentPage <= 1}
          className="p-1.5 rounded hover:bg-slate-700 text-slate-400 hover:text-white disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          title="上一页"
        >
          <ChevronLeft size={16} />
        </button>

        {/* 页码按钮 */}
        {pageNumbers.map((page, idx) =>
          page === 'ellipsis' ? (
            <span
              key={`ellipsis-${idx}`}
              className="px-1 text-slate-500 text-xs"
            >
              ...
            </span>
          ) : (
            <button
              key={page}
              onClick={() => onPageChange(page as number)}
              disabled={disabled}
              className={`min-w-[32px] px-2 py-1.5 rounded text-xs font-medium transition-colors ${
                currentPage === page
                  ? 'bg-cyan-600 text-white shadow-lg shadow-cyan-900/30'
                  : 'bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-white'
              } disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              {page}
            </button>
          )
        )}

        {/* 下一页按钮 */}
        <button
          onClick={() => onPageChange(currentPage + 1)}
          disabled={disabled || currentPage >= totalPages}
          className="p-1.5 rounded hover:bg-slate-700 text-slate-400 hover:text-white disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          title="下一页"
        >
          <ChevronRight size={16} />
        </button>
      </div>
    </div>
  );
};

export default PaginationControls;
