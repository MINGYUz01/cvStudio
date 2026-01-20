/**
 * 树形下拉选择器组件
 *
 * 用于选择带有层级结构的数据，如权重版本树
 */

import React, { useState, useRef, useEffect } from 'react';
import { ChevronRight, ChevronDown, Check, GitBranch, X } from 'lucide-react';

export interface WeightTreeSelectOption {
  id: number;
  name: string;
  display_name: string;
  version: string;
  source_type: 'uploaded' | 'trained';
  is_root: boolean;
  children?: WeightTreeSelectOption[];
}

interface WeightTreeSelectProps {
  options: WeightTreeSelectOption[];
  value?: number | null;
  onChange: (value: number | null) => void;
  placeholder?: string;
  disabled?: boolean;
  className?: string;
}

interface TreeNodeProps {
  node: WeightTreeSelectOption;
  level: number;
  selectedId?: number | null;
  onSelect: (id: number) => void;
  expandedIds: Set<number>;
  onToggleExpand: (id: number) => void;
}

const TreeNode: React.FC<TreeNodeProps> = ({
  node,
  level,
  selectedId,
  onSelect,
  expandedIds,
  onToggleExpand
}) => {
  const hasChildren = node.children && node.children.length > 0;
  const isExpanded = expandedIds.has(node.id);
  const isSelected = selectedId === node.id;

  const handleClick = () => {
    onSelect(node.id);
  };

  const handleExpandClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    onToggleExpand(node.id);
  };

  return (
    <div>
      <div
        className={`flex items-center py-2 px-3 cursor-pointer transition-colors ${
          isSelected
            ? 'bg-cyan-900/30 text-cyan-400'
            : 'hover:bg-slate-800 text-slate-300'
        }`}
        style={{ paddingLeft: `${level * 16 + 12}px` }}
        onClick={handleClick}
      >
        {/* 展开/折叠按钮 */}
        {hasChildren ? (
          <button
            className="mr-1 p-0.5 hover:bg-slate-700 rounded transition-colors"
            onClick={handleExpandClick}
          >
            {isExpanded ? (
              <ChevronDown size={12} />
            ) : (
              <ChevronRight size={12} />
            )}
          </button>
        ) : (
          <span className="w-4 mr-1" />
        )}

        {/* 选中标识 */}
        {isSelected && (
          <Check size={14} className="mr-2 text-cyan-400" />
        )}

        {/* 来源类型图标 */}
        <span className={`mr-2 ${node.source_type === 'trained' ? 'text-purple-400' : 'text-emerald-400'}`}>
          <GitBranch size={12} />
        </span>

        {/* 权重名称 */}
        <span className="flex-1 text-sm truncate">{node.display_name}</span>

        {/* 版本标签 */}
        <span className="ml-2 text-xs text-slate-500">v{node.version}</span>
      </div>

      {/* 子节点 */}
      {isExpanded && hasChildren && (
        <div>
          {node.children!.map((child) => (
            <TreeNode
              key={child.id}
              node={child}
              level={level + 1}
              selectedId={selectedId}
              onSelect={onSelect}
              expandedIds={expandedIds}
              onToggleExpand={onToggleExpand}
            />
          ))}
        </div>
      )}
    </div>
  );
};

const WeightTreeSelect: React.FC<WeightTreeSelectProps> = ({
  options,
  value,
  onChange,
  placeholder = '请选择预训练权重',
  disabled = false,
  className = ''
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [expandedIds, setExpandedIds] = useState<Set<number>>(new Set());
  const dropdownRef = useRef<HTMLDivElement>(null);

  // 获取选中的项
  const selectedOption = value
    ? findOptionById(options, value)
    : null;

  // 点击外部关闭下拉框
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleSelect = (id: number) => {
    onChange(id);
    setIsOpen(false);
  };

  const handleToggleExpand = (id: number) => {
    setExpandedIds((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(id)) {
        newSet.delete(id);
      } else {
        newSet.add(id);
      }
      return newSet;
    });
  };

  const handleClear = (e: React.MouseEvent) => {
    e.stopPropagation();
    onChange(null);
  };

  return (
    <div ref={dropdownRef} className={`relative ${className}`}>
      {/* 触发按钮 */}
      <div
        className={`flex items-center justify-between px-3 py-2 bg-slate-950 border rounded cursor-pointer transition-colors ${
          disabled
            ? 'border-slate-800 text-slate-600 cursor-not-allowed'
            : isOpen
            ? 'border-cyan-500 text-white'
            : 'border-slate-700 text-slate-300 hover:border-slate-600'
        }`}
        onClick={() => !disabled && setIsOpen(!isOpen)}
      >
        <div className="flex items-center flex-1 min-w-0">
          {selectedOption ? (
            <div className="flex items-center min-w-0">
              <GitBranch
                size={14}
                className={`mr-2 flex-shrink-0 ${
                  selectedOption.source_type === 'trained' ? 'text-purple-400' : 'text-emerald-400'
                }`}
              />
              <span className="text-sm truncate">{selectedOption.display_name}</span>
              <span className="ml-2 text-xs text-slate-500 flex-shrink-0">
                v{selectedOption.version}
              </span>
            </div>
          ) : (
            <span className="text-sm text-slate-500">{placeholder}</span>
          )}
        </div>

        <div className="flex items-center gap-1 ml-2">
          {selectedOption && !disabled && (
            <button
              className="p-1 hover:bg-slate-800 rounded transition-colors text-slate-400 hover:text-slate-200"
              onClick={handleClear}
            >
              <X size={12} />
            </button>
          )}
          <ChevronDown
            size={14}
            className={`transition-transform ${isOpen ? 'rotate-180' : ''} text-slate-400`}
          />
        </div>
      </div>

      {/* 下拉面板 */}
      {isOpen && !disabled && (
        <div className="absolute z-50 w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg shadow-xl max-h-64 overflow-y-auto">
          {options.length === 0 ? (
            <div className="py-4 text-center text-slate-500 text-sm">
              暂无可用权重
            </div>
          ) : (
            <div className="py-1">
              {/* 不使用选项 */}
              <div
                className={`flex items-center py-2 px-3 cursor-pointer transition-colors hover:bg-slate-800 ${
                  !value ? 'bg-cyan-900/30 text-cyan-400' : 'text-slate-300'
                }`}
                onClick={() => {
                  onChange(null);
                  setIsOpen(false);
                }}
              >
                <span className="w-4 mr-2" />
                <span className="flex-1 text-sm">不使用预训练权重</span>
              </div>

              {/* 权重树 */}
              {options.map((option) => (
                <TreeNode
                  key={option.id}
                  node={option}
                  level={0}
                  selectedId={value}
                  onSelect={handleSelect}
                  expandedIds={expandedIds}
                  onToggleExpand={handleToggleExpand}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// 递归查找选项
function findOptionById(
  options: WeightTreeSelectOption[],
  id: number
): WeightTreeSelectOption | null {
  for (const option of options) {
    if (option.id === id) {
      return option;
    }
    if (option.children) {
      const found = findOptionById(option.children, id);
      if (found) return found;
    }
  }
  return null;
}

export default WeightTreeSelect;
