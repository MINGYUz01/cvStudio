/**
 * 权重树形图可视化组件
 *
 * 用于展示权重版本关系的树形结构
 */

import React, { useState } from 'react';
import { ChevronRight, ChevronDown, Database, GitBranch, Clock } from 'lucide-react';
import { WeightTreeItem as WeightTreeItemType } from '../types';

interface TreeNodeProps {
  node: WeightTreeItemType;
  level?: number;
  selectedId?: number;
  onNodeClick?: (node: WeightTreeItemType) => void;
}

const TreeNode: React.FC<TreeNodeProps> = ({ node, level = 0, selectedId, onNodeClick }) => {
  const [expanded, setExpanded] = useState(level === 0);
  const hasChildren = node.children && node.children.length > 0;

  const handleClick = () => {
    if (hasChildren) {
      setExpanded(!expanded);
    }
    onNodeClick?.(node);
  };

  const handleExpandClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    setExpanded(!expanded);
  };

  return (
    <div className="tree-node">
      <div
        className={`tree-node-content ${node.id === selectedId ? 'selected' : ''}`}
        onClick={handleClick}
        style={{ paddingLeft: `${level * 16 + 8}px` }}
      >
        {hasChildren && (
          <button
            className="tree-expand-btn"
            onClick={handleExpandClick}
          >
            {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          </button>
        )}
        {!hasChildren && <span style={{ width: 20 }} />}
        <span className={`tree-node-icon ${node.source_type === 'trained' ? 'trained' : 'uploaded'}`}>
          {node.source_type === 'trained' ? <GitBranch size={14} /> : <Database size={14} />}
        </span>
        <span className="tree-node-name">{node.display_name}</span>
        <span className="tree-node-version">v{node.version}</span>
        {node.source_type === 'trained' && (
          <span className="tree-node-source">训练</span>
        )}
      </div>
      {expanded && hasChildren && (
        <div className="tree-node-children">
          {node.children.map((child) => (
            <TreeNode
              key={child.id}
              node={child}
              level={level + 1}
              selectedId={selectedId}
              onNodeClick={onNodeClick}
            />
          ))}
        </div>
      )}
    </div>
  );
};

interface WeightTreeViewProps {
  tree: WeightTreeItemType;
  onNodeClick?: (node: WeightTreeItemType) => void;
  selectedId?: number;
  showRootOnly?: boolean;
}

const WeightTreeView: React.FC<WeightTreeViewProps> = ({
  tree,
  onNodeClick,
  selectedId,
  showRootOnly = false
}) => {
  return (
    <div className="weight-tree-view">
      <TreeNode
        node={tree}
        selectedId={selectedId}
        onNodeClick={onNodeClick}
      />
    </div>
  );
};

export default WeightTreeView;
