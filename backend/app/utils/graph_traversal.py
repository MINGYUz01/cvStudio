"""
图遍历和分析算法模块

本模块提供模型图的解析、验证、拓扑排序和依赖分析功能。
支持检测循环依赖、确定节点执行顺序等核心功能。

作者: CV Studio 开发团队
日期: 2025-12-25
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, TYPE_CHECKING
from collections import deque
from enum import Enum
from loguru import logger

# 延迟导入避免循环依赖
if TYPE_CHECKING:
    from app.utils.shape_inference import TensorShape, NodeShapeInfo


# ==============================
# 数据结构定义
# ==============================

@dataclass
class Node:
    """模型节点"""
    id: str
    type: str
    params: dict
    label: Optional[str] = None

    def __repr__(self):
        return f"Node(id={self.id}, type={self.type})"


@dataclass
class Edge:
    """有向边"""
    source_id: str
    target_id: str

    def __repr__(self):
        return f"Edge({self.source_id} -> {self.target_id})"


@dataclass
class Graph:
    """计算图"""
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)
    adjacency_list: Dict[str, List[str]] = field(default_factory=dict)

    def add_node(self, node: Node):
        """添加节点"""
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge):
        """添加边"""
        self.edges.append(edge)
        if edge.source_id not in self.adjacency_list:
            self.adjacency_list[edge.source_id] = []
        self.adjacency_list[edge.source_id].append(edge.target_id)

    def get_predecessors(self, node_id: str) -> List[str]:
        """获取前驱节点"""
        predecessors = []
        for source, targets in self.adjacency_list.items():
            if node_id in targets:
                predecessors.append(source)
        return predecessors

    def get_successors(self, node_id: str) -> List[str]:
        """获取后继节点"""
        return self.adjacency_list.get(node_id, [])

    def __repr__(self):
        return f"Graph(nodes={len(self.nodes)}, edges={len(self.edges)})"


class NodeColor(Enum):
    """节点颜色标记（用于DFS检测环）"""
    WHITE = 0  # 未访问
    GRAY = 1   # 访问中
    BLACK = 2  # 已完成


# ==============================
# 支持的算子类型
# ==============================

SUPPORTED_LAYER_TYPES = {
    # IO层
    "Input",

    # 卷积层
    "Conv2d",
    "ConvTranspose2d",

    # 池化层
    "MaxPool2d",
    "AvgPool2d",
    "AdaptiveAvgPool2d",
    "AdaptiveAvg",  # AdaptiveAvgPool2d的别名（前端使用）

    # 全连接层
    "Linear",

    # 归一化层
    "BatchNorm2d",
    "LayerNorm",
    "GroupNorm",
    "InstanceNorm2d",

    # 激活函数
    "ReLU",
    "ReLU6",
    "LeakyReLU",
    "SiLU",
    "Sigmoid",
    "Softmax",
    "Tanh",
    "GELU",

    # Dropout
    "Dropout",
    "DropPath",

    # 注意力机制
    "MultiheadAttention",

    # 其他操作
    "Flatten",
    "Upsample",
    "Concat",
    "Add",
    "Identity",
}


# ==============================
# 图解析器
# ==============================

class GraphParser:
    """图解析器 - 将前端JSON格式转换为内部图结构"""

    @staticmethod
    def parse_graph(graph_json: dict) -> Graph:
        """
        解析前端传来的模型图JSON数据

        参数:
            graph_json: 前端ModelBuilder的JSON格式
                {
                    "nodes": [{"id": "n1", "type": "Conv2d", "data": {...}}, ...],
                    "connections": [{"source": "n1", "target": "n2"}, ...]
                }

        返回:
            Graph: 标准化的图结构对象
        """
        graph = Graph()

        # 解析节点
        nodes_data = graph_json.get("nodes", [])
        logger.debug(f"解析图数据，节点数量: {len(nodes_data)}")
        for node_data in nodes_data:
            node = Node(
                id=node_data["id"],
                type=node_data["type"],
                params=node_data.get("data", {}),
                label=node_data.get("label", node_data["type"])
            )
            graph.add_node(node)
            logger.debug(f"  节点: {node.id} ({node.type}), 参数: {node.params}")

        # 解析边
        connections_data = graph_json.get("connections", [])
        logger.debug(f"连接数量: {len(connections_data)}")
        for conn_data in connections_data:
            edge = Edge(
                source_id=conn_data["source"],
                target_id=conn_data["target"]
            )
            graph.add_edge(edge)
            logger.debug(f"  连接: {edge.source_id} -> {edge.target_id}")

        return graph


# ==============================
# 拓扑排序算法
# ==============================

def topological_sort(graph: Graph) -> List[str]:
    """
    使用Kahn算法进行拓扑排序

    参数:
        graph: 计算图

    返回:
        List[str]: 按拓扑顺序排列的节点ID列表

    异常:
        ValueError: 如果图中存在环
    """
    # 计算每个节点的入度
    in_degree = {node_id: 0 for node_id in graph.nodes}
    for source, targets in graph.adjacency_list.items():
        for target in targets:
            in_degree[target] += 1

    # 将入度为0的节点加入队列
    queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
    topo_order = []

    while queue:
        node_id = queue.popleft()
        topo_order.append(node_id)

        # 更新后继节点的入度
        for successor in graph.get_successors(node_id):
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                queue.append(successor)

    # 检查是否存在环
    if len(topo_order) != len(graph.nodes):
        # 存在环，找出环中的节点
        cycle_nodes = [node_id for node_id in graph.nodes if node_id not in topo_order]
        raise ValueError(f"检测到循环依赖，涉及的节点: {cycle_nodes}")

    return topo_order


# ==============================
# 循环依赖检测
# ==============================

def detect_cycles(graph: Graph) -> Optional[List[str]]:
    """
    使用DFS和三色标记法检测循环依赖

    参数:
        graph: 计算图

    返回:
        Optional[List[str]]: 如果存在环，返回环的路径；否则返回None
    """
    color = {node_id: NodeColor.WHITE for node_id in graph.nodes}
    parent = {}

    def dfs(node_id: str) -> Optional[List[str]]:
        color[node_id] = NodeColor.GRAY

        for successor in graph.get_successors(node_id):
            if color[successor] == NodeColor.GRAY:
                # 找到环，构建环路径
                cycle = [successor]
                current = node_id
                while current != successor:
                    cycle.append(current)
                    current = parent.get(current, successor)
                cycle.append(successor)
                return cycle[::-1]
            elif color[successor] == NodeColor.WHITE:
                parent[successor] = node_id
                cycle = dfs(successor)
                if cycle:
                    return cycle

        color[node_id] = NodeColor.BLACK
        return None

    for node_id in graph.nodes:
        if color[node_id] == NodeColor.WHITE:
            cycle = dfs(node_id)
            if cycle:
                return cycle

    return None


# ==============================
# 执行顺序确定
# ==============================

def determine_execution_order(graph: Graph, topo_order: List[str]) -> dict:
    """
    综合拓扑排序结果，确定节点的前向和反向执行顺序

    参数:
        graph: 计算图
        topo_order: 拓扑排序结果

    返回:
        dict: 包含执行顺序的字典
            {
                "forward_order": [...],    # 前向传播顺序
                "backward_order": [...],   # 反向传播顺序
                "layers": [...],          # 实际的层节点（排除Input等）
                "input_nodes": [...],     # 输入节点
                "output_nodes": [...],    # 输出节点（无后继的节点）
                "depth": int              # 网络深度
            }
    """
    # 确定层节点（排除Input等IO节点）
    layers = [
        node_id for node_id in topo_order
        if graph.nodes[node_id].type not in {"Input", "Identity"}
    ]

    # 确定输入节点（无前驱的节点，通常是Input）
    input_nodes = [
        node_id for node_id in topo_order
        if not graph.get_predecessors(node_id)
    ]

    # 确定输出节点（无后继的节点）
    output_nodes = [
        node_id for node_id in topo_order
        if not graph.get_successors(node_id)
    ]

    # 反向顺序（前向顺序的逆序）
    backward_order = topo_order[::-1]

    # 计算网络深度（最长路径）
    depth = calculate_graph_depth(graph, topo_order)

    return {
        "forward_order": topo_order,
        "backward_order": backward_order,
        "layers": layers,
        "input_nodes": input_nodes,
        "output_nodes": output_nodes,
        "depth": depth
    }


def calculate_graph_depth(graph: Graph, topo_order: List[str]) -> int:
    """
    计算图的深度（最长路径的边数）

    参数:
        graph: 计算图
        topo_order: 拓扑排序结果

    返回:
        int: 图的深度
    """
    # 使用动态规划计算最长路径
    depth = {node_id: 0 for node_id in graph.nodes}

    for node_id in topo_order:
        for predecessor in graph.get_predecessors(node_id):
            depth[node_id] = max(depth[node_id], depth[predecessor] + 1)

    return max(depth.values()) if depth else 0


# ==============================
# 图验证器
# ==============================

def validate_graph(graph: Graph, enable_advanced_validation: bool = True) -> dict:
    """
    验证模型图的合法性

    检查项：
    1. 连接完整性（源节点和目标节点必须存在）
    2. 节点类型验证（是否为支持的算子）
    3. 孤立节点检测（除了Input节点，所有节点都应该有连接）
    4. 参数完整性检查（必需参数是否存在）
    5. 循环依赖检测
    6. 类型兼容性验证（新增）
    7. 参数值范围验证（新增）
    8. 形状一致性预验证（新增）

    参数:
        graph: 计算图
        enable_advanced_validation: 是否启用高级验证规则

    返回:
        dict: 验证结果
            {
                "valid": bool,
                "errors": List[str],    # 错误列表（阻止保存）
                "warnings": List[str],   # 警告列表（允许保存）
                "validation_details": dict  # 验证详情
            }
    """
    logger.debug(f"开始验证图，节点数: {len(graph.nodes)}, 边数: {len(graph.edges)}")

    errors = []
    warnings = []
    validation_details = {
        "connection_integrity": {"passed": True, "errors": []},
        "node_type_support": {"passed": True, "errors": []},
        "isolated_nodes": {"passed": True, "warnings": []},
        "parameter_completeness": {"passed": True, "errors": []},
        "cycle_detection": {"passed": True, "errors": []},
        "type_compatibility": {"passed": True, "errors": []},
        "parameter_ranges": {"passed": True, "errors": []},
        "shape_consistency": {"passed": True, "errors": []},
    }

    # 1. 检查连接完整性
    connection_errors = []
    for edge in graph.edges:
        if edge.source_id not in graph.nodes:
            connection_errors.append(f"边的源节点 '{edge.source_id}' 不存在")
        if edge.target_id not in graph.nodes:
            connection_errors.append(f"边的目标节点 '{edge.target_id}' 不存在")

    if connection_errors:
        logger.debug(f"连接完整性检查失败: {connection_errors}")
        validation_details["connection_integrity"]["passed"] = False
        validation_details["connection_integrity"]["errors"] = connection_errors
        errors.extend([f"连接错误：{e}" for e in connection_errors])

    # 2. 检查节点类型
    type_errors = []
    for node_id, node in graph.nodes.items():
        if node.type not in SUPPORTED_LAYER_TYPES:
            type_errors.append(f"节点 '{node_id}' 的类型 '{node.type}' 不支持")

    if type_errors:
        logger.debug(f"节点类型检查失败: {type_errors}")
        validation_details["node_type_support"]["passed"] = False
        validation_details["node_type_support"]["errors"] = type_errors
        errors.extend([f"节点类型错误：{e}" for e in type_errors])

    # 3. 检查孤立节点
    isolated_warnings = []
    for node_id, node in graph.nodes.items():
        if node.type == "Input":
            continue

        predecessors = graph.get_predecessors(node_id)
        successors = graph.get_successors(node_id)

        if not predecessors and not successors:
            isolated_warnings.append(f"节点 '{node_id}' ({node.type}) 没有任何连接")
        elif not predecessors:
            isolated_warnings.append(f"节点 '{node_id}' ({node.type}) 没有输入连接")
        elif not successors and node.type not in {"Concat", "Add"}:
            isolated_warnings.append(f"节点 '{node_id}' ({node.type}) 没有输出连接")

    if isolated_warnings:
        logger.debug(f"孤立节点警告: {isolated_warnings}")
        validation_details["isolated_nodes"]["passed"] = False
        validation_details["isolated_nodes"]["warnings"] = isolated_warnings
        warnings.extend([f"孤立节点：{w}" for w in isolated_warnings])

    # 4. 检查参数完整性
    param_errors = []
    for node_id, node in graph.nodes.items():
        missing_params = check_required_params(node)
        if missing_params:
            param_errors.append(
                f"节点 '{node_id}' ({node.type}) 缺少必需参数: {', '.join(missing_params)}"
            )

    if param_errors:
        logger.debug(f"参数完整性检查失败: {param_errors}")
        validation_details["parameter_completeness"]["passed"] = False
        validation_details["parameter_completeness"]["errors"] = param_errors
        errors.extend([f"参数缺失：{e}" for e in param_errors])

    # 5. 检测循环依赖
    cycle = detect_cycles(graph)
    if cycle:
        cycle_str = " -> ".join(cycle)
        cycle_error = f"检测到循环依赖路径 {cycle_str}"
        logger.debug(f"循环依赖检测失败: {cycle_error}")
        validation_details["cycle_detection"]["passed"] = False
        validation_details["cycle_detection"]["errors"] = [cycle_error]
        errors.append(f"循环依赖：{cycle_error}")

    # 6. 高级验证规则
    if enable_advanced_validation:
        # 类型兼容性验证
        type_compat_errors = validate_type_compatibility(graph)
        if type_compat_errors:
            logger.debug(f"类型兼容性检查失败: {type_compat_errors}")
            validation_details["type_compatibility"]["passed"] = False
            validation_details["type_compatibility"]["errors"] = type_compat_errors
            errors.extend(type_compat_errors)

        # 参数值范围验证
        param_range_errors = validate_parameter_ranges(graph)
        if param_range_errors:
            logger.debug(f"参数范围检查失败: {param_range_errors}")
            validation_details["parameter_ranges"]["passed"] = False
            validation_details["parameter_ranges"]["errors"] = param_range_errors
            errors.extend(param_range_errors)

        # 形状一致性预验证
        shape_consistency_errors = validate_shape_consistency(graph)
        if shape_consistency_errors:
            logger.debug(f"形状一致性检查失败: {shape_consistency_errors}")
            validation_details["shape_consistency"]["passed"] = False
            validation_details["shape_consistency"]["errors"] = shape_consistency_errors
            errors.extend(shape_consistency_errors)

    # 确定验证结果
    valid = len(errors) == 0
    logger.debug(f"验证完成，valid={valid}, errors={len(errors)}, warnings={len(warnings)}")

    return {
        "valid": valid,
        "errors": errors,
        "warnings": warnings,
        "validation_details": validation_details
    }


def check_required_params(node: Node) -> List[str]:
    """
    检查节点的必需参数是否存在

    参数:
        node: 节点对象

    返回:
        List[str]: 缺失的必需参数列表
    """
    required_params = {
        "Conv2d": ["in", "out"],
        "ConvTranspose2d": ["in", "out"],
        # Linear支持两种参数命名方式
        "Linear": ["in", "out"],  # 或者 in_features, out_features
        "MaxPool2d": ["k"],  # kernel_size
        "AvgPool2d": ["k"],
        "BatchNorm2d": ["num_f"],  # num_features
        "GroupNorm": ["num_groups", "num_channels"],
        "InstanceNorm2d": ["num_features"],
        "LayerNorm": ["normalized_size"],
        "Input": ["c", "h", "w"],  # channels, height, width
        "Upsample": ["scale_factor"],
        "Flatten": [],
        "ReLU": [],
        "LeakyReLU": [],
        "SiLU": [],
        "Sigmoid": [],
        "Softmax": [],
        "Dropout": [],
        "Concat": [],
        "Add": [],
        "Identity": [],
        "MultiheadAttention": ["embed_dim", "num_heads"],
    }

    required = required_params.get(node.type, [])
    missing = []

    # 特殊处理Linear层，支持多种参数命名方式
    if node.type == "Linear":
        has_in = "in" in node.params and node.params["in"] is not None
        has_out = "out" in node.params and node.params["out"] is not None
        has_in_features = "in_features" in node.params and node.params["in_features"] is not None
        has_out_features = "out_features" in node.params and node.params["out_features"] is not None
        # 前端使用的简短命名方式
        has_in_f = "in_f" in node.params and node.params["in_f"] is not None
        has_out_f = "out_f" in node.params and node.params["out_f"] is not None

        if not (has_in or has_in_features or has_in_f):
            missing.append("in")
        if not (has_out or has_out_features or has_out_f):
            missing.append("out")
    else:
        for param in required:
            if param not in node.params or node.params[param] is None:
                missing.append(param)

    return missing


# ==============================
# 增强的图验证规则
# ==============================

def validate_type_compatibility(graph: Graph, shape_map: Dict[str, 'NodeShapeInfo'] = None) -> List[str]:
    """
    验证相邻节点之间的类型兼容性

    检查规则:
    1. Linear层前必须有Flatten或1D张量输入
    2. 2D卷积/池化/归一化层不能接Flatten后的1D张量
    3. Concat/Add节点至少需要2个输入
    4. 新增：支持基于形状的智能验证，允许 Linear -> ReLU -> Linear

    参数:
        graph: 计算图
        shape_map: 形状映射（可选），用于智能验证

    返回:
        List[str]: 错误列表
    """
    errors = []

    # 需要2D/3D/4D张量输入的层类型
    requires_spatial_input = {
        "Conv2d", "ConvTranspose2d", "MaxPool2d", "AvgPool2d",
        "AdaptiveAvgPool2d", "BatchNorm2d", "InstanceNorm2d", "GroupNorm"
    }

    # 可以接1D张量的层类型（可以接在Linear后面）
    can_follow_1d = {
        "Linear", "Flatten", "Dropout", "DropPath", "LayerNorm",
        "ReLU", "LeakyReLU", "SiLU", "Sigmoid", "Softmax", "Identity",
        "GELU", "Tanh", "ReLU6", "Hardswish"
    }

    for node_id, node in graph.nodes.items():
        predecessors = graph.get_predecessors(node_id)
        successors = graph.get_successors(node_id)

        # Concat至少需要2个输入
        if node.type == "Concat" and len(predecessors) < 2:
            errors.append(f"{node_id}: Concat至少需要2个输入，当前{len(predecessors)}个")

        # Add至少需要2个输入
        if node.type == "Add" and len(predecessors) < 2:
            errors.append(f"{node_id}: Add至少需要2个输入，当前{len(predecessors)}个")

        # Linear层智能验证 - 基于形状或前置节点类型
        if node.type == "Linear":
            for pred_id in predecessors:
                pred_node = graph.nodes[pred_id]

                # 1. 如果有形状信息，检查前置节点的输出形状
                if shape_map and pred_id in shape_map:
                    pred_shape_info = shape_map[pred_id]
                    # NodeShapeInfo.output_shape 是 TensorShape 对象
                    pred_shape = pred_shape_info.output_shape
                    # 如果前置节点的输出已经是1D（features存在），则可以连接
                    if pred_shape.features is not None:
                        continue  # 形状已经是1D，验证通过

                # 2. 检查前置节点的前置节点，看是否有Linear/Flatten
                # 这支持 Linear -> ReLU -> Linear 的模式
                pred_preds = graph.get_predecessors(pred_id)
                has_flatten_or_linear_before = any(
                    graph.nodes[p].type in ["Linear", "Flatten", "AdaptiveAvgPool2d", "AdaptiveAvg"]
                    for p in pred_preds
                )

                # 3. 如果激活函数前面有Linear/Flatten，或者是其他允许的类型，则合法
                if pred_node.type in can_follow_1d and has_flatten_or_linear_before:
                    continue

                # 4. 传统的类型检查（当以上条件都不满足时）
                if pred_node.type not in ["Flatten", "Linear", "AdaptiveAvgPool2d",
                                           "AdaptiveAvg", "LayerNorm", "Dropout",
                                           "DropPath", "Input"]:
                    errors.append(
                        f"{node_id}: Linear层应接Flatten或1D张量，当前接了{pred_node.type}。"
                        f"建议在Linear前添加Flatten层。"
                    )

        # 2D/3D操作不能接Flatten后的1D张量
        if node.type in requires_spatial_input:
            for pred_id in predecessors:
                pred_node = graph.nodes[pred_id]
                if pred_node.type == "Flatten":
                    errors.append(
                        f"{node_id}: {node.type}需要2D/3D张量输入，"
                        f"不能接Flatten后的1D张量"
                    )

        # MultiheadAttention需要特定输入
        if node.type == "MultiheadAttention":
            if len(predecessors) < 1:
                errors.append(f"{node_id}: MultiheadAttention至少需要1个输入（query）")

    return errors


def validate_parameter_ranges(graph: Graph) -> List[str]:
    """
    验证参数值在合理范围内

    检查规则:
    1. kernel_size > 0
    2. stride > 0
    3. 0 <= dropout_prob <= 1
    4. channels/features > 0
    5. num_groups必须能整除num_channels (GroupNorm)

    参数:
        graph: 计算图

    返回:
        List[str]: 错误列表
    """
    errors = []

    # 参数范围规则: (层类型, 参数名) -> (最小值, 最大值, 是否包含端点)
    range_rules = {
        ("Conv2d", "k"): (1, 11, True),
        ("Conv2d", "s"): (1, 5, True),
        ("Conv2d", "in"): (1, 10000, True),
        ("Conv2d", "out"): (1, 10000, True),
        ("Conv2d", "p"): (0, 10, True),

        ("ConvTranspose2d", "k"): (1, 11, True),
        ("ConvTranspose2d", "s"): (1, 5, True),
        ("ConvTranspose2d", "in"): (1, 10000, True),
        ("ConvTranspose2d", "out"): (1, 10000, True),
        ("ConvTranspose2d", "p"): (0, 10, True),

        ("Linear", "in"): (1, 100000, True),
        ("Linear", "out"): (1, 100000, True),

        ("MaxPool2d", "k"): (1, 10, True),
        ("MaxPool2d", "s"): (1, 5, True),
        ("AvgPool2d", "k"): (1, 10, True),
        ("AvgPool2d", "s"): (1, 5, True),

        ("BatchNorm2d", "num_f"): (1, 10000, True),
        ("InstanceNorm2d", "num_features"): (1, 10000, True),

        ("GroupNorm", "num_groups"): (1, 10000, True),
        ("GroupNorm", "num_channels"): (1, 10000, True),

        ("Dropout", "p"): (0, 1, True),

        ("Input", "c"): (1, 10000, True),
        ("Input", "h"): (1, 10000, True),
        ("Input", "w"): (1, 10000, True),

        ("MultiheadAttention", "embed_dim"): (1, 10000, True),
        ("MultiheadAttention", "num_heads"): (1, 100, True),
    }

    for node_id, node in graph.nodes.items():
        for (layer_type, param_name), (min_val, max_val, inclusive) in range_rules.items():
            # 首先检查节点类型是否匹配（修复：避免跨层类型的参数误判）
            if node.type != layer_type:
                continue

            # Linear层特殊处理：支持in/out、in_features/out_features、in_f/out_f
            if layer_type == "Linear" and param_name == "in":
                value = node.params.get("in") or node.params.get("in_features") or node.params.get("in_f")
            elif layer_type == "Linear" and param_name == "out":
                value = node.params.get("out") or node.params.get("out_features") or node.params.get("out_f")
            elif param_name in node.params:
                value = node.params[param_name]
            else:
                continue

            if value is None:
                continue

            # 检查是否为数值类型
            if not isinstance(value, (int, float)):
                errors.append(
                    f"{node_id}: 参数{param_name}应为数值类型，当前为{type(value).__name__}"
                )
                continue

            # 检查范围
            if inclusive:
                if value < min_val or value > max_val:
                    errors.append(
                        f"{node_id}: 参数{param_name}={value}超出范围[{min_val}, {max_val}]"
                    )
            else:
                if value <= min_val or value >= max_val:
                    errors.append(
                        f"{node_id}: 参数{param_name}={value}超出范围({min_val}, {max_val})"
                    )

        # GroupNorm特殊检查: num_groups必须能整除num_channels
        if node.type == "GroupNorm":
            num_groups = node.params.get("num_groups")
            num_channels = node.params.get("num_channels")
            if num_groups and num_channels and num_channels % num_groups != 0:
                errors.append(
                    f"{node_id}: GroupNorm的num_groups({num_groups})必须能整除num_channels({num_channels})"
                )

        # MultiheadAttention特殊检查: embed_dim必须能被num_heads整除
        if node.type == "MultiheadAttention":
            embed_dim = node.params.get("embed_dim")
            num_heads = node.params.get("num_heads")
            if embed_dim and num_heads and embed_dim % num_heads != 0:
                errors.append(
                    f"{node_id}: MultiheadAttention的embed_dim({embed_dim})必须能被num_heads({num_heads})整除"
                )

    return errors


def validate_shape_consistency(graph: Graph) -> List[str]:
    """
    在形状推断前进行快速形状一致性检查

    检查规则:
    1. Input节点必须指定c, h, w
    2. 多输入节点（Concat/Add）的输入数量检查
    3. 检测明显的形状冲突（通过参数推断）

    参数:
        graph: 计算图

    返回:
        List[str]: 错误列表
    """
    errors = []

    for node_id, node in graph.nodes.items():
        # Input节点必须指定完整形状
        if node.type == "Input":
            required = ["c", "h", "w"]
            for r in required:
                if r not in node.params or node.params[r] is None:
                    errors.append(f"{node_id}: Input节点缺少必需参数{r}")
                elif not isinstance(node.params[r], (int, float)) or node.params[r] <= 0:
                    errors.append(f"{node_id}: Input节点参数{r}必须为正数")

        # Concat至少需要2个输入
        if node.type == "Concat":
            predecessors = graph.get_predecessors(node_id)
            if len(predecessors) < 2:
                errors.append(f"{node_id}: Concat至少需要2个输入，当前{len(predecessors)}个")

        # Add至少需要2个输入
        if node.type == "Add":
            predecessors = graph.get_predecessors(node_id)
            if len(predecessors) < 2:
                errors.append(f"{node_id}: Add至少需要2个输入，当前{len(predecessors)}个")

    return errors


# ==============================
# 辅助函数
# ==============================

def print_graph_info(graph: Graph):
    """打印图的详细信息（用于调试）"""
    print(f"\n=== 图信息 ===")
    print(f"节点数量: {len(graph.nodes)}")
    print(f"边数量: {len(graph.edges)}")

    print(f"\n节点列表:")
    for node_id, node in graph.nodes.items():
        predecessors = graph.get_predecessors(node_id)
        successors = graph.get_successors(node_id)
        print(f"  {node_id} ({node.type}):")
        print(f"    前驱: {predecessors if predecessors else '无'}")
        print(f"    后继: {successors if successors else '无'}")
        print(f"    参数: {node.params}")

    print(f"\n边列表:")
    for edge in graph.edges:
        print(f"  {edge.source_id} -> {edge.target_id}")


def analyze_graph_structure(graph_json: dict) -> dict:
    """
    分析图结构的入口函数

    新增：集成形状推断，支持基于形状的智能验证

    参数:
        graph_json: 前端传来的模型图JSON

    返回:
        dict: 分析结果
            {
                "graph": Graph对象,
                "validation": 验证结果,
                "execution_order": 执行顺序
            }
    """
    # 延迟导入避免循环依赖
    from app.utils.shape_inference import infer_shapes_from_graph

    # 解析图
    graph = GraphParser.parse_graph(graph_json)

    # 第一步：基础验证（不包括类型兼容性验证，因为需要形状信息）
    logger.debug("开始基础验证（不包括类型兼容性）")
    validation = validate_graph(graph, enable_advanced_validation=False)

    # 如果基础验证失败，提前返回
    if not validation["valid"]:
        return {
            "graph": graph,
            "validation": validation,
            "execution_order": None
        }

    # 第二步：拓扑排序
    try:
        topo_order = topological_sort(graph)
    except ValueError as e:
        validation["errors"].append(f"拓扑排序失败: {str(e)}")
        validation["valid"] = False
        return {
            "graph": graph,
            "validation": validation,
            "execution_order": None
        }

    # 确定执行顺序
    execution_order = determine_execution_order(graph, topo_order)

    # 第三步：形状推断（用于智能类型兼容性验证）
    logger.debug("进行形状推断")
    try:
        shape_result = infer_shapes_from_graph(graph, execution_order)
        shape_map = shape_result.get("shape_map", {})
        logger.debug(f"形状推断完成，推断出 {len(shape_map)} 个节点的形状")
    except Exception as e:
        logger.warning(f"形状推断失败: {e}，将使用传统验证规则")
        shape_map = {}

    # 第四步：使用形状信息进行类型兼容性验证
    logger.debug("进行智能类型兼容性验证")
    type_compat_errors = validate_type_compatibility(graph, shape_map)
    if type_compat_errors:
        validation["validation_details"]["type_compatibility"] = {
            "passed": False,
            "errors": type_compat_errors
        }
        validation["errors"].extend(type_compat_errors)
        validation["valid"] = False

    return {
        "graph": graph,
        "validation": validation,
        "execution_order": execution_order
    }


# ==============================
# 主程序入口（用于测试）
# ==============================

if __name__ == "__main__":
    # 测试用例
    test_graph = {
        "nodes": [
            {"id": "n1", "type": "Input", "data": {"c": 3, "h": 640, "w": 640}},
            {"id": "n2", "type": "Conv2d", "data": {"in": 3, "out": 16, "k": 3, "s": 1, "p": 1}},
            {"id": "n3", "type": "ReLU", "data": {}},
            {"id": "n4", "type": "MaxPool2d", "data": {"k": 2, "s": 2}},
        ],
        "connections": [
            {"source": "n1", "target": "n2"},
            {"source": "n2", "target": "n3"},
            {"source": "n3", "target": "n4"},
        ]
    }

    # 分析图
    result = analyze_graph_structure(test_graph)

    # 打印结果
    print_graph_info(result["graph"])

    print(f"\n=== 验证结果 ===")
    print(f"有效: {result['validation']['valid']}")
    print(f"错误: {result['validation']['errors']}")
    print(f"警告: {result['validation']['warnings']}")

    if result["execution_order"]:
        print(f"\n=== 执行顺序 ===")
        print(f"前向: {result['execution_order']['forward_order']}")
        print(f"反向: {result['execution_order']['backward_order']}")
        print(f"层节点: {result['execution_order']['layers']}")
        print(f"输入节点: {result['execution_order']['input_nodes']}")
        print(f"输出节点: {result['execution_order']['output_nodes']}")
        print(f"深度: {result['execution_order']['depth']}")
