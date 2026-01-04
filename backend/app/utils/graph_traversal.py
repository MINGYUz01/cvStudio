"""
图遍历和分析算法模块

本模块提供模型图的解析、验证、拓扑排序和依赖分析功能。
支持检测循环依赖、确定节点执行顺序等核心功能。

作者: CV Studio 开发团队
日期: 2025-12-25
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from collections import deque
from enum import Enum


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

    # 池化层
    "MaxPool2d",
    "AvgPool2d",
    "AdaptiveAvgPool2d",

    # 全连接层
    "Linear",

    # 归一化层
    "BatchNorm2d",
    "LayerNorm",

    # 激活函数
    "ReLU",
    "LeakyReLU",
    "SiLU",
    "Sigmoid",
    "Softmax",

    # Dropout
    "Dropout",

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
        for node_data in nodes_data:
            node = Node(
                id=node_data["id"],
                type=node_data["type"],
                params=node_data.get("data", {}),
                label=node_data.get("label", node_data["type"])
            )
            graph.add_node(node)

        # 解析边
        connections_data = graph_json.get("connections", [])
        for conn_data in connections_data:
            edge = Edge(
                source_id=conn_data["source"],
                target_id=conn_data["target"]
            )
            graph.add_edge(edge)

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

def validate_graph(graph: Graph) -> dict:
    """
    验证模型图的合法性

    检查项：
    1. 孤立节点检测（除了Input节点，所有节点都应该有连接）
    2. 连接完整性（源节点和目标节点必须存在）
    3. 节点类型验证（是否为支持的算子）
    4. 参数完整性检查（必需参数是否存在）
    5. 循环依赖检测

    参数:
        graph: 计算图

    返回:
        dict: 验证结果
            {
                "valid": bool,
                "errors": List[str],    # 错误列表（阻止保存）
                "warnings": List[str]   # 警告列表（允许保存）
            }
    """
    errors = []
    warnings = []

    # 1. 检查连接完整性
    for edge in graph.edges:
        if edge.source_id not in graph.nodes:
            errors.append(f"连接错误：边的源节点 '{edge.source_id}' 不存在")
        if edge.target_id not in graph.nodes:
            errors.append(f"连接错误：边的目标节点 '{edge.target_id}' 不存在")

    # 2. 检查节点类型
    for node_id, node in graph.nodes.items():
        if node.type not in SUPPORTED_LAYER_TYPES:
            errors.append(f"节点类型错误：节点 '{node_id}' 的类型 '{node.type}' 不支持")

    # 3. 检查孤立节点
    for node_id, node in graph.nodes.items():
        if node.type == "Input":
            continue

        predecessors = graph.get_predecessors(node_id)
        successors = graph.get_successors(node_id)

        if not predecessors and not successors:
            warnings.append(f"孤立节点：节点 '{node_id}' ({node.type}) 没有任何连接")
        elif not predecessors:
            warnings.append(f"异常连接：节点 '{node_id}' ({node.type}) 没有输入连接")
        elif not successors and node.type not in {"Concat", "Add"}:
            # Concat和Add可能是中间节点
            warnings.append(f"输出节点：节点 '{node_id}' ({node.type}) 没有输出连接")

    # 4. 检查参数完整性
    for node_id, node in graph.nodes.items():
        missing_params = check_required_params(node)
        if missing_params:
            errors.append(
                f"参数缺失：节点 '{node_id}' ({node.type}) "
                f"缺少必需参数: {', '.join(missing_params)}"
            )

    # 5. 检测循环依赖
    cycle = detect_cycles(graph)
    if cycle:
        cycle_str = " -> ".join(cycle)
        errors.append(f"循环依赖：检测到循环依赖路径 {cycle_str}")

    # 确定验证结果
    valid = len(errors) == 0

    return {
        "valid": valid,
        "errors": errors,
        "warnings": warnings
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
        "Linear": ["in", "out"],  # 或者 in_features, out_features
        "MaxPool2d": ["k"],  # kernel_size
        "AvgPool2d": ["k"],
        "BatchNorm2d": ["num_f"],  # num_features
        "Input": ["c", "h", "w"],  # channels, height, width
        "Upsample": ["scale_factor"],
        "Flatten": [],
        "ReLU": [],
        "Sigmoid": [],
        "Softmax": [],
        "Dropout": [],
        "Concat": [],
        "Add": [],
        "Identity": [],
    }

    required = required_params.get(node.type, [])
    missing = []

    for param in required:
        if param not in node.params or node.params[param] is None:
            missing.append(param)

    return missing


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
    # 解析图
    graph = GraphParser.parse_graph(graph_json)

    # 验证图
    validation = validate_graph(graph)

    # 如果验证失败，提前返回
    if not validation["valid"]:
        return {
            "graph": graph,
            "validation": validation,
            "execution_order": None
        }

    # 拓扑排序
    try:
        topo_order = topological_sort(graph)
    except ValueError as e:
        validation["errors"].append(str(e))
        validation["valid"] = False
        return {
            "graph": graph,
            "validation": validation,
            "execution_order": None
        }

    # 确定执行顺序
    execution_order = determine_execution_order(graph, topo_order)

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
