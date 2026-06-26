"""生成和加载分布可控的合成图分类数据集。"""
from __future__ import annotations

import json
import re
import zlib
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.data import Data


# 合成数据集名称：旧名称保留为二分类，多分类扩展使用 SYN_ER_C2 这类稳定标识。
SYNTHETIC_BASE_DATASETS: List[str] = ["SYN_ER", "SYN_BA", "SYN_SBM", "SYN_WS", "SYN_REGULAR"]
SYNTHETIC_CLASS_COUNTS: List[int] = list(range(2, 9))
SYNTHETIC_DATASETS: List[str] = list(SYNTHETIC_BASE_DATASETS)
SYNTHETIC_MULTICLASS_DATASETS: List[str] = [
    f"{dataset_name}_C{num_classes}"
    for dataset_name in SYNTHETIC_BASE_DATASETS
    for num_classes in SYNTHETIC_CLASS_COUNTS
]

# 默认合成数据根目录：由 src.training 传入项目 data 目录下的子路径。
DEFAULT_PROFILE = "paper"


def stable_seed(*parts: object) -> int:
    """根据输入字符串生成跨进程稳定的随机种子。"""
    text = "::".join(str(part) for part in parts)
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


def resolve_synthetic_dataset(dataset_name: str) -> Tuple[str, int]:
    """把合成数据集名称解析为基础分布名称和类别数。"""
    if dataset_name in SYNTHETIC_BASE_DATASETS:
        return dataset_name, 2
    match = re.fullmatch(r"(SYN_(?:ER|BA|SBM|WS|REGULAR))_C([2-8])", dataset_name)
    if match is None:
        raise ValueError(f"Unsupported synthetic dataset: {dataset_name}")
    return match.group(1), int(match.group(2))


def class_fraction(label: int, num_classes: int) -> float:
    """把类别编号映射到 0 到 1 的连续位置，用于多分类结构参数插值。"""
    if num_classes <= 1:
        return 0.0
    return float(label) / float(num_classes - 1)


def load_torch_payload(path: Path) -> Dict[str, object]:
    """兼容不同 PyTorch 版本加载包含 PyG Data 对象的缓存文件。"""
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def normalize_edges(edges: Iterable[Tuple[int, int]], num_nodes: int) -> List[Tuple[int, int]]:
    """清理自环和越界边，并把无向边统一为有序二元组。"""
    normalized = set()
    for source, target in edges:
        if source == target:
            continue
        if not (0 <= source < num_nodes and 0 <= target < num_nodes):
            continue
        left, right = sorted((int(source), int(target)))
        normalized.add((left, right))
    return sorted(normalized)


def ensure_non_isolated(edges: Sequence[Tuple[int, int]], num_nodes: int, rng: np.random.Generator) -> List[Tuple[int, int]]:
    """为孤立节点补一条随机边，避免极小图在消息传递中退化过重。"""
    edge_set = set(normalize_edges(edges, num_nodes))
    degrees = np.zeros(num_nodes, dtype=int)
    for source, target in edge_set:
        degrees[source] += 1
        degrees[target] += 1
    for node in range(num_nodes):
        if degrees[node] > 0:
            continue
        candidate = int(rng.integers(0, max(num_nodes - 1, 1)))
        if candidate >= node:
            candidate += 1
        if candidate < num_nodes:
            edge_set.add(tuple(sorted((node, candidate))))
    return sorted(edge_set)


def edge_index_from_edges(edges: Sequence[Tuple[int, int]], num_nodes: int) -> torch.Tensor:
    """把无向边列表转换为 PyG 使用的双向 edge_index。"""
    directed_edges: List[Tuple[int, int]] = []
    for source, target in normalize_edges(edges, num_nodes):
        directed_edges.append((source, target))
        directed_edges.append((target, source))
    if not directed_edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(directed_edges, dtype=torch.long).t().contiguous()


def degree_vector(edges: Sequence[Tuple[int, int]], num_nodes: int) -> np.ndarray:
    """根据无向边列表计算每个节点的度。"""
    degrees = np.zeros(num_nodes, dtype=np.float32)
    for source, target in normalize_edges(edges, num_nodes):
        degrees[source] += 1.0
        degrees[target] += 1.0
    return degrees


def build_node_features(
    edges: Sequence[Tuple[int, int]],
    num_nodes: int,
    feature_dim: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    """构造不直接泄露标签的节点特征，包括归一化度和随机噪声维度。"""
    feature_dim = max(1, int(feature_dim))
    features = rng.normal(loc=0.0, scale=1.0, size=(num_nodes, feature_dim)).astype(np.float32)
    degrees = degree_vector(edges, num_nodes)
    features[:, 0] = degrees / max(num_nodes - 1, 1)
    if feature_dim > 1:
        features[:, 1] = rng.binomial(1, 0.5, size=num_nodes).astype(np.float32)
    return torch.tensor(features, dtype=torch.float32)


def sample_num_nodes(rng: np.random.Generator, min_nodes: int, max_nodes: int) -> int:
    """在指定节点数范围内采样单个图的节点数。"""
    return int(rng.integers(int(min_nodes), int(max_nodes) + 1))


def generate_er_edges(
    num_nodes: int,
    label: int,
    num_classes: int,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    """生成 ER 随机图，类别对应从低到高的 Bernoulli 边概率。"""
    probability = 0.04 + 0.18 * class_fraction(label, num_classes)
    edges = [
        (i, j)
        for i in range(num_nodes)
        for j in range(i + 1, num_nodes)
        if rng.random() < probability
    ]
    return ensure_non_isolated(edges, num_nodes, rng)


def generate_ba_edges(
    num_nodes: int,
    label: int,
    num_classes: int,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    """生成 BA 无标度图，类别对应逐步增大的新节点连接数和长尾强度。"""
    attachment_count = 1 + int(round(7 * class_fraction(label, num_classes)))
    attachment_count = max(1, min(attachment_count, num_nodes - 1))
    initial_nodes = min(num_nodes, attachment_count + 2)
    edges = {(i, j) for i in range(initial_nodes) for j in range(i + 1, initial_nodes)}
    degrees = np.ones(num_nodes, dtype=np.float64)
    for source, target in edges:
        degrees[source] += 1.0
        degrees[target] += 1.0
    for node in range(initial_nodes, num_nodes):
        candidates = np.arange(node)
        weights = degrees[:node] / degrees[:node].sum()
        chosen = rng.choice(candidates, size=min(attachment_count, node), replace=False, p=weights)
        for target in chosen:
            edges.add(tuple(sorted((node, int(target)))))
            degrees[node] += 1.0
            degrees[int(target)] += 1.0
    return sorted(edges)


def generate_sbm_edges(
    num_nodes: int,
    label: int,
    num_classes: int,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    """生成 SBM 社区图，类别对应由弱到强的社区分离程度。"""
    split = num_nodes // 2
    communities = np.zeros(num_nodes, dtype=int)
    communities[split:] = 1
    fraction = class_fraction(label, num_classes)
    p_in = 0.08 + 0.20 * fraction
    p_out = 0.06 - 0.05 * fraction
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            probability = p_in if communities[i] == communities[j] else p_out
            if rng.random() < probability:
                edges.append((i, j))
    return ensure_non_isolated(edges, num_nodes, rng)


def ring_lattice_edges(num_nodes: int, degree: int) -> List[Tuple[int, int]]:
    """生成偶数度环形格点图边列表。"""
    degree = max(2, min(int(degree), num_nodes - 1))
    if degree % 2 == 1:
        degree -= 1
    edges = set()
    for node in range(num_nodes):
        for step in range(1, degree // 2 + 1):
            edges.add(tuple(sorted((node, (node + step) % num_nodes))))
    return sorted(edges)


def generate_ws_edges(
    num_nodes: int,
    label: int,
    num_classes: int,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    """生成 WS 小世界图，类别对应从低到高的随机重连概率。"""
    degree = 4 if num_nodes > 6 else 2
    rewire_probability = 0.02 + 0.43 * class_fraction(label, num_classes)
    edges = set(ring_lattice_edges(num_nodes, degree))
    for source, target in list(edges):
        if rng.random() >= rewire_probability:
            continue
        edges.discard((source, target))
        forbidden = {source}
        forbidden.update(neighbor for edge in edges for neighbor in edge if source in edge)
        candidates = [node for node in range(num_nodes) if node not in forbidden]
        if not candidates:
            edges.add((source, target))
            continue
        new_target = int(rng.choice(candidates))
        edges.add(tuple(sorted((source, new_target))))
    return ensure_non_isolated(sorted(edges), num_nodes, rng)


def generate_regular_edges(
    num_nodes: int,
    label: int,
    num_classes: int,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    """生成近似随机正则图，类别对应逐步增大的均匀固定度。"""
    degree = 2 + 2 * int(round(7 * class_fraction(label, num_classes)))
    degree = min(degree, num_nodes - 1)
    if degree % 2 == 1:
        degree -= 1
    edges = ring_lattice_edges(num_nodes, max(2, degree))
    permutation = rng.permutation(num_nodes)
    permuted = [(int(permutation[source]), int(permutation[target])) for source, target in edges]
    return normalize_edges(permuted, num_nodes)


def generate_edges(
    dataset_name: str,
    num_nodes: int,
    label: int,
    num_classes: int,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    """按数据集名称分发到对应图结构生成器。"""
    if dataset_name == "SYN_ER":
        return generate_er_edges(num_nodes, label, num_classes, rng)
    if dataset_name == "SYN_BA":
        return generate_ba_edges(num_nodes, label, num_classes, rng)
    if dataset_name == "SYN_SBM":
        return generate_sbm_edges(num_nodes, label, num_classes, rng)
    if dataset_name == "SYN_WS":
        return generate_ws_edges(num_nodes, label, num_classes, rng)
    if dataset_name == "SYN_REGULAR":
        return generate_regular_edges(num_nodes, label, num_classes, rng)
    raise ValueError(f"Unsupported synthetic dataset: {dataset_name}")


def dataset_description(dataset_name: str, num_classes: int) -> Dict[str, object]:
    """返回合成数据集的结构分布说明和标签规则。"""
    descriptions = {
        "SYN_ER": {
            "graph_distribution": "Erdos-Renyi G(n,p)",
            "label_rule": f"{num_classes} classes use monotonically increasing Bernoulli edge probabilities",
        },
        "SYN_BA": {
            "graph_distribution": "Barabasi-Albert scale-free graph",
            "label_rule": f"{num_classes} classes use monotonically increasing attachment counts",
        },
        "SYN_SBM": {
            "graph_distribution": "two-block stochastic block model",
            "label_rule": f"{num_classes} classes use monotonically increasing community separation",
        },
        "SYN_WS": {
            "graph_distribution": "Watts-Strogatz small-world graph",
            "label_rule": f"{num_classes} classes use monotonically increasing rewiring probabilities",
        },
        "SYN_REGULAR": {
            "graph_distribution": "randomly permuted regular ring lattice",
            "label_rule": f"{num_classes} classes use monotonically increasing fixed degrees",
        },
    }
    return descriptions[dataset_name]


def generate_synthetic_graphs(
    dataset_name: str,
    graphs_per_class: int,
    min_nodes: int,
    max_nodes: int,
    feature_dim: int,
    seed: int,
) -> Tuple[List[Data], Dict[str, object]]:
    """生成一个平衡多分类合成图数据集及其元信息。"""
    base_dataset_name, num_classes = resolve_synthetic_dataset(dataset_name)
    rng = np.random.default_rng(stable_seed(dataset_name, seed))
    graphs: List[Data] = []
    for label in range(num_classes):
        for _ in range(int(graphs_per_class)):
            num_nodes = sample_num_nodes(rng, min_nodes, max_nodes)
            edges = generate_edges(base_dataset_name, num_nodes, label, num_classes, rng)
            graphs.append(
                Data(
                    x=build_node_features(edges, num_nodes, feature_dim, rng),
                    edge_index=edge_index_from_edges(edges, num_nodes),
                    y=torch.tensor([label], dtype=torch.long),
                )
            )
    order = rng.permutation(len(graphs))
    shuffled_graphs = [graphs[int(index)] for index in order]
    meta = {
        "dataset": dataset_name,
        "base_dataset": base_dataset_name,
        "num_graphs": len(shuffled_graphs),
        "num_classes": int(num_classes),
        "num_features": int(feature_dim),
        "graphs_per_class": int(graphs_per_class),
        "min_nodes": int(min_nodes),
        "max_nodes": int(max_nodes),
        "seed": int(seed),
        **dataset_description(base_dataset_name, num_classes),
    }
    return shuffled_graphs, meta


def synthetic_profile_defaults(profile: str) -> Dict[str, int]:
    """根据 profile 返回默认图数量、节点范围和特征维度。"""
    if profile == "smoke":
        return {"graphs_per_class": 12, "min_nodes": 16, "max_nodes": 28, "feature_dim": 8, "seed": 20260527}
    return {"graphs_per_class": 200, "min_nodes": 30, "max_nodes": 80, "feature_dim": 8, "seed": 20260527}


def dataset_cache_path(root: Path, dataset_name: str, profile: str) -> Path:
    """构造指定合成数据集和 profile 的缓存文件路径。"""
    return root / dataset_name / f"{profile}.pt"


def write_synthetic_dataset(
    root: Path,
    dataset_name: str,
    profile: str = DEFAULT_PROFILE,
    graphs_per_class: int | None = None,
    min_nodes: int | None = None,
    max_nodes: int | None = None,
    feature_dim: int | None = None,
    seed: int | None = None,
    force: bool = False,
) -> Path:
    """生成并缓存一个合成数据集，已存在时默认复用。"""
    defaults = synthetic_profile_defaults(profile)
    config = {
        "graphs_per_class": graphs_per_class if graphs_per_class is not None else defaults["graphs_per_class"],
        "min_nodes": min_nodes if min_nodes is not None else defaults["min_nodes"],
        "max_nodes": max_nodes if max_nodes is not None else defaults["max_nodes"],
        "feature_dim": feature_dim if feature_dim is not None else defaults["feature_dim"],
        "seed": seed if seed is not None else defaults["seed"],
    }
    path = dataset_cache_path(root, dataset_name, profile)
    if path.exists() and not force:
        return path
    graphs, meta = generate_synthetic_graphs(dataset_name=dataset_name, **config)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"graphs": graphs, "meta": {"profile": profile, **meta}}, path)
    path.with_suffix(".json").write_text(json.dumps({"profile": profile, **meta}, indent=2), encoding="utf-8")
    return path


class SyntheticGraphDataset:
    """轻量级 PyG 图数据集包装器，提供训练代码需要的 num_classes 和 num_features。"""

    def __init__(self, root: Path, dataset_name: str, profile: str = DEFAULT_PROFILE):
        """从缓存加载合成图数据；缓存不存在时使用 profile 默认配置生成。"""
        self.root = root
        self.dataset_name = dataset_name
        self.profile = profile
        path = write_synthetic_dataset(root=root, dataset_name=dataset_name, profile=profile)
        payload = load_torch_payload(path)
        self.graphs: List[Data] = list(payload["graphs"])
        self.meta: Dict[str, object] = dict(payload["meta"])
        self.num_classes = int(self.meta.get("num_classes", 2))
        self.num_features = int(self.meta.get("num_features", self.graphs[0].x.size(1) if self.graphs else 1))

    def __len__(self) -> int:
        """返回数据集图样本数量。"""
        return len(self.graphs)

    def __iter__(self) -> Iterator[Data]:
        """迭代返回图样本。"""
        return iter(self.graphs)

    def __getitem__(self, index: int) -> Data:
        """按索引返回单个图样本。"""
        return self.graphs[index]

    def shuffle(self) -> "SyntheticGraphDataset":
        """返回确定性打乱后的数据集副本。"""
        rng = np.random.default_rng(stable_seed(self.dataset_name, self.profile, "shuffle"))
        clone = object.__new__(SyntheticGraphDataset)
        clone.root = self.root
        clone.dataset_name = self.dataset_name
        clone.profile = self.profile
        clone.meta = dict(self.meta)
        clone.num_classes = self.num_classes
        clone.num_features = self.num_features
        order = rng.permutation(len(self.graphs))
        clone.graphs = [self.graphs[int(index)] for index in order]
        return clone
