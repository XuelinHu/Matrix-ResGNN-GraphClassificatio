"""实现数据加载、划分、训练、评估、机制分析导出和单配置实验执行。"""
from __future__ import annotations

import csv
import json
import random
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from src.experiment_catalog import dataset_family
from src.experiment_paths import (
    DEFAULT_EXPERIMENT_VERSION,
    analysis_dir,
    checkpoint_dir,
    ensure_version_manifest,
    log_dir,
    normalize_version,
    record_dir,
    run_dir,
)
from src.models import HorizontalResGNN, MatrixResGNN, MatrixResGatedGNN, PlainGNN, VerticalResGNN
from src.models.common import ResidualConfig


PROJECT_# 仓库根目录：用于把脚本中的相对路径统一定位到项目根路径。
ROOT = Path(__file__).resolve().parents[1]
# 数据根目录：PyG/TUDataset 等数据集会缓存在该目录下。
DATA_ROOT = PROJECT_ROOT / "data"
# 训练设备：优先使用 CUDA，否则自动回退到 CPU。
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dirs(version: str = DEFAULT_EXPERIMENT_VERSION) -> None:
    """创建数据、日志、运行记录、检查点和分析输出目录。"""
    paths = [
        DATA_ROOT,
        record_dir(PROJECT_ROOT, version),
        log_dir(PROJECT_ROOT, version),
        run_dir(PROJECT_ROOT, version),
        checkpoint_dir(PROJECT_ROOT, version),
        analysis_dir(PROJECT_ROOT, version),
    ]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    ensure_version_manifest(PROJECT_ROOT)


def set_seed(seed: int) -> None:
    """设置 Python、NumPy 和 PyTorch 随机种子，保证实验可复现。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def timestamp() -> str:
    """生成用于日志和产物文件名的时间戳。"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def prepare_batch(data: torch.Tensor) -> torch.Tensor:
    """补齐缺失节点特征，并把 batch 移动到当前设备。"""
    if data.x is None:
        data.x = torch.ones((data.num_nodes, 1), dtype=torch.float)
    return data.to(DEVICE)


def load_dataset(dataset_name: str) -> object:
    """根据数据集名称加载图分类数据集。"""
    family = dataset_family(dataset_name)
    if family == "tu":
        dataset = TUDataset(root=str(DATA_ROOT / "TUDataset"), name=dataset_name)
        return dataset.shuffle()
    raise ValueError(f"Unsupported dataset family for {dataset_name}.")


def graph_target(graph: torch.Tensor) -> torch.Tensor:
    """把图标签整理成交叉熵训练需要的一维 long tensor。"""
    return graph.y.view(-1).long()


def dataset_labels(dataset: Iterable[torch.Tensor]) -> List[int]:
    """提取数据集中每个图的整数标签。"""
    return [int(graph_target(graph)[0]) for graph in dataset]


def stratified_kfold_indices(labels: List[int], n_splits: int, seed: int) -> List[Tuple[List[int], List[int]]]:
    """按标签分层构造 K 折训练和测试索引。"""
    rng = random.Random(seed)
    label_to_indices: Dict[int, List[int]] = {}
    for index, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(index)

    folds: List[List[int]] = [[] for _ in range(n_splits)]
    for indices in label_to_indices.values():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        for offset, index in enumerate(shuffled):
            folds[offset % n_splits].append(index)

    split_pairs: List[Tuple[List[int], List[int]]] = []
    all_indices = set(range(len(labels)))
    for fold_indices in folds:
        test_indices = sorted(fold_indices)
        train_indices = sorted(all_indices - set(test_indices))
        split_pairs.append((train_indices, test_indices))
    return split_pairs


def stratified_train_val_indices(labels: List[int], val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    """在训练集内部按标签分层划分训练和验证索引。"""
    rng = random.Random(seed)
    label_to_indices: Dict[int, List[int]] = {}
    for index, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(index)

    train_indices: List[int] = []
    val_indices: List[int] = []
    for indices in label_to_indices.values():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        val_count = int(round(len(shuffled) * val_ratio))
        if len(shuffled) > 1:
            val_count = max(1, min(len(shuffled) - 1, val_count))
        else:
            val_count = 0
        val_indices.extend(shuffled[:val_count])
        train_indices.extend(shuffled[val_count:])
    return sorted(train_indices), sorted(val_indices)


def split_dataset(
    dataset: object,
    fold: int,
    dataset_name: str,
) -> Tuple[Sequence[torch.Tensor], Sequence[torch.Tensor], Optional[Sequence[torch.Tensor]], Dict[str, object]]:
    """按数据集协议生成训练集、测试集和可选官方验证集。"""
    family = dataset_family(dataset_name)
    if family == "tu":
        graph_list = list(dataset)
        labels = dataset_labels(graph_list)
        folds = stratified_kfold_indices(labels, n_splits=5, seed=0)
        train_indices, test_indices = folds[fold]
        train_dataset = [graph_list[index] for index in train_indices]
        test_dataset = [graph_list[index] for index in test_indices]
        return train_dataset, test_dataset, None, {
            "dataset_family": family,
            "split_protocol": "stratified_5fold_cv",
            "repeat_id": fold,
            "official_split": False,
        }
    raise ValueError(f"Unsupported dataset family for {dataset_name}.")


def split_train_val_dataset(
    train_dataset: Sequence[torch.Tensor],
    val_ratio: float,
    seed: int,
) -> Tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]:
    """从训练集内部切分验证集，用于早停和模型选择。"""
    if not train_dataset or val_ratio <= 0.0 or len(train_dataset) <= 1:
        return train_dataset, []
    labels = dataset_labels(train_dataset)
    if len(set(labels)) < 2:
        return train_dataset, []
    train_indices, val_indices = stratified_train_val_indices(labels, val_ratio=val_ratio, seed=seed)
    inner_train_dataset = [train_dataset[index] for index in train_indices]
    val_dataset = [train_dataset[index] for index in val_indices]
    return inner_train_dataset, val_dataset


def build_loader(dataset_slice: Sequence[torch.Tensor], batch_size: int, shuffle: bool) -> DataLoader:
    """根据图样本切片构造 PyG DataLoader。"""
    return DataLoader(dataset_slice, batch_size=batch_size, shuffle=shuffle)


def dataset_statistics(dataset: object, dataset_name: str) -> Dict[str, object]:
    """统计图数量、类别数、节点数、边数、度和类别分布。"""
    node_counts: List[int] = []
    edge_counts: List[float] = []
    avg_degrees: List[float] = []
    class_hist: Dict[int, int] = {}
    for graph in dataset:
        num_nodes = int(graph.num_nodes)
        directed_edges = int(graph.edge_index.size(1))
        undirected_edges = directed_edges / 2 if graph.is_undirected() else directed_edges
        avg_degree = directed_edges / max(num_nodes, 1)
        node_counts.append(num_nodes)
        edge_counts.append(undirected_edges)
        avg_degrees.append(avg_degree)
        label = int(graph_target(graph)[0])
        class_hist[label] = class_hist.get(label, 0) + 1
    return {
        "dataset": dataset_name,
        "dataset_family": dataset_family(dataset_name),
        "graphs": len(dataset),
        "classes": dataset.num_classes,
        "num_features": dataset.num_features if dataset.num_features > 0 else 1,
        "avg_nodes": float(np.mean(node_counts)),
        "avg_edges": float(np.mean(edge_counts)),
        "avg_degree": float(np.mean(avg_degrees)),
        "class_hist": class_hist,
    }


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """统计模型总参数、可训练参数和冻结参数数量。"""
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
    }


def gradient_norm(model: nn.Module) -> float:
    """计算当前模型梯度的全局 L2 范数。"""
    squared_norm = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        grad_value = float(parameter.grad.detach().norm(2).item())
        squared_norm += grad_value * grad_value
    return squared_norm ** 0.5


def tensor_stats(tensor: torch.Tensor) -> Dict[str, float]:
    """计算张量均值、标准差和最大值，用于训练诊断。"""
    if tensor.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "max": 0.0}
    detached = tensor.detach()
    return {
        "mean": float(detached.mean().item()),
        "std": float(detached.std(unbiased=False).item()) if detached.numel() > 1 else 0.0,
        "max": float(detached.max().item()),
    }


def collect_gate_values(model: nn.Module) -> Dict[str, float]:
    """收集模型中门控模块的当前数值。"""
    gate_values: Dict[str, float] = {}
    for module_name, module in model.named_modules():
        if hasattr(module, "forward") and module.__class__.__name__ in {"LearnableGate", "FixedGate"}:
            gate_values[module_name] = float(module().detach().item())
    return gate_values


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
    """在验证或测试 loader 上计算平均 loss 和 accuracy。"""
    model.eval()
    total_correct = 0
    total_loss = 0.0
    total_graphs = 0
    with torch.no_grad():
        for batch_data in loader:
            batch_data = prepare_batch(batch_data)
            targets = graph_target(batch_data)
            logits, _ = model(batch_data.x, batch_data.edge_index, batch_data.batch)
            loss = criterion(logits, targets)
            predictions = logits.argmax(dim=1)
            batch_size = targets.size(0)
            total_correct += int((predictions == targets).sum())
            total_loss += float(loss.item()) * batch_size
            total_graphs += batch_size
    return {
        "loss": total_loss / max(total_graphs, 1),
        "acc": total_correct / max(total_graphs, 1),
    }


def build_model(args, dataset: object) -> nn.Module:
    """根据命令行参数实例化对应残差拓扑模型。"""
    config = ResidualConfig(
        hidden_dim=args.dim,
        num_layers=args.h_layer,
        num_branches=args.num_branches,
        dropout=args.drop,
        operator=args.operator,
        gate_init=args.gate_init,
        gate_mode=args.gate_mode,
        fixed_gate_value=args.fixed_gate_value,
        residual_mode=args.residual_mode,
        topk_ratio=args.topk_ratio,
        sparse_lambda=args.sparse_lambda,
    )
    if args.model == "Plain":
        return PlainGNN(config=config, dataset=dataset)
    if args.model == "VerticalRes":
        return VerticalResGNN(config=config, dataset=dataset)
    if args.model == "HorizontalRes":
        return HorizontalResGNN(config=config, dataset=dataset)
    if args.model == "MatrixRes":
        return MatrixResGNN(config=config, dataset=dataset)
    if args.model == "MatrixResGated":
        return MatrixResGatedGNN(config=config, dataset=dataset)
    raise ValueError(f"Unsupported model: {args.model}")


def save_json(payload: Dict[str, object], target: Path) -> None:
    """把字典写入 JSON 文件，并自动创建父目录。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_checkpoint(payload: Dict[str, object], target: Path) -> None:
    """保存模型、优化器和调度器状态到检查点文件。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)


def save_rows(rows: List[Dict[str, object]], target: Path) -> None:
    """把字典行写入 CSV 文件，并自动创建父目录。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_matrix_csv(matrix: np.ndarray, row_labels: List[str], col_labels: List[str], target: Path) -> None:
    """把矩阵连同行列标签写入 CSV 文件。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", *col_labels])
        for label, row in zip(row_labels, matrix):
            writer.writerow([label, *[float(value) for value in row]])


def collect_test_outputs(model: nn.Module, loader: DataLoader) -> Dict[str, np.ndarray]:
    """收集测试集 embedding、logits、标签和预测结果。"""
    model.eval()
    embeddings: List[np.ndarray] = []
    logits_list: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for batch_data in loader:
            batch_data = prepare_batch(batch_data)
            logits, graph_embedding = model(batch_data.x, batch_data.edge_index, batch_data.batch)
            embeddings.append(graph_embedding.detach().cpu().numpy())
            logits_list.append(logits.detach().cpu().numpy())
            labels.append(graph_target(batch_data).detach().cpu().numpy())
            preds.append(logits.argmax(dim=1).detach().cpu().numpy())
    return {
        "embeddings": np.concatenate(embeddings, axis=0),
        "logits": np.concatenate(logits_list, axis=0),
        "labels": np.concatenate(labels, axis=0),
        "preds": np.concatenate(preds, axis=0),
    }


def cosine_similarity_matrix(vectors: Sequence[torch.Tensor]) -> np.ndarray:
    """计算一组表示向量之间的余弦相似度矩阵。"""
    if not vectors:
        return np.zeros((0, 0), dtype=float)
    matrix = np.stack([tensor.detach().cpu().numpy().reshape(-1) for tensor in vectors], axis=0)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normalized = matrix / norms
    return normalized @ normalized.T


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """计算两个表示矩阵之间的线性 CKA 相似度。"""
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    hsic_xy = np.linalg.norm(x.T @ y, ord="fro") ** 2
    hsic_xx = np.linalg.norm(x.T @ x, ord="fro") ** 2
    hsic_yy = np.linalg.norm(y.T @ y, ord="fro") ** 2
    denom = max(np.sqrt(hsic_xx * hsic_yy), 1e-12)
    return float(hsic_xy / denom)


def cka_matrix(states: Sequence[torch.Tensor]) -> np.ndarray:
    """计算一组表示之间两两线性 CKA 矩阵。"""
    if not states:
        return np.zeros((0, 0), dtype=float)
    arrays = [tensor.detach().cpu().numpy() for tensor in states]
    size = len(arrays)
    result = np.zeros((size, size), dtype=float)
    for i in range(size):
        for j in range(size):
            result[i, j] = linear_cka(arrays[i], arrays[j])
    return result


def branch_diversity(branch_embeddings: Sequence[torch.Tensor]) -> Dict[str, object]:
    """计算最终分支图表示之间的平均和最大 L2 距离。"""
    arrays = [tensor.detach().cpu().numpy() for tensor in branch_embeddings]
    if not arrays:
        return {"mean_pairwise_distance": 0.0, "max_pairwise_distance": 0.0, "pairwise_distances": []}
    pairwise: List[Dict[str, object]] = []
    max_distance = 0.0
    total_distance = 0.0
    count = 0
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            distance = float(np.linalg.norm(arrays[i] - arrays[j], axis=1).mean())
            pairwise.append({"branch_i": i, "branch_j": j, "mean_l2_distance": distance})
            total_distance += distance
            count += 1
            max_distance = max(max_distance, distance)
    return {
        "mean_pairwise_distance": total_distance / max(count, 1),
        "max_pairwise_distance": max_distance,
        "pairwise_distances": pairwise,
    }


def representative_gradient_rows(
    model: nn.Module,
    batch_data: torch.Tensor,
    criterion: nn.Module,
) -> List[Dict[str, object]]:
    """在代表 batch 上反传一次并导出各参数梯度范数。"""
    model.zero_grad(set_to_none=True)
    logits, _ = model(batch_data.x, batch_data.edge_index, batch_data.batch)
    loss = criterion(logits, graph_target(batch_data))
    loss.backward()
    rows: List[Dict[str, object]] = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        rows.append(
            {
                "parameter": name,
                "grad_l2_norm": float(parameter.grad.detach().norm(2).item()),
            }
        )
    model.zero_grad(set_to_none=True)
    return rows


def train_one_config(args) -> Dict[str, object]:
    """训练、早停、测试并导出单个实验配置的全部结果产物。"""
    version = normalize_version(getattr(args, "version", DEFAULT_EXPERIMENT_VERSION))
    ensure_dirs(version)
    set_seed(args.seed)
    dataset = load_dataset(args.dataset)
    stats = dataset_statistics(dataset, args.dataset)
    train_dataset, test_dataset, official_val_dataset, split_context = split_dataset(dataset, args.fold, args.dataset)
    if official_val_dataset is None:
        train_dataset, val_dataset = split_train_val_dataset(train_dataset, args.val_ratio, seed=args.seed + args.fold)
    else:
        val_dataset = official_val_dataset

    train_loader = build_loader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = build_loader(val_dataset, batch_size=args.batch_size, shuffle=False) if val_dataset else None
    test_loader = build_loader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(args, dataset).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    parameter_stats = count_parameters(model)

    history: List[Dict[str, object]] = []
    best_epoch = -1
    best_val_loss = float("inf")
    best_val_acc = -1.0
    best_state_dict = deepcopy(model.state_dict())
    best_optimizer_state = deepcopy(optimizer.state_dict())
    best_scheduler_state = deepcopy(scheduler.state_dict())
    best_aux_payload: Optional[Dict[str, object]] = None
    patience_counter = 0
    started = time.time()

    for epoch in range(args.ep):
        model.train()
        total_correct = 0
        total_loss = 0.0
        total_graphs = 0
        grad_norm_sum = 0.0
        grad_norm_steps = 0
        embedding_abs_mean_sum = 0.0
        embedding_abs_max = 0.0
        embedding_std_sum = 0.0

        for batch_data in train_loader:
            batch_data = prepare_batch(batch_data)
            targets = graph_target(batch_data)
            optimizer.zero_grad()
            logits, graph_embedding = model(batch_data.x, batch_data.edge_index, batch_data.batch)
            loss = criterion(logits, targets)
            loss.backward()
            batch_grad_norm = gradient_norm(model)
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            predictions = logits.argmax(dim=1)
            batch_size = targets.size(0)
            total_correct += int((predictions == targets).sum())
            total_loss += float(loss.item()) * batch_size
            total_graphs += batch_size
            grad_norm_sum += batch_grad_norm
            grad_norm_steps += 1

            embedding_stats = tensor_stats(graph_embedding.abs())
            embedding_abs_mean_sum += embedding_stats["mean"] * batch_size
            embedding_std_sum += float(graph_embedding.detach().std(unbiased=False).item()) * batch_size
            embedding_abs_max = max(embedding_abs_max, embedding_stats["max"])

        train_metrics = {
            "loss": total_loss / max(total_graphs, 1),
            "acc": total_correct / max(total_graphs, 1),
        }
        train_diagnostics = {
            "grad_norm": grad_norm_sum / max(grad_norm_steps, 1),
            "embedding_abs_mean": embedding_abs_mean_sum / max(total_graphs, 1),
            "embedding_std": embedding_std_sum / max(total_graphs, 1),
            "embedding_abs_max": embedding_abs_max,
        }
        val_metrics = evaluate(model, val_loader, criterion) if val_loader is not None else train_metrics
        scheduler.step(val_metrics["loss"])
        gate_values = collect_gate_values(model)
        improved = val_metrics["loss"] < (best_val_loss - args.min_delta)
        if improved:
            best_val_loss = val_metrics["loss"]
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch
            best_state_dict = deepcopy(model.state_dict())
            best_optimizer_state = deepcopy(optimizer.state_dict())
            best_scheduler_state = deepcopy(scheduler.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "grad_norm": train_diagnostics["grad_norm"],
                "embedding_abs_mean": train_diagnostics["embedding_abs_mean"],
                "embedding_std": train_diagnostics["embedding_std"],
                "embedding_abs_max": train_diagnostics["embedding_abs_max"],
                "lr": optimizer.param_groups[0]["lr"],
                "gate_values": gate_values,
            }
        )
        if patience_counter >= args.patience:
            break

    model.load_state_dict(best_state_dict)
    test_metrics = evaluate(model, test_loader, criterion)
    test_outputs = collect_test_outputs(model, test_loader)

    rep_loader = build_loader(test_dataset, batch_size=args.batch_size, shuffle=False)
    rep_batch = next(iter(rep_loader))
    rep_batch = prepare_batch(rep_batch)
    _, _, aux = model(rep_batch.x, rep_batch.edge_index, rep_batch.batch, return_aux=True)
    branch_graph_histories = [
        [tensor for tensor in branch_history]
        for branch_history in aux["branch_graph_history"]
    ]
    final_branch_graph_embeddings = aux["final_branch_graph_embeddings"]
    branch_labels = [f"branch{idx}" for idx in range(len(final_branch_graph_embeddings))]
    depth_labels = [f"layer{idx}" for idx in range(len(branch_graph_histories[0]))] if branch_graph_histories else []
    branch_cosine = cosine_similarity_matrix(final_branch_graph_embeddings)
    branch_cka = cka_matrix(final_branch_graph_embeddings)
    depth_cosine = cosine_similarity_matrix(branch_graph_histories[0]) if branch_graph_histories else np.zeros((0, 0))
    depth_cka = cka_matrix(branch_graph_histories[0]) if branch_graph_histories else np.zeros((0, 0))
    gradient_rows = representative_gradient_rows(model, rep_batch, criterion)
    diversity_payload = branch_diversity(final_branch_graph_embeddings)
    best_aux_payload = {
        "residual_stats": aux["residual_stats"],
        "branch_graph_shapes": [
            [list(tensor.shape) for tensor in branch_history]
            for branch_history in aux["branch_graph_history"]
        ],
        "branch_node_shapes": [
            [list(tensor.shape) for tensor in branch_history]
            for branch_history in aux["branch_node_history"]
        ],
        "branch_diversity": diversity_payload,
    }

    stem = (
        f"{args.dataset}__{args.model}__{args.operator}__fold{args.fold}"
        f"__B{args.num_branches}__{args.residual_mode}"
    )
    stamp = timestamp()
    payload = {
        "config": vars(args),
        "version": version,
        "dataset_stats": stats,
        "split_context": split_context,
        "parameter_stats": parameter_stats,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_test_acc": test_metrics["acc"],
        "test_loss": test_metrics["loss"],
        "runtime_seconds": time.time() - started,
    }

    save_json(payload, log_dir(PROJECT_ROOT, version) / f"result_{stem}__{stamp}.json")
    save_json({"history": history}, analysis_dir(PROJECT_ROOT, version) / f"history_{stem}__{stamp}.json")
    save_json(best_aux_payload, analysis_dir(PROJECT_ROOT, version) / f"residual_stats_{stem}__{stamp}.json")
    save_json(vars(args), record_dir(PROJECT_ROOT, version) / f"config_{stem}__{stamp}.json")
    save_rows(
        [
            {
                "index": index,
                "label": int(label),
                "pred": int(pred),
            }
            for index, (label, pred) in enumerate(zip(test_outputs["labels"], test_outputs["preds"]))
        ],
        analysis_dir(PROJECT_ROOT, version) / f"predictions_{stem}__{stamp}.csv",
    )
    np.save(analysis_dir(PROJECT_ROOT, version) / f"logits_{stem}__{stamp}.npy", test_outputs["logits"])
    np.save(analysis_dir(PROJECT_ROOT, version) / f"labels_{stem}__{stamp}.npy", test_outputs["labels"])
    np.save(analysis_dir(PROJECT_ROOT, version) / f"preds_{stem}__{stamp}.npy", test_outputs["preds"])
    np.savez_compressed(
        analysis_dir(PROJECT_ROOT, version) / f"graph_embeddings_{stem}__{stamp}.npz",
        embeddings=test_outputs["embeddings"],
    )
    torch.save(
        {
            "branch_node_history": aux["branch_node_history"],
            "branch_graph_history": aux["branch_graph_history"],
            "final_branch_graph_embeddings": aux["final_branch_graph_embeddings"],
        },
        analysis_dir(PROJECT_ROOT, version) / f"layer_states_{stem}__{stamp}.pt",
    )
    save_json(diversity_payload, analysis_dir(PROJECT_ROOT, version) / f"branch_diversity_{stem}__{stamp}.json")
    save_rows(gradient_rows, analysis_dir(PROJECT_ROOT, version) / f"gradient_by_layer_{stem}__{stamp}.csv")
    save_matrix_csv(
        branch_cosine,
        row_labels=branch_labels,
        col_labels=branch_labels,
        target=analysis_dir(PROJECT_ROOT, version) / f"cosine_branch_{stem}__{stamp}.csv",
    )
    save_matrix_csv(
        branch_cka,
        row_labels=branch_labels,
        col_labels=branch_labels,
        target=analysis_dir(PROJECT_ROOT, version) / f"cka_branch_{stem}__{stamp}.csv",
    )
    save_matrix_csv(
        depth_cosine,
        row_labels=depth_labels,
        col_labels=depth_labels,
        target=analysis_dir(PROJECT_ROOT, version) / f"cosine_depth_{stem}__{stamp}.csv",
    )
    save_matrix_csv(
        depth_cka,
        row_labels=depth_labels,
        col_labels=depth_labels,
        target=analysis_dir(PROJECT_ROOT, version) / f"cka_depth_{stem}__{stamp}.csv",
    )
    save_checkpoint(
        {
            "model_state_dict": best_state_dict,
            "optimizer_state_dict": best_optimizer_state,
            "scheduler_state_dict": best_scheduler_state,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "config": vars(args),
        },
        checkpoint_dir(PROJECT_ROOT, version) / f"checkpoint_{stem}__{stamp}.pt",
    )
    return payload
