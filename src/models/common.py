from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv, global_mean_pool


GridIndex = Tuple[int, int]


@dataclass(frozen=True)
class ResidualConfig:
    hidden_dim: int = 64
    num_layers: int = 4
    num_branches: int = 3
    dropout: float = 0.5
    operator: str = "GCNConv"
    gate_init: float = 0.8
    gate_mode: str = "learnable"
    fixed_gate_value: float = 0.8
    residual_mode: str = "identity"
    topk_ratio: float = 0.5
    sparse_lambda: float = 0.05


def gate_logit_from_probability(probability: float) -> float:
    clipped = min(max(probability, 1e-4), 1.0 - 1e-4)
    return float(np.log(clipped / (1.0 - clipped)))


class LearnableGate(nn.Module):
    def __init__(self, init_probability: float = 1.0):
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(gate_logit_from_probability(init_probability), dtype=torch.float))

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self.logit)


class FixedGate(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        clipped = min(max(value, 0.0), 1.0)
        self.register_buffer("value", torch.tensor(float(clipped), dtype=torch.float))

    def forward(self) -> torch.Tensor:
        return self.value


def build_gate(init_probability: float, gate_mode: str, fixed_gate_value: float) -> nn.Module:
    if gate_mode == "fixed":
        return FixedGate(fixed_gate_value)
    return LearnableGate(init_probability=init_probability)


def topk_mask(tensor: torch.Tensor, ratio: float) -> torch.Tensor:
    if tensor.numel() == 0:
        return torch.zeros_like(tensor)
    keep_ratio = min(max(ratio, 1e-4), 1.0)
    last_dim = tensor.size(-1)
    k = max(1, min(last_dim, int(np.ceil(last_dim * keep_ratio))))
    if k >= last_dim:
        return torch.ones_like(tensor)
    _, indices = torch.topk(tensor.abs(), k=k, dim=-1)
    mask = torch.zeros_like(tensor)
    mask.scatter_(-1, indices, 1.0)
    return mask


def apply_residual_mode(
    residual: torch.Tensor,
    residual_mode: str,
    topk_ratio: float,
    sparse_lambda: float,
) -> torch.Tensor:
    if residual_mode == "topk":
        return residual * topk_mask(residual, ratio=topk_ratio)
    if residual_mode == "sparse":
        return F.softshrink(residual, lambd=max(sparse_lambda, 0.0))
    return residual


def valid_branch(branch: int, num_branches: int) -> bool:
    return 0 <= branch < num_branches


def vertical_neighbors(branch: int, layer: int, num_branches: int) -> List[GridIndex]:
    del num_branches
    if layer <= 0:
        return []
    return [(branch, layer - 1)]


def horizontal_neighbors(branch: int, layer: int, num_branches: int) -> List[GridIndex]:
    neighbors: List[GridIndex] = []
    for other in (branch - 1, branch + 1):
        if valid_branch(other, num_branches):
            neighbors.append((other, layer))
    return neighbors


def matrix_neighbors(branch: int, layer: int, num_branches: int) -> List[GridIndex]:
    neighbors = vertical_neighbors(branch, layer, num_branches)
    neighbors.extend(horizontal_neighbors(branch, layer, num_branches))
    if layer > 0:
        for other in (branch - 1, branch + 1):
            if valid_branch(other, num_branches):
                neighbors.append((other, layer - 1))
    return neighbors


def build_neighbor_map(num_branches: int, num_layers: int, mode: str) -> Dict[GridIndex, List[GridIndex]]:
    if mode == "plain":
        fn = lambda b, l, nb: []  # noqa: E731
    elif mode == "vertical":
        fn = vertical_neighbors
    elif mode == "horizontal":
        fn = horizontal_neighbors
    elif mode == "matrix":
        fn = matrix_neighbors
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    mapping: Dict[GridIndex, List[GridIndex]] = {}
    for branch in range(num_branches):
        for layer in range(num_layers):
            mapping[(branch, layer)] = fn(branch, layer, num_branches)
    return mapping


def infer_input_dim(dataset: object) -> int:
    return dataset.num_features if dataset.num_features > 0 else 1


def build_mlp(input_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.ReLU(),
        nn.Linear(output_dim, output_dim),
    )


def build_operator(name: str, in_channels: int, out_channels: int) -> nn.Module:
    if name == "GCNConv":
        return GCNConv(in_channels, out_channels)
    if name == "GATConv":
        return GATConv(in_channels, out_channels)
    if name == "SAGEConv":
        return SAGEConv(in_channels, out_channels)
    if name == "GINConv":
        return GINConv(build_mlp(in_channels, out_channels))
    raise ValueError(f"Unsupported operator: {name}")


class BaseResidualGNN(nn.Module):
    topology_mode: str = "undefined"

    def __init__(self, config: ResidualConfig, dataset: object):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.input_dim = infer_input_dim(dataset)
        self.num_classes = dataset.num_classes
        self.neighbor_map = build_neighbor_map(
            num_branches=config.num_branches,
            num_layers=config.num_layers,
            mode=self.topology_mode,
        )
        self.input_layers = nn.ModuleList(
            [
                build_operator(config.operator, self.input_dim, config.hidden_dim)
                for _ in range(config.num_branches)
            ]
        )
        hidden_depth = max(config.num_layers - 1, 0)
        self.hidden_layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        build_operator(config.operator, config.hidden_dim, config.hidden_dim)
                        for _ in range(config.num_branches)
                    ]
                )
                for _ in range(hidden_depth)
            ]
        )
        self.residual_gate = build_gate(
            init_probability=config.gate_init,
            gate_mode=config.gate_mode,
            fixed_gate_value=config.fixed_gate_value,
        )
        self.classifier = nn.Linear(config.hidden_dim, self.num_classes)

    def residual_sources(self, branch: int, layer: int) -> Sequence[GridIndex]:
        return tuple(self.neighbor_map[(branch, layer)])

    def _fuse_residuals(
        self,
        branch: int,
        layer: int,
        proposals: Sequence[torch.Tensor],
        previous_states: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        fused = proposals[branch]
        gate = self.residual_gate()
        stats = {
            "residual_count": 0.0,
            "residual_norm_pre": 0.0,
            "residual_norm_post": 0.0,
            "active_ratio": 0.0,
            "gate": float(gate.detach().item()),
        }
        residual_sum = torch.zeros_like(proposals[branch])
        active_parts = 0.0
        for source_branch, source_layer in self.residual_sources(branch, layer):
            source_tensor = proposals[source_branch] if source_layer == layer else previous_states[source_branch]
            residual_sum = residual_sum + source_tensor
            filtered = apply_residual_mode(
                source_tensor,
                residual_mode=self.config.residual_mode,
                topk_ratio=self.config.topk_ratio,
                sparse_lambda=self.config.sparse_lambda,
            )
            residual_sum = residual_sum - source_tensor + filtered
            stats["residual_count"] += 1.0
            stats["residual_norm_pre"] += float(source_tensor.detach().norm(2).item())
            stats["residual_norm_post"] += float(filtered.detach().norm(2).item())
            active_parts += float((filtered.detach().abs() > 0).float().mean().item())

        if stats["residual_count"] > 0:
            stats["active_ratio"] = active_parts / stats["residual_count"]
            fused = fused + gate * residual_sum
        return fused, stats

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        return_aux: bool = False,
    ):
        branch_states: List[torch.Tensor] = [x for _ in range(self.config.num_branches)]
        branch_node_history: List[List[torch.Tensor]] = [[] for _ in range(self.config.num_branches)]
        branch_graph_history: List[List[torch.Tensor]] = [[] for _ in range(self.config.num_branches)]
        residual_stats: Dict[str, Dict[str, float]] = {}

        for layer in range(self.config.num_layers):
            proposals: List[torch.Tensor] = []
            for branch in range(self.config.num_branches):
                if layer == 0:
                    proposal = self.input_layers[branch](branch_states[branch], edge_index)
                else:
                    proposal = self.hidden_layers[layer - 1][branch](branch_states[branch], edge_index)
                proposals.append(F.relu(proposal))

            fused_states: List[torch.Tensor] = []
            for branch in range(self.config.num_branches):
                fused, stats = self._fuse_residuals(branch, layer, proposals, branch_states)
                fused = F.dropout(F.relu(fused), p=self.config.dropout, training=self.training)
                fused_states.append(fused)
                residual_stats[f"branch{branch}_layer{layer}"] = stats
                branch_node_history[branch].append(fused.detach().cpu())
                branch_graph_history[branch].append(global_mean_pool(fused, batch).detach().cpu())
            branch_states = fused_states

        graph_embeddings = [global_mean_pool(state, batch) for state in branch_states]
        stacked = torch.stack(graph_embeddings, dim=0)
        final_graph_embedding = stacked.mean(dim=0)
        logits = self.classifier(F.dropout(final_graph_embedding, p=self.config.dropout, training=self.training))

        if not return_aux:
            return logits, final_graph_embedding
        return logits, final_graph_embedding, {
            "branch_node_history": branch_node_history,
            "branch_graph_history": branch_graph_history,
            "residual_stats": residual_stats,
            "final_branch_graph_embeddings": [embedding.detach().cpu() for embedding in graph_embeddings],
        }
