from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MoEConfig, BalancerConfig
from .utils import compute_cosine_similarity, get_logger

logger = get_logger(__name__)


@dataclass
class ClusteringResult:
    cluster_assignments: torch.Tensor
    cluster_centers: List[List[int]]
    cluster_sizes: torch.Tensor
    similarity_matrix: torch.Tensor
    num_clusters: int
    cluster_avg_utilization: Optional[torch.Tensor] = None


class ExpertSimilarityClusterer:
    def __init__(
        self,
        moe_config: MoEConfig,
        balancer_config: BalancerConfig,
        device: Optional[torch.device] = None,
    ):
        self.moe_config = moe_config
        self.balancer_config = balancer_config
        self.device = device or torch.device("cpu")

        self.num_experts = moe_config.num_experts
        self.num_clusters = balancer_config.num_clusters
        self.similarity_top_k = balancer_config.similarity_top_k

        self.update_interval = balancer_config.cluster_update_interval

        self._similarity_matrix = None
        self._cluster_assignments = None
        self._cluster_centers: List[List[int]] = []
        self._last_update_step = -1

    def to(self, device: torch.device):
        self.device = device
        if self._similarity_matrix is not None:
            self._similarity_matrix = self._similarity_matrix.to(device)
        if self._cluster_assignments is not None:
            self._cluster_assignments = self._cluster_assignments.to(device)
        return self

    def compute_similarity_from_parameters(
        self,
        expert_representations: torch.Tensor,
    ) -> torch.Tensor:
        expert_representations = expert_representations.to(self.device)
        sim_matrix = compute_cosine_similarity(expert_representations)
        sim_matrix.fill_diagonal_(0.0)
        self._similarity_matrix = sim_matrix
        return sim_matrix

    def compute_similarity_from_activations(
        self,
        expert_activations: torch.Tensor,
    ) -> torch.Tensor:
        if expert_activations.dim() == 3:
            expert_activations = expert_activations.mean(dim=1)
        expert_activations = expert_activations.to(self.device)
        sim_matrix = compute_cosine_similarity(expert_activations)
        sim_matrix.fill_diagonal_(0.0)
        self._similarity_matrix = sim_matrix
        return sim_matrix

    def get_similarity_matrix(self) -> Optional[torch.Tensor]:
        return self._similarity_matrix

    def get_top_k_similar(self, expert_idx: int, k: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        k = k or self.similarity_top_k
        if self._similarity_matrix is None:
            raise RuntimeError("Similarity matrix not computed yet.")
        sims = self._similarity_matrix[expert_idx]
        top_k_vals, top_k_idx = torch.topk(sims, k)
        return top_k_idx, top_k_vals

    def cluster_experts_kmeans(
        self,
        similarity_matrix: Optional[torch.Tensor] = None,
        max_iter: int = 100,
        tol: float = 1e-4,
    ) -> ClusteringResult:
        if similarity_matrix is not None:
            self._similarity_matrix = similarity_matrix.to(self.device)
        if self._similarity_matrix is None:
            raise RuntimeError("Similarity matrix must be computed before clustering.")

        sim = self._similarity_matrix
        n = self.num_experts
        k = min(self.num_clusters, n)

        distance = 1.0 - sim

        center_idx = torch.randperm(n, device=self.device)[:k]
        prev_assignments = None

        for it in range(max_iter):
            cluster_distances = distance[:, center_idx]
            assignments = torch.argmin(cluster_distances, dim=1)

            if prev_assignments is not None:
                diff = (assignments != prev_assignments).float().mean().item()
                if diff < tol:
                    break
            prev_assignments = assignments.clone()

            new_centers = torch.zeros(k, device=self.device, dtype=torch.long)
            for ci in range(k):
                mask = (assignments == ci)
                if mask.sum() == 0:
                    new_centers[ci] = center_idx[ci]
                    continue
                cluster_dists = distance[mask][:, mask]
                total_dists = cluster_dists.sum(dim=1)
                best_local = torch.argmin(total_dists)
                local_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                new_centers[ci] = local_indices[best_local]

            if torch.equal(new_centers, center_idx):
                break
            center_idx = new_centers

        self._cluster_assignments = assignments

        centers: List[List[int]] = []
        for ci in range(k):
            members = torch.nonzero(assignments == ci, as_tuple=False).squeeze(-1).tolist()
            centers.append(members)

        cluster_sizes = torch.tensor([len(c) for c in centers], device=self.device, dtype=torch.long)

        self._cluster_centers = centers

        result = ClusteringResult(
            cluster_assignments=assignments.clone(),
            cluster_centers=centers,
            cluster_sizes=cluster_sizes,
            similarity_matrix=sim.clone(),
            num_clusters=k,
        )
        return result

    def get_cluster_for_expert(self, expert_idx: int) -> Optional[int]:
        if self._cluster_assignments is None:
            return None
        return int(self._cluster_assignments[expert_idx].item())

    def get_cluster_members(self, cluster_idx: int) -> List[int]:
        if cluster_idx < 0 or cluster_idx >= len(self._cluster_centers):
            return []
        return self._cluster_centers[cluster_idx]

    def generate_redistribution_mapping(
        self,
        overloaded_experts: torch.Tensor,
        underloaded_experts: torch.Tensor,
    ) -> Dict[int, List[int]]:
        if self._similarity_matrix is None:
            self._similarity_matrix = torch.randn(
                self.num_experts, self.num_experts, device=self.device
            )
            self._similarity_matrix = (self._similarity_matrix + self._similarity_matrix.T) / 2
            self._similarity_matrix.fill_diagonal_(0.0)
            self._similarity_matrix = self._similarity_matrix.softmax(dim=-1)

        mapping: Dict[int, List[int]] = {}
        sim = self._similarity_matrix

        for under_idx in underloaded_experts.tolist():
            sim_to_over = sim[under_idx][overloaded_experts]
            if len(sim_to_over) == 0:
                continue
            _, top_k = torch.topk(sim_to_over, k=min(self.similarity_top_k, len(sim_to_over)))
            src_experts = overloaded_experts[top_k].tolist()
            mapping[under_idx] = src_experts

        return mapping

    def compute_cluster_balance_penalty(
        self,
        expert_utilization: torch.Tensor,
    ) -> torch.Tensor:
        if self._cluster_assignments is None:
            return torch.tensor(0.0, device=self.device)

        expert_utilization = expert_utilization.to(self.device)
        k = self.num_clusters
        cluster_utils = torch.zeros(k, device=self.device)
        cluster_counts = torch.zeros(k, device=self.device).clamp_min(1)

        cluster_utils.index_add_(0, self._cluster_assignments, expert_utilization)
        cluster_counts.index_add_(
            0, self._cluster_assignments,
            torch.ones_like(expert_utilization)
        )

        cluster_avg = cluster_utils / cluster_counts
        mean_cluster_avg = cluster_avg.mean().clamp_min(1e-9)
        penalty = ((cluster_avg - mean_cluster_avg) ** 2).mean() / (mean_cluster_avg ** 2)
        return penalty

    def needs_update(self, current_step: int) -> bool:
        return (current_step - self._last_update_step) >= self.update_interval

    def mark_updated(self, step: int):
        self._last_update_step = step

    def state_dict(self) -> Dict:
        return {
            "similarity_matrix": self._similarity_matrix.clone().cpu() if self._similarity_matrix is not None else None,
            "cluster_assignments": self._cluster_assignments.clone().cpu() if self._cluster_assignments is not None else None,
            "cluster_centers": self._cluster_centers,
            "last_update_step": self._last_update_step,
        }

    def load_state_dict(self, state: Dict, map_location: Optional[torch.device] = None):
        dev = map_location or self.device
        if state["similarity_matrix"] is not None:
            self._similarity_matrix = state["similarity_matrix"].to(dev)
        if state["cluster_assignments"] is not None:
            self._cluster_assignments = state["cluster_assignments"].to(dev)
        self._cluster_centers = state["cluster_centers"]
        self._last_update_step = state["last_update_step"]
