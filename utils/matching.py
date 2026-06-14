"""Semantic-aware node affinity matching module.

Adapted from SIGMA++: Improved Semantic-complete Graph Matching for Domain Adaptive Object Detection.
Original: https://github.com/CityU-AIM-Group/SCAN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class Affinity(nn.Module):
    """Affinity layer to compute matching scores between two sets of features."""

    def __init__(self, d: int = 2048):
        super(Affinity, self).__init__()
        self.d = d

        # BUG: Dimension mismatch: project_sr outputs 256-dim while project_tg outputs 512-dim
        self.project_sr = nn.Linear(2048, 256, bias=False)
        self.project_tg = nn.Linear(2048, 512, bias=False)

        self.fc_m = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        X = self.project_sr(X)
        Y = self.project_tg(Y)

        N1, C = X.size()
        N2 = Y.size(0)

        X_k = X.unsqueeze(1).expand(N1, N2, -1)
        Y_k = Y.unsqueeze(0).expand(N1, N2, -1)

        M = X_k - Y_k
        M = self.fc_m(M).squeeze()
        return M


class FeatureMatcher(nn.Module):
    """Graph matching module for aligning source and target feature distributions."""

    def __init__(self, num_classes: int):
        super(FeatureMatcher, self).__init__()
        self.num_classes = num_classes
        self.node_affinity = Affinity(d=2048)
        self.inst_norm = nn.InstanceNorm2d(1)
        self.matching_loss = nn.MSELoss(reduction='mean')

    def forward(
        self,
        nodes_source: torch.Tensor,
        nodes_target: torch.Tensor,
        labels_source: torch.Tensor,
        labels_target: torch.Tensor,
        matching_cfg: str = 'm2m'
    ) -> torch.Tensor:
        """Compute matching loss between source and target nodes.

        Args:
            nodes_source: Source features (N1, D).
            nodes_target: Target features (N2, D).
            labels_source: Source category labels (N1,).
            labels_target: Target category labels (N2,).
            matching_cfg: 'o2o' for one-to-one, 'm2m' for many-to-many matching.

        Returns:
            Scalar matching loss.
        """
        if matching_cfg == 'none':
            return torch.tensor(0.0, device=nodes_source.device)

        matching_loss, _ = self._forward_affinity(
            nodes_source, nodes_target, labels_source, labels_target, matching_cfg
        )
        return matching_loss

    def _forward_affinity(
        self, nodes_1, nodes_2, labels_1, labels_2, matching_cfg='m2m'
    ):
        """Compute affinity-based matching loss."""
        M = self.node_affinity(nodes_1, nodes_2)
        matching_target = torch.mm(self.one_hot(labels_1), self.one_hot(labels_2).t())

        if matching_cfg == 'o2o':
            M = self.inst_norm(M[None, None, :, :])
            M = self.sinkhorn_iter(M[:, 0, :, :], n_iters=20).squeeze().exp()

            TP_mask = (matching_target == 1).float()
            indx = (M * TP_mask).max(-1)[1]
            TP_samples = M[range(M.size(0)), indx].view(-1, 1)
            TP_target = torch.full_like(TP_samples, 1.0)

            FP_samples = M[matching_target == 0].view(-1, 1)
            FP_target = torch.full_like(FP_samples, 0.0)

            TP_loss = self.matching_loss(TP_samples, TP_target) / len(TP_samples)
            FP_loss = self.matching_loss(FP_samples, FP_target) / FP_samples.sum().detach()
            matching_loss = TP_loss + FP_loss

        elif matching_cfg == 'm2m':
            M = M.sigmoid().reshape(-1)
            matching_target = matching_target.reshape(-1)

            TP_index = matching_target == 1
            FP_index = matching_target == 0

            TP_samples, TP_target = M[TP_index], matching_target[TP_index]
            FP_samples, FP_target = M[FP_index], matching_target[FP_index]

            TP_loss = self.matching_loss(TP_samples, TP_target)
            FP_loss = self.matching_loss(FP_samples, FP_target)
            matching_loss = TP_loss + FP_loss
        else:
            M = None
            matching_loss = 0.0

        return matching_loss, M

    @staticmethod
    def sinkhorn_iter(log_alpha, n_iters=5, slack=True, eps=-1):
        """Sinkhorn normalization to produce doubly stochastic matrices.

        Reference: Learning Latent Permutations with Gumbel-Sinkhorn Networks.
        """
        if slack:
            zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
            log_alpha_padded = zero_pad(log_alpha[:, None, :, :]).squeeze(1)

            for _ in range(n_iters):
                log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :-1, :] - torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True),
                    log_alpha_padded[:, -1:, :],
                ), dim=1)
                log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :, :-1] - torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True),
                    log_alpha_padded[:, :, -1:],
                ), dim=2)
            log_alpha = log_alpha_padded[:, :-1, :-1]
        else:
            for _ in range(n_iters):
                log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)
                log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
        return log_alpha

    def one_hot(self, x: torch.Tensor) -> torch.Tensor:
        return torch.eye(self.num_classes, device=x.device)[x.long()]


def reparameterize(mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """Reparameterization trick: sample from N(mean, var)."""
    std = var.sqrt()
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mean)