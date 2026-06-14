"""Statistical functions for computing covariances and domain distances."""
import torch


def covariance(features: torch.Tensor) -> torch.Tensor:
    """Compute the covariance matrix of features.

    Args:
        features: Tensor of shape (N, D).

    Returns:
        Covariance matrix of shape (D, D).
    """
    n = features.shape[0]
    tmp = torch.ones((1, n), device=features.device) @ features
    cov = (features.t() @ features - (tmp.t() @ tmp) / n) / n
    return cov


def coral(source_cov: torch.Tensor, target_cov: torch.Tensor) -> torch.Tensor:
    """CORAL loss (difference of covariance matrices)."""
    d = source_cov.shape[0]
    loss = (source_cov - target_cov).pow(2).sum() / (4.0 * d ** 2)
    return loss


def linear_mmd(source_mean: torch.Tensor, target_mean: torch.Tensor) -> torch.Tensor:
    """Linear MMD loss (difference of means)."""
    loss = (source_mean - target_mean).pow(2).mean()
    return loss