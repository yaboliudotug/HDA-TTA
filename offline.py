"""Offline feature summarization from source domain."""
import logging

import faiss
import numpy as np
import torch
import sklearn.cluster as cluster

from utils.stats import covariance
from utils.clustering import get_cluster_models

logger = logging.getLogger(__name__)


def run_kmeans(x: np.ndarray, num_clusters: int, knn: int = 1) -> list:
    """Run k-means clustering using FAISS.

    Args:
        x: Data array of shape (N, D).
        num_clusters: Number of clusters.
        knn: Number of nearest neighbors.

    Returns:
        List of cluster assignments.
    """
    n_data, d = x.shape
    num_clusters = int(num_clusters)
    clus = faiss.Clustering(d, num_clusters)
    clus.niter = 50
    clus.max_points_per_centroid = 10000000

    index = faiss.IndexFlatL2(d)
    clus.train(x, index)
    _, labels = index.search(x, knn)

    return [float(n[0]) for n in labels]


def summarize_source_features(args, loader, extractor, classifier, projection_head, num_classes=10):
    """Extract and summarize source domain features.

    Extracts features from the source dataset, organizes them by predicted class,
    computes per-class means and covariances, and optionally performs clustering.

    Args:
        args: Configuration with clustering options.
        loader: DataLoader for source data.
        extractor: Feature extractor network.
        classifier: Linear classifier.
        projection_head: Projection head for features.
        num_classes: Number of classes (10 for CIFAR-10).

    Returns:
        Tuple containing:
            - ext_mu: Per-class feature means from extractor.
            - ext_cov: Per-class feature covariances from extractor.
            - ssh_mu: Per-class feature means from projection head.
            - ssh_cov: Per-class feature covariances from projection head.
            - mu_src_ext: Global mean of extractor features.
            - cov_src_ext: Global covariance of extractor features.
            - mu_src_ssh: Global mean of projection features.
            - cov_src_ssh: Global covariance of projection features.
            - features_source: All source features (N, 2048).
            - category_labels: Category labels for each feature (N,).
            - cluster_labels: Cluster labels if clustering is used, else None.
            - cluster_models: Trained cluster models if clustering is used, else None.
            - ext_mu_cluster: Per-cluster feature means, or None.
            - ext_cov_cluster: Per-cluster feature covariances, or None.
    """
    extractor.eval()

    ext_features_per_class = [[] for _ in range(num_classes)]
    ssh_features_per_class = [[] for _ in range(num_classes)]

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            feat = extractor(inputs.cuda())
            logits = classifier(feat)
            ssh_feat = projection_head(feat)

            pseudo_label = logits.max(dim=1)[1]

            for label in labels.unique():
                label_mask = pseudo_label == label
                ext_features_per_class[label].extend(feat[label_mask])
                ssh_features_per_class[label].extend(ssh_feat[label_mask])

    ext_mu = []
    ext_cov = []
    ext_all = []
    category_labels_list = []

    ssh_mu = []
    ssh_cov = []
    ssh_all = []

    for c in range(num_classes):
        feats_c = torch.stack(ext_features_per_class[c])
        ext_mu.append(feats_c.mean(dim=0))
        ext_cov.append(covariance(feats_c))
        ext_all.extend(ext_features_per_class[c])
        category_labels_list.append(torch.ones(feats_c.shape[0]) * c)

        ssh_feats_c = torch.stack(ssh_features_per_class[c])
        ssh_mu.append(ssh_feats_c.mean(dim=0))
        ssh_cov.append(covariance(ssh_feats_c))
        ssh_all.extend(ssh_features_per_class[c])

    ext_all = torch.stack(ext_all)
    mu_src_ext = ext_all.mean(dim=0)
    cov_src_ext = covariance(ext_all)
    category_labels = torch.cat(category_labels_list, dim=0).cuda()

    ssh_all = torch.stack(ssh_all)
    mu_src_ssh = ssh_all.mean(dim=0)
    cov_src_ssh = covariance(ssh_all)

    # Optional clustering
    cluster_labels = None
    ext_mu_cluster = None
    ext_cov_cluster = None
    cluster_models = None

    if args.do_addtional_cluster:
        logger.info('Running additional clustering on all features...')
        labels_out = run_kmeans(ext_all.cpu().numpy(), args.num_all_clusters, knn=1)
        cluster_labels = torch.tensor(labels_out).cuda()
        _log_cluster_sizes(cluster_labels, args.num_all_clusters)

        ext_mu_cluster, ext_cov_cluster = _compute_cluster_stats(ext_all, cluster_labels, args.num_all_clusters)

    elif args.do_addtional_cluster_within_category:
        logger.info('Running additional clustering within categories...')
        all_features_list = []
        all_cluster_labels_list = []
        all_category_labels_list = []

        for natural_category in range(num_classes):
            cat_features = ext_all[category_labels == natural_category]
            labels_out = run_kmeans(cat_features.cpu().numpy(), args.num_clusters_per_category, knn=1)
            cluster_labels_cat = torch.tensor(labels_out).cuda()
            cluster_labels_cat += natural_category * args.num_clusters_per_category

            all_features_list.append(cat_features)
            all_cluster_labels_list.append(cluster_labels_cat)
            all_category_labels_list.append(torch.ones(cat_features.shape[0]) * natural_category)

            cluster_sizes = [(cluster_labels_cat == k).sum().item() for k in range(args.num_clusters_per_category)]
            logger.info(f'Category {natural_category}: clusters sizes {cluster_sizes}')

        ext_all = torch.cat(all_features_list, dim=0)
        cluster_labels = torch.cat(all_cluster_labels_list, dim=0)
        category_labels = torch.cat(all_category_labels_list, dim=0)

        ext_mu_cluster, ext_cov_cluster = _compute_cluster_stats(ext_all, cluster_labels, args.num_clusters)

    if args.do_addtional_cluster or args.do_addtional_cluster_within_category:
        cluster_models = get_cluster_models(args, ext_all.cpu(), cluster_labels.cpu())

    return (ext_mu, ext_cov, ssh_mu, ssh_cov,
            mu_src_ext, cov_src_ext, mu_src_ssh, cov_src_ssh,
            ext_all, category_labels, cluster_labels, cluster_models,
            ext_mu_cluster, ext_cov_cluster)


def _log_cluster_sizes(cluster_labels, num_clusters):
    sizes = [(cluster_labels == k).sum().item() for k in range(num_clusters)]
    logger.info(f'Cluster sizes: {sizes}')


def _compute_cluster_stats(features, cluster_labels, num_clusters):
    """Compute per-cluster means and covariances."""
    mu_list = []
    cov_list = []
    for k in range(num_clusters):
        idx = cluster_labels == k
        feats_k = features[idx]
        if len(feats_k.shape) == 1:
            feats_k = feats_k.unsqueeze(0)
        mu_list.append(feats_k.mean(dim=0).cuda())
        cov_list.append(covariance(feats_k).cuda())
    return mu_list, cov_list