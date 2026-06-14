"""Hierarchical Test-Time Adaptation (HTTAC) for CIFAR-10-C.

Main entry point for the single-pass (N-O protocol) adaptation.
"""
import argparse
import copy
import logging
import math
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from offline import summarize_source_features
from utils.datasets import create_test_loader, create_train_loader, seed_worker
from utils.helpers import make_directory
from utils.losses import SupervisedContrastiveLoss
from utils.matching import reparameterize, FeatureMatcher
from utils.model_utils import build_tta_model, load_pretrained, softmax_entropy


def parse_args():
    parser = argparse.ArgumentParser(
        description='Hierarchical Test-Time Adaptation (HTTAC) on CIFAR-10-C'
    )

    # Data
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--dataroot', default='./data', help='root directory for datasets')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size for SSL training')
    parser.add_argument('--batch_size_align', default=512, type=int, help='batch size for alignment')
    parser.add_argument('--workers', default=0, type=int, help='dataloader workers')
    parser.add_argument('--num_sample', default=1000000, type=int, help='max samples to use')

    # Optimization
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--iters', default=4, type=int, help='number of TTA iterations per batch')
    parser.add_argument('--outf', default='.', help='output directory')

    # Corruption
    parser.add_argument('--level', default=5, type=int, help='corruption severity level')
    parser.add_argument('--corruption', default='snow', help='corruption type')

    # Model
    parser.add_argument('--resume', default=None, help='path to pretrained model checkpoint')
    parser.add_argument('--ckpt', default=None, type=int, help='checkpoint epoch number')
    parser.add_argument('--model', default='resnet50', help='backbone architecture')

    # Self-supervised learning
    parser.add_argument('--ssl', default='contrastive', help='self-supervised task')
    parser.add_argument('--temperature', default=0.5, type=float, help='contrastive loss temperature')
    parser.add_argument('--with_ssl', action='store_true', default=False, help='use SSL loss')
    parser.add_argument('--with_shot', action='store_true', default=False, help='use SHOT entropy loss')
    parser.add_argument('--without_global', action='store_true', default=False)
    parser.add_argument('--without_mixture', action='store_true', default=False)

    # Filtering
    parser.add_argument(
        '--filter', default='ours', choices=['ours', 'posterior', 'none'],
        help='pseudo-label filtering strategy'
    )

    # Alignment
    parser.add_argument('--align_ext', action='store_true', help='align extractor features')
    parser.add_argument('--align_ssh', action='store_true', help='align projection head features')
    parser.add_argument('--fix_ssh', action='store_true', help='freeze projection head')

    # Hierarchical clustering
    parser.add_argument('--do_addtional_cluster', action='store_true', default=False,
                        help='cluster all features')
    parser.add_argument('--do_addtional_cluster_within_category', action='store_true', default=False,
                        help='cluster features within each category')
    parser.add_argument('--without_category_kl_loss', action='store_true', default=False)
    parser.add_argument('--use_cluster_kl_loss', action='store_true', default=False)
    parser.add_argument('--weight_cluster_kl_loss', default=0.001, type=float)
    parser.add_argument('--use_cluster_entropy_loss', action='store_true', default=False)
    parser.add_argument('--use_category_matching_loss', action='store_true', default=False)
    parser.add_argument('--use_cluster_matching_loss', action='store_true', default=False)
    parser.add_argument('--use_feature_matching_loss', action='store_true', default=False)
    parser.add_argument('--random_around_feature', action='store_true', default=False)
    parser.add_argument('--filter_with_cluster_within_category', action='store_true', default=False)
    parser.add_argument('--load_stored_features', action='store_true', default=False)
    parser.add_argument('--load_stored_cluster_models', action='store_true', default=False)

    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--num_all_clusters', default=50, type=int)
    parser.add_argument('--num_clusters_per_category', default=3, type=int)
    parser.add_argument('--num_samples_per_cluster', default=20, type=int)
    parser.add_argument('--cluster_train_epochs', default=5, type=int)
    parser.add_argument('--cluster_method', default='faiss')
    parser.add_argument('--cluster_model_mode', default='cls')

    parser.add_argument('--mather_lr', default=0.00025, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--marker', default='')

    return parser.parse_args()


def setup_logging(args):
    """Configure logging to file and console."""
    log_level = logging.DEBUG
    logger = logging.getLogger('main')
    logger.setLevel(log_level)

    # BUG: logs/ subdirectory doesn't exist — FileHandler will fail at startup
    fh = logging.FileHandler(os.path.join(args.outf, 'logs', 'tta.log'))
    fh.setLevel(log_level)

    ch = logging.StreamHandler()
    ch.setLevel(log_level)

    formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def configure_seed(seed: int):
    """Set deterministic seeds."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = True


def build_optimizer(args, ext, ssh, matcher, matcher_cluster, matcher_feature):
    """Build optimizer based on which modules are being trained.

    When fix_ssh is True, only the feature extractor (ext) is trained.
    When fix_ssh is False, the projection head (ssh) and optional matchers are trained.
    """
    if args.fix_ssh:
        if args.use_category_matching_loss:
            return optim.SGD(
                [{'params': ext.parameters()}, {'params': matcher.parameters(), 'lr': args.lr}],
                lr=args.lr, momentum=0.9
            )
        else:
            return optim.SGD(ext.parameters(), lr=args.lr, momentum=0.9)
    else:
        params = [{'params': ssh.parameters()}]
        if args.use_category_matching_loss:
            params.append({'params': matcher.parameters()})
        if args.use_cluster_matching_loss:
            params.append({'params': matcher_cluster.parameters()})
        if args.use_feature_matching_loss:
            params.append({'params': matcher_feature.parameters()})
        return optim.SGD(params, lr=args.lr, momentum=0.9)


def filter_pseudo_labels(feat_ext, feat_ssh, sample_alpha, sample_ema_logit,
                         pro, pseudo_label, filter_type, logger):
    """Filter pseudo-labels based on confidence and EMA consistency."""
    if filter_type == 'ours':
        mask = (sample_alpha == 1) & (pro > 0.9)
        return feat_ext[mask], feat_ssh[mask], pseudo_label[mask].cuda(), mask
    elif filter_type == 'none':
        return feat_ext, feat_ssh, pseudo_label.cuda(), torch.ones_like(pseudo_label, dtype=torch.bool)
    elif filter_type == 'posterior':
        raise NotImplementedError('Posterior filter not yet fully implemented')
    else:
        raise ValueError(f'Unknown filter type: {filter_type}')


def compute_cluster_assignments(feat_ext, cluster_models, args):
    """Compute cluster assignments using the cluster models."""
    if args.cluster_model_mode == 'cls':
        cluster_scores = cluster_models(feat_ext)
        cluster_label = cluster_scores.max(dim=1)[1]
    else:
        cluster_scores_list = [torch.sigmoid(m(feat_ext)) for m in cluster_models]
        cluster_scores = torch.cat(cluster_scores_list, dim=1)
        cluster_label = cluster_scores.max(dim=1)[1]
    return cluster_scores, cluster_label


def update_ema_distribution(ema_mu, ema_cov, ema_n, features, labels,
                            num_classes, ema_length, template_cov):
    """Update EMA-based per-class Gaussian distribution."""
    b, d = features.shape
    # Accumulate features by class
    features_by_class = torch.zeros(num_classes, b, d).cuda()
    features_by_class.scatter_add_(
        dim=0,
        index=labels[None, :, None].expand(-1, -1, d),
        src=features[None, :, :]
    )
    counts = torch.zeros(num_classes, b, dtype=torch.int).cuda()
    counts.scatter_add_(
        dim=0,
        index=labels[None, :],
        src=torch.ones_like(labels[None, :], dtype=torch.int)
    )
    ema_n += counts.sum(dim=1)
    alpha = torch.where(
        ema_n > ema_length,
        torch.ones(num_classes, dtype=torch.float).cuda() / ema_length,
        1.0 / (ema_n + 1e-10)
    )
    delta_pre = (features_by_class - ema_mu[:, None, :]) * counts[:, :, None]
    delta = alpha[:, None] * delta_pre.sum(dim=1)
    new_mean = ema_mu + delta
    new_cov = (
        ema_cov
        + alpha[:, None, None] * (
            (delta_pre.permute(0, 2, 1) @ delta_pre)
            - counts.sum(dim=1)[:, None, None] * ema_cov
        )
        - delta[:, :, None] @ delta[:, None, :]
    )
    return new_mean.detach(), new_cov.detach()


def update_global_distribution(ema_total_mu, ema_total_cov, ema_total_n,
                                features, alpha_min=1280):
    """Update EMA-based global Gaussian distribution."""
    b = features.shape[0]
    ema_total_n += b
    alpha = 1.0 / alpha_min if ema_total_n > alpha_min else 1.0 / ema_total_n
    delta_pre = features - ema_total_mu.cuda()
    delta = alpha * delta_pre.sum(dim=0)
    new_mu = ema_total_mu.cuda() + delta
    new_cov = (
        ema_total_cov.cuda()
        + alpha * (delta_pre.t() @ delta_pre - b * ema_total_cov.cuda())
        - delta[:, None] @ delta[None, :]
    )
    return new_mu.detach().cpu(), new_cov.detach().cpu(), ema_total_n


def main():
    args = parse_args()

    # Determine number of clusters
    args.num_clusters = args.num_classes
    if args.do_addtional_cluster:
        args.num_clusters = args.num_all_clusters
    elif args.do_addtional_cluster_within_category:
        args.num_clusters = args.num_classes * args.num_clusters_per_category

    # Setup output directory
    make_directory(args.outf)

    # Logging
    logger = setup_logging(args)
    logger.info(f'Arguments: {args}')

    # Seed
    configure_seed(args.seed)

    # Build models
    net, ext, head, ssh, classifier = build_tta_model(args)
    _, test_loader = create_test_loader(args)

    # Feature matchers
    matcher = FeatureMatcher(args.num_classes).cuda() if args.use_category_matching_loss else None
    matcher_cluster = FeatureMatcher(args.num_clusters).cuda() if args.use_cluster_matching_loss else None
    matcher_feature = FeatureMatcher(args.num_clusters).cuda() if args.use_feature_matching_loss else None

    # Data
    args.batch_size = min(args.batch_size, args.num_sample)
    args.batch_size_align = min(args.batch_size_align, args.num_sample)

    align_args = copy.deepcopy(args)
    align_args.ssl = None
    align_args.batch_size = args.batch_size_align

    train_dataset, _ = create_train_loader(args, args.num_sample)
    train_extra_dataset, _ = create_test_loader(align_args, num_sample=args.num_sample, shuffle=True)

    # Load pretrained model
    logger.info(f'Loading checkpoint from {args.resume}')
    load_pretrained(net, head, ssh, classifier, args)

    if torch.cuda.device_count() > 1:
        ext = torch.nn.DataParallel(ext)

    # Loss
    contrastive_criterion = SupervisedContrastiveLoss(temperature=args.temperature).cuda()

    # Optimizer
    optimizer = build_optimizer(args, ext, ssh, matcher, matcher_cluster, matcher_feature)

    # ---- Offline feature summarization ----
    _, offline_loader = create_train_loader(align_args)
    source_stats = summarize_source_features(
        args, offline_loader, ext, classifier, head, num_classes=args.num_classes
    )
    (ext_src_mu, ext_src_cov, ssh_src_mu, ssh_src_cov,
     mu_src_ext, cov_src_ext, mu_src_ssh, cov_src_ssh,
     features_source, category_labels, cluster_labels, cluster_models,
     ext_mu_cluster, ext_cov_cluster) = source_stats

    bias = cov_src_ext.max().item() / 30.0
    bias2 = cov_src_ssh.max().item() / 30.0
    template_ext_cov = torch.eye(2048).cuda() * bias
    template_ssh_cov = torch.eye(128).cuda() * bias2
    bias_cluster = cov_src_ext.max().item() / 30.0
    template_ext_cov_cluster = torch.eye(2048).cuda() * bias_cluster

    # ---- Initialize distributions ----
    ext_src_mu = torch.stack(ext_src_mu)
    ext_src_cov = torch.stack(ext_src_cov) + template_ext_cov[None, :, :]

    if ext_mu_cluster is None:
        ext_src_mu_cluster = ext_src_mu.clone()
        ext_src_cov_cluster = ext_src_cov.clone()
    else:
        ext_src_mu_cluster = torch.stack(ext_mu_cluster)
        ext_src_cov_cluster = torch.stack(ext_cov_cluster) + template_ext_cov_cluster[None, :, :]

    source_distribution = torch.distributions.MultivariateNormal(ext_src_mu, ext_src_cov)
    target_distribution = torch.distributions.MultivariateNormal(ext_src_mu, ext_src_cov)
    source_cluster_distribution = torch.distributions.MultivariateNormal(ext_src_mu_cluster, ext_src_cov_cluster)
    target_cluster_distribution = torch.distributions.MultivariateNormal(ext_src_mu_cluster, ext_src_cov_cluster)

    # ---- EMA tracking ----
    sample_predict_ema_logit = torch.zeros(len(train_dataset), args.num_classes, dtype=torch.float)
    sample_predict_alpha = torch.ones(len(train_dataset), dtype=torch.float)
    ema_alpha = 0.9

    ema_n = torch.zeros(args.num_classes).cuda()
    ema_ext_mu = ext_src_mu.clone()
    ema_ext_cov = ext_src_cov.clone()
    ema_n_cluster = torch.zeros(args.num_clusters).cuda()
    ema_ext_mu_cluster = ext_src_mu_cluster.clone()
    ema_ext_cov_cluster = ext_src_cov_cluster.clone()

    ema_ext_total_mu = torch.zeros(2048).float()
    ema_ext_total_cov = torch.zeros(2048, 2048).float()
    ema_ssh_total_mu = torch.zeros(128).float()
    ema_ssh_total_cov = torch.zeros(128, 128).float()
    ema_total_n = 0.0

    ema_length = 128
    mini_batch_length = 4096
    loss_scale = 0.05

    mini_batch_indices = []
    all_correct = []

    # ---- Main adaptation loop ----
    for batch_idx, (te_inputs, te_labels) in enumerate(test_loader):
        mini_batch_indices.extend(te_inputs[-1].tolist())
        mini_batch_indices = mini_batch_indices[-mini_batch_length:]

        # Create mini-batch data loaders from sliding window
        try:
            del train_dataset_subset, train_loader, train_extra_subset, train_extra_loader
        except NameError:
            pass

        train_dataset_subset = data.Subset(train_dataset, mini_batch_indices)
        train_loader = data.DataLoader(
            train_dataset_subset, batch_size=args.batch_size,
            shuffle=True, num_workers=args.workers,
            worker_init_fn=seed_worker, pin_memory=True, drop_last=True
        )
        train_extra_subset = data.Subset(train_extra_dataset, mini_batch_indices)
        train_extra_loader = data.DataLoader(
            train_extra_subset, batch_size=args.batch_size_align,
            shuffle=True, num_workers=args.workers,
            worker_init_fn=seed_worker, pin_memory=True, drop_last=False
        )
        train_extra_iter = iter(train_extra_loader)

        # Set training mode
        if args.fix_ssh:
            head.eval()
        else:
            head.train()
        ext.train()
        classifier.eval()

        for iter_id in range(min(args.iters, len(mini_batch_indices) // 256 + 1) + 1):
            if iter_id > 0:
                sample_predict_alpha = torch.where(
                    sample_predict_alpha < 1,
                    sample_predict_alpha + 0.2,
                    torch.ones_like(sample_predict_alpha)
                )

            for batch_data in train_loader:
                optimizer.zero_grad()

                # ---- Self-supervised loss ----
                if args.with_ssl:
                    images = torch.cat([batch_data[0], batch_data[1]], dim=0).cuda(non_blocking=True)
                    bsz = batch_data[1].shape[0]
                    backbone_features = ext(images)
                    features = F.normalize(head(backbone_features), dim=1)
                    f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    loss = contrastive_criterion(features)
                    loss.backward()
                    del loss

                # ---- Alignment (iter_id > 0) ----
                if iter_id > 0:
                    loss = 0.0
                    matching_loss = 0.0
                    kl_loss_val = 0.0
                    loss_message = ''

                    try:
                        inputs, labels = next(train_extra_iter)
                    except StopIteration:
                        train_extra_iter = iter(train_extra_loader)
                        inputs, labels = next(train_extra_iter)

                    inputs, indexes = inputs
                    inputs = inputs.cuda()

                    feat_ext = ext(inputs)
                    logit = classifier(feat_ext)
                    feat_ssh = head(feat_ext)

                    # Pseudo-label generation with EMA
                    with torch.no_grad():
                        ext.eval()
                        predict_logit = net(inputs)
                        softmax_logit = predict_logit.softmax(dim=1).cpu()

                        old_logit = sample_predict_ema_logit[indexes, :]
                        max_val, max_pos = softmax_logit.max(dim=1)
                        old_max_val = old_logit[torch.arange(max_pos.shape[0]), max_pos]
                        accept_mask = max_val > (old_max_val - 0.001)

                        sample_predict_alpha[indexes] = torch.where(
                            accept_mask,
                            sample_predict_alpha[indexes],
                            torch.zeros_like(accept_mask).float()
                        )

                        sample_predict_ema_logit[indexes, :] = torch.where(
                            sample_predict_ema_logit[indexes, :] == torch.zeros(args.num_classes),
                            softmax_logit,
                            (1 - ema_alpha) * sample_predict_ema_logit[indexes, :] + ema_alpha * softmax_logit
                        )

                        pro, pseudo_label = sample_predict_ema_logit[indexes].max(dim=1)
                        ext.train()
                        del predict_logit

                    # Filter pseudo-labels
                    feat_ext_filtered, feat_ssh_filtered, pseudo_label_filtered, mask = \
                        filter_pseudo_labels(
                            feat_ext, feat_ssh, sample_predict_alpha[indexes],
                            sample_predict_ema_logit[indexes], pro, pseudo_label,
                            args.filter, logger
                        )

                    # ---- Cluster assignments ----
                    has_clustering = args.do_addtional_cluster or args.do_addtional_cluster_within_category
                    if has_clustering:
                        cluster_scores, cluster_label = compute_cluster_assignments(
                            feat_ext, cluster_models, args
                        )
                        if args.do_addtional_cluster_within_category:
                            pseudo_label_from_cluster = cluster_label // args.num_clusters_per_category

                        cluster_label = cluster_label[mask]
                        cluster_scores = cluster_scores[mask]

                        if args.do_addtional_cluster_within_category:
                            pseudo_label_from_cluster = pseudo_label_from_cluster[mask]

                            if args.filter_with_cluster_within_category:
                                filter_idx = pseudo_label_filtered == pseudo_label_from_cluster
                                feat_ext_filtered = feat_ext_filtered[filter_idx]
                                feat_ssh_filtered = feat_ssh_filtered[filter_idx]
                                pseudo_label_filtered = pseudo_label_filtered[filter_idx]
                                cluster_scores = cluster_scores[filter_idx]
                                cluster_label = cluster_label[filter_idx]
                                pseudo_label_from_cluster = pseudo_label_from_cluster[filter_idx]

                        # Cluster entropy loss
                        if args.use_cluster_entropy_loss:
                            cluster_entropy = softmax_entropy(cluster_scores).mean(0)
                            loss = loss + cluster_entropy
                            loss_message += f' ClusterEntropy {cluster_entropy:.2f}'

                    # ---- Extractor distribution alignment ----
                    if args.align_ext:
                        if not args.without_mixture:
                            # Cluster-level alignment
                            if has_clustering:
                                new_mean, new_cov = update_ema_distribution(
                                    ema_ext_mu_cluster, ema_ext_cov_cluster, ema_n_cluster,
                                    feat_ext_filtered, cluster_label,
                                    args.num_clusters, ema_length, template_ext_cov_cluster
                                )
                                ema_ext_mu_cluster = new_mean
                                ema_ext_cov_cluster = new_cov

                                if (args.num_classes == 10 or len(mini_batch_indices) >= 4096) and \
                                   (iter_id > args.iters // 2 or args.filter == 'none'):
                                    target_cluster_distribution.loc = new_mean
                                    target_cluster_distribution.covariance_matrix = new_cov + template_ext_cov_cluster
                                    target_cluster_distribution._unbroadcasted_scale_tril = \
                                        torch.linalg.cholesky(new_cov + template_ext_cov_cluster)
                                    cluster_kl = torch.distributions.kl_divergence(
                                        source_cluster_distribution, target_cluster_distribution
                                    ) + torch.distributions.kl_divergence(
                                        target_cluster_distribution, source_cluster_distribution
                                    )
                                    cluster_kl = cluster_kl.mean() * args.weight_cluster_kl_loss
                                    loss = loss + cluster_kl
                                    loss_message += f' ClusterKL {cluster_kl:.2f}'

                            # Category-level alignment
                            if not args.without_category_kl_loss:
                                new_mean, new_cov = update_ema_distribution(
                                    ema_ext_mu, ema_ext_cov, ema_n,
                                    feat_ext_filtered, pseudo_label_filtered,
                                    args.num_classes, ema_length, template_ext_cov
                                )
                                ema_ext_mu = new_mean
                                ema_ext_cov = new_cov

                                if (args.num_classes == 10 or len(mini_batch_indices) >= 4096) and \
                                   (iter_id > args.iters // 2 or args.filter == 'none'):
                                    target_distribution.loc = new_mean
                                    target_distribution.covariance_matrix = new_cov + template_ext_cov
                                    target_distribution._unbroadcasted_scale_tril = \
                                        torch.linalg.cholesky(new_cov + template_ext_cov)
                                    category_kl = (torch.distributions.kl_divergence(
                                        source_distribution, target_distribution
                                    ) + torch.distributions.kl_divergence(
                                        target_distribution, source_distribution
                                    )).mean() * loss_scale
                                    loss = loss + category_kl
                                    loss_message += f' CategoryKL {category_kl:.2f}'

                            # Feature matching loss
                            if args.use_feature_matching_loss and \
                               (args.num_classes == 10 or len(mini_batch_indices) >= 4096) and \
                               (iter_id > args.iters // 2 or args.filter == 'none'):
                                src_per_category = []
                                tgt_per_category = []
                                src_labels = []
                                tgt_labels = []

                                for c in category_labels.unique():
                                    sr_idx = category_labels == c
                                    tg_idx = pseudo_label_filtered == c
                                    src_nodes = features_source[sr_idx]
                                    tgt_nodes = feat_ext_filtered[tg_idx]

                                    if len(src_nodes) > 0:
                                        max_len = min(args.num_samples_per_cluster, len(src_nodes))
                                        src_nodes = src_nodes[:max_len]
                                    if len(tgt_nodes) > 0:
                                        max_len = min(args.num_samples_per_cluster, len(tgt_nodes))
                                        tgt_nodes = tgt_nodes[:max_len]

                                    if len(src_nodes) > 0 and len(tgt_nodes) > 0:
                                        if args.random_around_feature:
                                            src_nodes = torch.normal(0, 0.01, size=src_nodes.size()).cuda() + src_nodes
                                            tgt_nodes = torch.normal(0, 0.01, size=tgt_nodes.size()).cuda() + tgt_nodes

                                        src_per_category.append(src_nodes)
                                        tgt_per_category.append(tgt_nodes)
                                        src_labels.append(src_nodes.new_ones(len(src_nodes)) * c)
                                        tgt_labels.append(tgt_nodes.new_ones(len(tgt_nodes)) * c)

                                    elif len(src_nodes) > 0:
                                        if args.random_around_feature:
                                            src_nodes = torch.normal(0, 0.01, size=src_nodes.size()).cuda() + src_nodes
                                        src_per_category.append(src_nodes)

                                        tgt_samples = [
                                            reparameterize(
                                                target_cluster_distribution.mean,
                                                target_cluster_distribution.variance
                                            ) for _ in range(args.num_samples_per_cluster)
                                        ]
                                        tgt_all = torch.stack(tgt_samples)
                                        tgt_c = tgt_all[:, c.long(), :].reshape(-1, 2048)
                                        tgt_per_category.append(tgt_c)

                                        src_labels.append(torch.ones(len(src_nodes), dtype=torch.float).cuda() * c)
                                        tgt_labels.append(torch.ones(len(tgt_c), dtype=torch.float).cuda() * c)

                                if src_per_category:
                                    nodes_sr = torch.cat(src_per_category, dim=0)
                                    nodes_tg = torch.cat(tgt_per_category, dim=0)
                                    labels_sr = torch.cat(src_labels, dim=0)
                                    labels_tg = torch.cat(tgt_labels, dim=0)

                                    fm_loss = matcher_feature(
                                        nodes_sr, nodes_tg, labels_sr, labels_tg,
                                        matching_cfg='m2m'
                                    )
                                    fm_loss = fm_loss * args.num_samples_per_cluster
                                    loss = loss + fm_loss
                                    loss_message += f' FeatureMatch {fm_loss:.2f}'

                            # Category matching loss
                            if args.use_category_matching_loss and \
                               (args.num_classes == 10 or len(mini_batch_indices) >= 4096) and \
                               (iter_id > args.iters // 2 or args.filter == 'none'):
                                d = feat_ext_filtered.shape[1]
                                source_samples = torch.stack([
                                    reparameterize(source_distribution.mean, source_distribution.variance)
                                    for _ in range(args.num_samples_per_cluster)
                                ]).permute(1, 0, 2).reshape(-1, d)
                                target_samples = torch.stack([
                                    reparameterize(target_distribution.mean, target_distribution.variance)
                                    for _ in range(args.num_samples_per_cluster)
                                ]).permute(1, 0, 2).reshape(-1, d)

                                matching_labels = torch.tensor(
                                    [k for k in range(args.num_classes) for _ in range(args.num_samples_per_cluster)]
                                )
                                cat_match_loss = matcher(
                                    source_samples, target_samples,
                                    matching_labels, matching_labels,
                                    matching_cfg='m2m'
                                ) * args.num_samples_per_cluster
                                loss = loss + cat_match_loss
                                loss_message += f' CategoryMatch {cat_match_loss:.2f}'

                            # Cluster matching loss
                            if args.use_cluster_matching_loss and \
                               (args.num_classes == 10 or len(mini_batch_indices) >= 4096) and \
                               (iter_id > args.iters // 2 or args.filter == 'none'):
                                d = feat_ext_filtered.shape[1]
                                source_samples = torch.stack([
                                    reparameterize(source_cluster_distribution.mean, source_cluster_distribution.variance)
                                    for _ in range(args.num_samples_per_cluster)
                                ]).permute(1, 0, 2).reshape(-1, d)
                                target_samples = torch.stack([
                                    reparameterize(target_cluster_distribution.mean, target_cluster_distribution.variance)
                                    for _ in range(args.num_samples_per_cluster)
                                ]).permute(1, 0, 2).reshape(-1, d)

                                matching_labels = torch.tensor(
                                    [k for k in range(args.num_clusters)
                                     for _ in range(args.num_samples_per_cluster)]
                                )
                                cl_match_loss = matcher_cluster(
                                    source_samples, target_samples,
                                    matching_labels, matching_labels,
                                    matching_cfg='m2m'
                                ) * args.num_samples_per_cluster
                                loss = loss + cl_match_loss
                                loss_message += f' ClusterMatch {cl_match_loss:.2f}'

                        if not args.without_global:
                            new_mu, new_cov, ema_total_n = update_global_distribution(
                                ema_ext_total_mu, ema_ext_total_cov, ema_total_n,
                                feat_ext, alpha_min=1280
                            )
                            ema_ext_total_mu = new_mu
                            ema_ext_total_cov = new_cov

                            source_domain = torch.distributions.MultivariateNormal(
                                mu_src_ext, cov_src_ext + template_ext_cov
                            )
                            target_domain = torch.distributions.MultivariateNormal(
                                new_mu, new_cov + template_ext_cov
                            )
                            global_kl = (torch.distributions.kl_divergence(source_domain, target_domain)
                                         + torch.distributions.kl_divergence(target_domain, source_domain)) * loss_scale
                            loss = loss + global_kl
                            loss_message += f' GlobalKL {global_kl:.2f}'

                        if args.without_mixture and args.without_global:
                            logit_filtered = logit[mask.cuda()]
                            loss += F.cross_entropy(logit_filtered, pseudo_label_filtered) * loss_scale * 2

                    # ---- Projection head alignment ----
                    if args.align_ssh:
                        new_mu, new_cov, ema_total_n = update_global_distribution(
                            ema_ssh_total_mu, ema_ssh_total_cov, ema_total_n,
                            feat_ssh, alpha_min=1280
                        )
                        ema_ssh_total_mu = new_mu
                        ema_ssh_total_cov = new_cov

                        source_domain = torch.distributions.MultivariateNormal(
                            mu_src_ssh, cov_src_ssh + template_ssh_cov
                        )
                        target_domain = torch.distributions.MultivariateNormal(
                            new_mu, new_cov + template_ssh_cov
                        )
                        loss += (torch.distributions.kl_divergence(source_domain, target_domain)
                                 + torch.distributions.kl_divergence(target_domain, source_domain)) * loss_scale

                    # ---- SHOT entropy loss ----
                    if args.with_shot:
                        ent_loss = softmax_entropy(logit).mean(0)
                        softmax_out = F.softmax(logit, dim=-1)
                        msoftmax = softmax_out.mean(dim=0)
                        ent_loss += torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
                        loss += ent_loss * loss_scale * 2

                    logger.info(
                        f'TTA iter {iter_id} batch {batch_idx}:{loss_message} Total:{loss:.4f}'
                    )

                    # Backward
                    try:
                        loss.backward()
                    except Exception as e:
                        logger.warning(f'Backward pass failed: {e}')
                    finally:
                        del loss, matching_loss

            if iter_id > 0:
                optimizer.step()
                optimizer.zero_grad()

        # ---- Evaluation ----
        net.eval()
        with torch.no_grad():
            outputs = net(te_inputs[0].cuda())
            _, predicted = outputs.max(1)
            all_correct.append(predicted.cpu().eq(te_labels))
        accuracy = 1 - torch.cat(all_correct).numpy().mean()
        logger.info(f'Batch {batch_idx} error rate: {accuracy:.4f}')
        net.train()

    final_error = 1 - torch.cat(all_correct).numpy().mean()
    logger.info(f'{args.corruption} Test-time adaptation error: {final_error:.4f}')
    print(f'Final error rate: {final_error:.4f}')


if __name__ == '__main__':
    main()