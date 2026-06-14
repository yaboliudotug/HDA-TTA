"""Cluster model training utilities for hierarchical alignment."""
import os
import copy
import random
import logging

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

from models.resnet import ClusterClassifier

logger = logging.getLogger(__name__)


class ClusterDataset(Dataset):
    """Dataset wrapper for cluster features and labels."""

    def __init__(self, features, labels):
        super().__init__()
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        feature = torch.from_numpy(self.features[index])
        label = torch.tensor(self.labels[index])
        return feature, label


def get_cluster_models(args, initial_features, initial_labels):
    """Train or load cluster assignment models.

    Args:
        args: Config with clustering parameters.
        initial_features: Numpy array of features (N, D).
        initial_labels: Tensor of initial cluster labels (N,).

    Returns:
        Cluster classifier model (or list of models, depending on mode).
    """
    load_stored = args.load_stored_cluster_models

    if args.cluster_model_mode == 'cls':
        cluster_model = train_cluster_classifier(
            args, initial_features, initial_labels,
            load_stored=load_stored, batch_size=2,
        )
        return cluster_model
    else:
        cluster_models = []
        for cat in range(args.num_clusters):
            model = train_cluster_classifier(
                args, initial_features, initial_labels,
                load_stored=load_stored, catch_category=cat, batch_size=2,
            )
            cluster_models.append(model)
        return cluster_models


def train_cluster_classifier(
    args, initial_features, initial_labels,
    load_stored=False, catch_category=3, batch_size=4,
    val_split=0.15,
):
    """Train a multi-class cluster classifier.

    Args:
        args: Config.
        initial_features: Numpy array of features (N, D).
        initial_labels: Tensor of cluster labels (N,).
        load_stored: Whether to load from cache.
        catch_category: For one-vs-all mode (unused in cls mode).
        batch_size: Training batch size.
        val_split: Fraction of data for validation.

    Returns:
        Trained cluster classifier.
    """
    model_path = os.path.join(args.outf, 'cluster_models', 'cluster_model_classification.pth')
    os.makedirs(os.path.join(args.outf, 'cluster_models'), exist_ok=True)

    if load_stored and os.path.exists(model_path):
        model = torch.load(model_path).cuda().eval()
        logger.info('Loaded cached cluster model')
        return model

    logger.info('Training cluster classifier...')
    cluster_classifier = ClusterClassifier(name=args.model, num_cluster=args.num_clusters)

    catched_features = initial_features
    catched_labels = initial_labels.long()
    num_val = int(catched_features.shape[0] * val_split)

    all_index = list(range(catched_features.shape[0]))
    random.shuffle(all_index)
    val_idx = all_index[:num_val]
    train_idx = all_index[num_val:]

    train_features = catched_features[train_idx].numpy()
    train_labels = catched_labels[train_idx].numpy()
    val_features = catched_features[val_idx].numpy()
    val_labels = catched_labels[val_idx].numpy()

    logger.info(f'Training data: {train_labels.shape[0]} samples, validation: {val_labels.shape[0]}')

    train_dataset = ClusterDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataset = ClusterDataset(val_features, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    ce_loss = nn.CrossEntropyLoss()
    cluster_classifier.cuda()
    optimizer = optim.SGD(cluster_classifier.parameters(), lr=0.01, momentum=0.9)

    best_model = None
    best_precision = 0.0

    for epoch in range(args.cluster_train_epochs):
        # Training
        cluster_classifier.train()
        for idx, (features, labels) in enumerate(train_loader):
            features, labels = features.cuda(), labels.cuda()
            optimizer.zero_grad()
            preds = cluster_classifier(features)
            loss = ce_loss(preds, labels)
            loss.backward()
            optimizer.step()

            if (idx + 1) % 1000 == 0:
                logger.info(f'Epoch {epoch} iter {idx + 1} loss {loss:.4f}')

        # Validation
        cluster_classifier.eval()
        all_correct = []
        all_preds = []
        all_gts = []
        for features, labels in val_loader:
            features, labels = features.cuda(), labels.cuda()
            preds = cluster_classifier(features)
            preds = torch.softmax(preds, dim=1)
            _, pred_label = preds.max(dim=1)
            all_correct.append(labels == pred_label)
            all_preds.extend(pred_label.cpu().detach().numpy())
            all_gts.extend(labels.cpu().detach().numpy())

        all_correct = torch.cat(all_correct, dim=0)
        precision = all_correct.sum().item() / len(all_correct)
        logger.info(f'Epoch {epoch} validation precision: {precision:.4f}')

        try:
            cm = confusion_matrix(all_gts, all_preds)
            logger.info(f'Confusion matrix:\n{cm}')
        except Exception:
            pass

        if best_precision < precision:
            best_precision = precision
            best_model = copy.deepcopy(cluster_classifier)

    logger.info(f'Best validation precision: {best_precision:.4f}')
    best_model.eval()
    torch.save(best_model, model_path)
    return best_model