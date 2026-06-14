"""Model building and checkpoint loading utilities."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_tta_model(args):
    """Build the TTA model: ResNet50 + projection head + linear classifier.

    Returns:
        net: Full model (extractor + classifier).
        extractor: Feature extractor (encoder).
        projection_head: projection head network.
        ssl_model: Full SSL model (encoder + projection head).
        classifier: Linear classifier.
    """
    from models.resnet import ContrastiveHead, LinearClassifier
    from models.heads import ExtractorHead

    num_classes = 10
    classifier = LinearClassifier(num_classes=num_classes).cuda()
    ssl_model = ContrastiveHead().cuda()
    projection_head = ssl_model.head
    extractor = ssl_model.encoder
    net = ExtractorHead(extractor, classifier).cuda()
    return net, extractor, projection_head, ssl_model, classifier


def load_pretrained(net, projection_head, ssl_model, classifier, args):
    """Load pretrained checkpoint for ResNet50 jointly trained on classification and SimCLR."""
    if args.checkpoint_step:
        filename = f'{args.resume}/ckpt_epoch_{args.checkpoint_step}.pth'
    else:
        filename = f'{args.resume}/ckpt.pth'
    checkpoint = torch.load(filename)
    state_dict = checkpoint['model']

    net_dict = {}
    head_dict = {}
    for k, v in state_dict.items():
        if k[:4] == 'head':
            k = k.replace('head.', '')
            head_dict[k] = v
        else:
            k = k.replace('encoder.', 'ext.')
            k = k.replace('fc.', 'head.fc.')
            net_dict[k] = v

    net.load_state_dict(net_dict)
    projection_head.load_state_dict(head_dict)
    print(f'Loaded model from {filename}')


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)