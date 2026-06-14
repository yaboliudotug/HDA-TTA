from torch import nn


class ViewFlatten(nn.Module):
    """Flatten layer for use in sequential modules."""
    def __init__(self):
        super(ViewFlatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class ExtractorHead(nn.Module):
    """Combines a feature extractor and a head network."""

    def __init__(self, extractor, head):
        super(ExtractorHead, self).__init__()
        self.extractor = extractor
        self.head = head

    def forward(self, x):
        return self.head(self.extractor(x))