from torch import nn
from configs import config


class ResNetBlock3d(nn.Module):
    def __init__(self, in_features, out_features, stride=1, downsample=None):
        super().__init__()

        self.conv1 = nn.Conv3d(in_features, out_features, 3, stride, 1, bias=False)
        self.bn1 = nn.InstanceNorm3d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=config.TRAIN.DROPOUT)

        self.conv2 = nn.Conv3d(out_features, out_features, 3, 1, 1, bias=False)
        self.bn2 = nn.InstanceNorm3d(out_features)
        self.dropout2 = nn.Dropout(p=config.TRAIN.DROPOUT)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        out = self.dropout2(out)

        return out


class ResNetDownsample(nn.Module):
    def __init__(self, in_features, out_features, stride=1):
        super().__init__()
        self.conv = nn.Conv3d(in_features, out_features, 1, stride, bias=False)
        self.norm = nn.InstanceNorm3d(out_features)

    def forward(self, x):
        # return self.conv(x)
        return self.norm(self.conv(x))


class ResNetBlockTranspose3d(nn.Module):
    def __init__(self, in_features, out_features, stride=1, upsample=None):
        super().__init__()

        self.conv1 = nn.ConvTranspose3d(in_features, out_features, 3, stride, 1, 1, bias=False)
        self.bn1 = nn.InstanceNorm3d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=config.TRAIN.DROPOUT)

        self.conv2 = nn.Conv3d(out_features, out_features, 3, 1, 1, bias=False)
        self.bn2 = nn.InstanceNorm3d(out_features)
        self.dropout2 = nn.Dropout(p=config.TRAIN.DROPOUT)

        self.upsample = upsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)


        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)
        # out = self.dropout2(out)

        return out


class ResNetTranspose(nn.Module):
    def __init__(self, in_features, out_features, stride=1):
        super().__init__()
        self.conv = nn.ConvTranspose3d(in_features, out_features, 3, stride, 1, 1, bias=False)

    def forward(self, x):
        return self.conv(x)
