""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from torch.nn import functional as F


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, num_output_features=64, num_bottleneck_features=64, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, num_output_features))
        self.down1 = (Down(num_output_features, num_output_features*2))
        self.down2 = (Down(num_output_features*2, num_output_features*4))
        self.down3 = (Down(num_output_features*4, num_output_features*8))
        factor = 2 if bilinear else 1
        self.down4 = (Down(num_output_features*8, num_output_features*16 // factor))
        self.up1 = (Up(num_output_features*16, num_output_features*8 // factor, bilinear))
        self.up2 = (Up(num_output_features*8, num_output_features*4 // factor, bilinear))
        self.up3 = (Up(num_output_features*4, num_output_features*2 // factor, bilinear))
        self.up4 = (Up(num_output_features*2, num_output_features, bilinear))
        self.outc = (OutConv(num_output_features, n_classes))
        self.softmax = nn.Softmax(dim=1)
        self.criterion = F.cross_entropy  #nn.CrossEntropyLoss(reduction="none")


    def forward(self, x, gt=None, weights=None):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        if gt is not None:
            loss = self.criterion(logits, gt.long(), weight=weights, reduction="mean", ignore_index=255)
        else:
            loss = None
        return logits, loss, x1, x2, x3, x4, x5
    
    def inference(self,x,only_features=False):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if only_features:
            return None, x1, x2, x3, x4, x5
        else:
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            logits = self.outc(x)
            out = self.softmax(logits)
            out = torch.argmax(out,dim=1)
            return out, x1, x2, x3, x4, x5

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


# """ Full assembly of the parts to form the complete network """

# from .unet_parts import *


# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, num_output_features=64, num_bottleneck_features=64, bilinear=False):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         self.inc = (DoubleConv(n_channels, num_output_features))
#         self.down1 = (Down(num_output_features, num_output_features*2))
#         self.down2 = (Down(num_output_features*2, num_output_features*4))
#         factor = 2 if bilinear else 1
#         self.down3 = (Down(num_output_features*4, num_output_features*8 // factor))
#         self.up1 = (Up(num_output_features*8, num_output_features*4 // factor, bilinear))
#         self.up2 = (Up(num_output_features*4, num_output_features*2 // factor, bilinear))
#         self.up3 = (Up(num_output_features*2, num_output_features, bilinear))
#         self.outc = (OutConv(num_output_features, n_classes))

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x = self.up1(x4, x3)
#         x = self.up2(x, x2)
#         x = self.up3(x, x1)
#         logits = self.outc(x)
#         return logits, x1, x2, x3, x4

#     def use_checkpointing(self):
#         self.inc = torch.utils.checkpoint(self.inc)
#         self.down1 = torch.utils.checkpoint(self.down1)
#         self.down2 = torch.utils.checkpoint(self.down2)
#         self.down3 = torch.utils.checkpoint(self.down3)
#         self.up1 = torch.utils.checkpoint(self.up1)
#         self.up2 = torch.utils.checkpoint(self.up2)
#         self.up3 = torch.utils.checkpoint(self.up3)
#         self.outc = torch.utils.checkpoint(self.outc)
