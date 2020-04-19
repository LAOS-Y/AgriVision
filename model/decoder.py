import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm, num_low_level_feat, interpolate_before_lastconv):
        super(Decoder, self).__init__()
        if backbone == 'resnet50' or backbone == 'resnet101' or backbone == 'drn' or backbone == 'sw-resnet101':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        elif backbone == 'efficientnet-b0':
            low_level_inplanes = 16   
        elif backbone == 'efficientnet-b6':
            low_level_inplanes = 32 
        else:
            raise NotImplementedError

        self.num_low_level_feat = num_low_level_feat
        self.interpolate_before_lastconv = interpolate_before_lastconv

        self.bottlenecks = nn.ModuleList()

        for i in range(self.num_low_level_feat):
            self.bottlenecks.append(nn.Sequential(
                nn.Conv2d(low_level_inplanes * 2**i, 48, 1, bias=False),
                BatchNorm(48),
                nn.ReLU()))

        if not self.interpolate_before_lastconv:
            self.last_conv = nn.Sequential(nn.Conv2d(256 + 48 * self.num_low_level_feat, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(256),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(256),
                                        nn.ReLU(),
                                        nn.Dropout(0.1),
                                        nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        else:
            self.last_conv = nn.Sequential(nn.Conv2d(256 + 48 * self.num_low_level_feat, 256, kernel_size=1, stride=1, padding=1, bias=False),
                                           BatchNorm(256),
                                           nn.ReLU(),
                                           nn.Dropout(0.1),
                                           nn.UpsamplingBilinear2d(scale_factor=4),
                                           nn.Conv2d(256, num_classes, kernel_size=3, stride=1))
        self._init_weight()


    def forward(self, x, low_level_feats):
        low_level_feats = low_level_feats[:self.num_low_level_feat]

        for i, bottleneck in enumerate(self.bottlenecks):
            low_level_feats[i] = bottleneck(low_level_feats[i])
            low_level_feats[i] = F.interpolate(low_level_feats[i], size=low_level_feats[0].size()[2:], mode='bilinear', align_corners=True)

        x = F.interpolate(x, size=low_level_feats[0].size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, *low_level_feats), dim=1)

        if self.interpolate_before_lastconv:
            x = F.interpolate(x, scale_factor=(4, 4), mode='bilinear', align_corners=True)

        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(num_classes, backbone, BatchNorm, num_low_level_feat, interpolate_before_lastconv):
    return Decoder(num_classes, backbone, BatchNorm, num_low_level_feat, interpolate_before_lastconv)