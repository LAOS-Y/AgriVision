import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
# from model.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from ..utils import build_norm_layer
'''
class IBN(nn.Module):
    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes/2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out
'''


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 norm_cfg=dict(type='BN'),
                 sw_cfg=None):
        super(BasicBlock, self).__init__()
        self.norm1_name, norm1 = build_norm_layer(
            sw_cfg if sw_cfg is not None else norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.add_module(self.norm2_name, norm2)
        self.downsample = downsample
        self.stride = stride

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 norm_cfg=dict(type='BN'),
                 sw_cfg=None):
        super(Bottleneck, self).__init__()
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            sw_cfg if sw_cfg is not None else norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * 4, postfix=3)

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.add_module(self.norm3_name, norm3)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(   self,
                    block,
                    layers,
                    output_stride,
                    num_classes = 1000,
                    norm_cfg=dict(type='BN', requires_grad=True),
                    sw_cfg=None,
                    stage_with_sw=(True, True, True, False)):
        self.inplanes = 64
        super(ResNet, self).__init__()
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        self.norm_cfg = norm_cfg
        self.sw_cfg = sw_cfg
        self.stage_with_sw = stage_with_sw
        self.norm1_name, norm1 = build_norm_layer(
            sw_cfg if sw_cfg is not None else norm_cfg, 64, postfix=1)

        # Modules
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.add_module(self.norm1_name, norm1)
        #self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], with_sw=stage_with_sw[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], with_sw=stage_with_sw[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], with_sw=stage_with_sw[0])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], with_sw=stage_with_sw[0])
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3])
        self.avgpool = nn.AvgPool2d(7)
        self._init_weight()
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)            

    def _make_layer(self, block, planes, blocks,  with_sw, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                build_norm_layer(self.norm_cfg, planes * block.expansion)[1],
            )

        layers = []
        layers.append(
            block(self.inplanes,
                  planes,
                  stride,
                  downsample,
                  norm_cfg=self.norm_cfg,
                  sw_cfg=None))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      norm_cfg=self.norm_cfg,
                      sw_cfg=self.sw_cfg if
                      (with_sw and i % 2 == 1) else None))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def _load_pretrained_model(model, url=None, path=None):
    if url is not None:
        pretrain_dict = model_zoo.load_url(url)
    else:
        pretrain_dict = torch.load(path)

    model_dict = {}
    state_dict = model.state_dict()
    for k, v in pretrain_dict.items():
        # print(k)
        if k in state_dict:
            model_dict[k] = v
        else:
            print(k)
    state_dict.update(model_dict)
    model.load_state_dict(state_dict)



def ResNet50(output_stride, ibn_mode='none', pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], output_stride, **kwargs)

    if pretrained:
        _load_pretrained_model(model, path='pretrained/resnet50_sw.pth')
    return model

def ResNet101(output_stride, ibn_mode='none', pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    assert ibn_mode in ['none', 'a']

    model = ResNet(Bottleneck, [3, 4, 23, 3], output_stride, **kwargs)

    if pretrained:
        _load_pretrained_model(model, path='pretrained/resnet101_sw.pth')

    return model


if __name__ == "__main__":
    model = ResNet50(output_stride=16, ibn_mode='a', pretrained=True)
    input = torch.rand(1, 3, 512, 512)
    output, low_level_feat = model(input)
    print(output.size())
    print(low_level_feat.size())