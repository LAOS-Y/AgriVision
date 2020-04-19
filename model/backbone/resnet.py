import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
# from model.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from .switchable_norm import SwitchNorm2d

TRACK_FEAT = False
SHARED_LIST = []

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


ibns = {
    'none': nn.BatchNorm2d,
    'a': IBN,
    'b': nn.BatchNorm2d,
    'ab':IBN,
    's': SwitchNorm2d
    }


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, ibn_mode='none'):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = ibns[ibn_mode](planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.IN = None
        if ibn_mode == 'b' or ibn_mode == 'ab':
            self.IN = nn.InstanceNorm2d(planes * 4, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.IN is not None:
            out = self.IN(out)
        out = self.relu(out)

        if TRACK_FEAT:
            SHARED_LIST.append(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, output_stride, ibn_mode='none',renet = 'resnet101'):
        self.inplanes = 64
        super(ResNet, self).__init__()
        blocks = [1, 2, 4]
        if output_stride == 32:
            strides = [1, 2, 2, 2]
            dilations = [1, 1, 1, 2]
        elif output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        if ibn_mode == 'b':
            self.bn1 = nn.InstanceNorm2d(64, affine=True)
        else:
            self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], ibn_mode=ibn_mode)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], ibn_mode=ibn_mode)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], ibn_mode=ibn_mode)
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], ibn_mode=ibn_mode)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3])
        self._init_weight()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, ibn_mode='none'):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []

        if planes == 512 or (planes == 256 and ibn_mode == 'b'):
            ibn_mode = 'none'

        if planes == 256 and ibn_mode == 'ab':
            ibn_mode = 'a'
        #ibn_a influence unless the last layer, ibn_b only influence the first and second layers
        
        if ibn_mode == 'none' or ibn_mode == 'a' or ibn_mode == 's':
            layers.append(block(self.inplanes, planes, stride, dilation, downsample, ibn_mode))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, dilation=dilation, ibn_mode=ibn_mode))
        elif ibn_mode == 'b':
            layers.append(block(self.inplanes, planes, stride, dilation, downsample))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks-1):
                layers.append(block(self.inplanes, planes, dilation=dilation, ibn_mode='none'))
            layers.append(block(self.inplanes, planes, dilation=dilation, ibn_mode=ibn_mode))

        elif ibn_mode == 'ab':
            layers.append(block(self.inplanes, planes, stride, dilation, downsample, ibn_mode='a'))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks-1):
                layers.append(block(self.inplanes, planes, dilation=dilation, ibn_mode='a'))
            layers.append(block(self.inplanes, planes, dilation=dilation, ibn_mode='ab'))
        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, ibn_mode='none'):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []

        if planes == 512:
            ibn_mode = 'none'

        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
                            downsample=downsample, ibn_mode=ibn_mode))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i]*dilation, ibn_mode=ibn_mode))

        return nn.Sequential(*layers)

    def forward(self, input):
        low_level_feats = []

        if TRACK_FEAT:
            SHARED_LIST.clear()

        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        if TRACK_FEAT:
            SHARED_LIST.append(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feats.append(x)
        x = self.layer2(x)
        low_level_feats.append(x)
        x = self.layer3(x)
        low_level_feats.append(x)
        x = self.layer4(x)

        return x, low_level_feats

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

def ResNet50(output_stride, ibn_mode='none', pretrained=True):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], output_stride, ibn_mode)

    if pretrained:
        if ibn_mode == 'none':
            _load_pretrained_model(model, url='https://download.pytorch.org/models/resnet50-19c8e357.pth')
        elif ibn_mode == 'a' or ibn_mode == 'ab' or ibn_mode == 's':
            _load_pretrained_model(model, path='pretrained/resnet50_ibn_a.pth')
        elif ibn_mode == 'b':
            _load_pretrained_model(model, path='pretrained/resnet50_ibn_b.pth')
        else:
            raise NotImplementedError

    return model

def ResNet101(output_stride, ibn_mode='none', pretrained=True):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], output_stride, ibn_mode)

    if pretrained:
        if ibn_mode == 'none':
            _load_pretrained_model(model, url='https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        elif ibn_mode == 'a' or ibn_mode == 'ab' or ibn_mode == 's':
            _load_pretrained_model(model, path='pretrained/resnet101_ibn_a.pth')
        elif ibn_mode == 'b':
            _load_pretrained_model(model, path='pretrained/resnet101_ibn_b.pth')
        else:
            raise NotImplementedError

    return model

if __name__ == "__main__":
    import torch
    model = ResNet101(output_stride=8, ibn_mode='a', pretrained=True)
    input = torch.rand(1, 3, 512, 512)
    output, low_level_feat = model(input)
    print(output.size())
    print(low_level_feat.size())
