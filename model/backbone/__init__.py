from . import resnet, xception, drn, mobilenet
from .efficientnet import EfficientNet
from .sw import backbones

def build_backbone(backbone, output_stride, ibn_mode, BatchNorm):
    if backbone == 'resnet50':
        return resnet.ResNet50(output_stride, ibn_mode)
    if backbone == 'resnet101':
        return resnet.ResNet101(output_stride, ibn_mode)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    elif backbone == 'efficientnet-b0':
        return EfficientNet.from_pretrained('efficientnet-b0')
    elif backbone == 'efficientnet-b6':
        return EfficientNet.from_pretrained('efficientnet-b6')
    elif backbone == 'sw-resnet101':
        return backbones.ResNet101(output_stride,    
            sw_cfg = dict(type='SW',
                    sw_type=2,
                    num_pergroup=16,
                    T=5,
                    tie_weight=False,
                    momentum=0.9,
                    affine=True))
    else:
        raise NotImplementedError
