# Author: Zylo117
import torch
from torch import nn

from detectron2.modeling.backbone.build import BACKBONE_REGISTRY, Backbone
# from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from projects.EfficientDet.efficient_det.efficientdet.model import BiFPN, EfficientNet

from detectron2.config import CfgNode as CN

__all__ = ["build_efficientnet_bifpn_backbone", "EfficientDetBackbone", "add_efficientnet_bifpn_config"]


class EfficientDetBackbone(Backbone):
    def __init__(self, num_classes=80, compound_coef=0, load_weights=False, **kwargs):
        super(EfficientDetBackbone, self).__init__()
        self.compound_coef = compound_coef

        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6, 7]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5, 5]
        self.pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6]
        self.anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5., 4.]
        self.aspect_ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])
        self.num_scales = len(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
        conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
            8: [80, 224, 640],
        }

        # num_anchors = len(self.aspect_ratios) * self.num_scales

        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[compound_coef],
                    True if _ == 0 else False,
                    attention=True if compound_coef < 6 else False,
                    use_p8=compound_coef > 7)
              for _ in range(self.fpn_cell_repeats[compound_coef])])

        self.num_classes = num_classes
        # self.regressor = Regressor(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
        #                            num_layers=self.box_class_repeats[self.compound_coef],
        #                            pyramid_levels=self.pyramid_levels[self.compound_coef])
        # self.classifier = Classifier(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
        #                              num_classes=num_classes,
        #                              num_layers=self.box_class_repeats[self.compound_coef],
        #                              pyramid_levels=self.pyramid_levels[self.compound_coef])
        #
        # self.anchors = Anchors(anchor_scale=self.anchor_scale[compound_coef],
        #                        pyramid_levels=(torch.arange(self.pyramid_levels[self.compound_coef]) + 3).tolist(),
        #                        **kwargs)
        self.backbone_net = EfficientNet(self.backbone_compound_coef[compound_coef], load_weights)

        # add attributes compatiable for default .out_shape()
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {}
        for s in range(2, 7):     # TODO: avoid hard-coding stage number
            self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)
        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: self.fpn_num_filters[compound_coef] for k in self._out_features}

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        """
        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        max_size = inputs.shape[-1]

        _, p3, p4, p5 = self.backbone_net(inputs)

        features = (p3, p4, p5)
        features = self.bifpn(features)

        # regression = self.regressor(features)
        # classification = self.classifier(features)
        # anchors = self.anchors(inputs, inputs.dtype)

        # return features, regression, classification, anchors
        assert len(self._out_features) == len(features)
        return dict(zip(self._out_features, features))

    def init_backbone(self, path):
        state_dict = torch.load(path)
        try:
            ret = self.load_state_dict(state_dict, strict=False)
            print(ret)
        except RuntimeError as e:
            print('Ignoring ' + str(e) + '"')


@BACKBONE_REGISTRY.register()
def build_efficientnet_bifpn_backbone(cfg, input_shape):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    num_classes = cfg.MODEL.RETINANET.NUM_CLASSES
    compound_coef = cfg.MODEL.EFFICIENTNET.COMPOUND_COEF
    backbone = EfficientDetBackbone(
        num_classes=num_classes,
        compound_coef=compound_coef,
        load_weights=False)
    return backbone


def add_efficientnet_bifpn_config(cfg):
    """
    Add config for efficientdet backbone.
    """
    cfg.MODEL.EFFICIENTNET = CN()
    cfg.MODEL.EFFICIENTNET.COMPOUND_COEF = 0
