import torch
from torch import nn
from .efficientdet import EfficientDet
from detectron2.modeling.roi_heads import build_roi_heads
from ..backbone import build_backbone
from ..postprocessing import detector_postprocess
from .build import META_ARCH_REGISTRY

__all__ = ["EfficientMask"]


@META_ARCH_REGISTRY.register()
class EfficientMask(nn.Module):
    """
    Implement RetinaNet in :paper:`RetinaNet`.
    """

    def __init__(self, cfg):
        super().__init__()
        # TODO: get shape directly without building backbone
        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()
        print("backbone_shape")
        print(backbone_shape)

        self.detector = EfficientDet(cfg)
        self.mask = build_roi_heads(cfg, backbone_shape)

    def forward(self, batched_inputs):
        self.detector.eval()
        with torch.no_grad():
            images, features_dict, detections = self.detector(batched_inputs)

        if self.training:
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            _, mask_losses = self.mask(images, features_dict, detections, gt_instances)

            return mask_losses
        else:
            processed_results = []
            results, _ = self.mask(images, features_dict, detections)

            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results



