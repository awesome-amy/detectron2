
import inspect
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.utils.events import get_event_storage
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou

from ..poolers import ROIPooler
from .box_head import build_box_head
from .fast_rcnn import FastRCNNOutputLayers
from .keypoint_head import build_keypoint_head
from .mask_head import build_mask_head
from ..proposal_generator.proposal_utils import add_ground_truth_to_proposals
from .roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads, select_foreground_proposals

@ROI_HEADS_REGISTRY.register()
class EfficientMaskROIHeads(StandardROIHeads):
    """
    Used in EfficientMask which takes predicted boxes from EfficientDet as proposals.
    """

    @configurable
    def __init__(
        self,
        *,
        mask_in_features: Optional[List[str]] = None,
        mask_pooler: Optional[ROIPooler] = None,
        mask_head: Optional[nn.Module] = None,
        train_on_pred_boxes: bool = False,
        **kwargs
    ):
        """
        NOTE: this interface is experimental.

        Args:
            mask_in_features (list[str]): list of feature names to use for the mask
                pooler or mask head. None if not using mask head.
            mask_pooler (ROIPooler): pooler to extract region features from image features.
                The mask head will then take region features to make predictions.
                If None, the mask head will directly take the dict of image features
                defined by `mask_in_features`
            mask_head (nn.Module): transform features to make mask predictions
            train_on_pred_boxes (bool): whether to use proposal boxes or
                predicted boxes from the box head to train other heads.
        """
        super(StandardROIHeads, self).__init__(**kwargs)
        # keep self.in_features for backward compatibility
        self.mask_on = mask_in_features is not None
        if self.mask_on:
            self.mask_in_features = mask_in_features
            self.mask_pooler = mask_pooler
            self.mask_head = mask_head

        self.train_on_pred_boxes = train_on_pred_boxes

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super(StandardROIHeads, cls).from_config(cfg)
        ret["train_on_pred_boxes"] = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # Subclasses that have not been updated to use from_config style construction
        # may have overridden _init_*_head methods. In this case, those overridden methods
        # will not be classmethods and we need to avoid trying to call them here.
        # We test for this with ismethod which only returns True for bound methods of cls.
        # Such subclasses will need to handle calling their overridden _init_*_head methods.
        if inspect.ismethod(cls._init_mask_head):
            ret.update(cls._init_mask_head(cfg, input_shape))
        return ret

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets
            for proposals_per_image in proposals:
                proposals_per_image.proposal_boxes = proposals_per_image.pred_boxes
            proposals = self.label_and_sample_proposals(proposals, targets)
            assert proposals[0].proposal_boxes
            assert proposals[0].gt_classes
            assert proposals[0].gt_masks
        del targets

        if self.training:
            losses = self._forward_mask(features, proposals)
            return proposals, losses

        else:
            pred_instances = self._forward_mask(features, proposals)
            return pred_instances, {}
