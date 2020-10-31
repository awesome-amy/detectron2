# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
This file contains components with some default boilerplate logic user may need
in training / testing. They will not work for everyone, but many users may find them useful.

The behavior of functions/classes in this file is subject to change,
since they are meant to represent the "common default behavior" people need in their projects.
"""

import argparse
import logging
import os
import sys
from collections import OrderedDict
import torch
import numpy as np
from fvcore.common.file_io import PathManager
from fvcore.nn.precise_bn import get_bn_modules
from torch.nn.parallel import DistributedDataParallel

from detectron2.data import DatasetMapper
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils import comm
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.env import seed_all_rng
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from detectron2.utils.logger import setup_logger
from detectron2.engine.defaults import default_argument_parser, default_setup, DefaultTrainer, DefaultPredictor

from detectron2.data.transforms.augmentation import Augmentation
from detectron2.data.transforms.augmentation_impl import ResizeShortestEdge
from detectron2.data.transforms.transform import ResizeTransform, TransformList, NoOpTransform
from PIL import Image
from fvcore.transforms.transform import Transform
from . import hooks
from .train_loop import SimpleTrainer

__all__ = ["PaddingTransform", "ResizeWithPadding", "Predictor", "Trainer"]


class PaddingTransform(NoOpTransform):
    """
        Pad an image at the bottom with d_h rows of zeros
        and to the right with d_w columns of zeros.
    """
    def __init__(self, d_h, d_w):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        return np.pad(img, ((0, self.d_h), (0, self.d_w), (0, 0)))


class ResizeWithPadding(Augmentation):
    """
    Resize image with the longest edge to a fixed target size
    and pad to square with zeros
    """
    def __init__(self, shape: int, interp=Image.BILINEAR):
        """
        Args:
            shape: int
            interp: PIL interpolation method
        """
        self._init(locals())

    def get_transform(self, image):
        height, width, _ = image.shape

        if height > width:
            scale = self.shape / height
            resized_height = self.shape
            resized_width = int(width * scale)
        else:
            scale = self.shape / width
            resized_height = int(height * scale)
            resized_width = self.shape

        padding_height = self.shape - resized_height
        padding_width = self.shape - resized_width

        return TransformList([ResizeTransform(
            image.shape[0], image.shape[1], resized_height, resized_width, self.interp
        ), PaddingTransform(
            padding_height, padding_width
        )])


class Predictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.

    Compared to using the model directly, this class does the following additions:

    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.

    If you'd like to do anything more fancy, please refer to its source code
    as examples to build and use the model manually.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.

    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        # self.aug = ResizeWithPadding(cfg.INPUT.MIN_SIZE_TEST)
        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions


class Trainer(DefaultTrainer):
    """
    A trainer with default training logic.
    It is a subclass of :class:`SimpleTrainer` and instantiates everything needed from the
    config. It does the following:

    1. Create model, optimizer, scheduler, dataloader from the given config.
    2. Load a checkpoint or `cfg.MODEL.WEIGHTS`, if exists, when
       `resume_or_load` is called.
    3. Register a few common hooks defined by the config.

    It is created to simplify the **standard model training workflow** and reduce code boilerplate
    for users who only need the standard training workflow, with standard features.
    It means this class makes *many assumptions* about your training logic that
    may easily become invalid in a new research. In fact, any assumptions beyond those made in the
    :class:`SimpleTrainer` are too much for research.

    The code of this class has been annotated about restrictive assumptions it makes.
    When they do not work for you, you're encouraged to:

    1. Overwrite methods of this class, OR:
    2. Use :class:`SimpleTrainer`, which only does minimal SGD training and
       nothing else. You can then add your own hooks if needed. OR:
    3. Write your own training loop similar to `tools/plain_train_net.py`.

    See the :doc:`/tutorials/training` tutorials for more details.

    Note that the behavior of this class, like other functions/classes in
    this file, is not stable, since it is meant to represent the "common default behavior".
    It is only guaranteed to work well with the standard models and training workflow in detectron2.
    To obtain more stable behavior, write your own training logic with other public APIs.

    Examples:
    ::
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load()  # load last checkpoint or MODEL.WEIGHTS
        trainer.train()

    Attributes:
        scheduler:
        checkpointer (DetectionCheckpointer):
        cfg (CfgNode):
    """

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))

        for name, param in model.named_parameters():
            logger.info("freezing parameters except for mask head ...")
            if not name.startswith("mask"):
                # if not name.startswith(("classifier", "regressor", "mask")):
                param.requires_grad = False

        def set_bn_eval(module):
            logger.info("freezing BatchNorm modules...")
            if isinstance(module, torch.nn.BatchNorm2d):
                logger.info(module)
                module.eval()

        model.apply(set_bn_eval)

        return model

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        if "build_efficientnet_bifpn_backbone" in cfg.MODEL.BACKBONE.NAME:
            mapper = DatasetMapper(cfg, is_train=True, augmentations=[
                T.ResizeShortestEdge(
                    [cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN], cfg.INPUT.MAX_SIZE_TRAIN
                )
            ])
            # ResizeWithPadding(cfg.INPUT.MIN_SIZE_TRAIN)
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        if "build_efficientnet_bifpn_backbone" in cfg.MODEL.BACKBONE.NAME:
            mapper = DatasetMapper(cfg, is_train=False, augmentations=[
                T.ResizeShortestEdge(
                    [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
                )
            ])
        # ResizeWithPadding(cfg.INPUT.MIN_SIZE_TEST)
        else:
            mapper = None
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)