"""Instance mask backends for crown segmentation.

The Detectree2 backend uses the official Detectree2 `setup_cfg` function and
Detectron2 `DefaultPredictor` API to infer per-instance masks from RGB images.
No SAM checkpoint is needed for this backend.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CrownInstance:
    box: np.ndarray     # xyxy pixels in input tile
    mask: np.ndarray    # Boolean HxW, same dimensions as input tile
    score: float


class Detectree2Segmenter:
    def __init__(self, weights, device: str, *, confidence: float,
                 nms_iou: float, base_model: str):
        try:
            from detectree2.models.train import setup_cfg
            from detectron2.engine import DefaultPredictor
        except ImportError as exc:
            raise RuntimeError(
                'Detectree2 mode requires detectree2 and detectron2 in the current '
                'Python environment. See README-DETECTREE2.md; on Windows, WSL2 '
                'is recommended because Detectron2 lacks official Windows support.'
            ) from exc

        self.device = device
        cfg = setup_cfg(base_model=base_model, update_model=str(weights),
                        imgmode='rgb', num_bands=3)
        cfg.MODEL.DEVICE = device
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_iou
        # DefaultPredictor accepts BGR and converts internally if required.
        cfg.INPUT.FORMAT = 'BGR'
        self.predictor = DefaultPredictor(cfg)
        print(f'Segmenter: Detectree2 weights={weights}; device={device}')

    def predict_masks(self, rgb: np.ndarray) -> list[CrownInstance]:
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError('Detectree2 RGB adapter needs HxWx3 uint8 image')
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        instances = self.predictor(bgr)['instances'].to('cpu')
        if len(instances) == 0:
            return []
        if not instances.has('pred_masks') or not instances.has('pred_boxes'):
            raise RuntimeError('Detectree2 checkpoint did not return instance masks and boxes. '
                               'Check that the weights are for Mask R-CNN instance segmentation.')
        boxes = instances.pred_boxes.tensor.detach().numpy()
        masks = instances.pred_masks.detach().numpy()
        scores = instances.scores.detach().numpy()
        h, w = rgb.shape[:2]
        result = []
        for box, mask, score in zip(boxes, masks, scores):
            if mask.shape != (h, w):
                raise ValueError(f'Detectree2 mask shape {mask.shape} does not match tile {(h, w)}')
            result.append(CrownInstance(np.asarray(box, dtype='float32'),
                                        np.asarray(mask, dtype=bool), float(score)))
        return result
