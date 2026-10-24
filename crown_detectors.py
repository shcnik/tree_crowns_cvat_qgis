"""Pluggable RGB bounding-box detectors for the tree crown pipeline.

The common output is a pandas DataFrame with pixel coordinates
xmin, ymin, xmax, ymax and a confidence score; SAM is run downstream.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd


COLUMNS = ('xmin', 'ymin', 'xmax', 'ymax', 'score')


def empty_predictions():
    return pd.DataFrame(columns=list(COLUMNS))


class DeepForestDetector:
    def __init__(self, device: str, score_thresh: float | None):
        from deepforest import main as df_main

        self.model = df_main.deepforest()
        self.model.load_model(model_name='weecology/deepforest-tree', revision='main')
        # DeepForest 2.x runs inference through a Lightning Trainer.
        # Only moving the underlying model to CUDA breaks device handling.
        self.model.create_trainer(
            accelerator='gpu' if device == 'cuda' else 'cpu', devices=1)
        self.model.model.eval()
        if score_thresh is not None:
            self.model.config.score_thresh = score_thresh
            self.model.model.score_thresh = score_thresh
        print('Detector: DeepForest')

    def predict(self, rgb: np.ndarray):
        result = self.model.predict_image(image=rgb.astype('float32'))
        return result if result is not None else empty_predictions()


class YoloDetector:
    def __init__(self, weights, device: str, *, confidence: float,
                 iou: float, imgsz: int, class_id: int | None):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.device = 0 if device == 'cuda' else 'cpu'
        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz
        self.class_id = class_id
        classes = self.model.names
        if class_id is not None and class_id not in classes:
            raise ValueError(f'YOLO class ID {class_id} absent from model class names: {classes}')
        print(f'Detector: YOLO weights={weights}; classes={classes}')
        if class_id is None and len(classes) > 1:
            print('WARNING: all YOLO classes will be treated as tree crowns. '
                  'Use --yolo-class-id to select the crown class.')

    def predict(self, rgb: np.ndarray):
        # Ultralytics expects NumPy input in BGR order; SAM expects RGB.
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        result = self.model.predict(
            source=bgr,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
            **({'classes': [self.class_id]} if self.class_id is not None else {}),
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return empty_predictions()
        coords = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        rows = [(*box.tolist(), float(score)) for box, score in zip(coords, scores)]
        return pd.DataFrame(rows, columns=list(COLUMNS))


def build_detector(args, device: str):
    if args.detector == 'deepforest':
        return DeepForestDetector(device, args.df_score_thresh)
    if args.detector == 'yolo':
        return YoloDetector(args.model_weights, device,
                            confidence=args.min_confidence,
                            iou=args.yolo_iou,
                            imgsz=args.yolo_imgsz,
                            class_id=args.yolo_class_id)
    raise ValueError('Unknown detector: ' + args.detector)
