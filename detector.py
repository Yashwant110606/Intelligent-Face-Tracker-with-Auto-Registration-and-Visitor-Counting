"""
Face Detection Module using YOLOv8 (yolov8n-face).
Loads the YOLOv8 face model and runs inference to detect bounding boxes
with high accuracy and speed on GPU/CPU.
"""

import os
import cv2
import torch
import numpy as np
from typing import List, Dict, Any, Tuple


class FaceDetector:
    """YOLOv8-based Face Detector."""

    def __init__(
        self,
        model_path: str = "models/yolov8n-face.pt",
        conf_threshold: float = 0.45,
        min_face_size: int = 25,
        device: str = None
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.min_face_size = min_face_size

        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"YOLOv8 face model not found at {self.model_path}")

        # Lazy import ultralytics so module can be imported anywhere
        from ultralytics import YOLO
        self.model = YOLO(self.model_path)
        # Warmup if on cuda
        if "cuda" in self.device:
            self.model.to(self.device)

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in a single video frame.
        
        Args:
            frame: BGR image (H, W, 3)
            
        Returns:
            List of dicts:
            [
                {
                    "bbox": (x1, y1, x2, y2),
                    "confidence": float,
                    "face_crop": np.ndarray (BGR)
                }, ...
            ]
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]

        # Run inference
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
            imgsz=640
        )

        detections = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            conf = float(box.conf[0].item()) if hasattr(box.conf[0], 'item') else float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)

            # Clip coordinates to frame boundary
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            bw = x2 - x1
            bh = y2 - y1

            # Filter out tiny or invalid detections
            if bw < self.min_face_size or bh < self.min_face_size:
                continue

            # Add slight margin around face crop (5%) for better embedding extraction
            margin_x = int(bw * 0.05)
            margin_y = int(bh * 0.05)
            crop_x1 = max(0, x1 - margin_x)
            crop_y1 = max(0, y1 - margin_y)
            crop_x2 = min(w, x2 + margin_x)
            crop_y2 = min(h, y2 + margin_y)

            face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": conf,
                "face_crop": face_crop
            })

        return detections
