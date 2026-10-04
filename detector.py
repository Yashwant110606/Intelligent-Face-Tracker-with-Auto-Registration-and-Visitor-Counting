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

def get_dominant_color(image: np.ndarray, k: int = 2) -> Tuple[int, int, int]:
    """Extracts the dominant BGR color from an image crop using K-Means."""
    if image is None or image.size == 0:
        return (0, 0, 0)
    
    # Resize to speed up k-means
    small_image = cv2.resize(image, (32, 32))
    pixels = np.float32(small_image.reshape(-1, 3))
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, palette = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]
    return tuple(map(int, dominant))


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
        self.face_model = YOLO(self.model_path)
        self.person_model = YOLO("yolov8n.pt")
        # Warmup if on cuda
        if "cuda" in self.device:
            self.face_model.to(self.device)
            self.person_model.to(self.device)

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

        # Run face inference
        face_results = self.face_model.predict(
            source=frame,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
            imgsz=640
        )
        
        # Run person inference
        person_results = self.person_model.predict(
            source=frame,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
            imgsz=640,
            classes=[0] # Only detect 'person' class
        )

        detections = []
        
        face_boxes = []
        if face_results and face_results[0].boxes is not None:
            for box in face_results[0].boxes:
                conf = float(box.conf[0].item()) if hasattr(box.conf[0], 'item') else float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                x1, y1, x2, y2 = max(0, min(x1, w-1)), max(0, min(y1, h-1)), max(0, min(x2, w-1)), max(0, min(y2, h-1))
                if (x2 - x1) >= self.min_face_size and (y2 - y1) >= self.min_face_size:
                    face_boxes.append({"bbox": (x1, y1, x2, y2), "conf": conf, "matched": False})

        person_boxes = []
        if person_results and person_results[0].boxes is not None:
            for box in person_results[0].boxes:
                conf = float(box.conf[0].item()) if hasattr(box.conf[0], 'item') else float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                x1, y1, x2, y2 = max(0, min(x1, w-1)), max(0, min(y1, h-1)), max(0, min(x2, w-1)), max(0, min(y2, h-1))
                person_boxes.append({"bbox": (x1, y1, x2, y2), "conf": conf})

        # Process faces (Primary)
        for fb in face_boxes:
            x1, y1, x2, y2 = fb["bbox"]
            bw, bh = x2 - x1, y2 - y1
            
            margin_x = int(bw * 0.05)
            margin_y = int(bh * 0.05)
            crop_x1, crop_y1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
            crop_x2, crop_y2 = min(w, x2 + margin_x), min(h, y2 + margin_y)
            face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()

            # Estimate Body Region for color extraction
            body_y1, body_y2 = y2, min(h, y2 + int(bh * 2.5))
            body_x1, body_x2 = max(0, x1 - int(bw * 0.5)), min(w, x2 + int(bw * 0.5))
            body_crop = frame[body_y1:body_y2, body_x1:body_x2]
            dominant_color = get_dominant_color(body_crop)

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": fb["conf"],
                "face_crop": face_crop,
                "dress_color": dominant_color,
                "type": "face"
            })
            
            # Mark overlapping person boxes as matched so we don't process them twice
            for pb in person_boxes:
                px1, py1, px2, py2 = pb["bbox"]
                if x1 >= px1 and y1 >= py1 and x2 <= px2 and y2 <= py2: # Face is inside person box
                    pb["matched"] = True
                
        # Process unmatched persons (Body Only, e.g., backs facing camera)
        for pb in person_boxes:
            if not pb.get("matched", False):
                x1, y1, x2, y2 = pb["bbox"]
                body_crop = frame[y1:y2, x1:x2].copy()
                dominant_color = get_dominant_color(body_crop)
                
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": pb["conf"],
                    "face_crop": None,  # No face visible!
                    "body_crop": body_crop,
                    "dress_color": dominant_color,
                    "type": "body"
                })

        return detections
