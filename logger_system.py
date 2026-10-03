"""
Logging System module for Intelligent Face Tracker.
Handles:
1. Structured filesystem storage for cropped face images (e.g., logs/entries/YYYY-MM-DD/).
2. Mandatory events.log file tracking all critical system events:
   - Embedding generation and registration
   - Face entry, recognition, tracking, and exit
"""

import os
import cv2
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional, Tuple
import numpy as np


class SystemLogger:
    """Manages system logging to events.log and face crop filesystem organization."""

    def __init__(
        self,
        log_file: str = "logs/events.log",
        entries_dir: str = "logs/entries",
        exits_dir: str = "logs/exits"
    ):
        self.log_file = log_file
        self.entries_dir = entries_dir
        self.exits_dir = exits_dir

        # Ensure base directories exist
        os.makedirs(os.path.dirname(os.path.abspath(self.log_file)), exist_ok=True)
        os.makedirs(self.entries_dir, exist_ok=True)
        os.makedirs(self.exits_dir, exist_ok=True)

        self._init_logger()

    def _init_logger(self):
        """Configures root/events logger with rotating file handler and console handler."""
        self.logger = logging.getLogger("FaceTracker")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if not self.logger.handlers:
            formatter = logging.Formatter(
                fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )

            # Rotating file handler (10MB per file, max 5 backups)
            file_handler = RotatingFileHandler(
                self.log_file,
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8"
            )
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            # Console stream handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def save_face_image(
        self,
        face_img: np.ndarray,
        visitor_id: str,
        event_type: str,
        timestamp: Optional[datetime] = None
    ) -> str:
        """
        Saves cropped face image to structured local folder:
        logs/entries/YYYY-MM-DD/<visitor_id>_<timestamp>.jpg or
        logs/exits/YYYY-MM-DD/<visitor_id>_<timestamp>.jpg
        Returns the absolute or relative file path.
        """
        if timestamp is None:
            timestamp = datetime.now()

        date_str = timestamp.strftime("%Y-%m-%d")
        time_str = timestamp.strftime("%H%M%S_%f")[:10]  # millisecond precision

        base_dir = self.entries_dir if event_type.lower() == "entry" else self.exits_dir
        target_dir = os.path.join(base_dir, date_str)
        os.makedirs(target_dir, exist_ok=True)

        filename = f"{visitor_id}_{event_type.lower()}_{time_str}.jpg"
        file_path = os.path.join(target_dir, filename)

        if face_img is not None and face_img.size > 0:
            # Normalize crop size for high-quality archival (e.g. 160x160)
            h, w = face_img.shape[:2]
            if h > 10 and w > 10:
                cv2.imwrite(file_path, face_img)
            else:
                # If too small, scale up slightly with linear interpolation
                resized = cv2.resize(face_img, (160, 160), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(file_path, resized)
        else:
            # Fallback placeholder 160x160 blank image
            blank = np.zeros((160, 160, 3), dtype=np.uint8)
            cv2.imwrite(file_path, blank)

        return os.path.normpath(file_path)

    # Event-specific logging methods
    def log_embedding_generation(self, track_id: int, dim: int = 512, elapsed_ms: float = 0.0):
        self.logger.info(
            f"[EMBEDDING] Generated {dim}-dim facial embedding for Track ID #{track_id} in {elapsed_ms:.1f}ms"
        )

    def log_registration(self, visitor_id: str, timestamp_str: str):
        self.logger.info(
            f"[REGISTRATION] Auto-registered new visitor '{visitor_id}' at {timestamp_str}"
        )

    def log_recognition(self, visitor_id: str, track_id: int, similarity: float):
        self.logger.info(
            f"[RECOGNITION] Matched Track ID #{track_id} to Visitor '{visitor_id}' (Similarity: {similarity:.3f})"
        )

    def log_entry(self, visitor_id: str, timestamp_str: str, image_path: str, confidence: float):
        self.logger.info(
            f"[ENTRY] Visitor '{visitor_id}' entered camera frame at {timestamp_str}. "
            f"Image saved: {image_path} (Confidence: {confidence:.2f})"
        )

    def log_tracking(self, track_id: int, visitor_id: str, bbox: Tuple[int, int, int, int], frame_idx: int):
        self.logger.debug(
            f"[TRACKING] Frame #{frame_idx}: Tracking Track ID #{track_id} as '{visitor_id}' at bbox {bbox}"
        )

    def log_exit(self, visitor_id: str, timestamp_str: str, image_path: str, duration_sec: float = 0.0):
        self.logger.info(
            f"[EXIT] Visitor '{visitor_id}' exited camera frame at {timestamp_str}. "
            f"Image saved: {image_path} (Duration in frame: {duration_sec:.1f}s)"
        )

    def log_info(self, msg: str):
        self.logger.info(msg)

    def log_warning(self, msg: str):
        self.logger.warning(msg)

    def log_error(self, msg: str):
        self.logger.error(msg)
