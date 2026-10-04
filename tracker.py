"""
Multi-Object Tracking and State Machine module.
Tracks faces across frames, integrates YOLO detection with configurable frame skip,
manages entry and exit states, auto-registration, and re-identification.
"""

import os
import cv2
import time
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any

from database import DatabaseManager
from logger_system import SystemLogger
from recognizer import FaceRecognizer


def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


class Track:
    """Represents a single continuous face track in the scene."""

    def __init__(
        self,
        track_id: int,
        visitor_id: str,
        bbox: Tuple[int, int, int, int],
        confidence: float,
        face_crop: Optional[np.ndarray],
        timestamp: datetime,
        embedding: Optional[np.ndarray] = None,
        dress_color: Tuple[int, int, int] = (0, 0, 0),
        body_crop: Optional[np.ndarray] = None
    ):
        self.track_id = track_id
        self.visitor_id = visitor_id
        self.bbox = bbox
        self.confidence = confidence
        self.face_crop = face_crop
        self.body_crop = body_crop
        self.best_crop = face_crop if face_crop is not None else body_crop
        self.embedding = embedding
        self.dress_color = dress_color

        self.first_seen = timestamp
        self.last_seen = timestamp
        self.disappeared = 0
        self.detection_hits = 1

        self.entry_logged = False
        self.exit_logged = False
        self.entry_image_path = None
        self.exit_image_path = None

    def update(self, bbox: Tuple[int, int, int, int], confidence: float, face_crop: Optional[np.ndarray], dress_color: Tuple[int, int, int], timestamp: datetime, body_crop: Optional[np.ndarray] = None):
        """Updates track with a new detection."""
        self.bbox = bbox
        self.confidence = confidence
        self.face_crop = face_crop
        self.body_crop = body_crop
        self.dress_color = dress_color
        self.last_seen = timestamp
        self.disappeared = 0
        self.detection_hits += 1

        # Keep best crop (highest resolution)
        crop_to_eval = face_crop if face_crop is not None else body_crop
        if crop_to_eval is not None and crop_to_eval.size > 0:
            if self.best_crop is None or crop_to_eval.size > self.best_crop.size:
                self.best_crop = crop_to_eval.copy()

    def mark_missed(self):
        """Increments disappeared counter."""
        self.disappeared += 1


class FaceTracker:
    """
    Orchestrates face tracking, detection skip frames, entry/exit logging,
    and visitor counting.
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        sys_logger: SystemLogger,
        recognizer: FaceRecognizer,
        skip_frames: int = 2,
        iou_threshold: float = 0.35,
        max_disappeared: int = 25,
        min_hits_for_entry: int = 2
    ):
        self.db = db_manager
        self.logger = sys_logger
        self.recognizer = recognizer
        self.skip_frames = skip_frames
        self.iou_threshold = iou_threshold
        self.max_disappeared = max_disappeared
        self.min_hits_for_entry = min_hits_for_entry

        self.active_tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        self.frame_count = 0

        # In-memory cache of registered embeddings for instant matching
        self.registered_embeddings = self.db.get_all_embeddings()
        self.logger.log_info(
            f"Loaded {len(self.registered_embeddings)} registered visitors from database."
        )

    def _generate_next_visitor_id(self) -> str:
        """Generates a sequential visitor ID (e.g., VISITOR_001, VISITOR_002)."""
        current_count = self.db.get_unique_visitor_count()
        candidate = f"VISITOR_{current_count + 1:04d}"
        # Ensure uniqueness
        while candidate in self.registered_embeddings:
            current_count += 1
            candidate = f"VISITOR_{current_count + 1:04d}"
        return candidate

    def process_frame(
        self,
        frame: np.ndarray,
        detections: Optional[List[Dict[str, Any]]] = None,
        is_detection_frame: bool = True
    ) -> List[Track]:
        """
        Processes one video frame.
        If is_detection_frame is True, matches detections to tracks or spawns new tracks.
        If False, updates tracks and checks for exits.
        """
        self.frame_count += 1
        now = datetime.now()

        if is_detection_frame and detections is not None:
            self._update_with_detections(frame, detections, now)
        else:
            self._update_without_detections()

        # Check for confirmed entries and process exits
        self._check_track_states(now)

        return list(self.active_tracks.values())

    def _update_with_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        now: datetime
    ):
        """Matches detections to active tracks using IoU matching."""
        active_ids = list(self.active_tracks.keys())
        det_indices = list(range(len(detections)))

        matched_tracks = set()
        matched_dets = set()

        if active_ids and det_indices:
            # Build IoU cost matrix
            iou_matrix = np.zeros((len(active_ids), len(det_indices)), dtype=np.float32)
            for i, tid in enumerate(active_ids):
                track_box = self.active_tracks[tid].bbox
                for j, dj in enumerate(det_indices):
                    iou_matrix[i, j] = compute_iou(track_box, detections[dj]["bbox"])

            # Greedy matching in descending order of IoU
            while True:
                max_iou = np.max(iou_matrix) if iou_matrix.size > 0 else 0.0
                if max_iou < self.iou_threshold:
                    break
                row, col = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                tid = active_ids[row]
                det_idx = det_indices[col]

                # Update track with detection
                det = detections[det_idx]
                self.active_tracks[tid].update(
                    bbox=det["bbox"],
                    confidence=det["confidence"],
                    face_crop=det.get("face_crop"),
                    body_crop=det.get("body_crop"),
                    dress_color=det.get("dress_color", (0, 0, 0)),
                    timestamp=now
                )
                matched_tracks.add(tid)
                matched_dets.add(det_idx)

                # Zero out row and col
                iou_matrix[row, :] = 0.0
                iou_matrix[:, col] = 0.0

        # Mark unmatched active tracks as missed
        for tid in active_ids:
            if tid not in matched_tracks:
                self.active_tracks[tid].mark_missed()

        # Handle unmatched detections -> Auto-Registration or Re-Identification
        for dj in det_indices:
            if dj not in matched_dets:
                self._handle_new_detection(detections[dj], now)

    def _handle_new_detection(self, det: Dict[str, Any], now: datetime):
        """
        Extracts ArcFace embedding, checks if already registered,
        auto-registers if new, and creates new Track.
        """
        face_crop = det.get("face_crop")
        body_crop = det.get("body_crop")
        track_id = self.next_track_id
        self.next_track_id += 1

        if face_crop is not None:
            # 1. Generate 512-dim embedding
            emb, elapsed_ms = self.recognizer.generate_embedding(face_crop)
            self.logger.log_embedding_generation(track_id=track_id, dim=len(emb), elapsed_ms=elapsed_ms)

            # 2. Match against registered visitors
            matched_id, similarity = self.recognizer.match_face(emb, self.registered_embeddings)

            if matched_id is not None:
                # Re-identified existing visitor!
                visitor_id = matched_id
                self.logger.log_recognition(visitor_id=visitor_id, track_id=track_id, similarity=similarity)
                self.db.update_visitor_activity(visitor_id, now.isoformat(), increment_visit=True)
                
                if similarity >= 0.45:
                    curr_emb = self.registered_embeddings[visitor_id]
                    refined_emb = 0.85 * curr_emb + 0.15 * emb
                    norm = np.linalg.norm(refined_emb)
                    if norm > 0:
                        refined_emb = (refined_emb / norm).astype(np.float32)
                    self.registered_embeddings[visitor_id] = refined_emb
                    self.db.update_visitor_embedding(visitor_id, refined_emb)
            else:
                # New face! Auto-register!
                visitor_id = self._generate_next_visitor_id()
                now_iso = now.isoformat()
                thumb_path = self.logger.save_face_image(face_crop, visitor_id, "entry", now)
                self.db.register_visitor(
                    visitor_id=visitor_id,
                    embedding=emb,
                    timestamp=now_iso,
                    thumbnail_path=thumb_path
                )
                self.registered_embeddings[visitor_id] = emb
                self.logger.log_registration(visitor_id, now_iso)
        else:
            # Body only detection (Walking away, no face)
            emb = np.zeros(512, dtype=np.float32)
            visitor_id = self._generate_next_visitor_id()
            now_iso = now.isoformat()
            thumb_path = self.logger.save_face_image(body_crop, visitor_id, "entry_back", now)
            self.db.register_visitor(
                visitor_id=visitor_id,
                embedding=emb,
                timestamp=now_iso,
                thumbnail_path=thumb_path
            )
            # We intentionally DO NOT cache the dummy embedding in registered_embeddings 
            # to prevent all back-facing people from matching each other (since distance between zero vectors is 0).
            self.logger.log_registration(visitor_id, now_iso)

        # Create active track
        new_track = Track(
            track_id=track_id,
            visitor_id=visitor_id,
            bbox=det["bbox"],
            confidence=det["confidence"],
            face_crop=face_crop,
            body_crop=body_crop,
            timestamp=now,
            embedding=emb,
            dress_color=det.get("dress_color", (0, 0, 0))
        )
        self.active_tracks[track_id] = new_track

    def _update_without_detections(self):
        """On skipped frames, increment missed counters for all active tracks."""
        for track in self.active_tracks.values():
            track.mark_missed()

    def _check_track_states(self, now: datetime):
        """
        Fires entry logging when track is confirmed,
        and fires exit logging when track disappears.
        """
        to_remove = []

        for tid, track in list(self.active_tracks.items()):
            # Log Entry once track passes min hits
            if not track.entry_logged and track.detection_hits >= self.min_hits_for_entry:
                entry_time_str = track.first_seen.strftime("%Y-%m-%d %H:%M:%S")
                # Save entry image
                img_path = self.logger.save_face_image(
                    face_img=track.best_crop,
                    visitor_id=track.visitor_id,
                    event_type="entry",
                    timestamp=track.first_seen
                )
                track.entry_image_path = img_path
                track.entry_logged = True

                # Record in Database
                self.db.log_event(
                    visitor_id=track.visitor_id,
                    event_type="entry",
                    timestamp=entry_time_str,
                    image_path=img_path,
                    confidence=track.confidence,
                    details=f"Track #{track.track_id} entered scene | Color: rgb({track.dress_color[2]},{track.dress_color[1]},{track.dress_color[0]})"
                )
                # Record in events.log
                self.logger.log_entry(
                    visitor_id=track.visitor_id,
                    timestamp_str=entry_time_str,
                    image_path=img_path,
                    confidence=track.confidence
                )

            # Check for Exit
            if track.disappeared >= self.max_disappeared:
                self._finalize_exit(track, now)
                to_remove.append(tid)

        for tid in to_remove:
            self.active_tracks.pop(tid, None)

    def _finalize_exit(self, track: Track, now: datetime):
        """Logs exit event exactly once when track leaves frame."""
        if track.entry_logged and not track.exit_logged:
            exit_time_str = track.last_seen.strftime("%Y-%m-%d %H:%M:%S")
            duration_sec = (track.last_seen - track.first_seen).total_seconds()

            # Save exit image
            img_path = self.logger.save_face_image(
                face_img=track.best_crop if track.best_crop is not None else track.face_crop,
                visitor_id=track.visitor_id,
                event_type="exit",
                timestamp=track.last_seen
            )
            track.exit_image_path = img_path
            track.exit_logged = True

            # Record in Database
            self.db.log_event(
                visitor_id=track.visitor_id,
                event_type="exit",
                timestamp=exit_time_str,
                image_path=img_path,
                confidence=track.confidence,
                details=f"Track #{track.track_id} exited scene (Duration: {duration_sec:.1f}s) | Color: rgb({track.dress_color[2]},{track.dress_color[1]},{track.dress_color[0]})"
            )
            # Record in events.log
            self.logger.log_exit(
                visitor_id=track.visitor_id,
                timestamp_str=exit_time_str,
                image_path=img_path,
                duration_sec=duration_sec
            )

    def reload_embeddings(self):
        """Reloads registered face embeddings from database."""
        self.registered_embeddings = self.db.get_all_embeddings()

    def remove_visitor(self, visitor_id: str):
        """Removes a visitor from in-memory cache and clears any active tracks with that ID."""
        self.registered_embeddings.pop(visitor_id, None)
        # Remove any active tracks associated with this visitor so they don't log exit for deleted visitor
        for tid in list(self.active_tracks.keys()):
            if self.active_tracks[tid].visitor_id == visitor_id:
                self.active_tracks.pop(tid, None)

    def flush_all(self):
        """Forces exit events for any remaining active tracks at end of stream."""
        now = datetime.now()
        for track in list(self.active_tracks.values()):
            self._finalize_exit(track, now)
        self.active_tracks.clear()

