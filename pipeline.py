"""
Video and Stream Processing Pipeline for Intelligent Face Tracker.
Supports both video files and live RTSP streams with auto-reconnection,
real-time HUD annotation, configurable frame skipping, and video recording.
"""

import os
import cv2
import time
import json
from datetime import datetime
from typing import Optional, Dict, Any

from database import DatabaseManager
from logger_system import SystemLogger
from detector import FaceDetector
from recognizer import FaceRecognizer
from tracker import FaceTracker, Track


class FaceTrackerPipeline:
    """End-to-end Face Tracking and Visitor Counting Pipeline."""

    def __init__(self, config_path: str = "config.json", shared_db=None):
        self.config_path = config_path
        self.config = self._load_config()

        # Use shared DatabaseManager if provided so dashboard and pipeline
        # share the same threading.Lock — eliminates SQLite write-lock conflicts.
        if shared_db is not None:
            self.db = shared_db
        else:
            self.db = DatabaseManager(self.config.get("database_path", "data/visitor_system.db"))

        self.sys_logger = SystemLogger(
            log_file=self.config.get("log_file", "logs/events.log"),
            entries_dir=self.config.get("entries_dir", "logs/entries"),
            exits_dir=self.config.get("exits_dir", "logs/exits")
        )

        self.sys_logger.log_info("Initializing Face Detection and Recognition Models...")
        self.detector = FaceDetector(
            model_path="models/yolov8n-face.pt",
            conf_threshold=float(self.config.get("confidence_threshold", 0.45)),
            min_face_size=int(self.config.get("min_face_size", 25))
        )
        self.recognizer = FaceRecognizer(
            model_path="models/w600k_mbf.onnx",
            similarity_threshold=float(self.config.get("similarity_threshold", 0.50))
        )

        self.tracker = FaceTracker(
            db_manager=self.db,
            sys_logger=self.sys_logger,
            recognizer=self.recognizer,
            skip_frames=int(self.config.get("detection_skip_frames", 2)),
            iou_threshold=float(self.config.get("iou_match_threshold", 0.35)),
            max_disappeared=int(self.config.get("max_disappeared_frames", 25)),
            min_hits_for_entry=int(self.config.get("min_hits_for_entry", 1))
        )
        self.stop_requested = False
        self.current_fps = 0.0
        self.active_tracks_count = 0


    def stop(self):
        """Signals the pipeline loop to cleanly stop and flush tracks."""
        self.stop_requested = True

    def remove_visitor(self, visitor_id: str):
        """Purges visitor from active tracker memory."""
        if hasattr(self, 'tracker') and self.tracker is not None:
            self.tracker.remove_visitor(visitor_id)


    def _load_config(self) -> Dict[str, Any]:
        """Loads configuration from config.json."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _open_stream(self, source: str) -> cv2.VideoCapture:
        """Opens video file, webcam, or RTSP stream."""
        # Check if source is integer (webcam index)
        if source.isdigit():
            cap = cv2.VideoCapture(int(source))
        else:
            # RTSP or video file
            cap = cv2.VideoCapture(source)
            if source.lower().startswith("rtsp://"):
                # Optimize RTSP buffer size and latency
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        return cap

    def draw_hud(
        self,
        frame,
        active_tracks,
        fps: float,
        is_detection_frame: bool
    ):
        """Renders information overlay and bounding boxes on frame with dynamic scaling."""
        h, w = frame.shape[:2]
        ui_scale = max(1.0, w / 1280.0)
        box_thickness = max(2, int(2 * ui_scale))
        font_scale = 0.55 * ui_scale
        font_thick = max(1, int(1.5 * ui_scale))

        # Draw Tracks
        for track in active_tracks:
            x1, y1, x2, y2 = track.bbox
            color = (0, 255, 0) if track.entry_logged else (0, 200, 255)
            # Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)

            # Label banner
            label = f"{track.visitor_id}"
            conf_str = f"({track.confidence:.2f})"
            full_label = f"{label} {conf_str}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), baseline = cv2.getTextSize(full_label, font, font_scale, font_thick)

            # Label background
            label_y1 = max(0, y1 - th - int(8 * ui_scale))
            cv2.rectangle(frame, (x1, label_y1), (x1 + tw + int(32 * ui_scale), label_y1 + th + int(8 * ui_scale)), color, -1)
            cv2.putText(
                frame,
                full_label,
                (x1 + int(4 * ui_scale), label_y1 + th + int(3 * ui_scale)),
                font,
                font_scale,
                (0, 0, 0),
                font_thick,
                cv2.LINE_AA
            )
            
            # Draw Dress Color Swatch
            swatch_size = int(12 * ui_scale)
            swatch_x1 = x1 + tw + int(12 * ui_scale)
            cv2.rectangle(frame, (swatch_x1, label_y1 + int(4 * ui_scale)), (swatch_x1 + swatch_size, label_y1 + int(4 * ui_scale) + swatch_size), (int(track.dress_color[0]), int(track.dress_color[1]), int(track.dress_color[2])), -1)
            cv2.rectangle(frame, (swatch_x1, label_y1 + int(4 * ui_scale)), (swatch_x1 + swatch_size, label_y1 + int(4 * ui_scale) + swatch_size), (255, 255, 255), 1)

        # Top-Left Dashboard Overlay
        unique_count = self.db.get_unique_visitor_count()
        active_count = len(active_tracks)

        box_w = int(360 * ui_scale)
        box_h = int(145 * ui_scale)

        overlay = frame.copy()
        cv2.rectangle(overlay, (int(10 * ui_scale), int(10 * ui_scale)), (box_w, box_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Header Text
        cv2.putText(frame, "INTELLIGENT FACE TRACKER", (int(20 * ui_scale), int(35 * ui_scale)), cv2.FONT_HERSHEY_DUPLEX, 0.65 * ui_scale, (255, 255, 255), max(1, int(ui_scale)))
        cv2.line(frame, (int(20 * ui_scale), int(45 * ui_scale)), (int((360 - 20) * ui_scale), int(45 * ui_scale)), (100, 100, 100), max(1, int(ui_scale)))

        # Metrics
        cv2.putText(frame, f"Unique Visitors: {unique_count}", (int(20 * ui_scale), int(70 * ui_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.60 * ui_scale, (0, 255, 128), max(2, int(2 * ui_scale)))
        cv2.putText(frame, f"Active in Frame : {active_count}", (int(20 * ui_scale), int(95 * ui_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * ui_scale, (255, 255, 0), max(1, int(ui_scale)))
        cv2.putText(frame, f"FPS: {fps:.1f}", (int(20 * ui_scale), int(115 * ui_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.50 * ui_scale, (200, 200, 200), max(1, int(ui_scale)))

        mode_badge = "DETECT" if is_detection_frame else f"SKIP ({self.tracker.skip_frames})"
        cv2.putText(frame, f"Cycle: {mode_badge}", (int(20 * ui_scale), int(135 * ui_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * ui_scale, (180, 180, 255), max(1, int(ui_scale)))

    def run(
        self,
        source_override: Optional[str] = None,
        max_frames: Optional[int] = None,
        frame_callback = None
    ):
        """
        Main loop processing the video source until completion or interruption.
        """
        self.stop_requested = False
        source = source_override or self.config.get("input_source", "data/video_sample1.mp4")
        self.sys_logger.log_info(f"Starting Face Tracker Pipeline with source: {source}")

        is_rtsp = str(source).lower().startswith("rtsp://")
        is_webcam = str(source).isdigit()
        reconnect_delay = float(self.config.get("rtsp_reconnect_delay", 2.0))
        save_video = bool(self.config.get("save_annotated_video", True))
        display_win = bool(self.config.get("display_window", True))
        output_path = self.config.get("output_video_path", "logs/output_annotated.mp4")

        video_writer = None
        frame_idx = 0
        skip_n = self.tracker.skip_frames
        fps_smooth = 0.0

        while True:
            if self.stop_requested:
                break

            cap = self._open_stream(source)
            if not cap.isOpened():
                self.sys_logger.log_error(f"Unable to open video stream: {source}")
                if is_rtsp:
                    time.sleep(reconnect_delay)
                    continue
                else:
                    break

            # Frame pacing for recorded video files to eliminate lag / slow-motion
            source_fps = 0.0
            if not is_rtsp and not is_webcam:
                try:
                    fps_prop = cap.get(cv2.CAP_PROP_FPS)
                    if fps_prop and fps_prop > 0 and fps_prop <= 120 and not (fps_prop != fps_prop):
                        source_fps = float(fps_prop)
                except Exception:
                    pass
                if source_fps <= 0:
                    source_fps = 30.0

            video_start_time = time.perf_counter()
            processed_video_frames = 0

            try:
                while True:
                    if self.stop_requested:
                        self.sys_logger.log_info("Pipeline stop requested.")
                        break

                    if max_frames is not None and frame_idx >= max_frames:
                        self.sys_logger.log_info(f"Reached max requested frames limit ({max_frames}). Stopping.")
                        break

                    # Frame pacing check for video files
                    if source_fps > 0:
                        frame_target_time = 1.0 / source_fps
                        expected_elapsed = processed_video_frames * frame_target_time
                        actual_elapsed = time.perf_counter() - video_start_time
                        
                        # If processing is running faster than video speed, sleep slightly to maintain real-time speed
                        if actual_elapsed < expected_elapsed:
                            time.sleep(expected_elapsed - actual_elapsed)
                        # If processing is falling behind real-time, skip intermediate frames to prevent slow-motion lag
                        elif actual_elapsed > expected_elapsed + frame_target_time:
                            frames_behind = int((actual_elapsed - expected_elapsed) / frame_target_time)
                            if frames_behind > 0:
                                drop_count = min(frames_behind, 4)
                                for _ in range(drop_count):
                                    grabbed = cap.grab()
                                    if not grabbed:
                                        break
                                    processed_video_frames += 1
                                    frame_idx += 1

                    t_start = time.perf_counter()
                    ret, frame = cap.read()
                    processed_video_frames += 1

                    if not ret:
                        if is_rtsp:
                            self.sys_logger.log_warning("RTSP stream interrupted. Attempting reconnect...")
                            break
                        else:
                            self.sys_logger.log_info("Video file processing completed.")
                            break

                    frame_idx += 1
                    # Configurable skip frames: run detection every (skip_n + 1) frames
                    is_det_frame = (frame_idx % (skip_n + 1) == 1) or (skip_n == 0)

                    detections = None
                    if is_det_frame:
                        detections = self.detector.detect(frame)

                    # Update Tracker
                    active_tracks = self.tracker.process_frame(
                        frame=frame,
                        detections=detections,
                        is_detection_frame=is_det_frame
                    )

                    # Calculate FPS
                    t_elapsed = time.perf_counter() - t_start
                    instant_fps = 1.0 / max(1e-5, t_elapsed)
                    fps_smooth = 0.9 * fps_smooth + 0.1 * instant_fps if fps_smooth > 0 else instant_fps
                    self.current_fps = fps_smooth
                    self.active_tracks_count = len(active_tracks)

                    # Draw Overlay
                    annotated = frame.copy()
                    self.draw_hud(annotated, active_tracks, fps_smooth, is_det_frame)

                    # Trigger frame callback for web stream if registered
                    if frame_callback is not None:
                        try:
                            frame_callback(annotated, active_tracks, fps_smooth, is_det_frame)
                        except Exception:
                            pass

                    # Video Writer setup
                    if save_video and video_writer is None:
                        h, w = frame.shape[:2]
                        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_writer = cv2.VideoWriter(output_path, fourcc, 25.0, (w, h))

                    if video_writer is not None:
                        video_writer.write(annotated)

                    # Real-time Display
                    if display_win:
                        disp_h, disp_w = annotated.shape[:2]
                        if disp_w > 1280:
                            scale_factor = 1280.0 / disp_w
                            preview = cv2.resize(annotated, (1280, int(disp_h * scale_factor)), interpolation=cv2.INTER_AREA)
                        else:
                            preview = annotated
                        cv2.imshow("Intelligent Face Tracker", preview)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            self.sys_logger.log_info("User requested exit ('q' pressed).")
                            return
                        elif key == ord('s'):
                            # Print summary
                            total_u = self.db.get_unique_visitor_count()
                            print(f"\n[SUMMARY] Unique Visitors: {total_u} | Active Tracks: {len(active_tracks)}")

                # If non-RTSP video ends or stop requested, exit outer loop
                if not is_rtsp or self.stop_requested:
                    break

            finally:
                cap.release()

        # Flush active tracks to generate clean exit events
        self.tracker.flush_all()
        if video_writer is not None:
            video_writer.release()
            self.sys_logger.log_info(f"Saved annotated output video to: {output_path}")

        if display_win:
            cv2.destroyAllWindows()

        final_count = self.db.get_unique_visitor_count()
        self.sys_logger.log_info(
            f"Pipeline terminated. Total unique visitors recorded: {final_count}"
        )
