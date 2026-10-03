"""
AetherVision - Standalone Enterprise Web Dashboard & Live Stream Server
for Intelligent Face Tracker with Auto-Registration and Visitor Counting.

Provides:
- Multi-threaded HTTP server (standard library socketserver.ThreadingMixIn)
- Real-time MJPEG live video streaming (/api/video_feed)
- Complete REST APIs for stats, visitors, audit events, charts, logs, and pipeline control
- Zero external web dependencies: runs out of the box with standard Python!
"""

import os
import sys
import json
import time
import shutil
import sqlite3
import urllib.parse
from datetime import datetime
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Optional, Dict, Any, List

import cv2
import numpy as np

from database import DatabaseManager
from pipeline import FaceTrackerPipeline

PORT = 8080
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP Server handling parallel stream and REST requests."""
    daemon_threads = True
    allow_reuse_address = True


class PipelineManager:
    """Manages background tracking pipeline, frame buffering, and MJPEG stream."""

    def __init__(self, config_path: str = CONFIG_PATH):
        self.config_path = config_path
        self.pipeline: Optional[FaceTrackerPipeline] = None
        self.thread: Optional[threading.Thread] = None
        self.is_running = False
        self.latest_frame_bytes: Optional[bytes] = None
        self.frame_lock = threading.Lock()
        self.fps = 0.0
        self.active_tracks: List[Dict[str, Any]] = []
        self.source = ""
        self.started_at: Optional[datetime] = None

    def _frame_callback(self, annotated, active_tracks, fps, is_det):
        """Encodes frame into JPEG bytes and caches active tracks."""
        try:
            _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            frame_bytes = buffer.tobytes()

            tracks_info = []
            for tr in active_tracks:
                tracks_info.append({
                    "track_id": tr.track_id,
                    "visitor_id": tr.visitor_id,
                    "confidence": float(tr.confidence),
                    "entry_logged": tr.entry_logged
                })

            with self.frame_lock:
                self.latest_frame_bytes = frame_bytes
                self.fps = fps
                self.active_tracks = tracks_info
        except Exception as e:
            pass

    def _worker(self, source: str):
        """Pipeline execution loop running in background daemon thread."""
        try:
            # Create fresh pipeline with display_window=False for server mode
            with open(self.config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

            cfg["display_window"] = False
            temp_config_path = os.path.join(BASE_DIR, "data", "temp_server_config.json")
            os.makedirs(os.path.dirname(temp_config_path), exist_ok=True)
            with open(temp_config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)

            self.pipeline = FaceTrackerPipeline(config_path=temp_config_path, shared_db=db_manager)
            self.source = source
            self.is_running = True
            self.started_at = datetime.now()

            self.pipeline.run(
                source_override=source,
                frame_callback=self._frame_callback
            )
        except Exception as e:
            print(f"[PIPELINE ERROR] {e}", file=sys.stderr)
        finally:
            self.is_running = False
            with self.frame_lock:
                self.active_tracks = []
                self.fps = 0.0

    def start(
        self,
        source: Optional[str] = None,
        skip_frames: Optional[int] = None,
        conf_thresh: Optional[float] = None,
        sim_thresh: Optional[float] = None
    ) -> bool:
        """Starts pipeline thread if not already running."""
        if self.is_running:
            return False

        # Update config.json if parameters provided
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if source:
                cfg["input_source"] = source
            if skip_frames is not None:
                cfg["detection_skip_frames"] = int(skip_frames)
            if conf_thresh is not None:
                cfg["confidence_threshold"] = float(conf_thresh)
            if sim_thresh is not None:
                cfg["similarity_threshold"] = float(sim_thresh)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)

        target_source = source or "data/video_sample1.mp4"
        self.thread = threading.Thread(target=self._worker, args=(target_source,), daemon=True)
        self.thread.start()
        return True

    def stop(self) -> bool:
        """Signals the pipeline loop to terminate cleanly."""
        if self.pipeline:
            self.pipeline.stop()
            self.is_running = False
            return True
        return False

    def remove_visitor(self, visitor_id: str):
        """Notifies running pipeline to purge visitor from active tracker memory."""
        if self.pipeline:
            try:
                self.pipeline.remove_visitor(visitor_id)
            except Exception:
                pass


    def get_latest_frame(self) -> bytes:
        """Returns the latest annotated JPEG frame or a high-tech standby screen."""
        with self.frame_lock:
            if self.is_running and self.latest_frame_bytes:
                return self.latest_frame_bytes

        # Generate high-tech standby frame
        return self._generate_standby_frame()

    def _generate_standby_frame(self) -> bytes:
        """Generates dynamic cyber-radar standby frame when pipeline is idle."""
        w, h = 1280, 720
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (18, 12, 8)  # Deep obsidian dark

        # Grid lines
        for x in range(0, w, 80):
            cv2.line(img, (x, 0), (x, h), (32, 24, 16), 1)
        for y in range(0, h, 80):
            cv2.line(img, (0, y), (w, y), (32, 24, 16), 1)

        cx, cy = w // 2, h // 2
        # Radar circles
        cv2.circle(img, (cx, cy), 160, (70, 50, 25), 1)
        cv2.circle(img, (cx, cy), 90, (140, 100, 30), 1)
        cv2.circle(img, (cx, cy), 30, (220, 160, 40), 1)
        cv2.line(img, (cx - 200, cy), (cx + 200, cy), (100, 70, 25), 1)
        cv2.line(img, (cx, cy - 200), (cx, cy + 200), (100, 70, 25), 1)

        # Tactical Titles
        cv2.putText(img, "AETHERVISION // INTELLIGENT FACE TRACKER", (cx - 280, cy - 100), cv2.FONT_HERSHEY_DUPLEX, 0.75, (254, 242, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "STANDBY: VIDEO ENGINE IDLE", (cx - 180, cy + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 242, 254), 1, cv2.LINE_AA)
        cv2.putText(img, "Click 'Start Tracking' or Select Video / Camera Source", (cx - 235, cy + 145), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1, cv2.LINE_AA)

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(img, f"Local Telemetry Clock: {now_str}", (cx - 150, cy + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)

        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return buf.tobytes()


# Global Pipeline Manager & Database Instance
pipeline_manager = PipelineManager()
db_manager = DatabaseManager("data/visitor_system.db")


class VisionDashboardHandler(SimpleHTTPRequestHandler):
    """Comprehensive HTTP Request Handler with REST endpoints and MJPEG streaming."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE_DIR, **kwargs)

    def _send_json(self, data: Any, status: int = 200):
        """Helper to send JSON response."""
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> Dict[str, Any]:
        """Helper to parse incoming JSON request body."""
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > 0:
                raw = self.rfile.read(length).decode("utf-8")
                return json.loads(raw)
        except Exception:
            pass
        return {}

    def do_OPTIONS(self):
        """Handle CORS pre-flight."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_DELETE(self):
        """Handle DELETE /api/visitor/<visitor_id> — removes visitor + all events from DB + face crop files."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path.startswith("/api/visitor/"):
            visitor_id = path[len("/api/visitor/"):].strip("/")
            if not visitor_id:
                self._send_json({"error": "visitor_id is required"}, status=400)
                return

            try:
                result = db_manager.delete_visitor(visitor_id)
            except Exception as e:
                print(f"[DELETE ERROR] {e}", file=sys.stderr)
                self._send_json({"error": f"Delete failed: {str(e)}"}, status=500)
                return

            if not result["deleted"]:
                self._send_json({"error": f"Visitor '{visitor_id}' not found"}, status=404)
                return

            # Clean up face crop image files from disk
            files_removed = 0
            for img_path in result["image_paths"]:
                try:
                    full_path = os.path.join(BASE_DIR, img_path) if not os.path.isabs(img_path) else img_path
                    if os.path.exists(full_path):
                        os.remove(full_path)
                        files_removed += 1
                except Exception:
                    pass

            # Notify active pipeline to purge visitor from in-memory cache
            pipeline_manager.remove_visitor(visitor_id)


            self._send_json({
                "success": True,
                "visitor_id": visitor_id,
                "events_deleted": result["events_deleted"],
                "files_removed": files_removed,
                "message": f"Visitor '{visitor_id}' and {result['events_deleted']} event(s) permanently deleted."
            })
            return

        self._send_json({"error": "Endpoint not found"}, status=404)


    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        # 1. Root / Single Page Application
        if path in ("/", "/index.html"):
            index_path = os.path.join(WEB_DIR, "index.html")
            if os.path.exists(index_path):
                with open(index_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return

        # 2. Static CSS and JS assets
        if path == "/style.css":
            css_path = os.path.join(WEB_DIR, "style.css")
            if os.path.exists(css_path):
                with open(css_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/css; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return

        if path == "/app.js":
            js_path = os.path.join(WEB_DIR, "app.js")
            if os.path.exists(js_path):
                with open(js_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/javascript; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return

        # 3. Live MJPEG Video Stream
        if path == "/api/video_feed":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            try:
                while True:
                    frame_bytes = pipeline_manager.get_latest_frame()
                    header = (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(frame_bytes)).encode("ascii") + b"\r\n\r\n"
                    )
                    self.wfile.write(header + frame_bytes + b"\r\n")
                    # Sleep slightly: 30 FPS when running, 5 FPS on standby
                    time.sleep(0.033 if pipeline_manager.is_running else 0.2)
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception:
                return

        # 4. REST API: Statistics & Telemetry
        if path == "/api/stats":
            u_count = db_manager.get_unique_visitor_count()
            en_count = db_manager.get_total_events_count("entry")
            ex_count = db_manager.get_total_events_count("exit")

            self._send_json({
                "unique_visitors": u_count,
                "total_entries": en_count,
                "total_exits": ex_count,
                "active_in_frame": len(pipeline_manager.active_tracks),
                "fps": pipeline_manager.fps,
                "pipeline_running": pipeline_manager.is_running,
                "active_tracks": pipeline_manager.active_tracks,
                "source": pipeline_manager.source
            })
            return

        # 5. REST API: Visitors Directory
        if path == "/api/visitors":
            search = query.get("search", [""])[0]
            sort = query.get("sort", ["newest"])[0]
            limit = int(query.get("limit", [100])[0])
            offset = int(query.get("offset", [0])[0])

            visitors = db_manager.get_visitors(limit=limit, offset=offset, search=search, sort=sort)
            self._send_json({"visitors": visitors})
            return

        # 6. REST API: Single Visitor Details + Event Timeline
        if path.startswith("/api/visitor/"):
            visitor_id = path.split("/api/visitor/")[-1]
            v = db_manager.get_visitor(visitor_id)
            if not v:
                self._send_json({"error": "Visitor not found"}, status=404)
                return
            events = db_manager.get_visitor_events(visitor_id)
            self._send_json({"visitor": v, "events": events})
            return

        # 7. REST API: Audit Events
        if path == "/api/events":
            ev_type = query.get("type", ["all"])[0]
            limit = int(query.get("limit", [50])[0])
            offset = int(query.get("offset", [0])[0])

            # Get recent events from DB
            all_events = db_manager.get_recent_events(limit=limit + offset)
            if ev_type in ("entry", "exit"):
                filtered = [e for e in all_events if e["event_type"] == ev_type]
            else:
                filtered = all_events

            paginated = filtered[offset: offset + limit]
            self._send_json({"events": paginated, "total": len(filtered)})
            return

        # 8. REST API: Footfall Charts Data
        if path == "/api/charts/footfall":
            footfall = db_manager.get_hourly_footfall()
            self._send_json({"footfall": footfall})
            return

        # 9. REST API: System Logs
        if path == "/api/logs":
            log_path = os.path.join(BASE_DIR, "logs", "events.log")
            lines = []
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    all_lines = f.readlines()
                    lines = [ln.strip() for ln in all_lines[-120:]]
            self._send_json({"logs": lines})
            return

        # 10. REST API: Configuration
        if path == "/api/config":
            if os.path.exists(CONFIG_PATH):
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                self._send_json(cfg)
            else:
                self._send_json({})
            return

        # 11. REST API: Export CSV Report
        if path == "/api/export_csv":
            events = db_manager.get_recent_events(limit=5000)
            csv_lines = ["Event ID,Visitor ID,Event Type,Timestamp,Confidence,Image Path,Details"]
            for ev in events:
                csv_lines.append(
                    f"{ev['id']},{ev['visitor_id']},{ev['event_type']},{ev['timestamp']},{ev['confidence'] or ''},\"{ev['image_path']}\",\"{ev['details'] or ''}\""
                )
            csv_data = "\n".join(csv_lines).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", f"attachment; filename=\"visitor_audit_export_{int(time.time())}.csv\"")
            self.send_header("Content-Length", str(len(csv_data)))
            self.end_headers()
            self.wfile.write(csv_data)
            return

        # 12. REST API: List Uploaded Videos
        if path == "/api/uploaded_videos":
            upload_dir = os.path.join(BASE_DIR, "data", "uploads")
            videos = []
            if os.path.exists(upload_dir):
                for fn in os.listdir(upload_dir):
                    if fn.lower().endswith((".mp4", ".avi", ".mkv", ".mov", ".m4v")):
                        fpath = f"data/uploads/{fn}"
                        full_p = os.path.join(upload_dir, fn)
                        sz_mb = round(os.path.getsize(full_p) / (1024 * 1024), 2)
                        videos.append({"filename": fn, "path": fpath, "size_mb": sz_mb})
            self._send_json({"videos": videos})
            return

        # 13. Fallback: Serve local files (face crops in logs/, samples in data/)
        super().do_GET()

    def _handle_file_upload(self):
        """Processes multipart/form-data or binary stream video file uploads."""
        try:
            content_type = self.headers.get("Content-Type", "")
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0:
                self._send_json({"error": "Empty upload payload"}, status=400)
                return

            upload_dir = os.path.join(BASE_DIR, "data", "uploads")
            os.makedirs(upload_dir, exist_ok=True)
            filename = self.headers.get("X-File-Name", "")

            if "multipart/form-data" in content_type:
                boundary_str = content_type.split("boundary=")[-1].strip()
                if boundary_str.startswith('"') and boundary_str.endswith('"'):
                    boundary_str = boundary_str[1:-1]
                boundary = ("--" + boundary_str).encode("ascii")

                raw_body = self.rfile.read(length)
                parts = raw_body.split(boundary)
                saved_path = None
                saved_filename = None

                for part in parts:
                    if b"filename=" in part:
                        sub_parts = part.split(b"\r\n\r\n", 1)
                        if len(sub_parts) == 2:
                            header_bytes, file_bytes = sub_parts
                            if file_bytes.endswith(b"\r\n--"):
                                file_bytes = file_bytes[:-4]
                            elif file_bytes.endswith(b"\r\n"):
                                file_bytes = file_bytes[:-2]

                            header_str = header_bytes.decode("utf-8", errors="ignore")
                            for line in header_str.split("\r\n"):
                                if "filename=" in line:
                                    fn_part = line.split("filename=")[-1].strip()
                                    saved_filename = fn_part.strip('"\'')
                                    saved_filename = os.path.basename(saved_filename)

                            if not saved_filename:
                                saved_filename = f"video_{int(time.time())}.mp4"

                            dest_path = os.path.join(upload_dir, saved_filename)
                            with open(dest_path, "wb") as f:
                                f.write(file_bytes)

                            saved_path = f"data/uploads/{saved_filename}"
                            break

                if saved_path:
                    self._send_json({
                        "success": True,
                        "filepath": saved_path,
                        "filename": saved_filename,
                        "message": f"Successfully uploaded '{saved_filename}'"
                    })
                else:
                    self._send_json({"error": "No file stream found in upload request"}, status=400)
            else:
                if not filename:
                    filename = f"video_{int(time.time())}.mp4"
                filename = os.path.basename(filename)
                file_bytes = self.rfile.read(length)
                dest_path = os.path.join(upload_dir, filename)
                with open(dest_path, "wb") as f:
                    f.write(file_bytes)

                saved_path = f"data/uploads/{filename}"
                self._send_json({
                    "success": True,
                    "filepath": saved_path,
                    "filename": filename,
                    "message": f"Successfully uploaded '{filename}'"
                })
        except Exception as e:
            self._send_json({"error": f"Upload processing failed: {str(e)}"}, status=500)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # File Upload Endpoint
        if path == "/api/upload_video":
            self._handle_file_upload()
            return

        body = self._read_json_body()

        # 1. Start Tracking Pipeline
        if path == "/api/pipeline/start":
            src = body.get("source", "data/video_sample1.mp4")
            skip = body.get("skip_frames", 2)
            conf = body.get("confidence_threshold", 0.45)
            sim = body.get("similarity_threshold", 0.50)

            started = pipeline_manager.start(
                source=src,
                skip_frames=skip,
                conf_thresh=conf,
                sim_thresh=sim
            )
            self._send_json({"success": started, "running": pipeline_manager.is_running})
            return

        # 2. Stop Tracking Pipeline
        if path == "/api/pipeline/stop":
            stopped = pipeline_manager.stop()
            self._send_json({"success": True, "stopped": stopped})
            return

        # 3. Update Visitor Custom Alias
        if path.startswith("/api/visitor/") and path.endswith("/alias"):
            parts = path.split("/")
            visitor_id = parts[3]
            new_alias = body.get("alias", "")
            updated = db_manager.update_visitor_alias(visitor_id, new_alias)
            self._send_json({"success": updated})
            return

        # 4. Save Config
        if path == "/api/config":
            if os.path.exists(CONFIG_PATH):
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                cfg.update(body)
                with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, indent=2)
                self._send_json({"success": True, "config": cfg})
            else:
                self._send_json({"error": "config.json not found"}, status=500)
            return

        # 5. Reset Database & Logs
        if path == "/api/reset_db":
            # Stop active pipeline first
            if pipeline_manager.is_running:
                pipeline_manager.stop()
                time.sleep(0.5)

            db_path = "data/visitor_system.db"
            entries_dir = "logs/entries"
            exits_dir = "logs/exits"
            log_file = "logs/events.log"

            if os.path.exists(db_path):
                try:
                    os.remove(db_path)
                except Exception:
                    pass

            if os.path.exists(entries_dir):
                shutil.rmtree(entries_dir, ignore_errors=True)
                os.makedirs(entries_dir, exist_ok=True)

            if os.path.exists(exits_dir):
                shutil.rmtree(exits_dir, ignore_errors=True)
                os.makedirs(exits_dir, exist_ok=True)

            if os.path.exists(log_file):
                try:
                    os.remove(log_file)
                except Exception:
                    pass

            # Re-init fresh tables
            db_manager.init_db()
            self._send_json({"success": True})
            return

        self._send_json({"error": "Endpoint not found"}, status=404)


def run_dashboard(port: int = PORT):
    os.chdir(BASE_DIR)
    server = ThreadedHTTPServer(("0.0.0.0", port), VisionDashboardHandler)
    print("\n" + "=" * 64)
    print("      AETHERVISION // INTELLIGENT FACE TRACKER WEB HUB")
    print("=" * 64)
    print(f" Web Dashboard URL      : http://localhost:{port}")
    print(f" Live Stream MJPEG Feed : http://localhost:{port}/api/video_feed")
    print(f" REST API Base          : http://localhost:{port}/api/")
    print(f" Registered Visitors    : {db_manager.get_unique_visitor_count()} unique profiles")
    print(" Zero External Web Deps : Powered by Python ThreadedHTTPServer")
    print("=" * 64)
    print(" Press Ctrl+C to terminate dashboard server.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[DASHBOARD] Stopping AetherVision dashboard server...")
        pipeline_manager.stop()
        server.server_close()
        print("[DASHBOARD] Server stopped cleanly.")


if __name__ == "__main__":
    run_dashboard()
