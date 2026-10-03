# AetherVision — Intelligent Face Tracker & Unique Visitor Counter

> **Real-time AI-powered biometric analytics system with YOLOv8 face detection, ArcFace deep re-identification, unique visitor counting, entry/exit event logging, and a live enterprise web dashboard.**

---

## Table of Contents

1. [Project Planning](#1-project-planning)
2. [Feature Documentation](#2-feature-documentation)
3. [Compute Load Estimation](#3-compute-load-estimation)
4. [AI Planning Document](#4-ai-planning-document)
5. [Setup Instructions](#5-setup-instructions)
6. [Assumptions Made](#6-assumptions-made)
7. [Sample config.json Structure](#7-sample-configjson-structure)
8. [Project Directory Structure](#8-project-directory-structure)
9. [Tech Stack](#9-tech-stack)

---

## 1. Project Planning

### 1.1 Problem Statement

Traditional video surveillance systems either:
- Count the same person multiple times (every time they pass the camera), inflating visitor counts.
- Fail entirely to re-identify returning visitors across sessions.

This project solves that by building a production-grade **unique visitor identification and counting pipeline** using state-of-the-art deep learning face recognition.

### 1.2 Planning Phases

```
Phase 1 — Architecture Design
  ├── Choose YOLOv8 for lightweight, fast, accurate face detection.
  ├── Choose ArcFace (w600k_mbf.onnx) for SOTA 512-dimensional facial embeddings.
  ├── Design IoU-based multi-object tracking state machine.
  ├── Plan SQLite schema for visitor profiles, embeddings, and events.
  └── Plan frame-skipping strategy for real-time throughput.

Phase 2 — Core Pipeline Development
  ├── detector.py   → YOLOv8 face detection module
  ├── recognizer.py → ArcFace ONNX embedding extractor + cosine similarity matching
  ├── tracker.py    → IoU tracker + entry/exit state machine + auto-registration
  ├── pipeline.py   → Video/RTSP orchestration engine with HUD rendering
  └── database.py   → Thread-safe SQLite manager

Phase 3 — Logging & Storage
  ├── logger_system.py → events.log + dated face crop filesystem store
  └── data/visitor_system.db → persistent biometric identity database

Phase 4 — Web Dashboard
  ├── dashboard.py  → Threaded HTTP server + MJPEG live stream + REST APIs
  ├── web/index.html → SPA dashboard UI (5 navigation tabs)
  ├── web/style.css  → Obsidian Cyber-Glassmorphism design system
  └── web/app.js    → Live polling, charts, visitor gallery, upload, config sync

Phase 5 — Optimization & Fixes
  ├── Frame pacing to eliminate slow-motion video lag
  ├── Adaptive embedding refinement on each re-identification pass
  ├── Video file upload endpoint for custom video processing
  └── Similarity threshold tuning (0.40) for dark lighting / glasses
```

---

## 2. Feature Documentation

### 2.1 Face Detection
- **Model**: YOLOv8-nano-face (`yolov8n-face.pt`)
- **Inference size**: 640×640 input (auto-resized)
- **Configurable confidence threshold** (`confidence_threshold`, default: `0.45`)
- **Minimum face size filter** (`min_face_size`, default: `25px`) eliminates tiny false positives
- **GPU acceleration**: Automatically uses CUDA if NVIDIA GPU is available; falls back to CPU
- **Output**: Bounding boxes `(x1, y1, x2, y2)` + confidence score + cropped face ROI

### 2.2 Face Recognition & Auto-Registration
- **Model**: ArcFace InsightFace MobileFaceNet (`w600k_mbf.onnx`)
- **Embedding dimensionality**: 512-dimensional L2-normalized facial feature vector
- **Similarity metric**: Cosine Similarity between unit embeddings
- **Threshold** (`similarity_threshold`, default: `0.40`)
- **Auto-Registration**: New faces automatically assigned unique sequential IDs (`VISITOR_0001`, `VISITOR_0002`, ...)
- **Adaptive Template Refinement**: On each successful re-identification, stored embedding is updated via exponential moving average (`0.85 × old + 0.15 × new`) to adapt to lighting/pose drift over time

### 2.3 Unique Visitor Counting (Core Feature)
- Maintained via `SELECT COUNT(*) FROM visitors` — one row per unique face
- Re-identification of returning visitors **never increments** the unique count
- Persistent across multiple sessions and stream restarts
- Retrievable via:
  - **Dashboard API**: `GET http://localhost:8080/api/stats`
  - **CLI**: `python main.py --stats`
  - **DB Inspector**: `python check_db.py`
  - **Raw SQL**: `SELECT COUNT(*) FROM visitors;`

### 2.4 Multi-Object Tracking (IoU-Based)
- **Algorithm**: Greedy IoU cost matrix matching
- **IoU Threshold** (`iou_match_threshold`, default: `0.35`)
- **Frame Skipping** (`detection_skip_frames`, default: `2`): full YOLO detection runs every N+1 frames
- **Max Disappeared Frames** (`max_disappeared_frames`, default: `25`): tracks persist through brief occlusions before exit is logged

### 2.5 Entry & Exit Event Logging
- **Entry Event**: Logged when track accumulates ≥2 detection hits
- **Exit Event**: Logged when track disappears for more than `max_disappeared_frames`
- Each event records: `visitor_id`, `event_type`, `timestamp`, `confidence`, `image_path`
- Face crops stored under `logs/entries/YYYY-MM-DD/` or `logs/exits/YYYY-MM-DD/`

### 2.6 Real-Time Frame Pacing (Anti-Lag)
- Reads native video FPS via `cv2.CAP_PROP_FPS`
- Wall-clock synchronization prevents slow-motion playback
- Dynamically drops intermediate frames when processing falls behind

### 2.7 Video File Upload
- Browser drag-and-drop or file picker for MP4 / AVI / MKV / MOV files
- Uploaded files saved to `data/uploads/`
- REST endpoint: `POST http://localhost:8080/api/upload_video`

### 2.8 Live RTSP Camera Support
- Connects to live RTSP cameras (`rtsp://username:password@ip:port/stream`)
- Automatic reconnection with configurable `rtsp_reconnect_delay`

### 2.9 Web Dashboard (AetherVision UI)
- **Tab 1 — Live Vision Feed**: MJPEG stream, HUD overlay, pipeline controls, upload, calibration sliders
- **Tab 2 — Analytics Hub**: KPI cards, hourly footfall bar chart, entry/exit donut chart
- **Tab 3 — Visitors Gallery**: All registered profiles with face thumbnails, visit counts, alias assignment
- **Tab 4 — Audit Trail**: Full event table with face snapshots, confidence scores, CSV export
- **Tab 5 — System Console**: Live events.log terminal, database reset

### 2.10 Database Persistence
- **SQLite** with WAL journal mode for concurrent read/write
- `visitors` table: unique profiles + 512-dim embedding BLOB
- `events` table: all entry/exit events with full metadata

### 2.11 Annotated Video Output
- Saves full annotated MP4 with bounding boxes, visitor IDs, and HUD overlay

### 2.12 Configuration & CLI
- All parameters managed via `config.json`
- CLI: `--source`, `--skip`, `--conf`, `--sim`, `--no-display`, `--stats`, `--reset-db`

---

## 3. Compute Load Estimation

### 3.1 CPU-Only Mode (Default)

| Component | Estimated CPU Load | Time per Frame |
| :--- | :---: | :---: |
| YOLOv8-nano Detection | ~60–80% single core | 80–150 ms |
| ArcFace Embedding Extraction | ~40–60% single core | 30–60 ms |
| IoU Tracking Matrix | ~5% | < 2 ms |
| SQLite Read/Write | ~5% | 2–8 ms |
| MJPEG JPEG Encoding | ~10% | 5–15 ms |
| **Total (full detection frame)** | **~80–120% CPU** | **~120–230 ms** |
| **Effective FPS (skip_frames=2)** | — | **~15–25 FPS** |

> Detection runs every 3rd frame. Skip frames cost only IoU tracking (~2 ms), maintaining 25–35 FPS display throughput.

### 3.2 GPU Mode (NVIDIA CUDA)

| Component | GPU Load | Time per Frame |
| :--- | :---: | :---: |
| YOLOv8-nano Detection | ~30–50% GPU | 8–20 ms |
| ArcFace Embedding Extraction | ~20–40% GPU | 5–15 ms |
| IoU Tracking + DB | ~5% CPU | 3–10 ms |
| **Total (full detection frame)** | **~40–70% GPU** | **~15–45 ms** |
| **Effective FPS (skip_frames=2)** | — | **~40–70 FPS** |

### 3.3 Memory Usage

| Resource | CPU Mode | GPU Mode |
| :--- | :---: | :---: |
| RAM | 400–700 MB | 400–700 MB |
| VRAM | — | 300–500 MB |
| SQLite DB (1000 visitors) | ~8–12 MB | ~8–12 MB |

### 3.4 Recommended Hardware

| Tier | Hardware | Expected FPS |
| :--- | :--- | :---: |
| Minimum | Intel Core i5 / 8GB RAM | 10–20 FPS |
| Recommended CPU | Intel Core i7/i9, 16GB RAM | 20–30 FPS |
| Optimal (GPU) | NVIDIA RTX 3060+, 16GB RAM | 40–70 FPS |

---

## 4. AI Planning Document

### 4.1 System Architecture

```
INPUT LAYER
  ├── Video File (MP4 / AVI / MKV)
  ├── Webcam (index 0, 1, ...)
  └── RTSP Network Camera

DETECTION LAYER (YOLOv8)
  ├── Detect all face bounding boxes per frame
  ├── Filter by confidence threshold (≥ 0.45)
  └── Filter by minimum face size (≥ 25px)

TRACKING LAYER (IoU Spatial Tracker)
  ├── Match detections to existing tracks via IoU matrix
  ├── Maintain track continuity across frame skip cycles
  └── Handle occlusion via max_disappeared counter

RECOGNITION LAYER (ArcFace Deep ReID)
  ├── Extract 512-dim normalized facial embedding
  ├── Compare via cosine similarity against all DB embeddings
  ├── IF similarity ≥ threshold → Re-identify (no new visitor)
  └── IF similarity < threshold → Auto-register new visitor

PERSISTENCE LAYER (SQLite)
  ├── Store unique visitor profiles + embeddings
  ├── Log all entry/exit events with timestamps + face crops
  └── COUNT(*) for unique visitor retrieval

PRESENTATION LAYER
  ├── HUD overlay on annotated video frames
  ├── MJPEG live stream to web browser
  ├── REST API for stats, visitors, events
  └── AetherVision Web Dashboard
```

### 4.2 Key AI Design Decisions

| Decision | Rationale |
| :--- | :--- |
| YOLOv8-nano | Optimal speed vs. accuracy for real-time face detection on CPU |
| ArcFace MobileFaceNet | SOTA facial recognition with minimal inference cost |
| Cosine Similarity | Rotation-invariant after L2-normalization; outperforms Euclidean for embeddings |
| IoU Tracking (not Kalman Filter) | Simpler, faster, sufficient for near-frontal indoor face tracking |
| SQLite | Zero-dependency, ACID-compliant, perfect for single-node deployments |
| Adaptive Embedding EMA | Prevents identity drift (85% old + 15% new) across lighting/pose changes |
| Frame Skipping | Decouples detection latency from display FPS for CPU-only deployments |
| Wall-Clock Frame Pacing | Ensures video files play at native speed without slow-motion artifacts |

### 4.3 Re-Identification Failure Modes & Mitigations

| Failure Mode | Root Cause | Mitigation |
| :--- | :--- | :--- |
| Same person registered twice | Dark lighting/glasses → similarity dips below threshold | Lowered threshold to 0.40; adaptive EMA refinement |
| False re-identification | Threshold too low → similar-looking faces collapse | Threshold floor at 0.38; visual monitoring via dashboard |
| Track fragmentation | Fast motion / small crops → low IoU | `min_face_size` filter + `max_disappeared_frames` buffer |
| High CPU at peak crowd | Many simultaneous ArcFace calls | Frame skipping reduces embedding calls by 60–70% |

---

## 5. Setup Instructions

### 5.1 Prerequisites
- Python 3.9 or higher
- pip package manager
- (Optional) NVIDIA GPU with CUDA 11.8+ for GPU acceleration

### 5.2 Create Virtual Environment

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# macOS / Linux:
source .venv/bin/activate
```

### 5.3 Install Dependencies

```bash
pip install -r requirements.txt
```

> For GPU support, install CUDA PyTorch instead:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

### 5.4 Verify Model Files

```
models/
├── yolov8n-face.pt       # YOLOv8 nano face detection model
└── w600k_mbf.onnx        # ArcFace MobileFaceNet recognition model
```

### 5.5 Running the System

**Option A — Web Dashboard (Recommended):**
```bash
python dashboard.py
# Open http://localhost:8080 in browser
```

**Option B — Command Line:**
```bash
python main.py --source data/video_sample1.mp4    # Video file
python main.py --source 0                          # Webcam
python main.py --source rtsp://user:pass@ip/stream # RTSP camera
python main.py --source data/video_sample1.mp4 --no-display  # Headless
```

**Option C — Utilities:**
```bash
python main.py --stats      # View statistics
python check_db.py          # Inspect database
python main.py --reset-db   # Reset for fresh benchmark
```

---

## 6. Assumptions Made

1. **Camera Angle**: Near-frontal or slight-angle perspective assumed. 90° profile views degrade ArcFace accuracy significantly.

2. **Lighting Conditions**: Moderate indoor lighting assumed. Extreme low-light or strong backlight can reduce re-identification similarity by 10–25%.

3. **Single Camera Instance**: Designed for single-node, single-camera deployments. Multi-camera federation requires shared embedding database infrastructure.

4. **Non-Adversarial Environment**: No disguises, face coverings, or identity spoofing attacks are considered. Designed for legitimate visitor counting.

5. **Face Size Minimum**: Faces smaller than 25×25 pixels are filtered out. Visitors assumed within ~2–5 meters of camera.

6. **SQLite Concurrency**: Designed for single-machine deployments. For multi-process or distributed inference, a proper RDBMS (PostgreSQL) is recommended at scale.

7. **Video FPS Accuracy**: Frame-pacing assumes `cv2.CAP_PROP_FPS` metadata is accurate. Files with incorrect FPS metadata may exhibit slight timing drift.

8. **Embedding Uniqueness**: ArcFace embeddings assumed unique per individual under normal conditions. Identical twins may occasionally share embedding neighborhoods.

9. **Similarity Threshold**: Default `0.40` tuned for indoor CCTV with moderate lighting. Dashboard calibration slider recommended for environment-specific fine-tuning.

10. **Internet Connectivity**: Google Fonts loaded from CDN in the dashboard. Removing the font import from `web/index.html` enables fully offline operation.

---

## 7. Sample config.json Structure

```json
{
  "detection_skip_frames": 2,
  "confidence_threshold": 0.45,
  "similarity_threshold": 0.40,
  "max_disappeared_frames": 25,
  "min_face_size": 25,
  "iou_match_threshold": 0.35,
  "input_source": "data/video_sample1.mp4",
  "database_path": "data/visitor_system.db",
  "log_file": "logs/events.log",
  "entries_dir": "logs/entries",
  "exits_dir": "logs/exits",
  "output_video_path": "logs/output_annotated.mp4",
  "save_annotated_video": true,
  "display_window": true,
  "rtsp_reconnect_delay": 2.0
}
```

### Parameter Reference

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `detection_skip_frames` | `int` | `2` | Frames to skip between YOLO detection cycles. `0` = detect every frame. |
| `confidence_threshold` | `float` | `0.45` | Minimum YOLO confidence (0–1) to accept a detection. |
| `similarity_threshold` | `float` | `0.40` | Minimum ArcFace cosine similarity to re-identify a visitor. Lower = more lenient. |
| `max_disappeared_frames` | `int` | `25` | Frames a track can be absent before marked as exited. |
| `min_face_size` | `int` | `25` | Minimum bounding box size in pixels. Smaller detections discarded. |
| `iou_match_threshold` | `float` | `0.35` | Minimum IoU to link detections to existing tracks across frames. |
| `input_source` | `string` | `"data/video_sample1.mp4"` | Video file path, webcam index (`"0"`), or RTSP URL. |
| `database_path` | `string` | `"data/visitor_system.db"` | SQLite database file path. |
| `log_file` | `string` | `"logs/events.log"` | System audit log file path. |
| `entries_dir` | `string` | `"logs/entries"` | Root directory for entry face crop images. |
| `exits_dir` | `string` | `"logs/exits"` | Root directory for exit face crop images. |
| `output_video_path` | `string` | `"logs/output_annotated.mp4"` | Output path for annotated video recording. |
| `save_annotated_video` | `bool` | `true` | Enable/disable annotated video file output. |
| `display_window` | `bool` | `true` | Show/hide OpenCV display window. Set `false` for headless server mode. |
| `rtsp_reconnect_delay` | `float` | `2.0` | Seconds to wait before reconnecting after RTSP stream interruption. |

---

## 8. Project Directory Structure

```
Intelligent Face Tracker/
├── config.json                    # Central system configuration
├── main.py                        # CLI entry point
├── dashboard.py                   # Enterprise web server + REST APIs + MJPEG stream
├── pipeline.py                    # Core video processing engine
├── detector.py                    # YOLOv8 face detection module
├── recognizer.py                  # ArcFace ONNX embedding extractor + matching
├── tracker.py                     # IoU multi-object tracker + state machine
├── database.py                    # SQLite thread-safe database manager
├── logger_system.py               # events.log + face crop filesystem manager
├── check_db.py                    # Database inspection utility
├── requirements.txt               # Python package dependencies
├── README.md                      # This documentation file
│
├── models/
│   ├── yolov8n-face.pt            # YOLOv8 nano face detection weights
│   ├── yolov8n-face.onnx          # ONNX version of YOLOv8 face model
│   └── w600k_mbf.onnx             # ArcFace MobileFaceNet recognition weights
│
├── data/
│   ├── visitor_system.db          # SQLite biometric database
│   ├── video_sample1.mp4          # Sample multi-visitor office test video
│   ├── record_sample.mp4          # Sample continuous walkthrough test video
│   └── uploads/                   # User-uploaded custom video files
│
├── logs/
│   ├── events.log                 # System audit event log
│   ├── output_annotated.mp4       # Annotated video output
│   ├── entries/
│   │   └── YYYY-MM-DD/            # Dated entry face crop images
│   └── exits/
│       └── YYYY-MM-DD/            # Dated exit face crop images
│
└── web/
    ├── index.html                 # Single-page dashboard application
    ├── style.css                  # Obsidian Cyber-Glassmorphism design system
    └── app.js                     # Frontend application logic
```

---

## 9. Tech Stack

| Module | Technology |
| :--- | :--- |
| **Face Detection** | YOLOv8-nano (`ultralytics`, `yolov8n-face.pt`) |
| **Face Recognition** | ArcFace InsightFace MobileFaceNet (`onnxruntime`, `w600k_mbf.onnx`) |
| **Tracking Algorithm** | Custom IoU-based Multi-Object Tracker + Deep ReID State Machine |
| **Video Processing** | OpenCV (`cv2`) |
| **Deep Learning Framework** | PyTorch (`torch`, `torchvision`) |
| **ONNX Inference Runtime** | `onnxruntime` (CPU + optional CUDA) |
| **Backend & Processing** | Python 3.9+ |
| **Database** | SQLite (WAL mode, thread-safe) |
| **Web Server** | Python `http.server` + `socketserver.ThreadingMixIn` |
| **Live Video Streaming** | MJPEG over HTTP multipart (`/api/video_feed`) |
| **Configuration** | JSON (`config.json`) |
| **Logging** | Python `logging` + `RotatingFileHandler` + filesystem face crop store |
| **Camera Input** | OpenCV `VideoCapture` (video files, webcam, RTSP URL) |
| **Frontend** | Vanilla HTML5, CSS3, JavaScript (zero framework dependencies) |

---

> This project is a part of a hackathon run by https://katomaran.com


An AI-driven unique visitor counter and real-time face tracking system designed for video files, live webcams, and RTSP camera streams.

---

## 🎯 Key Capabilities & Architecture

- **Face Detection (YOLOv8)**:
  - Powered by `yolov8n-face`, optimized for low-latency facial detection.
  - Automatically filters false positives using configurable confidence (`conf_threshold`) and minimum bounding box size (`min_face_size`).
  - Supports GPU (`cuda:0` / NVIDIA RTX) acceleration with CPU fallback.

- **Face Recognition & Auto-Registration (ArcFace / InsightFace)**:
  - Extracts 512-dimensional normalized facial embeddings using ArcFace (`w600k_mbf.onnx`).
  - **Auto-Registration**: Upon first sighting of an unrecognized face, the system automatically registers the visitor, generates a unique ID (`VISITOR_XXXX`), archives a reference crop, and persists the embedding into the database.
  - **Re-Identification**: Uses cosine similarity against database embeddings. Returning visitors are recognized instantly across frames and multiple sessions without duplicate registration.
  - **Strictly avoids legacy `face_recognition` library** in favor of production-grade ArcFace SOTA embeddings.

- **Unique Visitor Counting**:
  - Maintains a real-time, non-duplicating counter of unique individuals.
  - Multiple sightings or re-entries of the same person will never increment the unique visitor count.
  - Count is synced and retrievable directly from SQLite (`SELECT COUNT(*) FROM visitors`).

- **Configurable Frame Skipping**:
  - Parameter `detection_skip_frames` in `config.json` controls how many frames to skip between full YOLO detection cycles.
  - Intermediate frames maintain smooth tracking, delivering high FPS throughput (30–60+ FPS).

- **Logging System & Storage**:
  - **Structured Filesystem**: Every entry and exit generates a cropped face image stored neatly under:
    - `logs/entries/YYYY-MM-DD/<visitor_id>_entry_<timestamp>.jpg`
    - `logs/exits/YYYY-MM-DD/<visitor_id>_exit_<timestamp>.jpg`
  - **Database Persistence**: SQLite database (`data/visitor_system.db`) records visitor profiles, embedding vectors, entry/exit timestamps, confidence levels, and snapshot paths.
  - **Audit Log (`events.log`)**: Tracks all critical lifecycle events:
    - Embedding generation & timing
    - Auto-registration & ID assignment
    - Face entry and exit transitions
    - Real-time tracking & re-identification

- **Live RTSP Stream & Video Support**:
  - Seamlessly handles recorded video files (`.mp4`, `.avi`, `.mkv`), webcams (`0`), and live network RTSP camera streams (`rtsp://username:password@ip:port/stream`).
  - Includes automatic stream reconnection logic with exponential backoff for network drops.

---

## 📁 Project Directory Structure

```text
d:/Intelligent Face Tracker/
├── config.json               # Main configuration (thresholds, frame skip, paths)
├── database.py               # SQLite schema & thread-safe query manager
├── detector.py               # YOLOv8 face detector module
├── recognizer.py             # ArcFace / InsightFace 512-dim embedding extractor
├── tracker.py                # Multi-object tracker, state machine, entry/exit logic
├── pipeline.py               # RTSP / Video pipeline engine & HUD overlay
├── main.py                   # Main CLI entry point
├── dashboard.py              # Zero-dependency live web dashboard (http://localhost:8080)
├── requirements.txt          # Python dependencies
├── models/
│   ├── yolov8n-face.pt       # Pretrained YOLOv8 face detector
│   └── w600k_mbf.onnx        # SOTA ArcFace feature extractor
├── data/
│   ├── visitor_system.db     # SQLite database
│   └── video_sample1.mp4     # Sample company test video
└── logs/
    ├── events.log            # System event log
    ├── entries/              # Dated entry face crops
    └── exits/                # Dated exit face crops
```

---

## ⚙️ Configuration (`config.json`)

```json
{
  "detection_skip_frames": 2,
  "confidence_threshold": 0.45,
  "similarity_threshold": 0.50,
  "max_disappeared_frames": 25,
  "min_face_size": 25,
  "iou_match_threshold": 0.35,
  "input_source": "data/video_sample1.mp4",
  "database_path": "data/visitor_system.db",
  "log_file": "logs/events.log",
  "entries_dir": "logs/entries",
  "exits_dir": "logs/exits",
  "output_video_path": "logs/output_annotated.mp4",
  "save_annotated_video": true,
  "display_window": true,
  "rtsp_reconnect_delay": 2.0
}
```

---

## 🚀 Running the System

### 1. Process Video File (Development)
```bash
python main.py --source data/video_sample1.mp4
```

### 2. Process Live RTSP Camera (Interview / Production)
```bash
python main.py --source rtsp://admin:password123@192.168.1.100:554/h264Preview_01_main
```

### 3. Process Live Webcam
```bash
python main.py --source 0
```

### 4. Custom Frame Skipping & Headless Mode
```bash
# Skip 3 frames between YOLO detection cycles, run headless in background
python main.py --source data/video_sample1.mp4 --skip 3 --no-display
```

### 5. Check Visitor Statistics & Audit Log
```bash
python main.py --stats
```

### 6. Reset System for Fresh Benchmark
```bash
python main.py --reset-db
```

---

## 📊 Live Web Dashboard

Launch the zero-dependency web dashboard:
```bash
python dashboard.py
```
Open [http://localhost:8080](http://localhost:8080) in any browser to see:
- Real-time **Unique Visitor Count**
- Total Entries & Exits
- Gallery of Registered Visitors with face thumbnails
- Full Audit Log table with timestamped snapshots and entry/exit statuses
- Auto-refreshes every 5 seconds.
  
Logs:
1. System Events Log :
The primary plain-text log file that records the underlying operations of the AI pipeline.
•	Initialization & State: Logs pipeline startups, model initializations, and graceful shutdowns.
•	Biometric Performance: Records the latency (in milliseconds) required to generate the 512-dimensional facial embeddings for profiling.
•	Recognition Confidence: Logs whenever a live track successfully matches an existing biometric template, including the specific cosine similarity score.
•	Exception Handling: Captures dropped frames, unhandled connection aborts, and database locks for debugging purposes.

2. Visual Audit Snapshots :
To provide undeniable proof of events, the system logs cropped facial snapshots rather than relying strictly on text.
•	Entry Snapshots: When an individual enters the camera frame and meets the minimum tracking threshold, a localized face thumbnail is saved to a timestamped folder in logs/entries/.
•	Exit Snapshots: When the tracking algorithm determines a face has departed the frame (exceeding the max_disappeared_frames threshold), a final reference snapshot is written to logs/exits/.

3. Database Event Ledger :
The SQLite database maintains an immutable events table that acts as the backbone for the web dashboard's Audit Trail. Every physical transition creates a structured row containing:
•	The assigned visitor_id (e.g., VISITOR_0027)
•	The precise ISO-formatted timestamp of the event.
•	The event classification (entry or exit).
•	The relative filesystem path mapping directly to the corresponding visual audit snapshot.

4. Annotated Video Archiving :
Controlled by the save_annotated_video flag in config.json, the system can optionally render and export a permanent video record. This output file bakes the YOLO bounding boxes, assigned Track IDs, and real-time model confidence percentages directly into the video frames, providing a complete historical recreation of the session.

