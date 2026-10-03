"""
Main CLI entry-point for Intelligent Face Tracker with Auto-Registration and Visitor Counting.
"""

import sys
import os
import argparse
import json
import shutil

from database import DatabaseManager
from logger_system import SystemLogger


def print_stats(config_path: str = "config.json"):
    """Displays visitor and event statistics from SQLite."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    db_path = cfg.get("database_path", "data/visitor_system.db")
    db = DatabaseManager(db_path)

    unique_count = db.get_unique_visitor_count()
    entry_count = db.get_total_events_count("entry")
    exit_count = db.get_total_events_count("exit")

    print("\n" + "="*50)
    print("      INTELLIGENT FACE TRACKER STATISTICS")
    print("="*50)
    print(f" Database Path         : {db_path}")
    print(f" Total Unique Visitors : {unique_count}")
    print(f" Total Entry Events    : {entry_count}")
    print(f" Total Exit Events     : {exit_count}")
    print("="*50)

    events = db.get_recent_events(limit=10)
    if events:
        print("\nLast 10 Logged Events:")
        for ev in events:
            print(f" [{ev['timestamp']}] {ev['event_type'].upper():5s} | Visitor: {ev['visitor_id']} | Image: {ev['image_path']}")
    else:
        print("\nNo events logged yet.")
    print("="*50 + "\n")


def reset_database(config_path: str = "config.json"):
    """Clears SQLite database and image logs for fresh benchmarking."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    db_path = cfg.get("database_path", "data/visitor_system.db")
    entries_dir = cfg.get("entries_dir", "logs/entries")
    exits_dir = cfg.get("exits_dir", "logs/exits")
    log_file = cfg.get("log_file", "logs/events.log")

    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed database: {db_path}")

    if os.path.exists(entries_dir):
        shutil.rmtree(entries_dir)
        os.makedirs(entries_dir, exist_ok=True)
        print(f"Cleared entries directory: {entries_dir}")

    if os.path.exists(exits_dir):
        shutil.rmtree(exits_dir)
        os.makedirs(exits_dir, exist_ok=True)
        print(f"Cleared exits directory: {exits_dir}")

    if os.path.exists(log_file):
        os.remove(log_file)
        print(f"Cleared log file: {log_file}")

    print("System reset complete. Ready for new stream!")


def main():
    parser = argparse.ArgumentParser(description="Intelligent Face Tracker & Unique Visitor Counter")
    parser.add_argument("--source", type=str, default=None, help="Path to video file, webcam index (0), or RTSP URL")
    parser.add_argument("--skip", type=int, default=None, help="Number of frames to skip between YOLO detection cycles")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold for YOLOv8 face detection")
    parser.add_argument("--sim", type=float, default=None, help="Similarity threshold for ArcFace re-identification")
    parser.add_argument("--no-display", action="store_true", help="Run without graphical display window (headless mode)")
    parser.add_argument("--no-save", action="store_true", help="Do not save output annotated video")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames to process")
    parser.add_argument("--stats", action="store_true", help="Show database visitor counts and exit")
    parser.add_argument("--reset-db", action="store_true", help="Reset database and logs for fresh start")
    parser.add_argument("--dashboard", action="store_true", help="Launch the AetherVision Live Web UI on http://localhost:8080")
    parser.add_argument("--port", type=int, default=8080, help="Port for web dashboard server (default: 8080)")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")

    args = parser.parse_args()

    if args.dashboard:
        from dashboard import run_dashboard
        run_dashboard(port=args.port)
        return

    if args.stats:
        print_stats(args.config)
        return

    if args.reset_db:
        reset_database(args.config)
        return

    # Update config on the fly if CLI flags provided
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if args.source is not None:
        cfg["input_source"] = args.source
    if args.skip is not None:
        cfg["detection_skip_frames"] = args.skip
    if args.conf is not None:
        cfg["confidence_threshold"] = args.conf
    if args.sim is not None:
        cfg["similarity_threshold"] = args.sim
    if args.no_display:
        cfg["display_window"] = False
    if args.no_save:
        cfg["save_annotated_video"] = False

    # Save temp modified config
    with open(args.config, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # Launch pipeline
    from pipeline import FaceTrackerPipeline
    pipeline = FaceTrackerPipeline(config_path=args.config)
    pipeline.run(source_override=cfg["input_source"], max_frames=args.max_frames)


if __name__ == "__main__":
    main()
