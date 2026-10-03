"""
Database module for Intelligent Face Tracker.
Manages SQLite operations: visitor registration, embeddings, entry/exit events,
and accurate unique visitor counting.
"""

import sqlite3
import os
import threading
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager


class DatabaseManager:
    """Thread-safe SQLite manager for visitor tracking."""

    def __init__(self, db_path: str = "data/visitor_system.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._lock = threading.RLock()
        self.init_db()

    @contextmanager
    def get_connection(self):
        """Returns a connection with busy-timeout and foreign keys enabled."""
        conn = sqlite3.connect(self.db_path, timeout=60.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 30000;")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        try:
            yield conn
        finally:
            conn.close()


    def init_db(self):
        """Initializes tables and indexes."""
        with self._lock:
            with self.get_connection() as conn:
                conn.execute("PRAGMA journal_mode = WAL;")
                cursor = conn.cursor()

                # Visitors Table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS visitors (
                        visitor_id TEXT PRIMARY KEY,
                        first_seen TEXT NOT NULL,
                        last_seen TEXT NOT NULL,
                        total_visits INTEGER DEFAULT 1,
                        embedding BLOB NOT NULL,
                        thumbnail_path TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                # Events Table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS events (
                        visitor_id TEXT NOT NULL,
                        event_type TEXT NOT NULL CHECK(event_type IN ('entry', 'exit')),
                        timestamp TEXT NOT NULL,
                        image_path TEXT NOT NULL,
                        confidence REAL,
                        details TEXT,
                        FOREIGN KEY (visitor_id) REFERENCES visitors(visitor_id) ON DELETE CASCADE
                    );
                """)

                # Indexes for fast lookup
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_visitor ON events(visitor_id);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);")

                # Ensure optional alias column exists
                try:
                    cursor.execute("ALTER TABLE visitors ADD COLUMN alias TEXT;")
                except sqlite3.OperationalError:
                    pass
                conn.commit()

    def register_visitor(
        self,
        visitor_id: str,
        embedding: np.ndarray,
        timestamp: str,
        thumbnail_path: Optional[str] = None
    ) -> bool:
        """
        Auto-registers a new face upon first detection.
        Stores visitor_id, normalized embedding as BLOB, and initial timestamps.
        """
        emb_bytes = embedding.astype(np.float32).tobytes()
        with self._lock:
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO visitors (visitor_id, first_seen, last_seen, total_visits, embedding, thumbnail_path)
                        VALUES (?, ?, ?, 1, ?, ?);
                    """, (visitor_id, timestamp, timestamp, emb_bytes, thumbnail_path))
                    conn.commit()
                    return True
            except sqlite3.IntegrityError:
                return False
            except sqlite3.OperationalError as e:
                print(f"[DB ERROR] Could not register visitor: {e}")
                return False

    def update_visitor_activity(self, visitor_id: str, timestamp: str, increment_visit: bool = False):
        """Updates last_seen and optionally increments total_visits."""
        with self._lock:
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    if increment_visit:
                        cursor.execute("""
                            UPDATE visitors 
                            SET last_seen = ?, total_visits = total_visits + 1
                            WHERE visitor_id = ?;
                        """, (timestamp, visitor_id))
                    else:
                        cursor.execute("""
                            UPDATE visitors 
                            SET last_seen = ?
                            WHERE visitor_id = ?;
                        """, (timestamp, visitor_id))
                    conn.commit()
            except sqlite3.OperationalError as e:
                print(f"[DB ERROR] Could not update activity: {e}")

    def update_visitor_embedding(self, visitor_id: str, embedding: np.ndarray):
        """Updates stored 512-dim embedding for a visitor to refine matching model over time."""
        emb_bytes = embedding.astype(np.float32).tobytes()
        with self._lock:
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE visitors
                        SET embedding = ?
                        WHERE visitor_id = ?;
                    """, (emb_bytes, visitor_id))
                    conn.commit()
            except sqlite3.OperationalError as e:
                print(f"[DB ERROR] Could not update embedding: {e}")

    def log_event(
        self,
        visitor_id: str,
        event_type: str,
        timestamp: str,
        image_path: str,
        confidence: Optional[float] = None,
        details: Optional[str] = None
    ) -> None:
        """
        Logs exactly one entry or exit event per occurrence.
        """
        with self._lock:
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO events (visitor_id, event_type, timestamp, image_path, confidence, details)
                        VALUES (?, ?, ?, ?, ?, ?);
                    """, (visitor_id, event_type, timestamp, image_path, confidence, details))
                    conn.commit()
            except sqlite3.OperationalError as e:
                print(f"[DB ERROR] Could not log event: {e}")

    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Retrieves all registered visitors and their embeddings for fast in-memory matching.
        Returns a dictionary: {visitor_id: np.ndarray(512, dtype=float32)}
        """
        embeddings_dict = {}
        with self._lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT visitor_id, embedding FROM visitors;")
                rows = cursor.fetchall()
                for row in rows:
                    visitor_id = row["visitor_id"]
                    emb_bytes = row["embedding"]
                    emb_arr = np.frombuffer(emb_bytes, dtype=np.float32)
                    embeddings_dict[visitor_id] = emb_arr
        return embeddings_dict

    def get_unique_visitor_count(self) -> int:
        """Returns the total number of unique registered visitors."""
        with self._lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) AS total FROM visitors;")
                row = cursor.fetchone()
                return row["total"] if row else 0

    def get_total_events_count(self, event_type: Optional[str] = None) -> int:
        """Returns the total number of events (optionally filtered by 'entry' or 'exit')."""
        with self._lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                if event_type:
                    cursor.execute("SELECT COUNT(*) AS total FROM events WHERE event_type = ?;", (event_type,))
                else:
                    cursor.execute("SELECT COUNT(*) AS total FROM events;")
                row = cursor.fetchone()
                return row["total"] if row else 0

    def get_recent_events(self, limit: int = 50) -> List[dict]:
        """Returns recent entry/exit events for reporting or inspection."""
        with self._lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT visitor_id, event_type, timestamp, image_path, confidence, details
                    FROM events
                    ORDER BY timestamp DESC
                    LIMIT ?;
                """, (limit,))
                return [dict(row) for row in cursor.fetchall()]

    def get_visitor(self, visitor_id: str) -> Optional[dict]:
        """Fetches visitor details by ID."""
        with self._lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT visitor_id, first_seen, last_seen, total_visits, thumbnail_path, alias, created_at
                    FROM visitors
                    WHERE visitor_id = ?;
                """, (visitor_id,))
                row = cursor.fetchone()
                return dict(row) if row else None

    def get_visitors(
        self,
        limit: int = 100,
        offset: int = 0,
        search: Optional[str] = None,
        sort: str = "newest"
    ) -> List[dict]:
        """Returns paginated, searchable, sorted registered visitors."""
        order_clause = "created_at DESC"
        if sort == "visits":
            order_clause = "total_visits DESC, created_at DESC"
        elif sort == "oldest":
            order_clause = "created_at ASC"
        elif sort == "id":
            order_clause = "visitor_id ASC"

        with self._lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                if search and search.strip():
                    pattern = f"%{search.strip()}%"
                    cursor.execute(f"""
                        SELECT visitor_id, first_seen, last_seen, total_visits, thumbnail_path, alias, created_at
                        FROM visitors
                        WHERE visitor_id LIKE ? OR (alias IS NOT NULL AND alias LIKE ?)
                        ORDER BY {order_clause}
                        LIMIT ? OFFSET ?;
                    """, (pattern, pattern, limit, offset))
                else:
                    cursor.execute(f"""
                        SELECT visitor_id, first_seen, last_seen, total_visits, thumbnail_path, alias, created_at
                        FROM visitors
                        ORDER BY {order_clause}
                        LIMIT ? OFFSET ?;
                    """, (limit, offset))
                return [dict(row) for row in cursor.fetchall()]

    def get_visitor_events(self, visitor_id: str) -> List[dict]:
        """Returns all entry/exit events for a specific visitor, ordered chronologically."""
        with self._lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT visitor_id, event_type, timestamp, image_path, confidence, details
                    FROM events
                    WHERE visitor_id = ?
                    ORDER BY timestamp ASC;
                """, (visitor_id,))
                return [dict(row) for row in cursor.fetchall()]

    def update_visitor_alias(self, visitor_id: str, alias: str) -> bool:
        """Updates or assigns a friendly name/alias to a registered visitor."""
        with self._lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE visitors
                    SET alias = ?
                    WHERE visitor_id = ?;
                """, (alias.strip() if alias else None, visitor_id))
                conn.commit()
                return cursor.rowcount > 0

    def delete_visitor(self, visitor_id: str) -> dict:
        """
        Permanently deletes a visitor and all their associated events.
        """
        image_paths = []
        events_deleted = 0
        deleted = False

        with self._lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute(
                        "SELECT image_path FROM events WHERE visitor_id = ?;",
                        (visitor_id,)
                    )
                    image_paths = [row[0] for row in cursor.fetchall() if row[0]]

                    cursor.execute(
                        "SELECT thumbnail_path FROM visitors WHERE visitor_id = ?;",
                        (visitor_id,)
                    )
                    v_row = cursor.fetchone()
                    if v_row and v_row[0]:
                        image_paths.append(v_row[0])

                    cursor.execute("DELETE FROM events WHERE visitor_id = ?;", (visitor_id,))
                    events_deleted = cursor.rowcount
                    cursor.execute("DELETE FROM visitors WHERE visitor_id = ?;", (visitor_id,))
                    deleted = cursor.rowcount > 0
                    
                    # Reset sqlite auto-increment sequences if tables are fully empty
                    cursor.execute("SELECT COUNT(*) FROM events;")
                    if cursor.fetchone()[0] == 0:
                        cursor.execute("DELETE FROM sqlite_sequence WHERE name='events';")

                    cursor.execute("SELECT COUNT(*) FROM visitors;")
                    if cursor.fetchone()[0] == 0:
                        cursor.execute("DELETE FROM sqlite_sequence WHERE name='visitors';")

                    conn.commit()
                except Exception as e:
                    conn.rollback()
                    raise e


        return {
            "deleted": deleted,
            "events_deleted": events_deleted,
            "image_paths": image_paths
        }

    def get_hourly_footfall(self) -> List[dict]:
        """Returns entries and exits aggregated by hour or timestamp prefix."""
        with self._lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        SUBSTR(timestamp, 1, 13) as hour_group,
                        event_type,
                        COUNT(*) as count
                    FROM events
                    GROUP BY hour_group, event_type
                    ORDER BY hour_group ASC;
                """)
                return [dict(row) for row in cursor.fetchall()]

