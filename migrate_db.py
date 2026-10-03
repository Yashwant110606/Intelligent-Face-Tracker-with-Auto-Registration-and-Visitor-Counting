import sqlite3

def migrate():
    conn = sqlite3.connect('data/visitor_system.db')
    cursor = conn.cursor()
    
    # Create new table without ID
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events_new (
            visitor_id TEXT NOT NULL,
            event_type TEXT NOT NULL CHECK(event_type IN ('entry', 'exit')),
            timestamp TEXT NOT NULL,
            image_path TEXT NOT NULL,
            confidence REAL,
            details TEXT,
            FOREIGN KEY (visitor_id) REFERENCES visitors(visitor_id) ON DELETE CASCADE
        );
    """)
    
    # Copy data
    cursor.execute("""
        INSERT INTO events_new (visitor_id, event_type, timestamp, image_path, confidence, details)
        SELECT visitor_id, event_type, timestamp, image_path, confidence, details FROM events;
    """)
    
    # Drop old and rename
    cursor.execute("DROP TABLE events;")
    cursor.execute("ALTER TABLE events_new RENAME TO events;")
    
    # Recreate indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_visitor ON events(visitor_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);")
    
    conn.commit()
    conn.close()
    print("Migration successful")

if __name__ == '__main__':
    migrate()
