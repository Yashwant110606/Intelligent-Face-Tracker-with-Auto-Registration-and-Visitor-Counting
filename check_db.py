"""
SQLite Database Inspector & Verification Tool for Intelligent Face Tracker.
Run with: python check_db.py
"""

import os
import sqlite3

DB_PATH = "data/visitor_system.db"

def inspect_database():
    print("\n" + "=" * 60)
    print("        SQLITE DATABASE CONNECTION & DATA INSPECTOR")
    print("=" * 60)

    # 1. Check physical file existence
    if not os.path.exists(DB_PATH):
        print(f"[-] Status: Database file NOT found at: {DB_PATH}")
        print("    Run tracking once (python main.py) or dashboard to generate.")
        print("=" * 60 + "\n")
        return

    file_size_kb = os.path.getsize(DB_PATH) / 1024.0
    print(f"[+] Database File Found : {DB_PATH} ({file_size_kb:.1f} KB)")

    # 2. Test Connection
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        print(f"[+] SQLite Engine Version: {sqlite3.sqlite_version}")
        print("[+] Connection Status   : CONNECTED SUCCESSFULLY (Online)")
    except Exception as e:
        print(f"[-] Connection Failed   : {e}")
        return

    # 3. Check Pragma settings
    cur.execute("PRAGMA journal_mode;")
    mode = cur.fetchone()[0]
    cur.execute("PRAGMA foreign_keys;")
    fk = cur.fetchone()[0]
    print(f"[+] Journal Mode         : {mode.upper()} (Safe concurrent read/write)")
    print(f"[+] Foreign Keys Enforced: {'YES' if fk else 'NO'}")

    # 4. List Tables
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [row[0] for row in cur.fetchall()]
    print(f"[+] Existing Tables     : {', '.join(tables)}")

    print("-" * 60)

    # 5. Inspect 'visitors' table
    if 'visitors' in tables:
        cur.execute("SELECT COUNT(*) FROM visitors;")
        v_count = cur.fetchone()[0]
        print(f"👥 VISITORS TABLE: {v_count} unique registered visitors")
        
        cur.execute("""
            SELECT visitor_id, first_seen, last_seen, total_visits, alias 
            FROM visitors 
            ORDER BY created_at DESC 
            LIMIT 5;
        """)
        rows = cur.fetchall()
        if rows:
            print("   Top 5 Recent Visitors in SQLite:")
            for r in rows:
                alias_str = f" ('{r[4]}')" if r[4] else ""
                print(f"   - {r[0]}{alias_str} | Visits: {r[3]} | First: {r[1][:19]} | Last: {r[2][:19]}")
        else:
            print("   (Table is empty)")

    print("-" * 60)

    # 6. Inspect 'events' table
    if 'events' in tables:
        cur.execute("SELECT COUNT(*) FROM events;")
        e_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM events WHERE event_type='entry';")
        entry_cnt = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM events WHERE event_type='exit';")
        exit_cnt = cur.fetchone()[0]

        print(f"📋 EVENTS TABLE: {e_count} total events ({entry_cnt} entries, {exit_cnt} exits)")

        cur.execute("""
            SELECT id, visitor_id, event_type, timestamp, confidence 
            FROM events 
            ORDER BY id DESC 
            LIMIT 5;
        """)
        rows = cur.fetchall()
        if rows:
            print("   Last 5 Logged Events in SQLite:")
            for r in rows:
                conf_str = f"{r[4]*100:.1f}%" if r[4] is not None else "N/A"
                print(f"   - Event #{r[0]}: {r[2].upper():5s} | {r[1]} | {r[3]} | Conf: {conf_str}")
        else:
            print("   (Table is empty)")

    print("=" * 60 + "\n")
    conn.close()

if __name__ == "__main__":
    inspect_database()
