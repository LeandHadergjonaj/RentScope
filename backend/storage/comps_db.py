"""
SQLite comparables database layer.
Handles storage, retrieval, and maintenance of property comparables.
"""
import sqlite3
import os
import math
import shutil
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "comps.db")

def health_check_db(db_path: str) -> bool:
    """
    Check database integrity using PRAGMA integrity_check.
    Returns True if healthy, False if corrupted.
    """
    if not os.path.exists(db_path):
        return True  # Non-existent DB is not corrupted, just missing
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA integrity_check")
        result = cursor.fetchone()
        conn.close()
        
        # PRAGMA integrity_check returns ('ok',) if healthy
        if result and result[0] == "ok":
            return True
        else:
            logger.warning(f"DB integrity check failed: {result}")
            return False
    except sqlite3.DatabaseError as e:
        logger.warning(f"DB integrity check raised DatabaseError: {e}")
        return False
    except Exception as e:
        logger.warning(f"DB integrity check raised exception: {e}")
        return False

def ensure_db(db_path: str = DB_PATH) -> bool:
    """
    Create database and table if they don't exist.
    If DB is corrupted, backup and recreate it.
    Returns True if DB was recreated (fresh/empty), False otherwise.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Check if DB exists and is healthy
    db_recreated = False
    if os.path.exists(db_path):
        if not health_check_db(db_path):
            # DB is corrupted - backup and recreate
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{db_path}.corrupt.{timestamp}"
            try:
                shutil.move(db_path, backup_path)
                logger.warning(f"DB corrupted → backed up to {backup_path} and recreated")
                print(f"DB corrupted → backed up to {os.path.basename(backup_path)} and recreated")
                db_recreated = True
            except Exception as e:
                logger.error(f"Failed to backup corrupted DB: {e}")
                # Try to delete corrupted file
                try:
                    os.remove(db_path)
                except:
                    pass
                db_recreated = True
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create comparables table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS comparables (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                source_url TEXT UNIQUE,
                captured_at TEXT NOT NULL,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                postcode TEXT,
                borough TEXT,
                price_pcm REAL NOT NULL,
                bedrooms INTEGER NOT NULL,
                bathrooms INTEGER,
                property_type TEXT NOT NULL,
                floor_area_sqm REAL,
                floorplan_used INTEGER DEFAULT 0,
                furnished TEXT,
                quality TEXT,
                photo_condition_label TEXT,
                photo_condition_score REAL,
                parsing_confidence TEXT,
                location_source TEXT,
                location_precision_m REAL
            )
        """)
        
        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_comps_time ON comparables(captured_at)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_comps_geo ON comparables(lat, lon)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_comps_features ON comparables(bedrooms, property_type)
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"Database ensured at {db_path}")
        return db_recreated
    except sqlite3.DatabaseError as e:
        logger.error(f"Database error in ensure_db: {e}")
        # If we still get corruption errors, try to delete and recreate
        try:
            if os.path.exists(db_path):
                os.remove(db_path)
            conn = sqlite3.connect(db_path)
            conn.close()
            return True  # Recreated
        except Exception as e2:
            logger.error(f"Failed to recreate DB: {e2}")
            return False
    except Exception as e:
        logger.error(f"Error ensuring database: {e}")
        return False

def upsert_listing(db_path: str, item: Dict) -> bool:
    """
    Insert or replace listing by source_url.
    Returns True if successful, False otherwise.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if exists
        cursor.execute("SELECT id, first_seen_at FROM comparables WHERE source_url = ?", (item.get("source_url"),))
        existing = cursor.fetchone()
        
        if existing:
            # Update existing record, preserve first_seen_at
            cursor.execute("""
                UPDATE comparables SET
                    source = ?,
                    captured_at = ?,
                    last_seen_at = ?,
                    lat = ?,
                    lon = ?,
                    postcode = ?,
                    borough = ?,
                    price_pcm = ?,
                    bedrooms = ?,
                    bathrooms = ?,
                    property_type = ?,
                    floor_area_sqm = ?,
                    floorplan_used = ?,
                    furnished = ?,
                    quality = ?,
                    photo_condition_label = ?,
                    photo_condition_score = ?,
                    parsing_confidence = ?,
                    location_source = ?,
                    location_precision_m = ?
                WHERE source_url = ?
            """, (
                item.get("source"),
                item.get("captured_at"),
                item.get("last_seen_at"),
                item.get("lat"),
                item.get("lon"),
                item.get("postcode"),
                item.get("borough"),
                item.get("price_pcm"),
                item.get("bedrooms"),
                item.get("bathrooms"),
                item.get("property_type"),
                item.get("floor_area_sqm"),
                item.get("floorplan_used", 0),
                item.get("furnished"),
                item.get("quality"),
                item.get("photo_condition_label"),
                item.get("photo_condition_score"),
                item.get("parsing_confidence"),
                item.get("location_source"),
                item.get("location_precision_m"),
                item.get("source_url")
            ))
        else:
            # Insert new record
            cursor.execute("""
                INSERT INTO comparables (
                    source, source_url, captured_at, first_seen_at, last_seen_at,
                    lat, lon, postcode, borough, price_pcm, bedrooms, bathrooms,
                    property_type, floor_area_sqm, floorplan_used, furnished, quality,
                    photo_condition_label, photo_condition_score, parsing_confidence,
                    location_source, location_precision_m
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                item.get("source"),
                item.get("source_url"),
                item.get("captured_at"),
                item.get("first_seen_at", item.get("captured_at")),
                item.get("last_seen_at"),
                item.get("lat"),
                item.get("lon"),
                item.get("postcode"),
                item.get("borough"),
                item.get("price_pcm"),
                item.get("bedrooms"),
                item.get("bathrooms"),
                item.get("property_type"),
                item.get("floor_area_sqm"),
                item.get("floorplan_used", 0),
                item.get("furnished"),
                item.get("quality"),
                item.get("photo_condition_label"),
                item.get("photo_condition_score"),
                item.get("parsing_confidence"),
                item.get("location_source"),
                item.get("location_precision_m")
            ))
        
        conn.commit()
        conn.close()
        return True
    except sqlite3.DatabaseError as e:
        logger.warning(f"DB corruption detected in upsert_listing: {e}")
        return False
    except Exception as e:
        logger.error(f"Error upserting listing: {e}")
        return False

def purge_old(db_path: str, cutoff_date_iso: str) -> int:
    """
    Delete records older than cutoff_date.
    Returns number of deleted records.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM comparables 
            WHERE last_seen_at < ?
        """, (cutoff_date_iso,))
        
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        logger.info(f"Purged {deleted} old comparables (cutoff: {cutoff_date_iso})")
        return deleted
    except sqlite3.DatabaseError as e:
        logger.warning(f"DB corruption detected in purge_old: {e}")
        return 0
    except Exception as e:
        logger.error(f"Error purging old records: {e}")
        return 0

def query_recent(
    db_path: str,
    lat: float,
    lon: float,
    radius_m: float,
    days: int = 45,
    limit: int = 500
) -> List[Dict]:
    """
    Query recent comparables within radius.
    Returns list of dicts with all columns.
    """
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Calculate cutoff date
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Rough bounding box (1 degree ≈ 111km)
        lat_delta = radius_m / 111000.0
        lon_delta = radius_m / (111000.0 * abs(math.cos(math.radians(lat))))
        
        cursor.execute("""
            SELECT * FROM comparables
            WHERE last_seen_at >= ?
                AND lat BETWEEN ? AND ?
                AND lon BETWEEN ? AND ?
            LIMIT ?
        """, (
            cutoff_date,
            lat - lat_delta,
            lat + lat_delta,
            lon - lon_delta,
            lon + lon_delta,
            limit
        ))
        
        rows = cursor.fetchall()
        conn.close()
        
        # Convert to list of dicts
        result = []
        for row in rows:
            d = dict(row)
            # Calculate actual distance (will be filtered in KNN)
            result.append(d)
        
        return result
    except sqlite3.DatabaseError as e:
        logger.warning(f"DB corruption detected in query_recent: {e}")
        return []
    except Exception as e:
        logger.error(f"Error querying recent comparables: {e}")
        return []

def get_db_count_recent(db_path: str, days: int = 45) -> int:
    """Get count of recent comparables in database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        cursor.execute("SELECT COUNT(*) FROM comparables WHERE last_seen_at >= ?", (cutoff_date,))
        count = cursor.fetchone()[0]
        
        conn.close()
        return count
    except sqlite3.DatabaseError as e:
        logger.warning(f"DB corruption detected in get_db_count_recent: {e}")
        return 0
    except Exception as e:
        logger.error(f"Error getting DB count: {e}")
        return 0

