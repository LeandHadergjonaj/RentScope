#!/usr/bin/env python3
"""
One-off script to enrich the raw Rightmove dataset with inferred postcodes.
Targets: data/rightmove_data.csv
Output: data/rightmove_data_enriched.csv
"""

import pandas as pd
import numpy as np
import os
import sys
import math
from pathlib import Path

# Try to import scipy for KDTree (optional but recommended for speed)
try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    cKDTree = None

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two lat/lon points in meters using Haversine formula."""
    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

def normalize_postcode(postcode: str) -> str:
    """Normalize postcode to standard format: uppercase, single space before last 3 chars"""
    if not postcode or pd.isna(postcode):
        return ""
    # Remove all spaces and convert to uppercase
    clean = str(postcode).replace(" ", "").upper()
    if len(clean) >= 5:
        # Insert space before last 3 characters
        return clean[:-3] + " " + clean[-3:]
    return clean

def build_postcode_kdtree(postcode_df: pd.DataFrame):
    """
    Build KDTree for fast nearest postcode lookup.
    Returns (kdtree, coords, lookup_data) or (None, None, None) if unavailable.
    """
    try:
        # Filter to rows with valid lat/lon
        valid_mask = postcode_df['lat'].notna() & postcode_df['lon'].notna()
        valid_df = postcode_df[valid_mask].copy()
        
        if len(valid_df) == 0:
            return None, None, None
        
        # Extract coordinates as numpy array
        coords = valid_df[['lat', 'lon']].values.astype(np.float64)
        
        # Build KDTree if scipy available, otherwise return data for vectorized search
        kdtree = None
        if SCIPY_AVAILABLE and cKDTree is not None:
            kdtree = cKDTree(coords)
        
        # Store lookup data (postcode, ladcd, lat, lon)
        lookup_data = valid_df[['postcode', 'ladcd', 'lat', 'lon']].copy()
        
        return kdtree, coords, lookup_data
    except Exception as e:
        print(f"Warning: Could not build postcode KDTree: {e}")
        return None, None, None

def infer_postcode_from_latlon(lat: float, lon: float, kdtree, coords, lookup_data):
    """
    Infer nearest postcode from lat/lon coordinates using pre-built KDTree.
    Returns dict with {postcode, ladcd, dist_m} or None if lookup unavailable.
    Uses Haversine distance for accurate meter calculation.
    """
    if coords is None or lookup_data is None:
        return None
    
    if pd.isna(lat) or pd.isna(lon):
        return None
    
    try:
        query_point = np.array([[lat, lon]], dtype=np.float64)
        
        if kdtree is not None:
            # Use KDTree for fast lookup (returns distance in degrees)
            dist_deg, idx = kdtree.query(query_point, k=1)
            nearest_idx = int(idx[0])
        else:
            # Vectorized nearest search (fallback, returns distance in degrees)
            distances = np.sqrt(
                np.sum((coords - query_point) ** 2, axis=1)
            )
            nearest_idx = int(np.argmin(distances))
        
        # Get postcode and ladcd from lookup data
        nearest_row = lookup_data.iloc[nearest_idx]
        nearest_lat = float(nearest_row['lat'])
        nearest_lon = float(nearest_row['lon'])
        
        # Calculate accurate distance using Haversine
        dist_m = haversine_distance(lat, lon, nearest_lat, nearest_lon)
        
        postcode = str(nearest_row['postcode']).strip() if pd.notna(nearest_row['postcode']) else None
        ladcd = str(nearest_row['ladcd']).strip() if pd.notna(nearest_row['ladcd']) else None
        
        if postcode:
            postcode = normalize_postcode(postcode)
        
        return {
            'postcode': postcode,
            'ladcd': ladcd,
            'dist_m': dist_m
        }
    except Exception as e:
        return None

def main():
    # Paths
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "data"
    
    raw_data_file = data_dir / "rightmove_data.csv"
    postcode_lookup_file = data_dir / "postcode_lookup_clean.csv"
    output_file = data_dir / "rightmove_data_enriched.csv"
    
    if not raw_data_file.exists():
        print(f"Error: Raw data file not found at {raw_data_file}")
        sys.exit(1)
        
    if not postcode_lookup_file.exists():
        print(f"Error: Postcode lookup file not found at {postcode_lookup_file}")
        sys.exit(1)

    print(f"Loading raw data: {raw_data_file}")
    df = pd.read_csv(raw_data_file)
    original_row_count = len(df)
    print(f"Loaded {original_row_count} rows.")

    # Check for lat/lon columns
    lat_col = None
    lon_col = None
    for col in ['lat', 'latitude', 'LAT', 'LATITUDE']:
        if col in df.columns:
            lat_col = col
            break
            
    for col in ['lon', 'longitude', 'LON', 'LONGITUDE', 'lng', 'LNG']:
        if col in df.columns:
            lon_col = col
            break

    # If not found, look for them by value pattern in any column if needed?
    # But user says "DO have lat/lon", so they should be columns.
    if lat_col is None or lon_col is None:
        print("Warning: 'lat' and 'lon' columns not found in header.")
        print("Columns found:", df.columns.tolist())
        # We will proceed but enrichment might skip everything if columns missing
    else:
        print(f"Found coordinate columns: {lat_col}, {lon_col}")

    # Check for postcode column
    pc_col = None
    for col in ['postcode', 'POSTCODE', 'post_code', 'subdistrict_code']:
        if col in df.columns:
            pc_col = col
            break
            
    if pc_col is None:
        print("Warning: No postcode-like column found. Will create 'postcode' column.")
        df['postcode'] = None
        pc_col = 'postcode'
    else:
        print(f"Using '{pc_col}' as existing postcode column.")

    # Count missing postcodes
    missing_pc_before = df[pc_col].isna() | (df[pc_col].astype(str).str.strip() == "")
    missing_pc_count_before = missing_pc_before.sum()
    print(f"Rows with missing/empty postcode: {missing_pc_count_before}")

    # Load postcode lookup
    print(f"Loading postcode lookup: {postcode_lookup_file}")
    postcode_lookup = pd.read_csv(postcode_lookup_file)
    print(f"Loaded {len(postcode_lookup)} lookup entries.")

    # Build KDTree
    print("Building KDTree for fast spatial lookup...")
    kdtree, coords, lookup_data = build_postcode_kdtree(postcode_lookup)
    
    if coords is None:
        print("Error: Could not build coordinate lookup.")
        sys.exit(1)

    # Enrichment
    enriched_count = 0
    if lat_col and lon_col:
        print("Enriching missing postcodes from lat/lon...")
        
        # Filter for rows that CAN be enriched
        can_enrich_mask = missing_pc_before & df[lat_col].notna() & df[lon_col].notna()
        indices_to_enrich = df[can_enrich_mask].index
        
        print(f"Identified {len(indices_to_enrich)} rows eligible for enrichment.")
        
        # Prepare new columns if they don't exist
        if 'ladcd' not in df.columns:
            df['ladcd'] = None
        if 'inferred_postcode_dist_m' not in df.columns:
            df['inferred_postcode_dist_m'] = None
            
        # Iterate and infer
        for idx in indices_to_enrich:
            lat = df.at[idx, lat_col]
            lon = df.at[idx, lon_col]
            
            result = infer_postcode_from_latlon(lat, lon, kdtree, coords, lookup_data)
            
            if result:
                df.at[idx, pc_col] = result['postcode']
                df.at[idx, 'ladcd'] = result['ladcd']
                df.at[idx, 'inferred_postcode_dist_m'] = result['dist_m']
                enriched_count += 1
                
        print(f"Successfully enriched {enriched_count} rows.")
    else:
        print("Skipping enrichment due to missing lat/lon columns.")

    # Normalize existing postcodes too
    print("Normalizing all postcodes...")
    df[pc_col] = df[pc_col].apply(lambda x: normalize_postcode(str(x)) if pd.notna(x) and str(x).strip() != "" else x)

    # Drop rows ONLY if BOTH postcode is missing AND lat/lon are missing
    print("Applying final filter...")
    
    # Check if postcode is missing (after enrichment)
    pc_missing = df[pc_col].isna() | (df[pc_col].astype(str).str.strip() == "")
    
    # Check if lat/lon is missing
    if lat_col and lon_col:
        geo_missing = df[lat_col].isna() | df[lon_col].isna()
    else:
        geo_missing = pd.Series([True] * len(df))
        
    drop_mask = pc_missing & geo_missing
    df_final = df[~drop_mask].copy()
    
    final_row_count = len(df_final)
    print(f"Dropped {drop_mask.sum()} rows with no postcode and no coordinates.")
    
    # Save result
    print(f"Saving enriched data to: {output_file}")
    df_final.to_csv(output_file, index=False)
    
    print("\nSummary:")
    print(f"- Original row count: {original_row_count}")
    print(f"- Rows with missing postcode before: {missing_pc_count_before}")
    print(f"- Rows successfully enriched: {enriched_count}")
    print(f"- Final row count: {final_row_count}")
    print("Done.")

if __name__ == "__main__":
    main()

