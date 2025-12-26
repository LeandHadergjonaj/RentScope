#!/usr/bin/env python3
"""
One-off script to enrich comparables dataset with inferred postcodes.

Loads data/comparables.csv, infers missing postcodes from lat/lon using KDTree,
and saves enriched dataset as data/comparables_enriched.csv.

This script is idempotent and safe to run multiple times.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Try to import scipy for KDTree (optional)
try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    cKDTree = None

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two lat/lon points in meters using Haversine formula."""
    import math
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
    if kdtree is None or coords is None or lookup_data is None:
        return None
    
    if pd.isna(lat) or pd.isna(lon):
        return None
    
    try:
        query_point = np.array([[lat, lon]], dtype=np.float64)
        
        if SCIPY_AVAILABLE and kdtree is not None:
            # Use KDTree for fast lookup (returns distance in degrees)
            dist_deg, idx = kdtree.query(query_point, k=1)
            nearest_idx = int(idx[0])
        else:
            # Vectorized nearest search (fallback, returns distance in degrees)
            distances = np.sqrt(
                np.sum((coords - query_point) ** 2, axis=1)
            )
            nearest_idx = int(np.argmin(distances))
            dist_deg = distances[nearest_idx]
        
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
    # Get paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    data_dir = project_root / "data"
    
    comparables_file = data_dir / "comparables.csv"
    postcode_lookup_file = data_dir / "postcode_lookup_clean.csv"
    output_file = data_dir / "comparables_enriched.csv"
    
    if not comparables_file.exists():
        print(f"Error: Source file not found: {comparables_file}")
        sys.exit(1)
    
    if not postcode_lookup_file.exists():
        print(f"Error: Postcode lookup file not found: {postcode_lookup_file}")
        sys.exit(1)
    
    print("=" * 60)
    print("Enriching Comparables Dataset with Inferred Postcodes")
    print("=" * 60)
    
    # Load comparables dataset
    print(f"\nLoading comparables dataset: {comparables_file}")
    df = pd.read_csv(comparables_file)
    initial_count = len(df)
    print(f"Loaded {initial_count:,} rows")
    
    # Check initial state
    has_postcode_col = 'postcode' in df.columns
    if has_postcode_col:
        rows_with_postcode = ((df['postcode'].notna()) & (df['postcode'] != '')).sum()
        rows_without_postcode = initial_count - rows_with_postcode
        print(f"  - Rows with postcode: {rows_with_postcode:,}")
        print(f"  - Rows without postcode: {rows_without_postcode:,}")
    else:
        print(f"  - No postcode column found (all rows need inference)")
        rows_with_postcode = 0
        rows_without_postcode = initial_count
    
    # Check lat/lon availability
    rows_with_latlon = df['lat'].notna() & df['lon'].notna()
    rows_with_latlon_count = rows_with_latlon.sum()
    print(f"  - Rows with lat/lon: {rows_with_latlon_count:,}")
    
    # Load postcode lookup
    print(f"\nLoading postcode lookup: {postcode_lookup_file}")
    postcode_lookup = pd.read_csv(postcode_lookup_file)
    print(f"Loaded {len(postcode_lookup):,} postcodes")
    
    # Filter to London only (rgn == "E12000007")
    if "rgn" in postcode_lookup.columns:
        postcode_lookup = postcode_lookup[postcode_lookup["rgn"] == "E12000007"].copy()
        print(f"Filtered to London region: {len(postcode_lookup):,} postcodes")
    
    # Build KDTree for fast postcode lookup
    print("\nBuilding postcode KDTree...")
    postcode_kdtree, postcode_coords, postcode_lookup_data = build_postcode_kdtree(postcode_lookup)
    
    if postcode_kdtree is None and postcode_coords is None:
        print("Error: Could not build postcode KDTree")
        sys.exit(1)
    
    print(f"KDTree built: {len(postcode_lookup_data):,} postcodes indexed")
    
    # Initialize postcode, ladcd, and inferred_postcode_dist_m columns if they don't exist
    if 'postcode' not in df.columns:
        df['postcode'] = None
    if 'ladcd' not in df.columns:
        df['ladcd'] = None
    if 'inferred_postcode_dist_m' not in df.columns:
        df['inferred_postcode_dist_m'] = None
    
    # Identify rows that need postcode inference
    if has_postcode_col:
        # Rows with lat/lon but missing postcode
        needs_inference = (
            df['lat'].notna() & 
            df['lon'].notna() & 
            (df['postcode'].isna() | (df['postcode'] == ''))
        )
    else:
        # All rows with lat/lon need inference
        needs_inference = df['lat'].notna() & df['lon'].notna()
    
    rows_to_infer = needs_inference.sum()
    print(f"\nRows needing postcode inference: {rows_to_infer:,}")
    
    if rows_to_infer > 0:
        print("Inferring postcodes...")
        
        def infer_postcode_row(row):
            """Infer postcode for a single row"""
            if pd.isna(row['lat']) or pd.isna(row['lon']):
                return row
            
            # Skip if postcode already exists
            if has_postcode_col and pd.notna(row['postcode']) and row['postcode'] != '':
                return row
            
            result = infer_postcode_from_latlon(
                row['lat'],
                row['lon'],
                postcode_kdtree,
                postcode_coords,
                postcode_lookup_data
            )
            
            if result and result.get('postcode'):
                row['postcode'] = result['postcode']
                row['ladcd'] = result.get('ladcd')
                row['inferred_postcode_dist_m'] = result.get('dist_m')
            
            return row
        
        # Apply inference
        df.loc[needs_inference] = df.loc[needs_inference].apply(infer_postcode_row, axis=1)
        
        # Count results
        inferred_count = df.loc[needs_inference, 'postcode'].notna().sum()
        print(f"Successfully inferred postcodes for {inferred_count:,} rows")
        
        if inferred_count > 0:
            avg_dist = df.loc[needs_inference & df['inferred_postcode_dist_m'].notna(), 'inferred_postcode_dist_m'].mean()
            max_dist = df.loc[needs_inference & df['inferred_postcode_dist_m'].notna(), 'inferred_postcode_dist_m'].max()
            print(f"  - Average inference distance: {avg_dist:.1f}m")
            print(f"  - Maximum inference distance: {max_dist:.1f}m")
    
    # Drop rows where BOTH postcode and lat/lon are missing
    print("\nFiltering rows...")
    before_filter = len(df)
    
    # Keep rows that have either postcode OR lat/lon
    has_postcode = df['postcode'].notna() & (df['postcode'] != '')
    has_latlon = df['lat'].notna() & df['lon'].notna()
    keep_mask = has_postcode | has_latlon
    
    df = df[keep_mask].copy()
    after_filter = len(df)
    dropped = before_filter - after_filter
    
    print(f"  - Rows before filter: {before_filter:,}")
    print(f"  - Rows after filter: {after_filter:,}")
    print(f"  - Rows dropped (no postcode and no lat/lon): {dropped:,}")
    
    # Normalize all postcodes
    print("\nNormalizing postcodes...")
    df.loc[df['postcode'].notna(), 'postcode'] = df.loc[df['postcode'].notna(), 'postcode'].apply(normalize_postcode)
    
    # Final statistics
    print("\n" + "=" * 60)
    print("Final Statistics")
    print("=" * 60)
    final_count = len(df)
    rows_with_postcode_final = (df['postcode'].notna() & (df['postcode'] != '')).sum()
    rows_without_postcode_final = final_count - rows_with_postcode_final
    
    print(f"Total rows: {final_count:,}")
    print(f"  - Rows with postcode: {rows_with_postcode_final:,} ({rows_with_postcode_final/final_count*100:.1f}%)")
    print(f"  - Rows without postcode: {rows_without_postcode_final:,} ({rows_without_postcode_final/final_count*100:.1f}%)")
    
    if initial_count > 0:
        recovery_pct = (rows_with_postcode_final - rows_with_postcode) / initial_count * 100 if rows_with_postcode < rows_with_postcode_final else 0
        print(f"\nRecovery:")
        print(f"  - Initial rows with postcode: {rows_with_postcode:,}")
        print(f"  - Final rows with postcode: {rows_with_postcode_final:,}")
        print(f"  - Recovery: +{rows_with_postcode_final - rows_with_postcode:,} rows ({recovery_pct:.1f}% of original dataset)")
    
    # Ensure postcode, ladcd, inferred_postcode_dist_m are in the output
    output_cols = list(df.columns)
    
    # Reorder to put new columns near postcode if it exists, or at the end
    priority_cols = ['postcode', 'ladcd', 'inferred_postcode_dist_m']
    other_cols = [col for col in output_cols if col not in priority_cols]
    
    # Build final column order
    final_cols = []
    for col in priority_cols:
        if col in output_cols:
            final_cols.append(col)
    final_cols.extend(other_cols)
    
    df_output = df[final_cols].copy()
    
    # Save enriched dataset
    print(f"\nSaving enriched dataset: {output_file}")
    df_output.to_csv(output_file, index=False)
    print(f"✓ Successfully created comparables_enriched.csv with {len(df_output):,} rows")
    print("=" * 60)

if __name__ == "__main__":
    main()

