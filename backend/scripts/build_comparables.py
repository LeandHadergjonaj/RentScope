#!/usr/bin/env python3
"""
Convert rent_ads_rightmove_extended.csv to comparables.csv format
for use in RentScope comparables engine.
"""

import pandas as pd
import os
import sys
from datetime import datetime, timedelta
import re
import math
import numpy as np

def normalize_property_type(prop_type: str) -> str:
    """Normalize property type to flat|house|studio|room"""
    if pd.isna(prop_type):
        return "flat"
    
    prop_type_lower = str(prop_type).lower().strip()
    
    # Map to standard types
    if prop_type_lower in ["apartment", "flat", "maisonette", "penthouse"]:
        return "flat"
    elif prop_type_lower in ["house", "detached", "semi-detached", "semi detached", "terraced", "terrace", "cottage", "bungalow", "end of terrace"]:
        return "house"
    elif prop_type_lower in ["studio", "studios"]:
        return "studio"
    elif prop_type_lower in ["room", "rooms", "hmo", "shared", "house share"]:
        return "room"
    else:
        return "flat"  # default

def parse_furnished(furnish_type: str) -> bool:
    """Parse furnished status from Furnish Type column"""
    if pd.isna(furnish_type):
        return None
    
    furnish_lower = str(furnish_type).lower()
    if "furnished" in furnish_lower and "unfurnished" not in furnish_lower:
        return True
    elif "unfurnished" in furnish_lower:
        return False
    elif "flexible" in furnish_lower:
        return None  # Unknown for flexible
    else:
        return None

def parse_size_sqm(size_str: str) -> float:
    """Parse size from string and convert to sqm"""
    if pd.isna(size_str) or str(size_str).lower() in ["ask agent", "n/a", ""]:
        return None
    
    size_str = str(size_str).lower()
    
    # Extract number
    numbers = re.findall(r'\d+\.?\d*', size_str)
    if not numbers:
        return None
    
    size_value = float(numbers[0])
    
    # Check unit
    if "sq ft" in size_str or "sqft" in size_str or "sq.ft" in size_str:
        # Convert sq ft to sqm
        return size_value * 0.092903
    elif "sqm" in size_str or "sq m" in size_str or "m²" in size_str:
        return size_value
    else:
        # Assume sq ft if no unit specified
        return size_value * 0.092903

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

def infer_representative_postcode(district_lat: float, district_lon: float, postcode_lookup: pd.DataFrame) -> tuple:
    """
    Infer representative postcode for a district centroid.
    Returns (postcode, ladcd, dist_m) or (None, None, None) if not found.
    """
    if pd.isna(district_lat) or pd.isna(district_lon):
        return (None, None, None)
    
    # Filter to valid postcodes with lat/lon
    valid_postcodes = postcode_lookup[
        postcode_lookup['lat'].notna() & postcode_lookup['lon'].notna()
    ].copy()
    
    if len(valid_postcodes) == 0:
        return (None, None, None)
    
    # Compute distances (vectorized)
    valid_postcodes['dist_m'] = valid_postcodes.apply(
        lambda row: haversine_distance(district_lat, district_lon, row['lat'], row['lon']),
        axis=1
    )
    
    # Find nearest
    nearest_idx = valid_postcodes['dist_m'].idxmin()
    nearest_row = valid_postcodes.loc[nearest_idx]
    
    postcode = str(nearest_row['postcode']).strip() if pd.notna(nearest_row['postcode']) else None
    ladcd = str(nearest_row['ladcd']).strip() if pd.notna(nearest_row['ladcd']) else None
    dist_m = float(nearest_row['dist_m'])
    
    return (postcode, ladcd, dist_m)

def extract_district_from_address(address: str) -> str:
    """Extract district code from address string"""
    if pd.isna(address):
        return ""
    
    address_str = str(address).upper()
    
    # Try pattern at end: "..., SW1V" or "..., SW1V 1AA"
    # Pattern: ([A-Z]{1,2}\d[A-Z\d]?)
    patterns = [
        r'([A-Z]{1,2}\d[A-Z\d]?)$',  # At end
        r',\s*([A-Z]{1,2}\d[A-Z\d]?)(?:\s|$)',  # After comma
        r'\b([A-Z]{1,2}\d[A-Z\d]?)\b'  # Anywhere
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, address_str)
        if matches:
            return matches[-1].strip()  # Take last match
    
    return ""

def main():
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_dir = os.path.join(project_root, "data")
    
    source_file = os.path.join(data_dir, "rent_ads_rightmove_extended.csv")
    output_file = os.path.join(data_dir, "comparables.csv")
    postcode_lookup_file = os.path.join(data_dir, "postcode_lookup_clean.csv")
    
    if not os.path.exists(source_file):
        print(f"Error: Source file not found: {source_file}")
        sys.exit(1)
    
    if not os.path.exists(postcode_lookup_file):
        print(f"Error: Postcode lookup file not found: {postcode_lookup_file}")
        sys.exit(1)
    
    print("=" * 60)
    print("Building District Centroid Lookup")
    print("=" * 60)
    
    # Load postcode lookup
    print(f"Loading postcode lookup: {postcode_lookup_file}")
    postcode_lookup = pd.read_csv(postcode_lookup_file)
    print(f"Loaded {len(postcode_lookup)} postcodes")
    
    # Filter to London only (rgn == "E12000007")
    if "rgn" in postcode_lookup.columns:
        postcode_lookup = postcode_lookup[postcode_lookup["rgn"] == "E12000007"].copy()
        print(f"Filtered to London region: {len(postcode_lookup)} postcodes")
    
    # Extract district from postcode (first token before space)
    postcode_lookup["district"] = postcode_lookup["postcode"].str.split().str[0].str.upper()
    
    # Group by district and compute mean lat/lon
    district_centroids = postcode_lookup.groupby("district").agg({
        "lat": "mean",
        "lon": "mean"
    }).reset_index()
    district_centroids.columns = ["district", "district_lat", "district_lon"]
    
    print(f"Created {len(district_centroids)} district centroids")
    
    # Infer representative postcode for each district (D)
    print("Inferring representative postcodes for districts...")
    rep_postcode_results = district_centroids.apply(
        lambda row: infer_representative_postcode(row['district_lat'], row['district_lon'], postcode_lookup),
        axis=1,
        result_type='expand'
    )
    district_centroids['rep_postcode'] = rep_postcode_results[0]
    district_centroids['rep_ladcd'] = rep_postcode_results[1]
    district_centroids['rep_dist_m'] = rep_postcode_results[2]
    
    print(f"Inferred postcodes for {district_centroids['rep_postcode'].notna().sum()} districts")
    
    print("\n" + "=" * 60)
    print("Ingesting Rental Ads Data")
    print("=" * 60)
    
    # Load source data
    print(f"Reading source file: {source_file}")
    df = pd.read_csv(source_file)
    print(f"Loaded {len(df)} rows from source")
    
    # Create district key from subdistrict_code
    df["district"] = df["subdistrict_code"].fillna("").astype(str).str.strip().str.upper()
    
    # For rows with missing district, try to extract from address
    missing_district = df["district"] == ""
    if missing_district.sum() > 0:
        print(f"Extracting district from address for {missing_district.sum()} rows with missing subdistrict_code")
        df.loc[missing_district, "district"] = df.loc[missing_district, "address"].apply(extract_district_from_address)
    
    # Merge with district centroids
    print(f"Merging with district centroids...")
    df = df.merge(district_centroids, on="district", how="left")
    
    rows_with_location = df["district_lat"].notna().sum()
    print(f"Rows with district resolved: {rows_with_location} / {len(df)}")
    
    # Drop rows where lat/lon still missing after merge
    df = df[df["district_lat"].notna()].copy()
    print(f"Rows after centroid merge: {len(df)}")
    
    # Normalize columns
    print("\nNormalizing columns...")
    
    # price_pcm (from rent)
    df["price_pcm"] = pd.to_numeric(df["rent"], errors="coerce")
    
    # bedrooms (int from BEDROOMS)
    df["bedrooms"] = pd.to_numeric(df["BEDROOMS"], errors="coerce").astype("Int64")
    
    # bathrooms (int from BATHROOMS)
    df["bathrooms"] = pd.to_numeric(df["BATHROOMS"], errors="coerce").fillna(1).astype(int)
    
    # property_type (map PROPERTY TYPE)
    df["property_type"] = df["PROPERTY TYPE"].apply(normalize_property_type)
    
    # furnished (map Furnish Type)
    df["furnished"] = df["Furnish Type"].apply(parse_furnished)
    
    # floor_area_sqm (parse SIZE)
    df["floor_area_sqm"] = df["SIZE"].apply(parse_size_sqm)
    
    # lat/lon (from district centroid)
    df["lat"] = df["district_lat"]
    df["lon"] = df["district_lon"]
    
    # Add representative postcode and district (D)
    df = df.merge(
        district_centroids[["district", "rep_postcode"]],
        on="district",
        how="left"
    )
    df["postcode"] = df["rep_postcode"]
    df["postcode_district"] = df["district"]
    df["location_precision"] = "district_centroid"
    df = df.drop(columns=["rep_postcode"])  # Clean up
    
    # Set dates (today-30d and today)
    today = datetime.now()
    first_seen_default = (today - timedelta(days=30)).strftime("%Y-%m-%d")
    last_seen_default = today.strftime("%Y-%m-%d")
    df["first_seen"] = first_seen_default
    df["last_seen"] = last_seen_default
    
    print(f"Rows after normalization: {len(df)}")
    
    # Clean and filter
    print("\nCleaning and filtering...")
    initial_count = len(df)
    
    # Drop price outliers
    price_valid = (df["price_pcm"] > 200) & (df["price_pcm"] < 20000)
    df = df[price_valid].copy()
    print(f"After price filter (200 < price < 20000): {len(df)} rows (dropped {initial_count - len(df)})")
    
    # Drop bedroom outliers
    initial_count = len(df)
    bedrooms_valid = (df["bedrooms"] >= 0) & (df["bedrooms"] <= 10) & (df["bedrooms"].notna())
    df = df[bedrooms_valid].copy()
    print(f"After bedroom filter (0 <= bedrooms <= 10): {len(df)} rows (dropped {initial_count - len(df)})")
    
    # Ensure required columns exist
    required_cols = ["price_pcm", "bedrooms", "bathrooms", "property_type", "lat", "lon", "first_seen", "last_seen", "postcode", "postcode_district", "location_precision"]
    optional_cols = ["furnished", "floor_area_sqm"]
    
    # Select and order columns
    output_cols = required_cols + [col for col in optional_cols if col in df.columns]
    df_output = df[output_cols].copy()
    
    # Ensure correct dtypes
    df_output["price_pcm"] = df_output["price_pcm"].astype(float)
    df_output["bedrooms"] = df_output["bedrooms"].astype(int)
    df_output["bathrooms"] = df_output["bathrooms"].astype(int)
    df_output["property_type"] = df_output["property_type"].astype(str)
    df_output["lat"] = df_output["lat"].astype(float)
    df_output["lon"] = df_output["lon"].astype(float)
    df_output["first_seen"] = df_output["first_seen"].astype(str)
    df_output["last_seen"] = df_output["last_seen"].astype(str)
    
    print(f"\nFinal rows to write: {len(df_output)}")
    print(f"Price range: £{df_output['price_pcm'].min():.2f} - £{df_output['price_pcm'].max():.2f}/month")
    print(f"Bedrooms range: {df_output['bedrooms'].min()} - {df_output['bedrooms'].max()}")
    print(f"Property types: {df_output['property_type'].value_counts().to_dict()}")
    
    # Write output
    print(f"\nWriting output to: {output_file}")
    df_output.to_csv(output_file, index=False)
    print(f"✓ Successfully created comparables.csv with {len(df_output)} rows")
    print("=" * 60)

if __name__ == "__main__":
    main()
