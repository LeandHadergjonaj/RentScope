from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os
import math
import re
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from collections import Counter
import pickle
import io
import base64
from openai import OpenAI
import numpy as np

# Try to import scipy for fast nearest neighbor lookup
try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    cKDTree = None

# Import comparables database and KNN services
from storage.comps_db import ensure_db, upsert_listing, purge_old, query_recent, get_db_count_recent, DB_PATH
from services.comps_knn import estimate_from_comps

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, continue without it
    pass

app = FastAPI()

# Feature flags and configuration
ENABLE_PORTAL_ASSETS = os.getenv("ENABLE_PORTAL_ASSETS", "false").lower() == "true"
ENABLE_GEOCODING = os.getenv("ENABLE_GEOCODING", "false").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
openai_client: Optional[OpenAI] = None

if ENABLE_PORTAL_ASSETS:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is required when ENABLE_PORTAL_ASSETS=true. Please set it in backend/.env file.")
    else:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("Portal assets analysis enabled (OpenAI configured)")

# CORS configuration: localhost and 127.0.0.1 are treated as different origins by browsers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data storage
ons_data: Optional[pd.DataFrame] = None
postcode_data: Optional[pd.DataFrame] = None
tfl_stations_data: Optional[pd.DataFrame] = None
comps_data: Optional[pd.DataFrame] = None
valid_postcodes: set = set()  # Fast lookup for valid postcodes (normalized with space)
valid_postcode_nospace: set = set()  # Fast lookup for valid postcodes (no space)
# Nearest postcode lookup (A)
postcode_kdtree: Optional[Any] = None  # cKDTree or None
postcode_coords: Optional[np.ndarray] = None  # Array of (lat, lon) for all postcodes
postcode_lookup_data: Optional[pd.DataFrame] = None  # DataFrame with postcode, ladcd, lat, lon for each point
ml_model: Optional[Any] = None  # ML model pipeline (includes preprocessor)
ml_enabled: bool = False
ml_mae: float = 0.0

# Bedroom multipliers
BEDROOM_MULTIPLIERS = {
    "studio": 0.8,
    "1": 1.0,
    "2": 1.35,
    "3": 1.7,
    "4": 2.1,
    "5+": 2.6
}

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points using Haversine formula (returns meters)"""
    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

def get_transport_adjustment(distance_m: float) -> float:
    """Get transport accessibility adjustment factor based on distance to nearest station"""
    if distance_m <= 400:
        return 0.05  # +5%
    elif distance_m <= 800:
        return 0.02  # +2%
    elif distance_m <= 1500:
        return 0.00  # No adjustment
    else:
        return -0.03  # -3%

def normalize_borough_name(name: str) -> str:
    """Normalize borough name (strip whitespace, consistent casing)"""
    if pd.isna(name) or name == "":
        return "unknown"
    return name.strip().title()

def parse_property_type(prop_type: str) -> str:
    """Normalize property type values"""
    if pd.isna(prop_type):
        return "flat"
    prop_type_lower = str(prop_type).lower().strip()
    if prop_type_lower in ["flat", "apartment"]:
        return "flat"
    elif prop_type_lower in ["house", "terraced", "semi-detached", "detached"]:
        return "house"
    elif prop_type_lower in ["studio", "studios"]:
        return "studio"
    elif prop_type_lower in ["room", "rooms", "shared"]:
        return "room"
    else:
        return "flat"  # default

def is_active_or_recent(last_seen: Any) -> bool:
    """Check if comparable is active/recent (within last 45 days)"""
    if pd.isna(last_seen):
        return False
    try:
        last_seen_dt = pd.to_datetime(last_seen)
        cutoff = datetime.now() - timedelta(days=45)
        return last_seen_dt >= cutoff
    except:
        return False

def time_on_market_days(first_seen: Any, last_seen: Any) -> int:
    """Calculate time on market in days, clamped to >= 0"""
    try:
        first_dt = pd.to_datetime(first_seen)
        last_dt = pd.to_datetime(last_seen)
        days = (last_dt - first_dt).days
        return max(0, days)
    except:
        return 0

def get_furnished_adjustment(furnished: Optional[bool]) -> float:
    """Get furnished adjustment factor"""
    if furnished is True:
        return 0.04  # +4%
    elif furnished is False:
        return -0.02  # -2%
    else:
        return 0.00

def get_bathrooms_adjustment(bathrooms: Optional[int], bedrooms: int) -> float:
    """Get bathrooms adjustment factor"""
    if bathrooms is None:
        return 0.00
    if bathrooms >= bedrooms:
        return 0.03  # +3%
    elif bathrooms == 1 and bedrooms >= 3:
        return -0.03  # -3%
    else:
        return 0.00

def get_floor_area_adjustment(floor_area_sqm: Optional[float], bedrooms: int) -> float:
    """Get floor area adjustment factor"""
    if floor_area_sqm is None:
        return 0.00
    implied_sqm_per_bed = floor_area_sqm / max(bedrooms, 1)
    if implied_sqm_per_bed >= 25:
        return 0.04  # +4%
    elif implied_sqm_per_bed <= 15:
        return -0.04  # -4%
    else:
        return 0.00

def get_quality_adjustment(quality: Optional[str]) -> float:
    """Get quality adjustment factor"""
    if quality == "modern":
        return 0.06  # +6%
    elif quality == "dated":
        return -0.06  # -6%
    else:  # "average" or None
        return 0.00

def get_amenities_adjustment(
    has_lift: Optional[bool],
    has_parking: Optional[bool],
    has_balcony: Optional[bool],
    has_terrace: Optional[bool],
    has_concierge: Optional[bool],
    property_type: str
) -> float:
    """
    Get amenities adjustment factor.
    Small additive adjustments: +0.01 lift, +0.01 balcony, +0.015 terrace, +0.015 concierge, +0.01 parking.
    Cap total to ±0.05.
    """
    total_adj = 0.0
    
    # Lift: +0.01 (only for flats)
    if has_lift is True and property_type.lower() == "flat":
        total_adj += 0.01
    
    # Parking: +0.01
    if has_parking is True:
        total_adj += 0.01
    
    # Balcony: +0.01 (only if terrace is not True)
    if has_balcony is True and has_terrace is not True:
        total_adj += 0.01
    
    # Terrace: +0.015 (if terrace, don't add balcony)
    if has_terrace is True:
        total_adj += 0.015
    
    # Concierge: +0.015
    if has_concierge is True:
        total_adj += 0.015
    
    # Cap total amenities adjustment to ±0.05
    total_adj = max(-0.05, min(0.05, total_adj))
    
    return total_adj

def select_comparables(
    subject_lat: float,
    subject_lon: float,
    subject_bedrooms: int,
    subject_property_type: str,
    comps_df: pd.DataFrame,
    min_comps: int = 8,
    max_comps: int = 30
) -> Tuple[pd.DataFrame, float]:
    """Select and rank comparables for a subject property
    
    Returns:
        (selected_comps_df, radius_m)
    """
    if comps_df is None or len(comps_df) == 0:
        return pd.DataFrame(), 0.0
    
    # Filter by property type
    subject_prop_type = parse_property_type(subject_property_type)
    comps_filtered = comps_df[comps_df["property_type"].apply(parse_property_type) == subject_prop_type].copy()
    
    # Filter by bedrooms (within +/- 1, or +/- 2 for 4+)
    if subject_bedrooms >= 4:
        bedroom_tolerance = 2
    else:
        bedroom_tolerance = 1
    
    comps_filtered = comps_filtered[
        abs(comps_filtered["bedrooms"] - subject_bedrooms) <= bedroom_tolerance
    ]
    
    # Filter to recent comps (within 45 days)
    comps_filtered = comps_filtered[
        comps_filtered["last_seen"].apply(is_active_or_recent)
    ]
    
    if len(comps_filtered) == 0:
        return pd.DataFrame(), 0.0
    
    # Compute distances
    comps_filtered["distance_m"] = comps_filtered.apply(
        lambda row: haversine_distance(subject_lat, subject_lon, row["lat"], row["lon"]),
        axis=1
    )
    
    # Adaptive radius expansion: 500m -> 800m -> 1200m -> 2000m
    # Stop as soon as we have >= 12 comps
    radius_options = [500.0, 800.0, 1200.0, 2000.0]
    radius_m = 0.0
    comps_in_radius = pd.DataFrame()
    
    for radius in radius_options:
        radius_m = radius
        comps_in_radius = comps_filtered[comps_filtered["distance_m"] <= radius_m].copy()
        if len(comps_in_radius) >= 12:
            break
    
    if len(comps_in_radius) < min_comps:
        return pd.DataFrame(), radius_m
    
    # Compute similarity scores
    comps_in_radius["similarity_score"] = (
        comps_in_radius["distance_m"] / 800.0 +
        0.7 * abs(comps_in_radius["bedrooms"] - subject_bedrooms)
    )
    
    # Select top K comps
    comps_selected = comps_in_radius.nsmallest(max_comps, "similarity_score")
    
    return comps_selected, radius_m

def compute_comps_estimate(comps_df: pd.DataFrame) -> Tuple[float, List[float]]:
    """Compute expected median and range from comparables using time-on-market weighting
    
    Returns:
        (expected_median, [expected_lower, expected_upper])
    """
    if comps_df is None or len(comps_df) == 0:
        return 0.0, [0.0, 0.0]
    
    # Compute time on market and weights
    comps_df = comps_df.copy()
    comps_df["time_on_market_days"] = comps_df.apply(
        lambda row: time_on_market_days(row["first_seen"], row["last_seen"]),
        axis=1
    )
    comps_df["weight"] = 1.0 / (1.0 + comps_df["time_on_market_days"] / 14.0)
    
    prices_series = comps_df["price_pcm"]
    weights = comps_df["weight"].values
    
    # Unweighted median (use pandas Series median)
    unweighted_median = float(prices_series.median())
    
    # Weighted mean
    prices_array = prices_series.values
    weighted_mean = float((prices_array * weights).sum() / weights.sum())
    
    # Expected median = average of unweighted median and weighted mean
    expected_median = (unweighted_median + weighted_mean) / 2.0
    
    # Robust range using percentiles (25/75) - use pandas Series quantile
    expected_lower = float(prices_series.quantile(0.25))
    expected_upper = float(prices_series.quantile(0.75))
    
    return expected_median, [expected_lower, expected_upper]

# --- Nearest Postcode Lookup (A) ---
def build_postcode_kdtree(postcode_df: pd.DataFrame) -> Tuple[Optional[Any], Optional[np.ndarray], Optional[pd.DataFrame]]:
    """Build a fast nearest-neighbor index for postcode lookup from lat/lon."""
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
        if 'postcode_nospace' in valid_df.columns:
            lookup_data['postcode_nospace'] = valid_df['postcode_nospace']
        
        return kdtree, coords, lookup_data
    except Exception as e:
        print(f"Warning: Could not build postcode KDTree: {e}")
        return None, None, None

# --- Address/Postcode Resolution (A) ---
def resolve_location_from_listing(
    url: str,
    parsed_address_text: Optional[str],
    extracted_lat: Optional[float],
    extracted_lon: Optional[float]
) -> Dict[str, Any]:
    """
    Resolve location (lat, lon, postcode) from listing using priority order:
    1) Extracted lat/lon from listing
    2) OpenStreetMap Nominatim geocoding from address text
    3) None
    
    Returns dict with: lat, lon, postcode, postcode_valid, location_source, location_precision_m, warnings
    """
    result = {
        'lat': None,
        'lon': None,
        'postcode': None,
        'postcode_valid': False,
        'location_source': 'none',
        'location_precision_m': None,
        'warnings': []
    }
    
    # Priority 1: If lat/lon extracted from listing, infer postcode
    if extracted_lat is not None and extracted_lon is not None:
        inferred_result = infer_postcode_from_latlon(extracted_lat, extracted_lon)
        if inferred_result and inferred_result.get('postcode'):
            result['lat'] = extracted_lat
            result['lon'] = extracted_lon
            result['postcode'] = inferred_result['postcode']
            result['postcode_valid'] = True
            result['location_source'] = 'listing_latlon'
            result['location_precision_m'] = inferred_result.get('dist_m')
            if result['location_precision_m'] and result['location_precision_m'] > 500:
                result['warnings'].append(f"Postcode inferred from coordinates ({result['location_precision_m']:.0f}m from nearest)")
        else:
            result['lat'] = extracted_lat
            result['lon'] = extracted_lon
            result['location_source'] = 'listing_latlon'
            result['warnings'].append("Could not infer postcode from extracted coordinates")
        return result
    
    # Priority 2: OpenStreetMap Nominatim geocoding from address text
    if ENABLE_GEOCODING and parsed_address_text and parsed_address_text.strip():
        try:
            # Add "London, UK" context if not present
            address_query = parsed_address_text.strip()
            if 'london' not in address_query.lower():
                address_query = f"{address_query}, London, UK"
            
            headers = {
                'User-Agent': 'RentScope/1.0 (Property Evaluation Tool; contact@rentscope.example.com)'
            }
            
            nominatim_url = "https://nominatim.openstreetmap.org/search"
            params = {
                'q': address_query,
                'format': 'json',
                'addressdetails': 1,
                'limit': 1,
                'countrycodes': 'gb'  # UK only
            }
            
            response = requests.get(nominatim_url, params=params, headers=headers, timeout=8)
            response.raise_for_status()
            
            data = response.json()
            if data and len(data) > 0:
                first_result = data[0]
                
                # Extract lat/lon
                result['lat'] = float(first_result.get('lat', 0))
                result['lon'] = float(first_result.get('lon', 0))
                
                # Extract postcode from address details
                address_details = first_result.get('address', {})
                postcode_raw = (
                    address_details.get('postcode') or
                    address_details.get('postal_code')
                )
                
                if postcode_raw:
                    # Normalize and validate postcode
                    postcode_normalized = normalize_postcode(str(postcode_raw))
                    if validate_postcode_candidate(postcode_normalized):
                        result['postcode'] = postcode_normalized
                        result['postcode_valid'] = True
                    else:
                        result['warnings'].append(f"Geocoded postcode '{postcode_raw}' not found in lookup")
                
                # If postcode missing, infer from lat/lon
                if not result['postcode']:
                    inferred_result = infer_postcode_from_latlon(result['lat'], result['lon'])
                    if inferred_result and inferred_result.get('postcode'):
                        result['postcode'] = inferred_result['postcode']
                        result['postcode_valid'] = True
                        result['location_precision_m'] = inferred_result.get('dist_m')
                        if result['location_precision_m'] and result['location_precision_m'] > 500:
                            result['warnings'].append(f"Postcode inferred from geocoded coordinates ({result['location_precision_m']:.0f}m from nearest)")
                    else:
                        result['warnings'].append("Could not infer postcode from geocoded coordinates")
                
                result['location_source'] = 'nominatim'
            else:
                result['warnings'].append(f"No geocoding results for address: {parsed_address_text}")
        except requests.RequestException as e:
            result['warnings'].append(f"Geocoding request failed: {str(e)}")
        except Exception as e:
            result['warnings'].append(f"Geocoding error: {str(e)}")
    
    return result

def infer_postcode_from_latlon(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """
    Infer nearest postcode from lat/lon coordinates.
    Returns {postcode, ladcd, dist_m} or None if lookup unavailable.
    """
    global postcode_kdtree, postcode_coords, postcode_lookup_data
    
    if postcode_lookup_data is None or postcode_coords is None:
        return None
    
    try:
        query_point = np.array([[lat, lon]], dtype=np.float64)
        
        if SCIPY_AVAILABLE and postcode_kdtree is not None:
            # Use KDTree for fast lookup
            dist, idx = postcode_kdtree.query(query_point, k=1)
            dist_m = float(dist[0] * 111000)  # Rough conversion: 1 degree ≈ 111km
            nearest_idx = int(idx[0])
        else:
            # Vectorized nearest search (fallback)
            distances = np.sqrt(
                np.sum((postcode_coords - query_point) ** 2, axis=1)
            )
            nearest_idx = int(np.argmin(distances))
            # Convert to meters (rough approximation)
            dist_deg = float(distances[nearest_idx])
            dist_m = dist_deg * 111000
        
        # Get postcode and ladcd from lookup data
        nearest_row = postcode_lookup_data.iloc[nearest_idx]
        postcode = str(nearest_row['postcode']).strip() if pd.notna(nearest_row['postcode']) else None
        ladcd = str(nearest_row['ladcd']).strip() if pd.notna(nearest_row['ladcd']) else None
        
        return {
            'postcode': normalize_postcode(postcode) if postcode else None,
            'ladcd': ladcd,
            'dist_m': dist_m
        }
    except Exception as e:
        print(f"Warning: Error inferring postcode from lat/lon: {e}")
        return None

def normalize_postcode(postcode: str) -> str:
    """Normalize postcode to standard format: uppercase, single space before last 3 chars"""
    if not postcode:
        return ""
    # Remove all spaces and convert to uppercase
    clean = postcode.replace(" ", "").upper()
    if len(clean) >= 5:
        # Insert space before last 3 characters
        return clean[:-3] + " " + clean[-3:]
    return clean

def train_ml_model(comps_df: pd.DataFrame) -> Tuple[Optional[Any], Optional[Any], bool, float]:
    """
    Train Ridge regression model on comparables data using sklearn Pipeline.
    Returns (model_pipeline, None, enabled, mae)
    """
    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error
        
        if len(comps_df) < 2000:
            return None, None, False, 0.0
        
        # Prepare features
        df = comps_df.copy()
        
        # Fill missing values for required columns
        df['bathrooms'] = df.get('bathrooms', pd.Series([1] * len(df))).fillna(1)
        if 'floor_area_sqm' in df.columns:
            df['floor_area_sqm'] = df['floor_area_sqm'].fillna(df['floor_area_sqm'].median())
        else:
            df['floor_area_sqm'] = 50.0  # Default
        
        # Compute distance to nearest station for each comp
        if tfl_stations_data is not None and len(tfl_stations_data) > 0:
            def get_nearest_station_distance(lat, lon):
                if pd.isna(lat) or pd.isna(lon):
                    return 1500.0  # Default
                min_dist = float('inf')
                for _, station in tfl_stations_data.iterrows():
                    dist = haversine_distance(lat, lon, station['lat'], station['lon'])
                    if dist < min_dist:
                        min_dist = dist
                return min_dist if min_dist != float('inf') else 1500.0
            df['nearest_station_distance_m'] = df.apply(
                lambda row: get_nearest_station_distance(row['lat'], row['lon']), axis=1
            )
        else:
            df['nearest_station_distance_m'] = 1500.0  # Default
        
        # Derive postcode_district and ladcd from postcode if available (7)
        if 'postcode' in df.columns and df['postcode'].notna().any():
            # Extract district from postcode
            df['postcode_district'] = df['postcode'].apply(
                lambda x: str(x).split()[0].upper() if pd.notna(x) and " " in str(x) else 
                (str(x)[:4].upper() if pd.notna(x) and len(str(x)) >= 2 else "UNKNOWN")
            )
            # Try to get ladcd from postcode lookup
            if postcode_data is not None and 'ladcd' in postcode_data.columns and 'postcode' in postcode_data.columns:
                postcode_to_ladcd = dict(zip(postcode_data['postcode'].astype(str).str.upper(), postcode_data['ladcd'].astype(str)))
                df['ladcd'] = df['postcode'].apply(
                    lambda x: postcode_to_ladcd.get(str(x).upper().strip(), "UNKNOWN") if pd.notna(x) else "UNKNOWN"
                )
            else:
                df['ladcd'] = "UNKNOWN"
        elif 'postcode_district' in df.columns:
            # Use existing postcode_district if present
            df['postcode_district'] = df['postcode_district'].fillna("UNKNOWN")
            df['ladcd'] = "UNKNOWN"
        else:
            # Infer from lat/lon using nearest postcode lookup
            df['postcode_district'] = "UNKNOWN"  # Fallback
            df['ladcd'] = "UNKNOWN"
        
        # Ensure lat/lon are present
        if 'lat' not in df.columns:
            df['lat'] = 51.5074  # London default
        if 'lon' not in df.columns:
            df['lon'] = -0.1278  # London default
        
        # Add interaction feature: bedrooms * floor_area_sqm (7)
        df['bedrooms_floor_area'] = df['bedrooms'] * df['floor_area_sqm']
        
        # Define feature columns (7)
        numeric_features = ["bedrooms", "bathrooms", "floor_area_sqm", "nearest_station_distance_m", "lat", "lon", "bedrooms_floor_area"]
        categorical_features = ["property_type", "postcode_district", "ladcd"]
        
        # Ensure all required columns exist
        for col in numeric_features + categorical_features:
            if col not in df.columns:
                if col in numeric_features:
                    df[col] = 0.0 if col in ["lat", "lon"] else 1.0
                else:
                    df[col] = "UNKNOWN"
        
        # Build preprocessor (3)
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="median"))
                ]), numeric_features),
                ("cat", Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
                ]), categorical_features),
            ],
            remainder="drop"
        )
        
        # Build model pipeline (4)
        ml_model = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("ridge", Ridge(alpha=1.0, random_state=42))
        ])
        
        # Prepare X and y
        X = df[numeric_features + categorical_features].copy()
        y = df['price_pcm'].values
        
        # Train/test split (5)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        ml_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = ml_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"ML model trained: MAE = £{mae:.2f}/month")
        return ml_model, None, True, mae  # Return pipeline, not separate model/preprocessor
        
    except ImportError:
        print("Warning: scikit-learn not available, ML model disabled")
        return None, None, False, 0.0
    except Exception as e:
        print(f"Warning: ML model training failed: {e}")
        return None, None, False, 0.0

@app.on_event("startup")
async def load_data():
    """Load CSV files at startup and initialize comparables database"""
    global ons_data, postcode_data, tfl_stations_data, comps_data
    global valid_postcodes, valid_postcode_nospace
    global postcode_kdtree, postcode_coords, postcode_lookup_data
    global ml_model, ml_enabled, ml_mae
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    
    # Initialize comparables database (3)
    try:
        db_recreated = ensure_db(DB_PATH)
        if db_recreated:
            print(f"Comps DB recreated (fresh and empty) - will rebuild from new evaluations")
        else:
            cutoff_date = (datetime.now() - timedelta(days=45)).isoformat()
            purged_count = purge_old(DB_PATH, cutoff_date)
            db_count = get_db_count_recent(DB_PATH, days=45)
            print(f"Comps DB ready: path={DB_PATH}, purged={purged_count} old records, recent={db_count} records")
    except Exception as e:
        print(f"Warning: Failed to initialize comparables DB: {e}")
    
    try:
        ons_data = pd.read_csv(os.path.join(data_dir, "ons_clean.csv"))
        postcode_data = pd.read_csv(os.path.join(data_dir, "postcode_lookup_clean.csv"))
        tfl_stations_data = pd.read_csv(os.path.join(data_dir, "tfl_stations.csv"))
        
        print(f"Loaded ONS data: {len(ons_data)} rows")
        print(f"Loaded postcode data: {len(postcode_data)} rows")
        print(f"Loaded TFL stations data: {len(tfl_stations_data)} rows")
        
        # Build postcode lookup sets
        if 'postcode' in postcode_data.columns:
            valid_postcodes = set()
            valid_postcode_nospace = set()
            for pc in postcode_data['postcode'].dropna():
                pc_str = str(pc).strip().upper()
                normalized = normalize_postcode(pc_str)
                # Add normalized (with space) and nospace versions
                valid_postcodes.add(normalized)
                valid_postcode_nospace.add(normalized.replace(" ", ""))
            # Also add from postcode_nospace column if it exists
            if 'postcode_nospace' in postcode_data.columns:
                for pc in postcode_data['postcode_nospace'].dropna():
                    pc_str = str(pc).strip().upper()
                    normalized = normalize_postcode(pc_str)
                    valid_postcodes.add(normalized)
                    valid_postcode_nospace.add(normalized.replace(" ", ""))
            print(f"Built postcode lookup sets: {len(valid_postcodes)} normalized, {len(valid_postcode_nospace)} nospace entries")
        
        # Build nearest postcode lookup (A)
        postcode_kdtree, postcode_coords, postcode_lookup_data = build_postcode_kdtree(postcode_data)
        if postcode_kdtree is not None or postcode_coords is not None:
            print(f"Built postcode KDTree: {len(postcode_coords) if postcode_coords is not None else 0} points")
        else:
            print("Warning: Postcode KDTree not available (will use vectorized search)")
        
        # Load comparables if available
        comps_path = os.path.join(data_dir, "comparables.csv")
        if os.path.exists(comps_path):
            try:
                comps_data = pd.read_csv(comps_path)
                # Validate required columns
                required_cols = ["price_pcm", "bedrooms", "property_type", "lat", "lon", "first_seen", "last_seen"]
                missing_cols = [col for col in required_cols if col not in comps_data.columns]
                if missing_cols:
                    print(f"Warning: comparables.csv missing required columns: {missing_cols}")
                    comps_data = None
                else:
                    print(f"Loaded comparables: {len(comps_data)} rows")
                    # Train ML model if enough data
                    ml_model, _, ml_enabled, ml_mae = train_ml_model(comps_data)
            except Exception as e:
                print(f"Warning: Could not load comparables.csv: {e}")
                comps_data = None
        else:
            print("Warning: comparables.csv not found, continuing without comparables")
            comps_data = None
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# Request/Response models
class ParseListingRequest(BaseModel):
    url: str

class PostcodeCandidate(BaseModel):
    value: str
    source: str  # "jsonld" | "script" | "regex" | "address_text" | "latlon_inferred"
    valid: bool
    distance_m: Optional[float] = None

class ParseListingResponse(BaseModel):
    price_pcm: Optional[float] = None
    bedrooms: Optional[int] = None
    property_type: Optional[str] = None
    postcode: Optional[str] = None
    postcode_valid: bool = False
    postcode_source: str = "unknown"  # "jsonld" | "script" | "regex" | "unknown"
    postcode_candidates: List[PostcodeCandidate] = []
    chosen_postcode_source: str = "unknown"
    bathrooms: Optional[int] = None
    floor_area_sqm: Optional[float] = None
    furnished: Optional[str] = None  # "true" | "false" | "unknown" | None
    parsing_confidence: str = "low"  # "high" | "medium" | "low"
    extracted_fields: List[str] = []
    warnings: List[str] = []
    # Portal assets (B)
    image_urls: List[str] = []
    floorplan_url: Optional[str] = None
    asset_warnings: List[str] = []
    asset_extraction_confidence: str = "low"  # "high" | "medium" | "low"
    # Location extraction (B)
    lat: Optional[float] = None
    lon: Optional[float] = None
    location_source: str = "none"  # "jsonld" | "script" | "html" | "listing_latlon" | "nominatim" | "inferred" | "none"
    inferred_postcode: Optional[str] = None
    inferred_postcode_distance_m: Optional[float] = None
    address_text: Optional[str] = None  # Best-effort extracted address text (B)
    location_precision_m: Optional[float] = None  # Distance to inferred postcode if used (B)
    # Additional structured features (V23)
    floor_level: Optional[int] = None  # 0 for ground, 1+ for upper floors
    epc_rating: Optional[str] = None  # A-G
    has_lift: Optional[bool] = None
    has_parking: Optional[bool] = None
    has_balcony: Optional[bool] = None
    has_terrace: Optional[bool] = None
    has_concierge: Optional[bool] = None
    parsed_feature_warnings: List[str] = []
    # Debug output (only included when debug=true query param)
    debug_raw: Optional[Dict[str, Any]] = None

class EvaluateRequest(BaseModel):
    url: str = ""
    price_pcm: float
    bedrooms: int
    property_type: str  # "flat|house|studio|room"
    postcode: Optional[str] = None  # Optional - can infer from lat/lon
    bathrooms: Optional[int] = None
    floor_area_sqm: Optional[float] = None
    furnished: Optional[bool] = None
    quality: Optional[str] = "average"  # "dated", "average", "modern"
    # Portal assets (D)
    photo_condition_label: Optional[str] = None  # "dated" | "average" | "modern" | "luxury"
    photo_condition_score: Optional[int] = None  # 0-100
    photo_condition_confidence: Optional[str] = None  # "low" | "medium" | "high"
    floorplan_area_sqm: Optional[float] = None
    floorplan_confidence: Optional[str] = None  # "low" | "medium" | "high"
    # Location (C)
    lat: Optional[float] = None
    lon: Optional[float] = None
    # Comparables DB (5)
    save_as_comparable: bool = True  # Default true for auto-building DB
    # Amenities (3)
    has_lift: Optional[bool] = None
    has_parking: Optional[bool] = None
    has_balcony: Optional[bool] = None
    has_terrace: Optional[bool] = None
    has_concierge: Optional[bool] = None

class DebugInfo(BaseModel):
    estimate_quality_score: int
    quality_label: str  # "poor"|"ok"|"good"|"great"
    quality_reasons: List[str]
    model_components: Dict[str, Any]
    location_source: str = "none"  # Never null, defaults to "none" (3)
    location_precision_m: float = 0.0  # Never null, defaults to 0.0 (3)
    geocoding_used: bool = False  # Never null, defaults to False (3)

class EvaluateResponse(BaseModel):
    borough: str
    listed_price_pcm: float
    expected_median_pcm: float
    expected_range_pcm: List[float]  # Wide statistical range (quartiles/IQR)
    most_likely_range_pcm: Optional[List[float]] = None  # Tight range when evidence supports (C)
    most_likely_range_basis: Optional[str] = None  # Explanation of range basis
    deviation_pct: float
    classification: str  # "undervalued|fair|overvalued"
    confidence: str  # "low|medium|high"
    explanations: List[str]
    nearest_station_distance_m: float
    transport_adjustment_pct: float
    used_borough_fallback: bool
    extra_adjustment_pct: float
    adjustments_breakdown: dict
    comps_used: bool
    comps_sample_size: int
    comps_radius_m: float
    comps_expected_median_pcm: float
    comps_expected_range_pcm: List[float]
    deviation_pct_damped: float
    confidence_adjusted: bool
    adaptive_radius_used: bool
    strong_similarity: bool
    similarity_dampening_factor: float
    ml_expected_median_pcm: Optional[float] = None
    # Portal assets (D)
    photo_adjustment_pct: float = 0.0
    floorplan_used: bool = False
    floorplan_area_sqm_used: Optional[float] = None
    area_used: bool = False
    area_source: str = "none"  # "floorplan" | "none"
    area_used_sqm: Optional[float] = None
    debug: DebugInfo

def compute_ml_prediction(
    bedrooms: int,
    bathrooms: Optional[int],
    floor_area_sqm: Optional[float],
    property_type: str,
    distance_to_station_m: float,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    postcode_district: Optional[str] = None,
    ladcd: Optional[str] = None
) -> Optional[float]:
    """Compute ML prediction for expected rent using sklearn Pipeline."""
    if not ml_enabled or ml_model is None:
        return None
    
    try:
        # Prepare feature values (7)
        bathrooms_val = bathrooms if bathrooms is not None else 1
        floor_area_val = floor_area_sqm if floor_area_sqm is not None else 50.0
        lat_val = lat if lat is not None else 51.5074  # London default
        lon_val = lon if lon is not None else -0.1278  # London default
        postcode_district_val = postcode_district if postcode_district else "UNKNOWN"
        ladcd_val = ladcd if ladcd else "UNKNOWN"
        
        # Add interaction feature: bedrooms * floor_area_sqm (7)
        bedrooms_floor_area = bedrooms * floor_area_val
        
        # Create single-row DataFrame with EXACT columns matching training (7)
        numeric_features = ["bedrooms", "bathrooms", "floor_area_sqm", "nearest_station_distance_m", "lat", "lon", "bedrooms_floor_area"]
        categorical_features = ["property_type", "postcode_district", "ladcd"]
        
        df_row = pd.DataFrame([{
            'bedrooms': bedrooms,
            'bathrooms': bathrooms_val,
            'floor_area_sqm': floor_area_val,
            'nearest_station_distance_m': distance_to_station_m,
            'lat': lat_val,
            'lon': lon_val,
            'bedrooms_floor_area': bedrooms_floor_area,
            'property_type': property_type,
            'postcode_district': postcode_district_val,
            'ladcd': ladcd_val
        }], columns=numeric_features + categorical_features)
        
        # Predict directly using pipeline (6)
        prediction = ml_model.predict(df_row)[0]
        
        if ml_enabled:
            print("ML prediction ok")
        
        return max(0.0, float(prediction))
    except Exception as e:
        print(f"ML prediction error: {e}")
        return None

def create_default_debug_info() -> DebugInfo:
    """Create default debug info for error cases."""
    return DebugInfo(
        estimate_quality_score=0,
        quality_label="poor",
        quality_reasons=["Insufficient data for evaluation"],
        model_components={
            "ons_used": False,
            "comps_used": False,
            "ml_used": False,
            "comps_sample_size": 0,
            "comps_radius_m": 0.0,
            "strong_similarity": False,
            "ml_mae": None,
            "ml_expected_median_pcm": None,
            "portal_assets_used": False,
            "photos_used": 0,
            "floorplan_used": False
        }
    )

def compute_estimate_quality_score(
    comps_used: bool,
    comps_sample_size: int,
    comps_radius_m: float,
    strong_similarity: bool,
    confidence: str,
    structural_divergence_triggered: bool,
    ons_used: bool,
    ml_used: bool,
    ml_mae: Optional[float] = None,
    postcode_inferred: bool = False,
    inferred_postcode_dist_m: Optional[float] = None
) -> Tuple[int, str, List[str]]:
    """
    Compute estimate quality score heuristically.
    Returns (score, label, reasons)
    """
    score = 50  # Start at 50
    reasons = []
    
    # Positive factors
    if comps_used and comps_sample_size >= 15:
        score += 15
        reasons.append(f"+15: Comparables used with {comps_sample_size} samples (>=15)")
    
    if comps_radius_m <= 1200:
        score += 10
        reasons.append(f"+10: Comparables radius {comps_radius_m:.0f}m (<=1200m)")
    
    if strong_similarity:
        score += 10
        reasons.append("+10: Strong similarity detected (exact bedroom match, close radius, sufficient comps)")
    
    if ml_used and ml_mae is not None:
        score += 10
        reasons.append(f"+10: ML model used (MAE: £{ml_mae:.2f}/month)")
    
    if postcode_inferred and inferred_postcode_dist_m is not None and inferred_postcode_dist_m <= 500:
        score += 10
        reasons.append(f"+10: Postcode inferred from coordinates (distance: {inferred_postcode_dist_m:.0f}m, reliable)")
    
    if confidence == "high":
        score += 5
        reasons.append("+5: High confidence")
    elif confidence == "medium":
        reasons.append("+0: Medium confidence")
    
    # Negative factors
    if confidence == "low":
        score -= 10
        reasons.append("-10: Low confidence")
    
    if comps_sample_size < 10 and comps_used:
        score -= 10
        reasons.append(f"-10: Comparables sample size {comps_sample_size} (<10)")
    
    if comps_radius_m > 1200 and comps_used:
        score -= 10
        reasons.append(f"-10: Comparables radius {comps_radius_m:.0f}m (>1200m)")
    
    if structural_divergence_triggered:
        score -= 10
        reasons.append("-10: Structural divergence triggered (large ONS/comps mismatch)")
    
    # Clamp to [0, 100]
    score = max(0, min(100, score))
    
    # Determine label
    if score <= 39:
        label = "poor"
    elif score <= 59:
        label = "ok"
    elif score <= 79:
        label = "good"
    else:
        label = "great"
    
    return score, label, reasons

@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_property(request: EvaluateRequest):
    """Evaluate a property listing"""
    
    if ons_data is None or postcode_data is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    # Step 0: Strict area gating - only use floorplan area if confidence is sufficient (A)
    area_used = False
    area_used_sqm = None
    area_source = "none"
    floorplan_used = False
    floorplan_area_sqm_used = None
    
    # Decide area usage strictly: only if floorplan qualifies
    if request.floorplan_area_sqm and request.floorplan_area_sqm > 0 and request.floorplan_confidence in ["high", "medium"]:
        area_used = True
        area_used_sqm = float(request.floorplan_area_sqm)
        area_source = "floorplan"
        floorplan_used = True
        floorplan_area_sqm_used = float(request.floorplan_area_sqm)
    else:
        # Ignore request.floor_area_sqm entirely - bedrooms-only mode
        area_used = False
        area_used_sqm = None
        area_source = "none"
        floorplan_used = False
        floorplan_area_sqm_used = None
    
    # Step 1: Get lat/lon and postcode (C) - lat/lon-first approach
    lat = None
    lon = None
    ladcd = None
    postcode_inferred = False
    inferred_postcode_dist_m = None
    # Track location resolution path for debug (2)
    location_resolution_source = "none"  # "listing_latlon" | "inferred" | "postcode_lookup" | "nominatim" | "none"
    location_resolution_precision_m = 0.0
    geocoding_used = False
    
    # If lat/lon provided, use them directly
    if request.lat is not None and request.lon is not None:
        lat = float(request.lat)
        lon = float(request.lon)
        location_resolution_source = "listing_latlon"  # Track that lat/lon was provided (2)
        
        # Infer postcode from lat/lon if postcode missing or invalid
        if not request.postcode or not request.postcode.strip():
            inferred_result = infer_postcode_from_latlon(lat, lon)
            if inferred_result and inferred_result.get('postcode'):
                request.postcode = inferred_result['postcode']
                ladcd = inferred_result.get('ladcd')
                postcode_inferred = True
                inferred_postcode_dist_m = inferred_result.get('dist_m')
                location_resolution_source = "inferred"  # Postcode was inferred (2)
                location_resolution_precision_m = float(inferred_postcode_dist_m) if inferred_postcode_dist_m else 0.0
            else:
                raise HTTPException(status_code=400, detail="Could not infer postcode from coordinates — cannot evaluate.")
        else:
            # Validate postcode and get ladcd
            postcode_normalized = request.postcode.replace(" ", "").upper()
            postcode_row = postcode_data[
                postcode_data["postcode_nospace"].str.upper() == postcode_normalized
            ]
            
            if postcode_row.empty:
                postcode_row = postcode_data[
                    postcode_data["postcode"].str.upper() == request.postcode.upper()
                ]
            
            if not postcode_row.empty:
                ladcd = str(postcode_row.iloc[0]["ladcd"]).strip() if pd.notna(postcode_row.iloc[0]["ladcd"]) else None
                # Postcode was provided and validated, keep "listing_latlon" as source
            else:
                # Postcode not found, try to infer
                inferred_result = infer_postcode_from_latlon(lat, lon)
                if inferred_result and inferred_result.get('postcode'):
                    request.postcode = inferred_result['postcode']
                    ladcd = inferred_result.get('ladcd')
                    postcode_inferred = True
                    inferred_postcode_dist_m = inferred_result.get('dist_m')
                    location_resolution_source = "inferred"  # Postcode was inferred (2)
                    location_resolution_precision_m = float(inferred_postcode_dist_m) if inferred_postcode_dist_m else 0.0
    
    # If no lat/lon but postcode provided, lookup lat/lon from postcode
    elif request.postcode and request.postcode.strip():
        postcode_normalized = request.postcode.replace(" ", "").upper()
        postcode_row = postcode_data[
            postcode_data["postcode_nospace"].str.upper() == postcode_normalized
        ]
        
        if postcode_row.empty:
            postcode_row = postcode_data[
                postcode_data["postcode"].str.upper() == request.postcode.upper()
            ]
        
        if postcode_row.empty:
            raise HTTPException(status_code=400, detail="Postcode not found / invalid — cannot evaluate.")
        
        lat = float(postcode_row.iloc[0]["lat"])
        lon = float(postcode_row.iloc[0]["lon"])
        ladcd = str(postcode_row.iloc[0]["ladcd"]).strip() if pd.notna(postcode_row.iloc[0]["ladcd"]) else None
        location_resolution_source = "postcode_lookup"  # Track that postcode was used to lookup lat/lon (2)
        # Precision is 0 for postcode lookup (exact match)
        location_resolution_precision_m = 0.0
    else:
        # Neither postcode nor lat/lon provided
        raise HTTPException(status_code=400, detail="Need postcode or lat/lon to evaluate.")
    
    # Extract postcode district for prime central London check
    postcode_district = ""
    if request.postcode:
        if " " in request.postcode:
            postcode_district = request.postcode.split()[0].upper()
        else:
            # Try to extract district from postcode (first 2-4 chars)
            postcode_clean = request.postcode.replace(" ", "").upper()
            if len(postcode_clean) >= 2:
                # Match pattern like SW1A, W1, EC1, etc.
                match = re.match(r'([A-Z]{1,2}\d[A-Z]?)', postcode_clean)
                if match:
                    postcode_district = match.group(1)
    
    # Step 2: lat/lon -> nearest borough by matching postcode's borough column if present
    # Using ladcd to match with ons_clean, with fallback to area name matching
    borough = "unknown"
    ons_row = None
    used_borough_fallback = False
    missing_fields = []
    
    if ladcd and pd.notna(ladcd):
        ons_row = ons_data[ons_data["ladcd"] == ladcd]
        if not ons_row.empty:
            borough = normalize_borough_name(str(ons_row.iloc[0]["area"]))
    
    # Check if postcode_lookup_clean has a borough/local authority name column
    # (e.g., "ladnm" or similar) - if it exists, use it as fallback
    # Note: Current CSV structure doesn't have this, but we check for it
    if (ons_row is None or ons_row.empty) and pd.notna(ladcd):
        # Check for borough name columns in postcode data
        borough_name_cols = [col for col in postcode_data.columns if 'name' in col.lower() or 'borough' in col.lower() or 'ladnm' in col.lower()]
        if borough_name_cols:
            borough_name = postcode_row.iloc[0].get(borough_name_cols[0])
            if pd.notna(borough_name):
                # Try to match by area name in ONS data
                ons_row = ons_data[ons_data["area"].str.lower() == str(borough_name).lower().strip()]
                if not ons_row.empty:
                    borough = normalize_borough_name(str(ons_row.iloc[0]["area"]))
                    used_borough_fallback = True
    
    # Step 3: borough -> median/lower/upper quartile from ons_clean
    # Check if we have lat/lon for comps/ML even if ONS is missing
    has_geo = lat is not None and lon is not None and not (pd.isna(lat) or pd.isna(lon))
    
    # If no ONS data and no geo, we can't evaluate
    if (ons_row is None or ons_row.empty) and borough == "unknown" and not has_geo:
        raise HTTPException(status_code=422, detail="Insufficient data for evaluation at this location.")
    
    # If we have ONS row, extract data
    median_rent = None
    lower_quartile = None
    upper_quartile = None
    count_rents = None
    
    if ons_row is not None and not ons_row.empty:
        median_rent_val = ons_row.iloc[0].get("median_rent")
        lower_quartile_val = ons_row.iloc[0].get("lower_quartile_rent")
        upper_quartile_val = ons_row.iloc[0].get("upper_quartile_rent")
        count_rents_val = ons_row.iloc[0].get("count_rents")
        
        # Convert to native Python types
        median_rent = float(median_rent_val) if pd.notna(median_rent_val) else None
        lower_quartile = float(lower_quartile_val) if pd.notna(lower_quartile_val) else None
        upper_quartile = float(upper_quartile_val) if pd.notna(upper_quartile_val) else None
        count_rents = int(count_rents_val) if pd.notna(count_rents_val) else None
    
    # Track missing fields for confidence adjustment
    if lower_quartile is not None and (pd.isna(lower_quartile) or lower_quartile == 0):
        missing_fields.append("lower_quartile")
    if upper_quartile is not None and (pd.isna(upper_quartile) or upper_quartile == 0):
        missing_fields.append("upper_quartile")
    
    # If no ONS data and no comps/ML available, we can't evaluate
    if (median_rent is None or pd.isna(median_rent) or median_rent == 0):
        # Check if we can use comps or ML
        can_use_comps = comps_data is not None and len(comps_data) > 0 and has_geo
        can_use_ml = ml_enabled and has_geo
        
        if not can_use_comps and not can_use_ml:
            raise HTTPException(status_code=422, detail="Insufficient data for evaluation at this location.")
        
        # Set defaults for ONS if we're using comps/ML only
        if median_rent is None or pd.isna(median_rent):
            median_rent = 0.0
        if lower_quartile is None or pd.isna(lower_quartile):
            lower_quartile = 0.0
        if upper_quartile is None or pd.isna(upper_quartile):
            upper_quartile = 0.0
        if count_rents is None or pd.isna(count_rents):
            count_rents = 0
    
    # Step 4: Get bedroom multiplier
    bedrooms_key = str(request.bedrooms) if request.bedrooms < 5 else "5+"
    if request.property_type.lower() == "studio":
        bedrooms_key = "studio"
    
    multiplier = BEDROOM_MULTIPLIERS.get(bedrooms_key, 1.0)
    
    # Calculate ONS-based expected values (if ONS data available)
    expected_median_ons = 0.0
    expected_lower_ons = 0.0
    expected_upper_ons = 0.0
    
    if median_rent is not None and not pd.isna(median_rent) and median_rent > 0:
        expected_median_ons = median_rent * multiplier
        if lower_quartile is not None and not pd.isna(lower_quartile) and lower_quartile > 0:
            expected_lower_ons = lower_quartile * multiplier
        else:
            expected_lower_ons = expected_median_ons * 0.85  # Fallback
        if upper_quartile is not None and not pd.isna(upper_quartile) and upper_quartile > 0:
            expected_upper_ons = upper_quartile * multiplier
        else:
            expected_upper_ons = expected_median_ons * 1.15  # Fallback
    
    # Step 5: Calculate nearest station distance and transport adjustment
    nearest_station_distance_m = float('inf')
    if tfl_stations_data is not None and len(tfl_stations_data) > 0:
        for _, station in tfl_stations_data.iterrows():
            if pd.notna(station["lat"]) and pd.notna(station["lon"]):
                distance = haversine_distance(lat, lon, station["lat"], station["lon"])
                if distance < nearest_station_distance_m:
                    nearest_station_distance_m = distance
    
    if nearest_station_distance_m == float('inf'):
        nearest_station_distance_m = 0.0
    
    transport_adjustment_pct = get_transport_adjustment(nearest_station_distance_m)
    
    # Step 6.5: Try to get comparables estimate (6 - DB KNN first, CSV fallback)
    comps_used = False
    comps_sample_size = 0
    comps_neff = 0.0  # Effective sample size (n_eff)
    comps_radius_m = 0.0
    comps_expected_median = 0.0
    comps_expected_range = [0.0, 0.0]
    adaptive_radius_used = False
    strong_similarity = False
    similarity_dampening_factor = 0.0
    similarity_ratio = 0.0
    comps_source = "none"  # "db" | "csv" | "none"
    comps_db_count_recent = 0
    comps_top10_weight_share = 0.0
    comps_weighted_quantiles_used = "none"
    
    # Try DB KNN first (6)
    if lat and lon:
        try:
            # Query DB for recent comps
            db_comps_rows = query_recent(DB_PATH, lat, lon, radius_m=2000.0, days=45, limit=500)
            comps_db_count_recent = len(db_comps_rows)
            
            if len(db_comps_rows) >= 10:
                # Build target dict for KNN
                target_dict = {
                    "lat": lat,
                    "lon": lon,
                    "bedrooms": request.bedrooms,
                    "property_type": request.property_type,
                    "bathrooms": request.bathrooms,
                    "floorplan_used": 1 if floorplan_used else 0,
                    "floor_area_sqm": area_used_sqm
                }
                
                # Try adaptive radius: 500m -> 800m -> 1200m -> 2000m
                radius_options = [500.0, 800.0, 1200.0, 2000.0]
                db_result = None
                
                for radius in radius_options:
                    db_result = estimate_from_comps(
                        target_dict,
                        db_comps_rows,
                        radius_m=radius,
                        min_comps=10,
                        max_comps=40,
                        weight_threshold=0.01
                    )
                    
                    # Use n_eff >= 10 instead of sample_size >= 12 (2)
                    if db_result["comps_used"] and db_result.get("comps_neff", db_result["sample_size"]) >= 10.0:
                        break
                
                if db_result and db_result["comps_used"]:
                    comps_used = True
                    comps_sample_size = db_result["sample_size"]
                    comps_neff = db_result.get("comps_neff", float(comps_sample_size))  # Use n_eff from KNN
                    comps_radius_m = db_result["radius_m"]
                    comps_expected_median = db_result["comps_estimate_pcm"]
                    comps_expected_range = db_result["comps_range_pcm"]
                    strong_similarity = db_result["strong_similarity"]
                    similarity_ratio = db_result["similarity_ratio"]
                    comps_source = "db"
                    comps_top10_weight_share = db_result.get("comps_top10_weight_share", 0.0)
                    comps_weighted_quantiles_used = db_result.get("comps_weighted_quantiles_used", "none")
        except Exception as e:
            # Fall back to CSV if DB fails
            pass
    
    # Fallback to CSV comps if DB insufficient (6)
    if not comps_used and comps_data is not None and len(comps_data) > 0:
        comps_selected, comps_radius_m = select_comparables(
            lat, lon, request.bedrooms, request.property_type, comps_data
        )
        
        # Check if adaptive radius was used (radius > 500m)
        adaptive_radius_used = bool(comps_radius_m > 500.0)
        
        if len(comps_selected) >= 8:
            comps_expected_median, comps_expected_range = compute_comps_estimate(comps_selected)
            comps_used = True
            comps_sample_size = int(len(comps_selected))
            # For CSV fallback, estimate n_eff (approximate: use sample_size as baseline)
            comps_neff = float(comps_sample_size)  # CSV doesn't have weights, so n_eff ≈ sample_size
            comps_top10_weight_share = 0.0  # Not computed for CSV
            comps_weighted_quantiles_used = "25-75"  # Default for CSV
            comps_source = "csv"  # Set comps_source when CSV comps are used (1)
            
            # Compute improved similarity strength score (5)
            # A comparable is "similar" if:
            # - bedrooms match exactly
            # - property_type match exactly
            # - bathrooms within ±1 (if present)
            # - floor_area_sqm within ±15% (if available)
            similar_count = 0
            similarity_rules_applied = []
            
            subject_bathrooms = request.bathrooms if request.bathrooms else None
            similarity_uses_area = area_used  # Track if area is used in similarity (B.3)
            
            for _, comp in comps_selected.iterrows():
                is_similar = True
                comp_rules = []
                
                # Bedrooms must match exactly
                if comp["bedrooms"] != request.bedrooms:
                    is_similar = False
                else:
                    comp_rules.append("bedrooms")
                
                # Property type must match exactly
                comp_prop_type = parse_property_type(str(comp.get("property_type", "")))
                subject_prop_type = parse_property_type(request.property_type)
                if comp_prop_type != subject_prop_type:
                    is_similar = False
                else:
                    comp_rules.append("property_type")
                
                # Bathrooms within ±1 (if both present)
                if subject_bathrooms is not None:
                    comp_bathrooms = comp.get("bathrooms")
                    if comp_bathrooms is not None and pd.notna(comp_bathrooms):
                        if abs(float(comp_bathrooms) - float(subject_bathrooms)) > 1:
                            is_similar = False
                        else:
                            comp_rules.append("bathrooms")
                
                # Floor area within ±15% (ONLY if area_used == True) (B.3)
                if area_used and area_used_sqm is not None:
                    comp_floor_area = comp.get("floor_area_sqm")
                    if comp_floor_area is not None and pd.notna(comp_floor_area) and comp_floor_area > 0:
                        ratio = abs(float(comp_floor_area) - float(area_used_sqm)) / max(float(comp_floor_area), float(area_used_sqm))
                        if ratio > 0.15:
                            is_similar = False
                        else:
                            comp_rules.append("floor_area")
                
                if is_similar:
                    similar_count += 1
                    if comp_rules:
                        similarity_rules_applied.extend(comp_rules)
            
            similarity_ratio = float(similar_count / len(comps_selected) if len(comps_selected) > 0 else 0.0)
            total = len(comps_selected)
            # strong_similarity requires similarity_ratio >= 0.60, total >= 20, and radius <= 1200m
            strong_similarity = bool(similarity_ratio >= 0.60 and total >= 20 and comps_radius_m <= 1200)
            
            # Deduplicate rules for debug
            similarity_rules_applied = list(set(similarity_rules_applied))
        else:
            strong_similarity = False
            similarity_ratio = 0.0
            similarity_rules_applied = []
    else:
        strong_similarity = False
        similarity_ratio = 0.0
        similarity_rules_applied = []
    
    # Step 6.6: Blend comps with ONS baseline (with prime central London and divergence safeguards)
    prime_central_districts = {"SW1A", "SW1", "SW1X", "W1", "W1J", "W1K", "W1S", "WC2", "EC1", "EC2", "EC3", "EC4"}
    is_prime_central = bool(postcode_district in prime_central_districts)
    
    # Structural divergence safeguard
    structural_divergence = False
    if comps_used and expected_median_ons > 0 and comps_expected_median > 0:
        ratio_comps_to_ons = comps_expected_median / expected_median_ons
        ratio_ons_to_comps = expected_median_ons / comps_expected_median
        if ratio_comps_to_ons > 2.5 or ratio_ons_to_comps > 2.5:
            structural_divergence = True
    structural_divergence = bool(structural_divergence)
    
    # Determine blending weights
    if structural_divergence:
        # Ignore ONS entirely
        ons_weight = 0.0
        comps_weight = 1.0
    elif is_prime_central and comps_used:
        # Prime central: reduce ONS weight to max 15%
        if comps_sample_size >= 15:
            ons_weight = 0.0
        else:
            ons_weight = 0.15 if expected_median_ons > 0 else 0.0
        comps_weight = 1.0 - ons_weight
    elif comps_used:
        # Standard blend: 70% comps, 30% ONS (if ONS available)
        if expected_median_ons > 0:
            comps_weight = 0.7
            ons_weight = 0.3
        else:
            comps_weight = 1.0
            ons_weight = 0.0
    else:
        # Fall back to ONS only (if available)
        if expected_median_ons > 0:
            comps_weight = 0.0
            ons_weight = 1.0
        else:
            # No ONS, no comps - should have been caught earlier, but handle gracefully
            comps_weight = 0.0
            ons_weight = 0.0
    
    # Compute ML prediction if available
    ml_expected_median = None
    if ml_enabled:
        # Extract postcode district for ML prediction
        postcode_district_ml = "UNKNOWN"
        if request.postcode:
            if " " in request.postcode:
                postcode_district_ml = request.postcode.split()[0].upper()
            else:
                postcode_clean = request.postcode.replace(" ", "").upper()
                if len(postcode_clean) >= 2:
                    match = re.match(r'([A-Z]{1,2}\d[A-Z]?)', postcode_clean)
                    if match:
                        postcode_district_ml = match.group(1)
        
        # ML features: ONLY include area if area_used == True (B.5)
        ml_floor_area = area_used_sqm if area_used else None
        
        ml_expected_median = compute_ml_prediction(
            request.bedrooms,
            request.bathrooms,
            ml_floor_area,
            request.property_type,
            nearest_station_distance_m,
            lat=lat,
            lon=lon,
            postcode_district=postcode_district_ml,
            ladcd=ladcd if ladcd else None
        )
    
    # Blend: Include ML if available (B.9 - smarter blending based on strong_similarity)
    if comps_used and ml_expected_median is not None:
        # If strong_similarity: overweight comps, underweight ONS
        if strong_similarity:
            # 85% comps, 15% ML (do not overweight ONS baseline)
            expected_median_base = 0.85 * comps_expected_median + 0.15 * ml_expected_median
            expected_lower_base = 0.85 * comps_expected_range[0] + 0.15 * ml_expected_median * 0.85
            expected_upper_base = 0.85 * comps_expected_range[1] + 0.15 * ml_expected_median * 1.15
        else:
            # Standard blend: 60% comps, 25% ML, 15% ONS
            if ons_weight > 0:
                expected_median_base = 0.60 * comps_expected_median + 0.25 * ml_expected_median + 0.15 * expected_median_ons
                expected_lower_base = 0.60 * comps_expected_range[0] + 0.25 * ml_expected_median * 0.85 + 0.15 * expected_lower_ons
                expected_upper_base = 0.60 * comps_expected_range[1] + 0.25 * ml_expected_median * 1.15 + 0.15 * expected_upper_ons
            else:
                # ONS ignored (prime central or divergence)
                expected_median_base = 0.70 * comps_expected_median + 0.30 * ml_expected_median
                expected_lower_base = 0.70 * comps_expected_range[0] + 0.30 * ml_expected_median * 0.85
                expected_upper_base = 0.70 * comps_expected_range[1] + 0.30 * ml_expected_median * 1.15
    elif comps_used:
        # Existing logic: comps + ONS
        expected_median_base = comps_weight * comps_expected_median + ons_weight * expected_median_ons
        expected_lower_base = comps_weight * comps_expected_range[0] + ons_weight * expected_lower_ons
        expected_upper_base = comps_weight * comps_expected_range[1] + ons_weight * expected_upper_ons
    elif ml_expected_median is not None:
        # ML + ONS (comps not used)
        if ons_weight > 0:
            expected_median_base = 0.70 * ml_expected_median + 0.30 * expected_median_ons
            expected_lower_base = 0.70 * ml_expected_median * 0.85 + 0.30 * expected_lower_ons
            expected_upper_base = 0.70 * ml_expected_median * 1.15 + 0.30 * expected_upper_ons
        else:
            # ONS ignored, use ML only
            expected_median_base = ml_expected_median
            expected_lower_base = ml_expected_median * 0.85
            expected_upper_base = ml_expected_median * 1.15
    else:
        # Fallback to ONS only
        expected_median_base = expected_median_ons
        expected_lower_base = expected_lower_ons
        expected_upper_base = expected_upper_ons
    
    # Final safeguard: if we have no valid estimate, return error
    if expected_median_base <= 0:
        raise HTTPException(status_code=422, detail="Insufficient data for evaluation at this location.")
    
    # Apply transport adjustment
    expected_median_after_transport = expected_median_base * (1 + transport_adjustment_pct)
    expected_lower_after_transport = expected_lower_base * (1 + transport_adjustment_pct)
    expected_upper_after_transport = expected_upper_base * (1 + transport_adjustment_pct)
    
    # Step 7: Calculate extra adjustments (furnished, bathrooms, floor area, quality, amenities)
    furnished_adj_base = get_furnished_adjustment(request.furnished)
    bathrooms_adj_base = get_bathrooms_adjustment(request.bathrooms, request.bedrooms)
    # Size adjustment: ONLY if area_used == True (B.4)
    if area_used and area_used_sqm is not None:
        floor_area_adj_base = get_floor_area_adjustment(area_used_sqm, request.bedrooms)
    else:
        floor_area_adj_base = 0.0  # Must be 0 if area not used
    quality_adj_base = get_quality_adjustment(request.quality)
    amenities_adj_base = get_amenities_adjustment(
        request.has_lift,
        request.has_parking,
        request.has_balcony,
        request.has_terrace,
        request.has_concierge,
        request.property_type
    )
    
    # Apply similarity dampening if strong_similarity is True (4)
    if strong_similarity:
        similarity_dampening_factor = 0.4
        # Dampen extra adjustments only (not transport)
        furnished_adj = furnished_adj_base * similarity_dampening_factor
        bathrooms_adj = bathrooms_adj_base * similarity_dampening_factor
        floor_area_adj = floor_area_adj_base * similarity_dampening_factor
        quality_adj = quality_adj_base * similarity_dampening_factor
        amenities_adj = amenities_adj_base * similarity_dampening_factor  # Dampen amenities too (4)
    else:
        furnished_adj = furnished_adj_base
        bathrooms_adj = bathrooms_adj_base
        floor_area_adj = floor_area_adj_base
        quality_adj = quality_adj_base
        amenities_adj = amenities_adj_base
    
    # Step 7.5: Calculate photo condition adjustment (D, 6)
    photo_adjustment_pct = 0.0
    photos_used_count = 0
    photo_condition_confidence_actual = None
    
    if request.photo_condition_label and request.photo_condition_score is not None:
        photos_used_count = 1  # Simplified - actual count from analysis
        photo_condition_confidence_actual = request.photo_condition_confidence
        
        # Base adjustment by label
        label_adjustments = {
            "dated": -0.08,
            "average": 0.00,
            "modern": +0.06,
            "luxury": +0.12
        }
        base_adj = label_adjustments.get(request.photo_condition_label, 0.0)
        
        # Scale by score (score-50)/50, so score 0 = -1x, score 50 = 0x, score 100 = +1x
        score_factor = (request.photo_condition_score - 50) / 50.0
        photo_adjustment_pct = base_adj * score_factor
        
        # Cap to [-0.10, +0.15] normally, but max +5% if photos_used < 2 or confidence is low (6)
        max_adj = 0.15
        if photos_used_count < 2 or photo_condition_confidence_actual == "low":
            max_adj = 0.05
            photo_condition_confidence_actual = "low"  # Force low if insufficient photos
        
        photo_adjustment_pct = max(-0.10, min(max_adj, photo_adjustment_pct))
    
    # Combine extra adjustments and cap to [-0.12, +0.12]
    extra_adjustment_pct = furnished_adj + bathrooms_adj + floor_area_adj + quality_adj + amenities_adj
    extra_adjustment_pct = max(-0.12, min(0.12, extra_adjustment_pct))
    
    # Build adjustments breakdown (B.4: size MUST be 0 if area not used) (5)
    adjustments_breakdown = {
        "transport": transport_adjustment_pct,
        "furnished": furnished_adj,
        "bathrooms": bathrooms_adj,
        "size": floor_area_adj if area_used else 0.0,  # Must be 0 if area not used
        "quality": quality_adj,
        "amenities": amenities_adj,  # Add amenities to breakdown (5)
        "photo": photo_adjustment_pct
    }
    
    # Build list of amenities that are True for explanations (5)
    amenities_list = []
    if request.has_lift is True:
        amenities_list.append("lift")
    if request.has_parking is True:
        amenities_list.append("parking")
    if request.has_balcony is True and request.has_terrace is not True:
        amenities_list.append("balcony")
    if request.has_terrace is True:
        amenities_list.append("terrace")
    if request.has_concierge is True:
        amenities_list.append("concierge")
    
    # Apply extra adjustment + photo adjustment
    total_extra_adj = extra_adjustment_pct + photo_adjustment_pct
    expected_median = expected_median_after_transport * (1 + total_extra_adj)
    expected_lower = expected_lower_after_transport * (1 + total_extra_adj)
    expected_upper = expected_upper_after_transport * (1 + total_extra_adj)
    expected_range = [expected_lower, expected_upper]
    
    # Step 8: Calculate deviation
    deviation_pct = (request.price_pcm - expected_median) / expected_median
    
    # Step 9: Confidence (improved scoring)
    # Start from ONS count_rents thresholds
    if pd.isna(count_rents) or count_rents < 300 or borough == "unknown":
        confidence = "low"
    elif count_rents >= 1000:
        confidence = "high"
    else:
        confidence = "medium"
    
    # Reduce confidence by 1 level if postcode mapping was fuzzy (fallback borough name, not ladcd)
    if used_borough_fallback:
        if confidence == "high":
            confidence = "medium"
        elif confidence == "medium":
            confidence = "low"
    
    # Reduce confidence by 1 level if any key field was missing or defaulted
    if missing_fields:
        if confidence == "high":
            confidence = "medium"
        elif confidence == "medium":
            confidence = "low"
    
    # Reduce confidence by 1 level if structural divergence detected
    confidence_adjusted = False
    if structural_divergence:
        confidence_adjusted = True
        if confidence == "high":
            confidence = "medium"
        elif confidence == "medium":
            confidence = "low"
    confidence_adjusted = bool(confidence_adjusted)
    
    # Step 9.5: Confidence-aware deviation damping
    deviation_pct_damped = deviation_pct
    if confidence == "low":
        deviation_pct_damped = deviation_pct * 0.65
        # Expand expected range by ±25% around median
        range_expansion = 0.25
        expected_lower = expected_median * (1 - range_expansion)
        expected_upper = expected_median * (1 + range_expansion)
        expected_range = [expected_lower, expected_upper]
    elif confidence == "medium":
        deviation_pct_damped = deviation_pct * 0.85
    # high confidence: no damping
    
    # Step 10: Classification (recompute after damping)
    if deviation_pct_damped <= -0.10:
        classification = "undervalued"
    elif deviation_pct_damped >= 0.10:
        classification = "overvalued"
    else:
        classification = "fair"
    
    # Generate explanations with transparent breakdown
    explanations = []
    location_note = ""
    if postcode_inferred:
        location_note = f" (postcode inferred from coordinates, {inferred_postcode_dist_m:.0f}m from nearest)"
    explanations.append(f"Property located in {borough}{location_note}")
    if median_rent and median_rent > 0:
        # User-facing: simplified bedroom explanation
        explanations.append(f"Bedrooms and property type match local market")
    
    # User-facing explanations (B.10 - less technical)
    if comps_used:
        explanations.append(f"Based on {comps_sample_size} similar rentals within ~{comps_radius_m:.0f}m")
        
        if structural_divergence:
            explanations.append("Using nearby rentals only (borough baseline not representative for this area)")
        elif is_prime_central:
            if ons_weight == 0.0:
                explanations.append("Using nearby rentals only (borough baseline less reliable in prime central locations)")
            else:
                explanations.append(f"Combined nearby rentals with borough baseline")
        else:
            explanations.append(f"Combined nearby rentals with borough baseline")
    else:
        explanations.append("Using borough baseline (nearby rentals not available)")
    
    if transport_adjustment_pct != 0:
        if transport_adjustment_pct > 0:
            explanations.append(f"Adjusted slightly for nearby transport ({nearest_station_distance_m:.0f}m to nearest station)")
        else:
            explanations.append(f"Adjusted slightly for transport distance ({nearest_station_distance_m:.0f}m to nearest station)")
    
    # User-facing explanations (B.10 - less technical, simplified) (5)
    if extra_adjustment_pct != 0:
        adj_summary = []
        if adjustments_breakdown["furnished"] != 0:
            adj_summary.append("furnishing")
        if adjustments_breakdown["bathrooms"] != 0:
            adj_summary.append("bathrooms")
        if area_used and adjustments_breakdown["size"] != 0:
            adj_summary.append("size")
        if adjustments_breakdown["quality"] != 0:
            adj_summary.append("property condition")
        # Add amenities explanation if present (5)
        if adjustments_breakdown.get("amenities", 0) != 0 and amenities_list:
            explanations.append(f"Adjusted for amenities ({', '.join(amenities_list)})")
        if adj_summary:
            explanations.append(f"Adjusted for property features ({', '.join(adj_summary)})")
    
    if strong_similarity:
        explanations.append("Bedrooms and property type match local market closely")
    
    # Simplified final explanation (user-facing)
    explanations.append(f"Estimated fair rent: £{expected_median:.2f}/month")
    
    # Compute tight "Most-Likely Range" when evidence supports it (C)
    most_likely_range_pcm = None
    most_likely_range_basis = None
    
    if comps_used and comps_sample_size > 0:
        # Step 1: Evidence gates
        postcode_valid_check = (
            (request.postcode and request.postcode.strip()) or
            (postcode_inferred and inferred_postcode_dist_m and inferred_postcode_dist_m <= 500)
        )
        
        # Use n_eff instead of raw sample_size for gates (4)
        evidence_gates = {
            'comps_used': comps_used,
            'sample_size': comps_neff >= 18.0,  # Use n_eff >= 18 instead of sample_size >= 20
            'radius': comps_radius_m <= 800,
            'strong_similarity': strong_similarity,
            'location_precision': postcode_valid_check or (request.lat is not None and request.lon is not None),
            'floor_area': (
                (request.floor_area_sqm is not None and request.floor_area_sqm > 0) or
                (floorplan_used and request.floorplan_confidence and request.floorplan_confidence != "low")
            )
        }
        
        all_gates_pass = all(evidence_gates.values())
        
        # Step 2: Measure local dispersion from final selected comps
        # Use DB comps if available, otherwise CSV comps
        comps_for_dispersion = None
        comps_prices = None
        
        if comps_source == "db" and lat and lon:
            # Use DB comps for dispersion
            try:
                db_comps_rows = query_recent(DB_PATH, lat, lon, radius_m=comps_radius_m, days=45, limit=500)
                if len(db_comps_rows) > 0:
                    # Get prices from DB comps
                    comps_prices = np.array([row.get("price_pcm", 0) for row in db_comps_rows[:min(25, len(db_comps_rows))]])
            except Exception:
                pass
        elif comps_data is not None and lat is not None and lon is not None:
            # Fallback to CSV comps
            try:
                comps_selected, _ = select_comparables(
                    lat, lon, request.bedrooms, request.property_type, comps_data
                )
                if len(comps_selected) > 0:
                    comps_for_dispersion = comps_selected
                    top_k = min(25, len(comps_for_dispersion))
                    comps_prices = comps_for_dispersion.head(top_k)["price_pcm"].values
            except Exception:
                pass
        
        if comps_prices is not None and len(comps_prices) > 0:
            
            if len(comps_prices) > 0 and comps_expected_median > 0:
                # Compute median and MAD (median absolute deviation)
                median_comp = float(np.median(comps_prices))
                mad = float(np.median(np.abs(comps_prices - median_comp)))
                # Convert MAD to robust sigma
                sigma = 1.4826 * mad if mad > 0 else 0.0
                relative_sigma = sigma / median_comp if median_comp > 0 else 0.0
                
                # Step 3: Determine tightness using n_eff instead of sample_size (4)
                if comps_source == "db" and strong_similarity and comps_radius_m <= 800 and comps_neff >= 18.0:
                    # Strong DB evidence: aim ±3.5% to ±5%
                    width_pct = max(0.035, min(0.05, 1.0 * relative_sigma))
                elif strong_similarity and comps_radius_m <= 800 and comps_neff >= 18.0:
                    # Strong evidence: aim ±3.5% to ±5%
                    width_pct = max(0.035, min(0.05, 1.0 * relative_sigma))
                elif comps_radius_m <= 1200 and comps_neff >= 15.0:
                    # Medium evidence: ±8%
                    width_pct = 0.08
                else:
                    # Standard range: ±12%
                    width_pct = 0.12
                
                # Step 5: If confidence is low, never allow tighter than ±7%
                if confidence == "low":
                    width_pct = max(width_pct, 0.07)
                
                # Step 4: Compute most_likely_range using n_eff (4)
                if comps_neff >= 10.0:  # Use n_eff >= 10 instead of sample_size >= 12
                    most_likely_low = expected_median * (1 - width_pct)
                    most_likely_high = expected_median * (1 + width_pct)
                    most_likely_range_pcm = [float(most_likely_low), float(most_likely_high)]
                    
                    # Step 6: Set basis string using n_eff
                    if strong_similarity and comps_radius_m <= 800 and comps_neff >= 18.0:
                        most_likely_range_basis = "tight range (strong local evidence)"
                    elif comps_radius_m <= 1200 and comps_neff >= 15.0:
                        most_likely_range_basis = "standard range (moderate local evidence)"
                    else:
                        most_likely_range_basis = "wider range (limited local evidence)"
    
    # Compute quality score and debug info
    quality_score, quality_label, quality_reasons = compute_estimate_quality_score(
        comps_used=comps_used,
        comps_sample_size=comps_sample_size,
        comps_radius_m=comps_radius_m,
        strong_similarity=strong_similarity,
        confidence=confidence,
        structural_divergence_triggered=structural_divergence,
        ons_used=ons_weight > 0,
        ml_used=ml_expected_median is not None
    )
    
    # Determine location source and precision from request (2)
    # Use the tracked location resolution path
    location_source_debug = location_resolution_source if location_resolution_source != "none" else "none"
    location_precision_m_debug = location_resolution_precision_m if location_resolution_precision_m > 0 else 0.0
    geocoding_used_debug = geocoding_used  # Geocoding is only used in /parse-listing, not /evaluate
    
    debug_info = DebugInfo(
        estimate_quality_score=quality_score,
        quality_label=quality_label,
        quality_reasons=quality_reasons,
        model_components={
            "ons_used": ons_weight > 0,
            "comps_used": comps_used,
            "ml_used": ml_expected_median is not None,
            "comps_sample_size": comps_sample_size,
            "comps_neff": float(comps_neff),  # Effective sample size (5)
            "comps_radius_m": comps_radius_m,
            "strong_similarity": strong_similarity,
            "similarity_ratio": float(similarity_ratio),  # Always ensure it's a float
            "similarity_rules_applied": similarity_rules_applied if 'similarity_rules_applied' in locals() else [],
            "similarity_uses_area": similarity_uses_area if 'similarity_uses_area' in locals() else False,
            "area_used": area_used,
            "area_source": area_source,
            "ml_mae": float(ml_mae) if ml_enabled and ml_mae is not None else None,  # Ensure float or None
            "ml_expected_median_pcm": ml_expected_median,
            "portal_assets_used": bool(request.photo_condition_label or request.floorplan_area_sqm),
            "photos_used": photos_used_count if 'photos_used_count' in locals() else (1 if request.photo_condition_label else 0),
            "floorplan_used": bool(floorplan_used),
            "comps_source": comps_source,  # Always set: "db" | "csv" | "none" (1)
            "comps_db_count_recent": comps_db_count_recent if 'comps_db_count_recent' in locals() else 0,
            "comps_top10_weight_share": float(comps_top10_weight_share),  # (5)
            "comps_weighted_quantiles_used": comps_weighted_quantiles_used,  # (5)
            "amenities_used": amenities_list if 'amenities_list' in locals() else [],  # List of amenities that are True (6)
            "amenities_adjustment_pct": float(amenities_adj) if 'amenities_adj' in locals() else 0.0  # Numeric adjustment (6)
        },
        location_source=location_source_debug,  # Never null, defaults to "none" (3)
        location_precision_m=float(location_precision_m_debug),  # Never null, defaults to 0.0 (3)
        geocoding_used=bool(geocoding_used_debug)  # Never null, defaults to False (3)
    )
    
    return EvaluateResponse(
        borough=borough,
        listed_price_pcm=request.price_pcm,
        expected_median_pcm=float(expected_median),
        expected_range_pcm=[float(expected_range[0]), float(expected_range[1])],
        most_likely_range_pcm=most_likely_range_pcm,
        most_likely_range_basis=most_likely_range_basis,
        deviation_pct=float(deviation_pct),
        classification=classification,
        confidence=confidence,
        explanations=explanations,
        nearest_station_distance_m=nearest_station_distance_m,
        transport_adjustment_pct=transport_adjustment_pct,
        used_borough_fallback=bool(used_borough_fallback),
        extra_adjustment_pct=float(extra_adjustment_pct),
        adjustments_breakdown={k: float(v) for k, v in adjustments_breakdown.items()},
        comps_used=bool(comps_used),
        comps_sample_size=int(comps_sample_size),
        comps_radius_m=float(comps_radius_m),
        comps_expected_median_pcm=float(comps_expected_median),
        comps_expected_range_pcm=[float(comps_expected_range[0]), float(comps_expected_range[1])],
        deviation_pct_damped=float(deviation_pct_damped),
        confidence_adjusted=bool(confidence_adjusted),
        adaptive_radius_used=bool(adaptive_radius_used),
        strong_similarity=bool(strong_similarity),
        similarity_dampening_factor=float(similarity_dampening_factor),
        ml_expected_median_pcm=ml_expected_median,
        photo_adjustment_pct=float(photo_adjustment_pct),
        floorplan_used=bool(floorplan_used),
        floorplan_area_sqm_used=float(floorplan_area_sqm_used) if floorplan_area_sqm_used else None,
        area_used=bool(area_used),
        area_source=area_source,
        area_used_sqm=float(area_used_sqm) if area_used_sqm else None,
        debug=debug_info
    )

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"ok": True}

def extract_price_from_jsonld(html: str) -> Optional[Tuple[float, bool]]:
    """
    Extract price from JSON-LD structured data.
    Returns (price_pcm, is_weekly) or None.
    """
    try:
        soup = BeautifulSoup(html, 'html.parser')
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                # Handle arrays
                if isinstance(data, list):
                    for item in data:
                        result = _extract_price_from_json_object(item)
                        if result:
                            return result
                elif isinstance(data, dict):
                    result = _extract_price_from_json_object(data)
                    if result:
                        return result
            except (json.JSONDecodeError, AttributeError):
                continue
    except Exception:
        pass
    return None

def extract_latlon_from_jsonld(html: str) -> Optional[Tuple[float, float]]:
    """Extract latitude/longitude from JSON-LD structured data. Returns (lat, lon) or None."""
    try:
        soup = BeautifulSoup(html, 'html.parser')
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                # Handle arrays
                if isinstance(data, list):
                    for item in data:
                        result = _extract_latlon_from_json_object(item)
                        if result:
                            return result
                elif isinstance(data, dict):
                    result = _extract_latlon_from_json_object(data)
                    if result:
                        return result
            except (json.JSONDecodeError, AttributeError):
                continue
    except Exception:
        pass
    return None

def _extract_latlon_from_json_object(obj: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """Helper to extract lat/lon from a JSON object."""
    if not isinstance(obj, dict):
        return None
    
    # Check geo.latitude/longitude
    if 'geo' in obj:
        geo = obj['geo']
        if isinstance(geo, dict):
            if 'latitude' in geo and 'longitude' in geo:
                try:
                    lat = float(geo['latitude'])
                    lon = float(geo['longitude'])
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        return (lat, lon)
                except (ValueError, TypeError):
                    pass
    
    # Check direct latitude/longitude fields
    if 'latitude' in obj and 'longitude' in obj:
        try:
            lat = float(obj['latitude'])
            lon = float(obj['longitude'])
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return (lat, lon)
        except (ValueError, TypeError):
            pass
    
    # Check location object
    if 'location' in obj:
        return _extract_latlon_from_json_object(obj['location'])
    
    return None

def extract_latlon_from_scripts(html: str) -> Optional[Tuple[float, float]]:
    """Extract latitude/longitude from embedded JavaScript variables."""
    try:
        soup = BeautifulSoup(html, 'html.parser')
        scripts = soup.find_all('script')
        
        patterns = [
            r'["\']latitude["\']\s*:\s*([+-]?\d+\.?\d*)',
            r'["\']longitude["\']\s*:\s*([+-]?\d+\.?\d*)',
            r'["\']lat["\']\s*:\s*([+-]?\d+\.?\d*)',
            r'["\']lng["\']\s*:\s*([+-]?\d+\.?\d*)',
            r'["\']lon["\']\s*:\s*([+-]?\d+\.?\d*)',
        ]
        
        lat = None
        lon = None
        
        for script in scripts:
            if not script.string:
                continue
            script_text = script.string
            
            # Try to find both lat and lon in same script
            for pattern in patterns:
                matches = re.findall(pattern, script_text, re.IGNORECASE)
                if matches:
                    for match in matches:
                        try:
                            val = float(match)
                            if 'lat' in pattern.lower() or 'latitude' in pattern.lower():
                                if lat is None and -90 <= val <= 90:
                                    lat = val
                            elif 'lon' in pattern.lower() or 'lng' in pattern.lower() or 'longitude' in pattern.lower():
                                if lon is None and -180 <= val <= 180:
                                    lon = val
                        except (ValueError, TypeError):
                            continue
            
            # Also try to find location object
            location_patterns = [
                r'["\']location["\']\s*:\s*\{[^}]*["\']latitude["\']\s*:\s*([+-]?\d+\.?\d*)[^}]*["\']longitude["\']\s*:\s*([+-]?\d+\.?\d*)',
                r'["\']location["\']\s*:\s*\{[^}]*["\']lat["\']\s*:\s*([+-]?\d+\.?\d*)[^}]*["\']lng["\']\s*:\s*([+-]?\d+\.?\d*)',
            ]
            
            for pattern in location_patterns:
                matches = re.findall(pattern, script_text, re.IGNORECASE | re.DOTALL)
                if matches and len(matches[0]) == 2:
                    try:
                        lat_val = float(matches[0][0])
                        lon_val = float(matches[0][1])
                        if -90 <= lat_val <= 90 and -180 <= lon_val <= 180:
                            lat = lat_val
                            lon = lon_val
                            break
                    except (ValueError, TypeError, IndexError):
                        continue
            
            if lat is not None and lon is not None:
                return (lat, lon)
    except Exception:
        pass
    
    return (lat, lon) if (lat is not None and lon is not None) else None

def _extract_price_from_json_object(obj: Dict[str, Any]) -> Optional[Tuple[float, bool]]:
    """Helper to extract price from a JSON object."""
    if not isinstance(obj, dict):
        return None
    
    def check_if_weekly(price_spec: Any) -> bool:
        """Check if price specification indicates weekly pricing."""
        if not isinstance(price_spec, dict):
            return False
        spec_str = json.dumps(price_spec).lower()
        return any(term in spec_str for term in ['week', 'pw', 'per week', 'weekly'])
    
    # Check offers.price
    if 'offers' in obj and isinstance(obj['offers'], dict):
        offers = obj['offers']
        if 'price' in offers:
            price_val = offers['price']
            if isinstance(price_val, (int, float)):
                # Check if weekly from priceSpecification
                is_weekly = check_if_weekly(offers.get('priceSpecification', {}))
                return (float(price_val), is_weekly)
        if 'priceSpecification' in offers and isinstance(offers['priceSpecification'], dict):
            ps = offers['priceSpecification']
            if 'price' in ps:
                price_val = ps['price']
                if isinstance(price_val, (int, float)):
                    is_weekly = check_if_weekly(ps)
                    return (float(price_val), is_weekly)
    
    # Check direct price field
    if 'price' in obj:
        price_val = obj['price']
        if isinstance(price_val, (int, float)):
            # Check surrounding context for weekly indicators
            obj_str = json.dumps(obj).lower()
            is_weekly = any(term in obj_str for term in ['week', 'pw', 'per week', 'weekly'])
            return (float(price_val), is_weekly)
    
    return None

def extract_price_from_scripts(html: str) -> Optional[float]:
    """
    Extract price from embedded JavaScript variables.
    Returns price_pcm or None.
    """
    try:
        # Look for patterns like "price":2600 or "rent":2600 in script tags
        price_patterns = [
            r'["\']price["\']\s*:\s*["\']?(\d+(?:,\d{3})*)["\']?',
            r'["\']rent["\']\s*:\s*["\']?(\d+(?:,\d{3})*)["\']?',
            r'price["\']?\s*=\s*["\']?(\d+(?:,\d{3})*)["\']?',
            r'rent["\']?\s*=\s*["\']?(\d+(?:,\d{3})*)["\']?',
        ]
        
        soup = BeautifulSoup(html, 'html.parser')
        scripts = soup.find_all('script')
        
        for script in scripts:
            if not script.string:
                continue
            script_text = script.string
            
            for pattern in price_patterns:
                matches = re.findall(pattern, script_text, re.IGNORECASE)
                if matches:
                    try:
                        # Take first match, convert to float
                        price_str = matches[0].replace(',', '')
                        price = float(price_str)
                        # Sanity check: reasonable rental price
                        if 200 <= price <= 20000:
                            return price
                    except (ValueError, IndexError):
                        continue
    except Exception:
        pass
    return None

def extract_postcode_candidates_from_jsonld(html: str) -> List[str]:
    """Extract all postcode candidates from JSON-LD structured data."""
    candidates = []
    try:
        soup = BeautifulSoup(html, 'html.parser')
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                # Handle arrays
                if isinstance(data, list):
                    for item in data:
                        result = _extract_postcode_from_json_object(item)
                        if result and result not in candidates:
                            candidates.append(result)
                elif isinstance(data, dict):
                    result = _extract_postcode_from_json_object(data)
                    if result and result not in candidates:
                        candidates.append(result)
            except (json.JSONDecodeError, AttributeError):
                continue
    except Exception:
        pass
    return candidates

def _extract_postcode_from_json_object(obj: Dict[str, Any]) -> Optional[str]:
    """Helper to extract postcode from a JSON object."""
    if not isinstance(obj, dict):
        return None
    
    # Check address.postalCode
    if 'address' in obj:
        address = obj['address']
        if isinstance(address, dict) and 'postalCode' in address:
            return str(address['postalCode']).strip()
        elif isinstance(address, str):
            # Sometimes address is a string, try to extract postcode
            postcode_match = re.search(r'\b([A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2})\b', address, re.IGNORECASE)
            if postcode_match:
                return postcode_match.group(1)
    
    # Check direct postalCode field
    if 'postalCode' in obj:
        return str(obj['postalCode']).strip()
    
    return None

def extract_postcode_candidates_from_scripts(html: str) -> List[str]:
    """Extract all postcode candidates from embedded JavaScript variables."""
    candidates = []
    try:
        soup = BeautifulSoup(html, 'html.parser')
        scripts = soup.find_all('script')
        
        for script in scripts:
            if not script.string:
                continue
            script_text = script.string
            
            # Look for postcode/postalCode in JSON-like structures
            patterns = [
                r'["\']postalCode["\']\s*:\s*["\']([^"\']+)["\']',
                r'["\']postcode["\']\s*:\s*["\']([^"\']+)["\']',
                r'postalCode["\']?\s*=\s*["\']([^"\']+)["\']',
                r'postcode["\']?\s*=\s*["\']([^"\']+)["\']',
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, script_text, re.IGNORECASE)
                for match in matches:
                    postcode_str = match.strip()
                    # Validate it looks like a postcode
                    if re.match(r'^[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}$', postcode_str, re.IGNORECASE):
                        if postcode_str not in candidates:
                            candidates.append(postcode_str)
    except Exception:
        pass
    return candidates

def extract_postcode_candidates_from_regex(html: str) -> List[str]:
    """Extract all postcode candidates using regex pattern."""
    candidates = []
    try:
        postcode_pattern = r'\b([A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2})\b'
        matches = re.findall(postcode_pattern, html, re.IGNORECASE)
        for match in matches:
            postcode_str = match.strip()
            if postcode_str not in candidates:
                candidates.append(postcode_str)
    except Exception:
        pass
    return candidates

def validate_postcode_candidate(candidate: str) -> bool:
    """Validate a postcode candidate against lookup sets."""
    normalized = normalize_postcode(candidate)
    normalized_nospace = normalized.replace(" ", "")
    return normalized in valid_postcodes or normalized_nospace in valid_postcode_nospace

# --- Portal Assets Helpers (A) ---
def safe_fetch_image(url: str, max_size_mb: int = 8, timeout: int = 8) -> Optional[bytes]:
    """Safely fetch an image URL with size and timeout limits. Returns bytes or None."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        if not content_type.startswith('image/'):
            return None
        
        # Check size
        max_bytes = max_size_mb * 1024 * 1024
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > max_bytes:
            return None
        
        # Read with size limit
        data = b''
        for chunk in response.iter_content(chunk_size=8192):
            data += chunk
            if len(data) > max_bytes:
                return None
        
        return data
    except Exception:
        return None

def extract_image_urls_from_jsonld(html: str) -> List[str]:
    """Extract image URLs from JSON-LD structured data."""
    image_urls = []
    try:
        soup = BeautifulSoup(html, 'html.parser')
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                if isinstance(data, list):
                    for item in data:
                        urls = _extract_images_from_json_object(item)
                        image_urls.extend(urls)
                elif isinstance(data, dict):
                    urls = _extract_images_from_json_object(data)
                    image_urls.extend(urls)
            except (json.JSONDecodeError, AttributeError):
                continue
    except Exception:
        pass
    return image_urls

def _extract_images_from_json_object(obj: Dict[str, Any]) -> List[str]:
    """Recursively extract image URLs from JSON object."""
    urls = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key.lower() in ['image', 'photo', 'photos', 'imageurl', 'propertyimages']:
                if isinstance(value, str) and value.startswith('http'):
                    urls.append(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str) and item.startswith('http'):
                            urls.append(item)
                        elif isinstance(item, dict) and 'url' in item:
                            urls.append(item['url'])
            else:
                urls.extend(_extract_images_from_json_object(value))
    elif isinstance(obj, list):
        for item in obj:
            urls.extend(_extract_images_from_json_object(item))
    return urls

def extract_image_urls_from_scripts(html: str) -> List[str]:
    """Extract image URLs from embedded JavaScript."""
    image_urls = []
    try:
        soup = BeautifulSoup(html, 'html.parser')
        scripts = soup.find_all('script')
        
        patterns = [
            r'["\']image["\']\s*:\s*["\']([^"\']+)["\']',
            r'["\']imageUrl["\']\s*:\s*["\']([^"\']+)["\']',
            r'["\']photos["\']\s*:\s*\[([^\]]+)\]',
            r'["\']propertyImages["\']\s*:\s*\[([^\]]+)\]',
        ]
        
        for script in scripts:
            if not script.string:
                continue
            script_text = script.string
            
            for pattern in patterns:
                matches = re.findall(pattern, script_text, re.IGNORECASE)
                for match in matches:
                    # Try to extract URLs from match
                    url_matches = re.findall(r'https?://[^\s"\'<>]+', match)
                    image_urls.extend(url_matches)
    except Exception:
        pass
    return image_urls

def extract_floorplan_url(html: str) -> Optional[str]:
    """Extract floorplan URL from HTML."""
    floorplan_url = None
    try:
        # Search in scripts for floorplan keywords
        script_patterns = [
            r'["\']floorplan["\']\s*:\s*["\']([^"\']+)["\']',
            r'["\']floorPlan["\']\s*:\s*["\']([^"\']+)["\']',
            r'["\']floor_plan["\']\s*:\s*["\']([^"\']+)["\']',
        ]
        
        soup = BeautifulSoup(html, 'html.parser')
        scripts = soup.find_all('script')
        for script in scripts:
            if not script.string:
                continue
            for pattern in script_patterns:
                matches = re.findall(pattern, script.string, re.IGNORECASE)
                for match in matches:
                    if 'floorplan' in match.lower() and match.startswith('http'):
                        floorplan_url = match
                        break
                if floorplan_url:
                    break
            if floorplan_url:
                break
        
        # Also search for links/hrefs containing floorplan
        if not floorplan_url:
            for link in soup.find_all('a', href=True):
                href = link['href']
                if 'floorplan' in href.lower() and href.startswith('http'):
                    floorplan_url = href
                    break
        
        # Search img tags with floorplan in src or alt
        if not floorplan_url:
            for img in soup.find_all('img', src=True):
                src = img.get('src', '')
                alt = img.get('alt', '').lower()
                if 'floorplan' in src.lower() or 'floorplan' in alt:
                    if src.startswith('http'):
                        floorplan_url = src
                        break
    except Exception:
        pass
    
    return floorplan_url

def normalize_image_urls(urls: List[str]) -> List[str]:
    """Normalize and deduplicate image URLs."""
    normalized = []
    seen = set()
    
    for url in urls:
        if not url or not url.startswith('https://'):
            continue
        
        # Remove query params for deduplication (keep original for fetching)
        url_base = url.split('?')[0]
        if url_base not in seen:
            seen.add(url_base)
            normalized.append(url)
    
    # Prefer larger images if obvious from URL
    def url_priority(url: str) -> int:
        priority = 0
        url_lower = url.lower()
        if any(keyword in url_lower for keyword in ['large', 'full', '1024', '1280', '1920']):
            priority += 10
        if any(keyword in url_lower for keyword in ['thumb', 'small', '64', '128']):
            priority -= 10
        return priority
    
    normalized.sort(key=url_priority, reverse=True)
    return normalized[:10]  # Max 10 for UI

def choose_best_postcode(
    candidates: List[Dict[str, Any]], 
    lat: Optional[float] = None, 
    lon: Optional[float] = None
) -> Tuple[Optional[str], str, Optional[float]]:
    global postcode_data, postcode_lookup_data
    """
    Choose the best postcode from candidates.
    Returns (chosen_postcode, source, distance_m)
    
    If multiple valid candidates exist and lat/lon is available:
    - Compute distance from listing lat/lon to each candidate's postcode centroid
    - Choose the closest one (smallest haversine distance)
    
    If lat/lon is missing, fall back to priority rules (explicit sources > inferred).
    """
    # Separate explicit sources from inferred
    explicit_candidates = [c for c in candidates if c.get('valid', False) and c.get('source') != 'latlon_inferred']
    inferred_candidates = [c for c in candidates if c.get('valid', False) and c.get('source') == 'latlon_inferred']
    
    # Priority 1: Prefer explicit sources
    valid_candidates = explicit_candidates if explicit_candidates else inferred_candidates
    
    # Fallback: If no valid candidates but lat/lon exists, try inference
    if not valid_candidates and lat is not None and lon is not None:
        inferred_result = infer_postcode_from_latlon(lat, lon)
        if inferred_result and inferred_result.get('postcode'):
            pc_norm = normalize_postcode(str(inferred_result['postcode']))
            is_valid = validate_postcode_candidate(pc_norm)
            if is_valid:
                return pc_norm, "latlon_inferred", inferred_result.get('dist_m')
    
    if not valid_candidates:
        return None, "unknown", None
    
    # If multiple valid candidates and lat/lon exists, choose closest (1, 2)
    if len(valid_candidates) > 1 and lat is not None and lon is not None:
        # Compute distance for each candidate
        candidates_with_distance = []
        for candidate in valid_candidates:
            value = candidate['value']
            source = candidate['source']
            candidate_distance_m = candidate.get('distance_m')  # May already be set for latlon_inferred
            
            # If not set, compute from postcode centroid
            if candidate_distance_m is None:
                # Use postcode_data (original CSV) to find postcode centroid
                pc_norm = normalize_postcode(str(value))
                pc_nospace = pc_norm.replace(" ", "")
                
                # Try postcode_data first (has postcode_nospace column)
                postcode_row = None
                if postcode_data is not None and 'postcode_nospace' in postcode_data.columns:
                    postcode_row = postcode_data[postcode_data['postcode_nospace'].str.upper() == pc_nospace.upper()]
                
                # Fallback to postcode_lookup_data
                if (postcode_row is None or len(postcode_row) == 0) and postcode_lookup_data is not None:
                    if 'postcode' in postcode_lookup_data.columns:
                        lookup_normalized = postcode_lookup_data['postcode'].apply(lambda x: normalize_postcode(str(x)) if pd.notna(x) else "")
                        postcode_row = postcode_lookup_data[lookup_normalized == pc_norm]
                    elif 'postcode_nospace' in postcode_lookup_data.columns:
                        postcode_row = postcode_lookup_data[postcode_lookup_data['postcode_nospace'] == pc_nospace]
                
                if postcode_row is not None and len(postcode_row) > 0:
                    candidate_lat = float(postcode_row.iloc[0]['lat'])
                    candidate_lon = float(postcode_row.iloc[0]['lon'])
                    candidate_distance_m = haversine_distance(lat, lon, candidate_lat, candidate_lon)
            
            candidates_with_distance.append({
                'value': value,
                'source': source,
                'distance_m': candidate_distance_m
            })
        
        # Sort by distance (closest first)
        candidates_with_distance.sort(key=lambda c: c['distance_m'] if c['distance_m'] is not None else float('inf'))
        best = candidates_with_distance[0]
        return best['value'], best['source'], best.get('distance_m')
    
    # Fallback to priority rules if lat/lon missing or single candidate (3)
    # Score each candidate
    scored_candidates = []
    for candidate in valid_candidates:
        score = 0.0
        source = candidate['source']
        value = candidate['value']
        
        # Base score: +100 if valid
        score += 100.0
        
        # Source bonus
        if source == 'address_text':
            score += 50.0
        elif source in ['script', 'jsonld']:
            score += 30.0
        elif source == 'regex':
            score += 0.0
        elif source == 'latlon_inferred':
            score += 40.0
        
        scored_candidates.append({
            'value': value,
            'source': source,
            'score': score,
            'distance_m': candidate.get('distance_m')
        })
    
    # Sort by score (highest first)
    scored_candidates.sort(key=lambda c: c['score'], reverse=True)
    
    if scored_candidates:
        best = scored_candidates[0]
        return best['value'], best['source'], best.get('distance_m')
    
    return None, "unknown", None

def extract_price_from_text(html: str) -> Optional[float]:
    """
    Extract price from visible text (last resort).
    Returns price_pcm or None.
    """
    try:
        # Look for monthly prices only (ignore weekly unless explicitly marked)
        monthly_patterns = [
            r'£\s*(\d{1,3}(?:,\d{3})*)\s*(?:per\s*month|/pcm|pcm|/month)',
            r'(\d{1,3}(?:,\d{3})*)\s*£\s*(?:per\s*month|/pcm|pcm|/month)',
        ]
        
        for pattern in monthly_patterns:
            matches = re.findall(pattern, html, re.IGNORECASE)
            if matches:
                try:
                    price_str = matches[0].replace(',', '')
                    price = float(price_str)
                    if 200 <= price <= 20000:
                        return price
                except (ValueError, IndexError):
                    continue
    except Exception:
        pass
    return None

def extract_from_key_facts(soup: BeautifulSoup) -> Dict[str, Any]:
    """
    Extract bathrooms, floor_area_sqm, and furnished from structured Key Facts section.
    Returns dict with extracted values and extraction method used.
    """
    result = {
        'bathrooms': None,
        'floor_area_sqm': None,
        'furnished': None,
        'method': None
    }
    
    try:
        # Look for Key Facts sections - common patterns in Rightmove
        # Try various selectors and structures
        key_facts_section = None
        
        # Try by class/id with various patterns
        key_facts_selectors = [
            {'class': re.compile('key.*fact', re.I)},
            {'id': re.compile('key.*fact', re.I)},
            {'data-test': re.compile('key.*fact', re.I)},
            {'class': re.compile('property.*detail', re.I)},
            {'class': re.compile('summary|overview', re.I)},
        ]
        
        for selector in key_facts_selectors:
            key_facts_section = soup.find(attrs=selector)
            if key_facts_section:
                break
        
        # Also try finding <dl>, <ul>, or <div> with key facts patterns
        if not key_facts_section:
            key_facts_section = soup.find('dl', class_=re.compile('key|fact|detail', re.I))
        if not key_facts_section:
            key_facts_section = soup.find('ul', class_=re.compile('key|fact|detail', re.I))
        if not key_facts_section:
            key_facts_section = soup.find('div', class_=re.compile('key|fact|detail|summary', re.I))
        
        if not key_facts_section:
            return result
        
        result['method'] = 'key_facts'
        text_content = key_facts_section.get_text().lower()
        
        # Extract bathrooms
        bathroom_patterns = [
            r'(\d+)\s*(?:bathroom|bath)',
            r'bathroom[s]?[:\s]+(\d+)',
        ]
        for pattern in bathroom_patterns:
            matches = re.findall(pattern, text_content, re.IGNORECASE)
            if matches:
                try:
                    bathrooms = int(matches[0])
                    if 0 <= bathrooms <= 10:
                        result['bathrooms'] = bathrooms
                        break
                except (ValueError, IndexError):
                    continue
        
        # Extract floor area
        area_patterns = [
            r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:sq\s*ft|sqft|square\s*feet)',
            r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:sq\s*m|sqm|square\s*meters?|m²)',
            r'floor\s*area[:\s]+(\d+(?:,\d{3})*(?:\.\d+)?)',
        ]
        for pattern in area_patterns:
            matches = re.findall(pattern, text_content, re.IGNORECASE)
            if matches:
                try:
                    area_str = matches[0].replace(',', '')
                    area_val = float(area_str)
                    # Check if it's sq ft (typically > 100) or sqm (typically < 500)
                    if 'sq ft' in pattern.lower() or 'sqft' in pattern.lower() or 'square feet' in pattern.lower():
                        # Convert sq ft to sqm
                        area_val = area_val * 0.092903
                    result['floor_area_sqm'] = round(area_val, 2)
                    break
                except (ValueError, IndexError):
                    continue
        
        # Extract furnished status
        furnished_keywords = {
            'furnished': True,
            'unfurnished': False,
            'part furnished': 'unknown',
            'partially furnished': 'unknown',
        }
        for keyword, value in furnished_keywords.items():
            if keyword in text_content:
                result['furnished'] = str(value).lower() if isinstance(value, bool) else value
                break
        
    except Exception:
        pass
    
    return result

def extract_from_embedded_js(html: str) -> Dict[str, Any]:
    """
    Extract bathrooms, floor_area_sqm, and furnished from embedded JavaScript.
    Returns dict with extracted values.
    """
    result = {
        'bathrooms': None,
        'floor_area_sqm': None,
        'furnished': None,
        'method': None
    }
    
    try:
        soup = BeautifulSoup(html, 'html.parser')
        scripts = soup.find_all('script')
        
        for script in scripts:
            if not script.string:
                continue
            script_text = script.string
            
            # Extract bathrooms
            if not result['bathrooms']:
                bathroom_patterns = [
                    r'["\']bathrooms?["\']\s*:\s*["\']?(\d+)["\']?',
                    r'["\']bath["\']\s*:\s*["\']?(\d+)["\']?',
                    r'["\']bathCount["\']\s*:\s*["\']?(\d+)["\']?',
                    r'bathrooms?\s*=\s*["\']?(\d+)["\']?',
                ]
                for pattern in bathroom_patterns:
                    matches = re.findall(pattern, script_text, re.IGNORECASE)
                    if matches:
                        try:
                            bathrooms = int(matches[0])
                            if 0 <= bathrooms <= 10:
                                result['bathrooms'] = bathrooms
                                result['method'] = 'embedded_js'
                                break
                        except (ValueError, IndexError):
                            continue
            
            # Extract floor area
            if not result['floor_area_sqm']:
                area_patterns = [
                    r'["\']floorArea["\']\s*:\s*["\']?(\d+(?:\.\d+)?)["\']?',
                    r'["\']floor_area["\']\s*:\s*["\']?(\d+(?:\.\d+)?)["\']?',
                    r'["\']size["\']\s*:\s*["\']?(\d+(?:\.\d+)?)["\']?',
                    r'floorArea\s*=\s*["\']?(\d+(?:\.\d+)?)["\']?',
                ]
                for pattern in area_patterns:
                    matches = re.findall(pattern, script_text, re.IGNORECASE)
                    if matches:
                        try:
                            area_val = float(matches[0])
                            # Assume sqm if reasonable, otherwise might be sqft
                            if area_val > 1000:  # Likely sqft
                                area_val = area_val * 0.092903
                            result['floor_area_sqm'] = round(area_val, 2)
                            if not result['method']:
                                result['method'] = 'embedded_js'
                            break
                        except (ValueError, IndexError):
                            continue
            
            # Extract furnished status
            if not result['furnished']:
                furnished_patterns = [
                    r'["\']furnishedType["\']\s*:\s*["\']([^"\']+)["\']',
                    r'["\']furnished["\']\s*:\s*["\']?([^"\',\s]+)["\']?',
                ]
                for pattern in furnished_patterns:
                    matches = re.findall(pattern, script_text, re.IGNORECASE)
                    if matches:
                        furnished_str = matches[0].lower()
                        if 'furnished' in furnished_str and 'un' not in furnished_str:
                            result['furnished'] = 'true'
                        elif 'unfurnished' in furnished_str:
                            result['furnished'] = 'false'
                        elif 'part' in furnished_str:
                            result['furnished'] = 'unknown'
                        if not result['method']:
                            result['method'] = 'embedded_js'
                        break
        
    except Exception:
        pass
    
    return result

def extract_from_description(html: str) -> Dict[str, Any]:
    """
    Extract bathrooms, floor_area_sqm, and furnished from description text (lowest confidence).
    Returns dict with extracted values.
    """
    result = {
        'bathrooms': None,
        'floor_area_sqm': None,
        'furnished': None,
        'method': None
    }
    
    try:
        # Look for description sections
        soup = BeautifulSoup(html, 'html.parser')
        description_selectors = [
            {'class': re.compile('description', re.I)},
            {'id': re.compile('description', re.I)},
            {'data-test': re.compile('description', re.I)},
        ]
        
        description_text = ''
        for selector in description_selectors:
            desc_elem = soup.find(attrs=selector)
            if desc_elem:
                description_text = desc_elem.get_text()
                break
        
        # Fallback to body text if no description section found
        if not description_text:
            body = soup.find('body')
            if body:
                description_text = body.get_text()
        
        description_lower = description_text.lower()
        
        # Extract bathrooms (be conservative - only if unambiguous)
        bathroom_pattern = r'(\d+)\s*(?:bathroom|bath)(?!\s*(?:and|or|\+))'
        matches = re.findall(bathroom_pattern, description_lower, re.IGNORECASE)
        if matches and len(matches) == 1:  # Only if single unambiguous match
            try:
                bathrooms = int(matches[0])
                if 0 <= bathrooms <= 10:
                    result['bathrooms'] = bathrooms
                    result['method'] = 'description'
            except (ValueError, IndexError):
                pass
        
        # Extract floor area (be conservative)
        area_patterns = [
            r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:sq\s*ft|sqft)',
            r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:sq\s*m|sqm|m²)',
        ]
        for pattern in area_patterns:
            matches = re.findall(pattern, description_lower, re.IGNORECASE)
            if matches and len(matches) == 1:  # Only if single unambiguous match
                try:
                    area_str = matches[0].replace(',', '')
                    area_val = float(area_str)
                    # Convert sq ft to sqm if needed
                    if 'sq ft' in pattern.lower() or 'sqft' in pattern.lower():
                        area_val = area_val * 0.092903
                    result['floor_area_sqm'] = round(area_val, 2)
                    if not result['method']:
                        result['method'] = 'description'
                    break
                except (ValueError, IndexError):
                    continue
        
        # Extract furnished status (be conservative)
        if 'furnished' in description_lower and 'unfurnished' not in description_lower:
            result['furnished'] = 'true'
            if not result['method']:
                result['method'] = 'description'
        elif 'unfurnished' in description_lower:
            result['furnished'] = 'false'
            if not result['method']:
                result['method'] = 'description'
        elif 'part' in description_lower and 'furnished' in description_lower:
            result['furnished'] = 'unknown'
            if not result['method']:
                result['method'] = 'description'
        
    except Exception:
        pass
    
    return result

def extract_epc_rating(html: str, soup: Any) -> Tuple[Optional[str], str]:
    """
    Extract EPC rating (A-G) with priority: JSON-LD > HTML meta/key-facts > regex.
    Returns (rating, source) or (None, "none").
    """
    # Priority 1: JSON-LD
    try:
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                if isinstance(data, list):
                    data = data[0] if data else {}
                if isinstance(data, dict):
                    # Look for EPC in various JSON-LD fields
                    epc_fields = ['epcRating', 'epc_rating', 'energyRating', 'energy_rating']
                    for field in epc_fields:
                        if field in data:
                            rating = str(data[field]).strip().upper()
                            if len(rating) == 1 and rating in 'ABCDEFG':
                                return rating, 'jsonld'
            except (json.JSONDecodeError, AttributeError, KeyError):
                continue
    except Exception:
        pass
    
    # Priority 2: HTML key facts / structured sections
    try:
        key_facts_section = None
        key_facts_selectors = [
            {'class': re.compile('key.*fact', re.I)},
            {'id': re.compile('key.*fact', re.I)},
            {'class': re.compile('property.*detail', re.I)},
        ]
        for selector in key_facts_selectors:
            key_facts_section = soup.find(attrs=selector)
            if key_facts_section:
                break
        
        if key_facts_section:
            text = key_facts_section.get_text()
            # Look for EPC patterns
            epc_patterns = [
                r'EPC\s*rating[:\s]+([A-G])',
                r'Energy\s*rating[:\s]+([A-G])',
                r'EPC[:\s]+([A-G])',
            ]
            for pattern in epc_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    rating = matches[0].upper()
                    if rating in 'ABCDEFG':
                        return rating, 'html'
    except Exception:
        pass
    
    # Priority 3: Regex fallback on full text
    try:
        epc_patterns = [
            r'EPC\s*rating[:\s]+([A-G])',
            r'Energy\s*rating[:\s]+([A-G])',
            r'EPC[:\s]+([A-G])\b',
        ]
        for pattern in epc_patterns:
            matches = re.findall(pattern, html, re.IGNORECASE)
            if matches:
                rating = matches[0].upper()
                if rating in 'ABCDEFG':
                    return rating, 'regex'
    except Exception:
        pass
    
    return None, 'none'

def extract_floor_level(html: str, soup: Any) -> Tuple[Optional[int], str]:
    """
    Extract floor level (0 for ground, 1+ for upper floors) with priority: JSON-LD > HTML > regex.
    Returns (level, source) or (None, "none").
    """
    # Priority 1: JSON-LD
    try:
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                if isinstance(data, list):
                    data = data[0] if data else {}
                if isinstance(data, dict):
                    floor_fields = ['floorLevel', 'floor_level', 'floorNumber', 'floor_number', 'level']
                    for field in floor_fields:
                        if field in data:
                            try:
                                level = int(data[field])
                                if level >= 0:
                                    return level, 'jsonld'
                            except (ValueError, TypeError):
                                continue
            except (json.JSONDecodeError, AttributeError, KeyError):
                continue
    except Exception:
        pass
    
    # Priority 2: HTML key facts / structured sections
    try:
        key_facts_section = None
        key_facts_selectors = [
            {'class': re.compile('key.*fact', re.I)},
            {'id': re.compile('key.*fact', re.I)},
            {'class': re.compile('property.*detail', re.I)},
        ]
        for selector in key_facts_selectors:
            key_facts_section = soup.find(attrs=selector)
            if key_facts_section:
                break
        
        if key_facts_section:
            text = key_facts_section.get_text().lower()
            # Map floor descriptions to numbers
            floor_map = {
                'ground floor': 0, 'ground': 0, 'lower ground': -1,
                'first floor': 1, '1st floor': 1, 'second floor': 2, '2nd floor': 2,
                'third floor': 3, '3rd floor': 3, 'fourth floor': 4, '4th floor': 4,
            }
            for desc, level in floor_map.items():
                if desc in text:
                    return level, 'html'
            
            # Try numeric patterns
            floor_patterns = [
                r'(\d+)(?:st|nd|rd|th)?\s*floor',
                r'floor[:\s]+(\d+)',
            ]
            for pattern in floor_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    try:
                        level = int(matches[0])
                        if level >= 0:
                            return level, 'html'
                    except (ValueError, IndexError):
                        continue
    except Exception:
        pass
    
    # Priority 3: Regex fallback
    try:
        html_lower = html.lower()
        floor_map = {
            'ground floor': 0, 'ground': 0,
            'first floor': 1, '1st floor': 1,
            'second floor': 2, '2nd floor': 2,
        }
        for desc, level in floor_map.items():
            if desc in html_lower:
                return level, 'regex'
        
        floor_pattern = r'(\d+)(?:st|nd|rd|th)?\s*floor'
        matches = re.findall(floor_pattern, html_lower, re.IGNORECASE)
        if matches:
            try:
                level = int(matches[0])
                if level >= 0:
                    return level, 'regex'
            except (ValueError, IndexError):
                pass
    except Exception:
        pass
    
    return None, 'none'

def extract_feature_booleans(html: str, soup: Any) -> Dict[str, Tuple[Optional[bool], str]]:
    """
    Extract feature booleans (lift, parking, balcony, terrace, concierge) from structured sections.
    Only sets True when found in key features list; otherwise keeps None.
    Returns dict mapping feature_name -> (value, source).
    """
    result = {
        'has_lift': (None, 'none'),
        'has_parking': (None, 'none'),
        'has_balcony': (None, 'none'),
        'has_terrace': (None, 'none'),
        'has_concierge': (None, 'none'),
    }
    
    # Feature keywords mapping
    feature_keywords = {
        'has_lift': ['lift', 'elevator', 'lift access'],
        'has_parking': ['parking', 'garage', 'off-street parking', 'off street parking'],
        'has_balcony': ['balcony', 'balconies'],
        'has_terrace': ['terrace', 'roof terrace', 'private terrace'],
        'has_concierge': ['concierge', 'porter', '24-hour concierge', '24 hour concierge', 'doorman'],
    }
    
    # Priority 1: JSON-LD / embedded JSON
    try:
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                if isinstance(data, list):
                    data = data[0] if data else {}
                if isinstance(data, dict):
                    # Check amenities/features arrays
                    amenity_fields = ['amenities', 'amenityFeature', 'features', 'facilities']
                    for field in amenity_fields:
                        if field in data:
                            amenities = data[field]
                            if isinstance(amenities, list):
                                amenity_text = ' '.join(str(a).lower() for a in amenities)
                            else:
                                amenity_text = str(amenities).lower()
                            
                            for feature_name, keywords in feature_keywords.items():
                                if any(kw in amenity_text for kw in keywords):
                                    result[feature_name] = (True, 'jsonld')
            except (json.JSONDecodeError, AttributeError, KeyError):
                continue
        
        # Also check embedded scripts for feature flags
        scripts = soup.find_all('script')
        for script in scripts:
            if not script.string:
                continue
            script_text = script.string.lower()
            for feature_name, keywords in feature_keywords.items():
                if result[feature_name][0] is None:
                    for kw in keywords:
                        pattern = rf'["\']{kw}["\']\s*:\s*true'
                        if re.search(pattern, script_text, re.IGNORECASE):
                            result[feature_name] = (True, 'script')
                            break
    except Exception:
        pass
    
    # Priority 2: HTML key features section (most reliable for boolean features)
    try:
        key_features_section = None
        key_features_selectors = [
            {'class': re.compile('key.*feature', re.I)},
            {'class': re.compile('feature', re.I)},
            {'class': re.compile('amenity', re.I)},
            {'id': re.compile('feature', re.I)},
        ]
        for selector in key_features_selectors:
            key_features_section = soup.find(attrs=selector)
            if key_features_section:
                break
        
        # Also try finding <ul> or <ol> with feature items
        if not key_features_section:
            for ul in soup.find_all(['ul', 'ol']):
                ul_class = ul.get('class', [])
                ul_id = ul.get('id', '')
                if any('feature' in str(c).lower() or 'amenity' in str(c).lower() for c in ul_class) or 'feature' in str(ul_id).lower():
                    key_features_section = ul
                    break
        
        if key_features_section:
            feature_text = key_features_section.get_text().lower()
            for feature_name, keywords in feature_keywords.items():
                if result[feature_name][0] is None:
                    if any(kw in feature_text for kw in keywords):
                        result[feature_name] = (True, 'html')
    except Exception:
        pass
    
    # Priority 3: Regex fallback on full HTML text (case-insensitive)
    # Only set True when confident; otherwise keep None
    try:
        html_lower = html.lower()
        for feature_name, keywords in feature_keywords.items():
            if result[feature_name][0] is None:  # Only if not already found
                # Check for keywords in HTML text
                for kw in keywords:
                    # For multi-word phrases, use simple substring search
                    # For single words, use word boundaries to avoid partial matches
                    if ' ' in kw or '-' in kw:
                        # Multi-word phrase: simple case-insensitive search
                        if kw.lower() in html_lower:
                            result[feature_name] = (True, 'regex')
                            break
                    else:
                        # Single word: use word boundaries
                        pattern = r'\b' + re.escape(kw) + r'\b'
                        if re.search(pattern, html_lower, re.IGNORECASE):
                            result[feature_name] = (True, 'regex')
                            break  # Found one keyword, move to next feature
    except Exception:
        pass
    
    return result

@app.post("/parse-listing", response_model=ParseListingResponse)
async def parse_listing(request: ParseListingRequest, debug: bool = Query(False, description="Include debug_raw in response")):
    """
    This endpoint is for user-provided, single-listing convenience only.
    Extracts basic property details from Rightmove or Zoopla listing URLs.
    Does not store HTML or extracted data.
    """
    warnings = []
    price_pcm = None
    bedrooms = None
    property_type = None
    postcode = None
    postcode_valid = False
    postcode_source = "unknown"
    postcode_candidates_response = []
    chosen_postcode = None
    chosen_source = "unknown"
    chosen_distance_m = None
    parsing_confidence = "low"
    extracted_fields = []
    # Initialize all response variables
    bathrooms = None
    floor_area_sqm = None
    furnished = None
    address_text = None
    image_urls = []
    floorplan_url = None
    asset_warnings = []
    asset_extraction_confidence = "low"
    lat = None
    lon = None
    location_source = "none"
    inferred_postcode = None
    inferred_postcode_distance_m = None
    location_resolution = None
    latlon_inference_result = None  # Store lat/lon inference result for inferred_postcode fields
    floor_level = None
    epc_rating = None
    has_lift = None
    has_parking = None
    has_balcony = None
    has_terrace = None
    has_concierge = None
    parsed_feature_warnings = []
    
    # Debug collection (only if debug=true)
    debug_data = {} if debug else None
    if debug_data is not None:
        debug_data['url'] = request.url
        debug_data['candidates'] = {}
        debug_data['chosen'] = {}
        debug_data['snippets'] = {}
        
        # Helper to extract text snippets (max 200 chars, no full HTML)
        def get_snippet(text: str, max_len: int = 200) -> str:
            if not text:
                return ""
            text_clean = re.sub(r'\s+', ' ', text.strip())
            if len(text_clean) > max_len:
                return text_clean[:max_len] + "..."
            return text_clean
    
    try:
        # Validate URL
        parsed_url = urlparse(request.url)
        hostname = parsed_url.netloc.lower()
        
        if 'rightmove.co.uk' not in hostname and 'zoopla.co.uk' not in hostname:
            raise HTTPException(status_code=400, detail="URL must be from Rightmove or Zoopla")
        
        # Fetch HTML once (no retries, no storage)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        response = requests.get(request.url, headers=headers, timeout=10)
        response.raise_for_status()
        html = response.text
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        
        # Collect text snippets for debug (if enabled)
        if debug_data is not None:
            # Extract a few key text snippets from HTML (not full HTML)
            try:
                # Get title/heading
                title_elem = soup.find('title') or soup.find('h1')
                if title_elem:
                    debug_data['snippets']['title'] = get_snippet(title_elem.get_text())
                
                # Get address snippet if available
                addr_elem = soup.find(attrs={'class': re.compile('address', re.I)}) or soup.find(attrs={'itemprop': 'address'})
                if addr_elem:
                    debug_data['snippets']['address'] = get_snippet(addr_elem.get_text())
                
                # Get price snippet
                price_elems = soup.find_all(string=re.compile(r'£\s*\d+', re.I))
                if price_elems:
                    debug_data['snippets']['price_text'] = get_snippet(price_elems[0]) if price_elems else None
            except Exception:
                pass
        
        # Extract price with priority: JSON-LD > Scripts > Text
        price_source = None
        if 'rightmove.co.uk' in hostname:
            # Method A: JSON-LD (preferred)
            jsonld_result = extract_price_from_jsonld(html)
            if jsonld_result:
                price_val, is_weekly = jsonld_result
                if is_weekly:
                    price_pcm = price_val * (52 / 12)  # Convert weekly to monthly
                else:
                    price_pcm = price_val
                parsing_confidence = "high"
                extracted_fields.append("price")
                price_source = "jsonld"
            
            # Method B: Embedded scripts (fallback)
            if not price_pcm:
                script_price = extract_price_from_scripts(html)
                if script_price:
                    price_pcm = script_price
                    parsing_confidence = "medium"
                    extracted_fields.append("price")
                    price_source = "script"
            
            # Method C: Visible text (last resort)
            if not price_pcm:
                text_price = extract_price_from_text(html)
                if text_price:
                    price_pcm = text_price
                    parsing_confidence = "low"
                    extracted_fields.append("price")
                    price_source = "text"
            
            if debug_data is not None:
                debug_data['chosen']['price'] = {'value': price_pcm, 'source': price_source}
        
        elif 'zoopla.co.uk' in hostname:
            # Zoopla: try JSON-LD first, then fallback to existing method
            jsonld_result = extract_price_from_jsonld(html)
            if jsonld_result:
                price_val, is_weekly = jsonld_result
                if is_weekly:
                    price_pcm = price_val * (52 / 12)
                else:
                    price_pcm = price_val
                parsing_confidence = "high"
                extracted_fields.append("price")
                price_source = "jsonld"
            else:
                # Fallback to existing Zoopla parsing
                price_elem = soup.find('p', class_=re.compile('price|rent', re.I))
                if not price_elem:
                    price_elem = soup.find(attrs={'data-testid': re.compile('price', re.I)})
                
                if price_elem:
                    price_text = price_elem.get_text()
                    price_match = re.search(r'[\d,]+', price_text.replace(',', ''))
                    if price_match:
                        price_pcm = float(price_match.group().replace(',', ''))
                        parsing_confidence = "medium"
                        extracted_fields.append("price")
                        price_source = "html"
            
            if debug_data is not None:
                debug_data['chosen']['price'] = {'value': price_pcm, 'source': price_source}
        
        if not price_pcm:
            warnings.append("Could not reliably extract price — please enter manually")
            parsing_confidence = "low"
        
        # Extract bedrooms
        bedroom_patterns = [
            r'(\d+)\s*bed',
            r'(\d+)\s*bedroom',
            r'bedrooms?[:\s]+(\d+)',
        ]
        
        bedroom_found = False
        for pattern in bedroom_patterns:
            matches = re.findall(pattern, html, re.IGNORECASE)
            if matches:
                try:
                    bedrooms = int(matches[0])
                    if 0 <= bedrooms <= 10:
                        bedroom_found = True
                        extracted_fields.append("bedrooms")
                        break
                except:
                    continue
        
        if not bedroom_found:
            warnings.append("Could not reliably extract bedrooms — please check")
        
        # Extract property type
        property_type_map = {
            'flat': ['flat', 'apartment', 'maisonette'],
            'house': ['house', 'terraced', 'semi-detached', 'detached', 'cottage'],
            'studio': ['studio'],
            'room': ['room', 'hmo', 'shared']
        }
        
        html_lower = html.lower()
        for prop_type, keywords in property_type_map.items():
            if any(keyword in html_lower for keyword in keywords):
                property_type = prop_type
                extracted_fields.append("property_type")
                break
        
        if not property_type:
            warnings.append("Could not determine property type — defaulting to 'flat'")
            property_type = "flat"
        
        # Extract address text (best-effort) (C)
        address_text = None
        try:
            # Try to find address in common locations
            address_selectors = [
                {'class': re.compile('address', re.I)},
                {'data-test': re.compile('address', re.I)},
                {'itemprop': 'address'},
            ]
            
            for selector in address_selectors:
                addr_elem = soup.find(attrs=selector)
                if addr_elem:
                    address_text = addr_elem.get_text(strip=True)
                    break
            
            # Fallback: look for structured address in JSON-LD
            if not address_text:
                json_ld_scripts = soup.find_all('script', type='application/ld+json')
                for script in json_ld_scripts:
                    try:
                        data = json.loads(script.string)
                        if isinstance(data, list):
                            data = data[0] if data else {}
                        if isinstance(data, dict):
                            if 'address' in data:
                                addr_obj = data['address']
                                if isinstance(addr_obj, dict):
                                    # Build address string from components
                                    addr_parts = []
                                    if 'streetAddress' in addr_obj:
                                        addr_parts.append(str(addr_obj['streetAddress']))
                                    if 'addressLocality' in addr_obj:
                                        addr_parts.append(str(addr_obj['addressLocality']))
                                    if addr_parts:
                                        address_text = ', '.join(addr_parts)
                                        break
                    except (json.JSONDecodeError, AttributeError, KeyError):
                        continue
        except Exception:
            pass
        
        # Extract lat/lon (B)
        lat = None
        lon = None
        location_source = "none"
        
        if 'rightmove.co.uk' in hostname or 'zoopla.co.uk' in hostname:
            # Priority 1: JSON-LD
            jsonld_latlon = extract_latlon_from_jsonld(html)
            if jsonld_latlon:
                lat, lon = jsonld_latlon
                location_source = "jsonld"
                extracted_fields.append("location")
            
            # Priority 2: Embedded scripts
            if lat is None or lon is None:
                script_latlon = extract_latlon_from_scripts(html)
                if script_latlon:
                    lat, lon = script_latlon
                    location_source = "script"
                    extracted_fields.append("location")
            
            # Priority 3: HTML meta tags (rare)
            if lat is None or lon is None:
                meta_lat = soup.find('meta', property='geo:latitude')
                meta_lon = soup.find('meta', property='geo:longitude')
                if meta_lat and meta_lon:
                    try:
                        lat = float(meta_lat.get('content', ''))
                        lon = float(meta_lon.get('content', ''))
                        location_source = "html"
                        extracted_fields.append("location")
                    except (ValueError, TypeError):
                        pass
        
        # Extract postcode candidates from all sources
        postcode_candidates_raw = []
        
        if 'rightmove.co.uk' in hostname:
            # Collect candidates from all sources
            jsonld_candidates = extract_postcode_candidates_from_jsonld(html)
            if debug_data is not None:
                debug_data['candidates']['postcode_jsonld'] = jsonld_candidates[:5]  # Top 5
            for candidate in jsonld_candidates:
                normalized = normalize_postcode(candidate)
                postcode_candidates_raw.append({
                    'value': normalized,
                    'source': 'jsonld',
                    'original': candidate
                })
            
            script_candidates = extract_postcode_candidates_from_scripts(html)
            if debug_data is not None:
                debug_data['candidates']['postcode_script'] = script_candidates[:5]
            for candidate in script_candidates:
                normalized = normalize_postcode(candidate)
                postcode_candidates_raw.append({
                    'value': normalized,
                    'source': 'script',
                    'original': candidate
                })
            
            regex_candidates = extract_postcode_candidates_from_regex(html)
            if debug_data is not None:
                debug_data['candidates']['postcode_regex'] = regex_candidates[:5]
            for candidate in regex_candidates:
                normalized = normalize_postcode(candidate)
                postcode_candidates_raw.append({
                    'value': normalized,
                    'source': 'regex',
                    'original': candidate
                })
        
        # Extract postcode from address_text (2) - robust extraction
        if address_text:
            try:
                # UK postcode regex (case-insensitive)
                postcode_pattern = r'\b([A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2})\b'
                address_matches = re.findall(postcode_pattern, address_text, re.IGNORECASE)
                if address_matches:
                    # Take first match
                    raw_pc = address_matches[0]
                    pc_norm = normalize_postcode(str(raw_pc))
                    is_valid = validate_postcode_candidate(pc_norm)
                    postcode_candidates_raw.append({
                        'value': pc_norm,
                        'source': 'address_text',
                        'original': raw_pc,
                        'valid': is_valid  # Pre-validate
                    })
            except Exception:
                pass  # Don't crash if address_text extraction fails
        
        # Extract postcode from lat/lon inferred (1) - ALWAYS attempt if lat/lon exists
        # Store inference result for inferred_postcode/inferred_postcode_distance_m fields
        if lat is not None and lon is not None:
            latlon_inference_result = infer_postcode_from_latlon(lat, lon)
            if latlon_inference_result and latlon_inference_result.get('postcode'):
                normalized = normalize_postcode(latlon_inference_result['postcode'])
                is_valid = validate_postcode_candidate(normalized)
                postcode_candidates_raw.append({
                    'value': normalized,
                    'source': 'latlon_inferred',
                    'original': latlon_inference_result['postcode'],
                    'distance_m': latlon_inference_result.get('dist_m'),
                    'valid': is_valid  # Pre-validate for choose_best_postcode
                })
        
        # Validate all candidates
        postcode_candidates = []
        seen_values = set()
        for candidate in postcode_candidates_raw:
            value = candidate['value']
            if value in seen_values:
                continue
            seen_values.add(value)
            # Use pre-validated value if available (for latlon_inferred), otherwise validate
            is_valid = candidate.get('valid')
            if is_valid is None:
                is_valid = validate_postcode_candidate(value)
            candidate_dict = {
                'value': value,
                'source': candidate['source'],
                'valid': is_valid,
                'distance_m': candidate.get('distance_m')
            }
            postcode_candidates.append(candidate_dict)
        
        # Choose best postcode using distance-based selection (1, 2)
        chosen_postcode, chosen_source, chosen_distance_m = choose_best_postcode(postcode_candidates, lat, lon)
        
        # Check for conflicts and add debug warning with distances (4)
        valid_candidates = [c for c in postcode_candidates if c.get('valid', False)]
        if len(valid_candidates) > 1:
            # Compute distances for all candidates if lat/lon available (for warning message)
            candidates_with_distances = []
            for c in valid_candidates:
                value = c['value']
                source = c['source']
                dist_m = c.get('distance_m')
                
                # Compute distance if not already set and lat/lon available
                if dist_m is None and lat is not None and lon is not None:
                    pc_norm = normalize_postcode(str(value))
                    pc_nospace = pc_norm.replace(" ", "")
                    
                    postcode_row = None
                    if postcode_data is not None and 'postcode_nospace' in postcode_data.columns:
                        postcode_row = postcode_data[postcode_data['postcode_nospace'].str.upper() == pc_nospace.upper()]
                    
                    if (postcode_row is None or len(postcode_row) == 0) and postcode_lookup_data is not None:
                        if 'postcode' in postcode_lookup_data.columns:
                            lookup_normalized = postcode_lookup_data['postcode'].apply(lambda x: normalize_postcode(str(x)) if pd.notna(x) else "")
                            postcode_row = postcode_lookup_data[lookup_normalized == pc_norm]
                        elif 'postcode_nospace' in postcode_lookup_data.columns:
                            postcode_row = postcode_lookup_data[postcode_lookup_data['postcode_nospace'] == pc_nospace]
                    
                    if postcode_row is not None and len(postcode_row) > 0:
                        candidate_lat = float(postcode_row.iloc[0]['lat'])
                        candidate_lon = float(postcode_row.iloc[0]['lon'])
                        dist_m = haversine_distance(lat, lon, candidate_lat, candidate_lon)
                
                candidates_with_distances.append({
                    'value': value,
                    'source': source,
                    'distance_m': dist_m
                })
            
            # Build conflict message with distances (4)
            other_candidates = [c for c in candidates_with_distances if c['value'] != chosen_postcode]
            if other_candidates:
                other_parts = []
                for c in other_candidates:
                    if c['distance_m'] is not None:
                        other_parts.append(f"{c['source']}={c['value']} ({int(c['distance_m'])}m)")
                    else:
                        other_parts.append(f"{c['source']}={c['value']}")
                
                chosen_dist_str = ""
                chosen_candidate = next((c for c in candidates_with_distances if c['value'] == chosen_postcode), None)
                if chosen_candidate and chosen_candidate.get('distance_m') is not None:
                    chosen_dist_str = f" ({int(chosen_candidate['distance_m'])}m)"
                
                conflict_msg = f"Postcode conflict resolved: {', '.join(other_parts)}, chosen={chosen_postcode}{chosen_dist_str}"
                warnings.append(conflict_msg)
                if debug_data is not None:
                    debug_data['warnings'] = debug_data.get('warnings', []) + [conflict_msg]
        
        # Guard: if chosen postcode >1500m away, fall back to inferred (5)
        if chosen_postcode and lat is not None and lon is not None:
            # Compute distance if not already set
            if chosen_distance_m is None:
                pc_norm = normalize_postcode(str(chosen_postcode))
                pc_nospace = pc_norm.replace(" ", "")
                
                postcode_row = None
                if postcode_data is not None and 'postcode_nospace' in postcode_data.columns:
                    postcode_row = postcode_data[postcode_data['postcode_nospace'].str.upper() == pc_nospace.upper()]
                
                if (postcode_row is None or len(postcode_row) == 0) and postcode_lookup_data is not None:
                    if 'postcode' in postcode_lookup_data.columns:
                        lookup_normalized = postcode_lookup_data['postcode'].apply(lambda x: normalize_postcode(str(x)) if pd.notna(x) else "")
                        postcode_row = postcode_lookup_data[lookup_normalized == pc_norm]
                    elif 'postcode_nospace' in postcode_lookup_data.columns:
                        postcode_row = postcode_lookup_data[postcode_lookup_data['postcode_nospace'] == pc_nospace]
                
                if postcode_row is not None and len(postcode_row) > 0:
                    candidate_lat = float(postcode_row.iloc[0]['lat'])
                    candidate_lon = float(postcode_row.iloc[0]['lon'])
                    chosen_distance_m = haversine_distance(lat, lon, candidate_lat, candidate_lon)
            
            # Check if distance > 1500m and fall back to inferred
            if chosen_distance_m is not None and chosen_distance_m > 1500:
                # Fall back to inferred postcode
                inferred_result = infer_postcode_from_latlon(lat, lon)
                if inferred_result and inferred_result.get('postcode'):
                    pc_norm = normalize_postcode(str(inferred_result['postcode']))
                    is_valid = validate_postcode_candidate(pc_norm)
                    if is_valid:
                        old_distance = int(chosen_distance_m) if chosen_distance_m else "unknown"
                        chosen_postcode = pc_norm
                        chosen_source = "latlon_inferred"
                        chosen_distance_m = inferred_result.get('dist_m')
                        warnings.append(f"Chosen postcode was {old_distance}m away; using inferred postcode instead")
        
        postcode = chosen_postcode
        postcode_source = chosen_source
        # postcode_valid should be True if chosen_postcode exists and is valid
        if chosen_postcode:
            # Check if the chosen postcode is valid
            postcode_valid = validate_postcode_candidate(chosen_postcode)
        else:
            postcode_valid = False
        
        # Update postcode_candidates with distance_m for response (4)
        postcode_candidates_response = []
        for c in postcode_candidates:
            candidate_dict = {
                'value': c['value'],
                'source': c['source'],
                'valid': c['valid']
            }
            # Add distance_m if available
            if c.get('distance_m') is not None:
                candidate_dict['distance_m'] = c['distance_m']
            elif lat is not None and lon is not None and postcode_lookup_data is not None:
                # Compute distance for this candidate
                pc_norm = normalize_postcode(str(c['value']))
                pc_nospace = pc_norm.replace(" ", "")
                
                # Find postcode in lookup - normalize lookup postcode column values for comparison
                postcode_row = None
                if 'postcode' in postcode_lookup_data.columns:
                    lookup_normalized = postcode_lookup_data['postcode'].apply(lambda x: normalize_postcode(str(x)) if pd.notna(x) else "")
                    postcode_row = postcode_lookup_data[lookup_normalized == pc_norm]
                
                # Also try postcode_nospace if available
                if (postcode_row is None or len(postcode_row) == 0) and 'postcode_nospace' in postcode_lookup_data.columns:
                    postcode_row = postcode_lookup_data[postcode_lookup_data['postcode_nospace'] == pc_nospace]
                
                if postcode_row is not None and len(postcode_row) > 0:
                    candidate_lat = float(postcode_row.iloc[0]['lat'])
                    candidate_lon = float(postcode_row.iloc[0]['lon'])
                    candidate_dict['distance_m'] = haversine_distance(lat, lon, candidate_lat, candidate_lon)
            postcode_candidates_response.append(PostcodeCandidate(**candidate_dict))
        
        # Collect debug info for postcode
        if debug_data is not None:
            debug_data['chosen']['postcode'] = {
                'value': postcode,
                'source': chosen_source,
                'valid': postcode_valid,
                'candidates_count': len(postcode_candidates),
                'distance_m': chosen_distance_m
            }
        
        # Resolve location using helper (A, C) - only if postcode missing/invalid
        location_resolution = None
        if not postcode_valid:
            location_resolution = resolve_location_from_listing(
                request.url,
                address_text,
                lat,
                lon
            )
            
            # Use resolved location if postcode is missing/invalid
            if location_resolution['postcode'] and location_resolution['postcode_valid']:
                # Use resolved postcode
                postcode = location_resolution['postcode']
                postcode_source = location_resolution['location_source']
                postcode_valid = True
                if "postcode" not in extracted_fields:
                    extracted_fields.append("postcode")
                # Update lat/lon if resolved
                if location_resolution['lat'] and location_resolution['lon']:
                    lat = location_resolution['lat']
                    lon = location_resolution['lon']
                    location_source = location_resolution['location_source']
            elif location_resolution['lat'] and location_resolution['lon']:
                # Use resolved lat/lon even if postcode not found
                lat = location_resolution['lat']
                lon = location_resolution['lon']
                location_source = location_resolution['location_source']
                warnings.extend(location_resolution['warnings'])
            else:
                warnings.append("Could not confidently extract a valid postcode; please enter manually.")
        else:
            # Postcode is valid, but update lat/lon if resolved location has better coordinates
            if not lat or not lon:
                location_resolution = resolve_location_from_listing(
                    request.url,
                    address_text,
                    lat,
                    lon
                )
                if location_resolution['lat'] and location_resolution['lon']:
                    lat = location_resolution['lat']
                    lon = location_resolution['lon']
                    location_source = location_resolution['location_source']
        
        # Set inferred postcode and distance (4) - always set if lat/lon inference ran
        # Use latlon_inference_result if available (from candidate extraction)
        if latlon_inference_result and latlon_inference_result.get('postcode'):
            inferred_postcode = latlon_inference_result['postcode']
            inferred_postcode_distance_m = latlon_inference_result.get('dist_m')
        elif location_resolution:
            inferred_postcode = location_resolution.get('postcode') if not postcode_valid else None
            inferred_postcode_distance_m = location_resolution.get('location_precision_m')
        else:
            inferred_postcode = None
            inferred_postcode_distance_m = None
        
        if postcode_valid:
            if "postcode" not in extracted_fields:
                extracted_fields.append("postcode")
        
        # Extract additional fields: bathrooms, floor_area_sqm, furnished
        # Using layered approach: Key Facts > Embedded JS > Description
        # Note: bathrooms, floor_area_sqm, furnished already initialized at function start
        extraction_methods = []  # Track which methods were used
        
        # Method A: Structured HTML Key Facts (preferred)
        if 'rightmove.co.uk' in hostname:
            key_facts_result = extract_from_key_facts(soup)
            if key_facts_result['method'] == 'key_facts':
                if key_facts_result['bathrooms'] is not None:
                    bathrooms = key_facts_result['bathrooms']
                    extracted_fields.append("bathrooms")
                if key_facts_result['floor_area_sqm'] is not None:
                    floor_area_sqm = key_facts_result['floor_area_sqm']
                    extracted_fields.append("floor_area_sqm")
                if key_facts_result['furnished'] is not None:
                    furnished = key_facts_result['furnished']
                    extracted_fields.append("furnished")
                extraction_methods.append('key_facts')
            
            # Method B: Embedded JS (fallback)
            if bathrooms is None or floor_area_sqm is None or furnished is None:
                js_result = extract_from_embedded_js(html)
                if js_result['method'] == 'embedded_js':
                    if bathrooms is None and js_result['bathrooms'] is not None:
                        bathrooms = js_result['bathrooms']
                        extracted_fields.append("bathrooms")
                    if floor_area_sqm is None and js_result['floor_area_sqm'] is not None:
                        floor_area_sqm = js_result['floor_area_sqm']
                        extracted_fields.append("floor_area_sqm")
                    if furnished is None and js_result['furnished'] is not None:
                        furnished = js_result['furnished']
                        extracted_fields.append("furnished")
                    if 'embedded_js' not in extraction_methods:
                        extraction_methods.append('embedded_js')
            
            # Method C: Description text (lowest confidence)
            if bathrooms is None or floor_area_sqm is None or furnished is None:
                desc_result = extract_from_description(html)
                if desc_result['method'] == 'description':
                    if bathrooms is None and desc_result['bathrooms'] is not None:
                        bathrooms = desc_result['bathrooms']
                        extracted_fields.append("bathrooms")
                    if floor_area_sqm is None and desc_result['floor_area_sqm'] is not None:
                        floor_area_sqm = desc_result['floor_area_sqm']
                        extracted_fields.append("floor_area_sqm")
                    if furnished is None and desc_result['furnished'] is not None:
                        furnished = desc_result['furnished']
                        extracted_fields.append("furnished")
                    if 'description' not in extraction_methods:
                        extraction_methods.append('description')
            
            if debug_data is not None:
                debug_data['chosen']['bathrooms'] = {'value': bathrooms, 'method': extraction_methods[0] if extraction_methods else None}
                debug_data['chosen']['floor_area_sqm'] = {'value': floor_area_sqm, 'method': extraction_methods[0] if extraction_methods else None}
                debug_data['chosen']['furnished'] = {'value': furnished, 'method': extraction_methods[0] if extraction_methods else None}
        
        # Update parsing_confidence based on extraction methods
        # high: price + beds + postcode + at least one of (bathrooms or size) from key_facts
        # medium: price + beds + postcode + any additional fields from embedded_js
        # low: anything relying on description or missing key fields
        
        has_key_fields = price_pcm and bedrooms and postcode
        has_additional_from_key_facts = (bathrooms is not None or floor_area_sqm is not None) and 'key_facts' in extraction_methods
        has_additional_from_js = (bathrooms is not None or floor_area_sqm is not None or furnished is not None) and 'embedded_js' in extraction_methods
        used_description = 'description' in extraction_methods
        
        # Update parsing_confidence based on postcode quality (4)
        if postcode_valid:
            # Check if postcode source is high quality
            postcode_high_quality = False
            if postcode_source == 'address_text':
                postcode_high_quality = True
            elif postcode_source == 'latlon_inferred' and chosen_distance_m is not None and chosen_distance_m <= 500:
                postcode_high_quality = True
            
            if postcode_high_quality:
                # Boost confidence if postcode is from address_text or lat/lon inferred with distance <= 500m
                if parsing_confidence == "low":
                    parsing_confidence = "medium"
                if parsing_confidence == "medium":
                    parsing_confidence = "high"
        
        if has_key_fields and has_additional_from_key_facts:
            # High confidence: key fields + additional from structured Key Facts
            if parsing_confidence == "low":
                parsing_confidence = "medium"
            if parsing_confidence == "medium":
                parsing_confidence = "high"
        elif has_key_fields and has_additional_from_js and not used_description:
            # Medium confidence: key fields + additional from embedded JS
            if parsing_confidence == "low":
                parsing_confidence = "medium"
        elif used_description or not has_key_fields:
            # Low confidence: used description method or missing key fields
            parsing_confidence = "low"
        
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch listing: {str(e)}")
    except Exception as e:
        warnings.append(f"Extraction error: {str(e)}")
    
    # Extract portal assets (B)
    # Note: image_urls, floorplan_url, asset_warnings, asset_extraction_confidence already initialized at function start
    
    try:
        if 'rightmove.co.uk' in hostname or 'zoopla.co.uk' in hostname:
            # Extract images from multiple sources
            jsonld_images = extract_image_urls_from_jsonld(html)
            script_images = extract_image_urls_from_scripts(html)
            
            # Also check og:image and gallery img tags
            soup = BeautifulSoup(html, 'html.parser')
            og_image = soup.find('meta', property='og:image')
            if og_image and og_image.get('content'):
                jsonld_images.append(og_image['content'])
            
            # Get gallery images
            gallery_imgs = soup.find_all('img', {'class': re.compile('gallery|photo|image', re.I)})
            for img in gallery_imgs[:10]:  # Limit
                src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
                if src and src.startswith('http'):
                    script_images.append(src)
            
            # Combine and normalize
            all_images = jsonld_images + script_images
            image_urls = normalize_image_urls(all_images)
            
            # Extract floorplan
            floorplan_url = extract_floorplan_url(html)
            
            # Determine confidence
            if len(jsonld_images) > 0 and len(image_urls) >= 3:
                asset_extraction_confidence = "high"
            elif len(image_urls) >= 2:
                asset_extraction_confidence = "medium"
            else:
                asset_extraction_confidence = "low"
            
            if len(image_urls) == 0:
                asset_warnings.append("No listing images found")
            if not floorplan_url:
                asset_warnings.append("No floorplan found")
            
            if debug_data is not None:
                debug_data['chosen']['assets'] = {
                    'image_urls_count': len(image_urls),
                    'floorplan_url_present': floorplan_url is not None,
                    'asset_extraction_confidence': asset_extraction_confidence
                }
    except Exception as e:
        asset_warnings.append(f"Asset extraction error: {str(e)}")
    
    # Extract additional structured features (V23)
    # Note: floor_level, epc_rating, has_lift, has_parking, has_balcony, has_terrace, has_concierge, parsed_feature_warnings already initialized at function start
    try:
        # Extract EPC rating
        epc_rating, epc_source = extract_epc_rating(html, soup)
        if epc_rating:
            extracted_fields.append("epc_rating")
        if debug_data is not None:
            debug_data['chosen']['epc_rating'] = {'value': epc_rating, 'source': epc_source}
        
        # Extract floor level
        floor_level, floor_source = extract_floor_level(html, soup)
        if floor_level is not None:
            extracted_fields.append("floor_level")
        if debug_data is not None:
            debug_data['chosen']['floor_level'] = {'value': floor_level, 'source': floor_source}
        
        # Extract feature booleans
        feature_results = extract_feature_booleans(html, soup)
        if feature_results['has_lift'][0] is not None:
            has_lift = feature_results['has_lift'][0]
            extracted_fields.append("has_lift")
        if feature_results['has_parking'][0] is not None:
            has_parking = feature_results['has_parking'][0]
            extracted_fields.append("has_parking")
        if feature_results['has_balcony'][0] is not None:
            has_balcony = feature_results['has_balcony'][0]
            extracted_fields.append("has_balcony")
        if feature_results['has_terrace'][0] is not None:
            has_terrace = feature_results['has_terrace'][0]
            extracted_fields.append("has_terrace")
        if feature_results['has_concierge'][0] is not None:
            has_concierge = feature_results['has_concierge'][0]
            extracted_fields.append("has_concierge")
        
        if debug_data is not None:
            debug_data['chosen']['features'] = {
                'has_lift': {'value': has_lift, 'source': feature_results['has_lift'][1]},
                'has_parking': {'value': has_parking, 'source': feature_results['has_parking'][1]},
                'has_balcony': {'value': has_balcony, 'source': feature_results['has_balcony'][1]},
                'has_terrace': {'value': has_terrace, 'source': feature_results['has_terrace'][1]},
                'has_concierge': {'value': has_concierge, 'source': feature_results['has_concierge'][1]},
            }
        
        # Minor confidence boost if we extracted structured features
        if (epc_rating or floor_level is not None or 
            any(f[0] is not None for f in feature_results.values())):
            if parsing_confidence == "low" and len(extracted_fields) >= 5:
                # Small boost but don't inflate massively
                pass  # Keep as low if other fields are weak
    except Exception as e:
        parsed_feature_warnings.append(f"Feature extraction error: {str(e)}")
    
    # Build response
    response = ParseListingResponse(
        price_pcm=price_pcm,
        bedrooms=bedrooms,
        property_type=property_type,
        postcode=postcode,
        postcode_valid=postcode_valid,
        postcode_source=postcode_source,
        postcode_candidates=postcode_candidates_response,
        chosen_postcode_source=chosen_source,
        bathrooms=bathrooms,
        floor_area_sqm=floor_area_sqm,
        furnished=furnished,
        parsing_confidence=parsing_confidence,
        extracted_fields=extracted_fields,
        warnings=warnings,
        image_urls=image_urls,
        floorplan_url=floorplan_url,
        asset_warnings=asset_warnings,
        asset_extraction_confidence=asset_extraction_confidence,
        lat=lat,
        lon=lon,
        location_source=location_source,
        inferred_postcode=inferred_postcode,
        inferred_postcode_distance_m=inferred_postcode_distance_m,
        address_text=address_text,
        location_precision_m=location_resolution.get('location_precision_m') if location_resolution else None,
        # Additional structured features (V23)
        floor_level=floor_level,
        epc_rating=epc_rating,
        has_lift=has_lift,
        has_parking=has_parking,
        has_balcony=has_balcony,
        has_terrace=has_terrace,
        has_concierge=has_concierge,
        parsed_feature_warnings=parsed_feature_warnings,
        debug_raw=debug_data
    )
    
    # Print structured JSON log line (1)
    log_data = {
        'url': request.url,
        'parsing_confidence': parsing_confidence,
        'extracted_fields': extracted_fields,
        'postcode': postcode,
        'postcode_valid': postcode_valid,
        'postcode_source': postcode_source,
        'bathrooms': bathrooms,
        'furnished': furnished,
        'floor_area_sqm': floor_area_sqm,
        'lat': lat,
        'lon': lon,
        'location_source': location_source,
        'address_text': address_text,
        'image_urls_count': len(image_urls),
        'floorplan_url_present': floorplan_url is not None,
        'epc_rating': epc_rating,
        'floor_level': floor_level,
        'has_lift': has_lift,
        'has_parking': has_parking,
        'has_balcony': has_balcony,
        'has_terrace': has_terrace,
        'has_concierge': has_concierge,
    }
    print(json.dumps(log_data))
    
    return response

class PhotoAnalysisResponse(BaseModel):
    condition_score: int  # 0-100
    condition_label: str  # "poor"|"ok"|"good"|"great"
    warnings: List[str]

# --- Portal Assets Analysis (C) ---
class AnalyzeListingAssetsRequest(BaseModel):
    url: str

class ConditionAnalysis(BaseModel):
    label: str  # "dated" | "average" | "modern" | "luxury"
    score: int  # 0-100
    confidence: str  # "low" | "medium" | "high"
    signals: List[str]

class FloorplanExtracted(BaseModel):
    estimated_area_sqm: Optional[float] = None
    layout_notes: List[str] = []
    bedroom_count_hint: Optional[int] = None
    bathroom_count_hint: Optional[int] = None

class FloorplanAnalysis(BaseModel):
    present: bool
    confidence: str  # "low" | "medium" | "high"
    extracted: FloorplanExtracted
    warnings: List[str] = []

class AssetsUsed(BaseModel):
    photos_used: int
    floorplan_used: bool

class AnalyzeListingAssetsResponse(BaseModel):
    condition: ConditionAnalysis
    floorplan: FloorplanAnalysis
    assets_used: AssetsUsed
    warnings: List[str] = []

@app.get("/dev/test-parsing")
async def test_parsing(url: str):
    """
    Dev-only endpoint to test parsing on a provided URL.
    Returns detailed extraction results including new structured features.
    """
    try:
        parse_request = ParseListingRequest(url=url)
        result = await parse_listing(parse_request)
        
        return {
            "url": url,
            "extracted_fields": result.extracted_fields,
            "parsing_confidence": result.parsing_confidence,
            "price_pcm": result.price_pcm,
            "bedrooms": result.bedrooms,
            "property_type": result.property_type,
            "postcode": result.postcode,
            "bathrooms": result.bathrooms,
            "floor_area_sqm": result.floor_area_sqm,
            "furnished": result.furnished,
            # New structured features
            "floor_level": result.floor_level,
            "epc_rating": result.epc_rating,
            "has_lift": result.has_lift,
            "has_parking": result.has_parking,
            "has_balcony": result.has_balcony,
            "has_terrace": result.has_terrace,
            "has_concierge": result.has_concierge,
            "warnings": result.warnings,
            "parsed_feature_warnings": result.parsed_feature_warnings,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parsing test failed: {str(e)}")

@app.post("/analyze-listing-assets", response_model=AnalyzeListingAssetsResponse)
async def analyze_listing_assets(request: AnalyzeListingAssetsRequest):
    """Analyze listing photos and floorplan using AI vision."""
    if not ENABLE_PORTAL_ASSETS:
        raise HTTPException(status_code=400, detail="Portal assets disabled")
    
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client not configured")
    
    warnings = []
    
    try:
        # Step 1: Parse listing to get asset URLs and HTML for photo selection (6)
        parse_request = ParseListingRequest(url=request.url)
        parse_response = await parse_listing(parse_request)
        
        # Fetch HTML for photo selection heuristic
        html = ""
        try:
            response = requests.get(request.url, headers={'User-Agent': 'RentScope/1.0'}, timeout=8)
            if response.ok:
                html = response.text
        except:
            pass  # Continue without HTML if fetch fails
        
        # Improve photo selection: prioritize interior-like URLs (6)
        all_image_urls = parse_response.image_urls
        html_lower = html.lower() if html else ""
        interior_keywords = ["kitchen", "bathroom", "living", "interior", "bedroom", "dining"]
        
        # Score images by proximity to interior keywords
        image_scores = []
        for url in all_image_urls:
            score = 0
            url_lower = url.lower()
            # Avoid floorplan and map images
            if "floorplan" in url_lower or "map" in url_lower:
                score -= 10
            # Avoid thumbnails
            if "thumb" in url_lower or "small" in url_lower:
                score -= 5
            # Prefer larger images
            if any(size in url_lower for size in ["large", "full", "1024", "1280", "1920"]):
                score += 5
            
            # Check HTML context near image URL for interior keywords
            try:
                url_index = html_lower.find(url_lower[:50])  # Find URL in HTML
                if url_index >= 0:
                    context = html_lower[max(0, url_index-200):url_index+200]
                    for keyword in interior_keywords:
                        if keyword in context:
                            score += 3
            except:
                pass
            
            image_scores.append((score, url))
        
        # Sort by score (descending) and take top 4
        image_scores.sort(reverse=True, key=lambda x: x[0])
        image_urls = [url for _, url in image_scores[:4]]
        floorplan_url = parse_response.floorplan_url
        
        if not image_urls and not floorplan_url:
            warnings.append("No images or floorplan found in listing")
            return AnalyzeListingAssetsResponse(
                condition=ConditionAnalysis(label="average", score=50, confidence="low", signals=[]),
                floorplan=FloorplanAnalysis(present=False, confidence="low", extracted=FloorplanExtracted(), warnings=warnings),
                assets_used=AssetsUsed(photos_used=0, floorplan_used=False),
                warnings=warnings
            )
        
        # Step 2: Download assets in-memory
        photo_data = []
        for url in image_urls:
            data = safe_fetch_image(url)
            if data:
                photo_data.append(data)
            else:
                warnings.append(f"Could not download image: {url[:50]}...")
        
        floorplan_data = None
        if floorplan_url:
            floorplan_data = safe_fetch_image(floorplan_url)
            if not floorplan_data:
                warnings.append(f"Could not download floorplan: {floorplan_url[:50]}...")
        
        if not photo_data and not floorplan_data:
            warnings.append("No assets could be downloaded")
            return AnalyzeListingAssetsResponse(
                condition=ConditionAnalysis(label="average", score=50, confidence="low", signals=[]),
                floorplan=FloorplanAnalysis(present=bool(floorplan_data), confidence="low", extracted=FloorplanExtracted(), warnings=warnings),
                assets_used=AssetsUsed(photos_used=0, floorplan_used=bool(floorplan_data)),
                warnings=warnings
            )
        
        # Step 3: Prepare images for OpenAI (base64 encode)
        vision_images = []
        for img_data in photo_data:
            base64_img = base64.b64encode(img_data).decode('utf-8')
            vision_images.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
            })
        
        if floorplan_data:
            base64_floorplan = base64.b64encode(floorplan_data).decode('utf-8')
            vision_images.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_floorplan}"}
            })
        
        # Step 4: Call OpenAI vision API
        prompt = """Analyze these property listing images. Return a JSON object with this exact structure:
{
  "condition": {
    "label": "dated" | "average" | "modern" | "luxury",
    "score": 0-100,
    "confidence": "low" | "medium" | "high",
    "signals": ["short concrete signal 1", "signal 2", ...]
  },
  "floorplan": {
    "present": true/false,
    "confidence": "low" | "medium" | "high",
    "extracted": {
      "estimated_area_sqm": number or null,
      "layout_notes": ["note 1", ...],
      "bedroom_count_hint": number or null,
      "bathroom_count_hint": number or null
    },
    "warnings": ["warning 1", ...]
  }
}

Rules:
- Do NOT infer location or address
- For floorplan area: only extract if clearly printed/visible, else null
- Keep signals short and concrete (e.g., "modern kitchen", "dated bathroom", "high ceilings")
- If no floorplan visible, set floorplan.present=false
- Be conservative with confidence levels"""

        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        *vision_images
                    ]}
                ],
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content
            result_json = json.loads(result_text)
            
            # Parse response
            condition_data = result_json.get("condition", {})
            floorplan_data_result = result_json.get("floorplan", {})
            
            condition = ConditionAnalysis(
                label=condition_data.get("label", "average"),
                score=int(condition_data.get("score", 50)),
                confidence=condition_data.get("confidence", "low"),
                signals=condition_data.get("signals", [])
            )
            
            floorplan_extracted = FloorplanExtracted(
                estimated_area_sqm=floorplan_data_result.get("extracted", {}).get("estimated_area_sqm"),
                layout_notes=floorplan_data_result.get("extracted", {}).get("layout_notes", []),
                bedroom_count_hint=floorplan_data_result.get("extracted", {}).get("bedroom_count_hint"),
                bathroom_count_hint=floorplan_data_result.get("extracted", {}).get("bathroom_count_hint")
            )
            
            floorplan = FloorplanAnalysis(
                present=floorplan_data_result.get("present", False),
                confidence=floorplan_data_result.get("confidence", "low"),
                extracted=floorplan_extracted,
                warnings=floorplan_data_result.get("warnings", [])
            )
            
            return AnalyzeListingAssetsResponse(
                condition=condition,
                floorplan=floorplan,
                assets_used=AssetsUsed(photos_used=len(photo_data), floorplan_used=bool(floorplan_data)),
                warnings=warnings
            )
            
        except Exception as e:
            warnings.append(f"OpenAI API error: {str(e)}")
            return AnalyzeListingAssetsResponse(
                condition=ConditionAnalysis(label="average", score=50, confidence="low", signals=[]),
                floorplan=FloorplanAnalysis(present=bool(floorplan_data), confidence="low", extracted=FloorplanExtracted(), warnings=warnings),
                assets_used=AssetsUsed(photos_used=len(photo_data), floorplan_used=bool(floorplan_data)),
                warnings=warnings
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@app.post("/analyze-photos", response_model=PhotoAnalysisResponse)
async def analyze_photos(files: List[UploadFile] = File(...)):
    """
    Analyze property photos to estimate condition.
    TODO: Integrate a real vision model here (e.g., fine-tuned ResNet, CLIP, or commercial API).
    For now, returns a placeholder stub.
    """
    if len(files) > 6:
        raise HTTPException(status_code=400, detail="Maximum 6 images allowed")
    
    # Validate file types
    allowed_types = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
    for file in files:
        if file.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}. Allowed: jpg, png, webp")
    
    # TODO: Implement actual vision model here
    # Example integration points:
    # 1. Load images: from PIL import Image; img = Image.open(io.BytesIO(await file.read()))
    # 2. Preprocess for model
    # 3. Run inference
    # 4. Aggregate scores across multiple images
    # 5. Map to condition_score (0-100) and condition_label
    
    # Placeholder implementation
    return PhotoAnalysisResponse(
        condition_score=50,
        condition_label="ok",
        warnings=["Photo analysis is beta and currently uses a placeholder model."]
    )

@app.get("/debug/comps")
async def debug_comps(
    postcode: str = Query(..., description="Postcode to search"),
    bedrooms: int = Query(..., description="Number of bedrooms"),
    property_type: str = Query(..., description="Property type")
):
    """Debug endpoint to see selected comparables (top 10)"""
    if comps_data is None or len(comps_data) == 0:
        return {"error": "Comparables data not loaded"}
    
    if postcode_data is None:
        return {"error": "Postcode data not loaded"}
    
    # Find postcode location
    postcode_normalized = postcode.replace(" ", "").upper()
    postcode_row = postcode_data[
        postcode_data["postcode_nospace"].str.upper() == postcode_normalized
    ]
    
    if postcode_row.empty:
        postcode_row = postcode_data[
            postcode_data["postcode"].str.upper() == postcode.upper()
        ]
    
    if postcode_row.empty:
        return {"error": "Postcode not found"}
    
    lat = postcode_row.iloc[0]["lat"]
    lon = postcode_row.iloc[0]["lon"]
    
    # Select comparables
    comps_selected, radius_m = select_comparables(
        lat, lon, bedrooms, property_type, comps_data, min_comps=1, max_comps=10
    )
    
    if len(comps_selected) == 0:
        return {
            "postcode": postcode,
            "lat": lat,
            "lon": lon,
            "bedrooms": bedrooms,
            "property_type": property_type,
            "comps_found": 0,
            "radius_m": radius_m,
            "comps": []
        }
    
    # Prepare response
    comps_list = []
    for _, comp in comps_selected.head(10).iterrows():
        time_on_market = time_on_market_days(comp["first_seen"], comp["last_seen"])
        comps_list.append({
            "price_pcm": float(comp["price_pcm"]),
            "distance_m": float(comp["distance_m"]),
            "time_on_market_days": time_on_market,
            "bedrooms": int(comp["bedrooms"]),
            "property_type": str(comp["property_type"])
        })
    
    return {
        "postcode": postcode,
        "lat": lat,
        "lon": lon,
        "bedrooms": bedrooms,
        "property_type": property_type,
        "comps_found": len(comps_selected),
        "radius_m": float(radius_m),
        "comps": comps_list
    }

