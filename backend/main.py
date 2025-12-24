from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os
import math
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple

app = FastAPI()

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
    
    # Try radius 800m first, expand to 2000m if needed
    radius_m = 800.0
    comps_in_radius = comps_filtered[comps_filtered["distance_m"] <= radius_m]
    
    if len(comps_in_radius) < min_comps:
        radius_m = 2000.0
        comps_in_radius = comps_filtered[comps_filtered["distance_m"] <= radius_m]
    
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
    
    prices = comps_df["price_pcm"].values
    weights = comps_df["weight"].values
    
    # Unweighted median
    unweighted_median = float(prices.median())
    
    # Weighted mean
    weighted_mean = float((prices * weights).sum() / weights.sum())
    
    # Expected median = average of unweighted median and weighted mean
    expected_median = (unweighted_median + weighted_mean) / 2.0
    
    # Robust range using percentiles (25/75)
    expected_lower = float(prices.quantile(0.25))
    expected_upper = float(prices.quantile(0.75))
    
    return expected_median, [expected_lower, expected_upper]

@app.on_event("startup")
async def load_data():
    """Load CSV files at startup"""
    global ons_data, postcode_data, tfl_stations_data, comps_data
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    
    try:
        ons_data = pd.read_csv(os.path.join(data_dir, "ons_clean.csv"))
        postcode_data = pd.read_csv(os.path.join(data_dir, "postcode_lookup_clean.csv"))
        tfl_stations_data = pd.read_csv(os.path.join(data_dir, "tfl_stations.csv"))
        
        print(f"Loaded ONS data: {len(ons_data)} rows")
        print(f"Loaded postcode data: {len(postcode_data)} rows")
        print(f"Loaded TFL stations data: {len(tfl_stations_data)} rows")
        
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
class EvaluateRequest(BaseModel):
    url: str = ""
    price_pcm: float
    bedrooms: int
    property_type: str  # "flat|house|studio|room"
    postcode: str
    bathrooms: Optional[int] = None
    floor_area_sqm: Optional[float] = None
    furnished: Optional[bool] = None
    quality: Optional[str] = "average"  # "dated", "average", "modern"

class EvaluateResponse(BaseModel):
    borough: str
    listed_price_pcm: float
    expected_median_pcm: float
    expected_range_pcm: List[float]
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

@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_property(request: EvaluateRequest):
    """Evaluate a property listing"""
    
    if ons_data is None or postcode_data is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    # Step 1: postcode -> lat/lon using postcode_lookup_clean
    postcode_normalized = request.postcode.replace(" ", "").upper()
    postcode_row = postcode_data[
        postcode_data["postcode_nospace"].str.upper() == postcode_normalized
    ]
    
    if postcode_row.empty:
        # Try with space
        postcode_row = postcode_data[
            postcode_data["postcode"].str.upper() == request.postcode.upper()
        ]
    
    if postcode_row.empty:
        return EvaluateResponse(
            borough="unknown",
            listed_price_pcm=request.price_pcm,
            expected_median_pcm=0.0,
            expected_range_pcm=[0.0, 0.0],
            deviation_pct=0.0,
            classification="fair",
            confidence="low",
            explanations=["Postcode not found in database"],
            nearest_station_distance_m=0.0,
            transport_adjustment_pct=0.0,
            used_borough_fallback=False,
            extra_adjustment_pct=0.0,
            adjustments_breakdown={"transport": 0.0, "furnished": 0.0, "bathrooms": 0.0, "size": 0.0, "quality": 0.0},
            comps_used=False,
            comps_sample_size=0,
            comps_radius_m=0.0,
            comps_expected_median_pcm=0.0,
            comps_expected_range_pcm=[0.0, 0.0]
        )
    
    lat = postcode_row.iloc[0]["lat"]
    lon = postcode_row.iloc[0]["lon"]
    ladcd = postcode_row.iloc[0]["ladcd"]
    
    # Step 2: lat/lon -> nearest borough by matching postcode's borough column if present
    # Using ladcd to match with ons_clean, with fallback to area name matching
    borough = "unknown"
    ons_row = None
    used_borough_fallback = False
    missing_fields = []
    
    if pd.notna(ladcd):
        ons_row = ons_data[ons_data["ladcd"] == ladcd]
        if not ons_row.empty:
            borough = normalize_borough_name(ons_row.iloc[0]["area"])
    
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
                    borough = normalize_borough_name(ons_row.iloc[0]["area"])
                    used_borough_fallback = True
    
    # Step 3: borough -> median/lower/upper quartile from ons_clean
    if ons_row is None or ons_row.empty:
        return EvaluateResponse(
            borough=borough,
            listed_price_pcm=request.price_pcm,
            expected_median_pcm=0.0,
            expected_range_pcm=[0.0, 0.0],
            deviation_pct=0.0,
            classification="fair",
            confidence="low",
            explanations=["Borough data not available in ONS dataset"],
            nearest_station_distance_m=0.0,
            transport_adjustment_pct=0.0,
            used_borough_fallback=used_borough_fallback,
            extra_adjustment_pct=0.0,
            adjustments_breakdown={"transport": 0.0, "furnished": 0.0, "bathrooms": 0.0, "size": 0.0, "quality": 0.0},
            comps_used=False,
            comps_sample_size=0,
            comps_radius_m=0.0,
            comps_expected_median_pcm=0.0,
            comps_expected_range_pcm=[0.0, 0.0]
        )
    
    median_rent = ons_row.iloc[0]["median_rent"]
    lower_quartile = ons_row.iloc[0]["lower_quartile_rent"]
    upper_quartile = ons_row.iloc[0]["upper_quartile_rent"]
    count_rents = ons_row.iloc[0]["count_rents"]
    
    # Track missing fields for confidence adjustment
    if pd.isna(lower_quartile) or lower_quartile == 0:
        missing_fields.append("lower_quartile")
    if pd.isna(upper_quartile) or upper_quartile == 0:
        missing_fields.append("upper_quartile")
    
    # Handle missing values
    if pd.isna(median_rent) or median_rent == 0:
        return EvaluateResponse(
            borough=borough,
            listed_price_pcm=request.price_pcm,
            expected_median_pcm=0.0,
            expected_range_pcm=[0.0, 0.0],
            deviation_pct=0.0,
            classification="fair",
            confidence="low",
            explanations=["Median rent data not available for this borough"],
            nearest_station_distance_m=0.0,
            transport_adjustment_pct=0.0,
            used_borough_fallback=used_borough_fallback,
            extra_adjustment_pct=0.0,
            adjustments_breakdown={"transport": 0.0, "furnished": 0.0, "bathrooms": 0.0, "size": 0.0, "quality": 0.0},
            comps_used=False,
            comps_sample_size=0,
            comps_radius_m=0.0,
            comps_expected_median_pcm=0.0,
            comps_expected_range_pcm=[0.0, 0.0]
        )
    
    # Step 4: Get bedroom multiplier
    bedrooms_key = str(request.bedrooms) if request.bedrooms < 5 else "5+"
    if request.property_type.lower() == "studio":
        bedrooms_key = "studio"
    
    multiplier = BEDROOM_MULTIPLIERS.get(bedrooms_key, 1.0)
    
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
    
    # Step 6: Calculate ONS baseline expected values
    expected_median_ons = median_rent * multiplier
    expected_lower_ons = (lower_quartile * multiplier) if pd.notna(lower_quartile) else expected_median_ons * 0.8
    expected_upper_ons = (upper_quartile * multiplier) if pd.notna(upper_quartile) else expected_median_ons * 1.2
    
    # Step 6.5: Try to get comparables estimate
    comps_used = False
    comps_sample_size = 0
    comps_radius_m = 0.0
    comps_expected_median = 0.0
    comps_expected_range = [0.0, 0.0]
    
    if comps_data is not None and len(comps_data) > 0:
        comps_selected, comps_radius_m = select_comparables(
            lat, lon, request.bedrooms, request.property_type, comps_data
        )
        
        if len(comps_selected) >= 8:
            comps_expected_median, comps_expected_range = compute_comps_estimate(comps_selected)
            comps_used = True
            comps_sample_size = len(comps_selected)
    
    # Step 6.6: Blend comps with ONS baseline
    if comps_used:
        # Blend: 70% comps, 30% ONS
        expected_median_base = 0.7 * comps_expected_median + 0.3 * expected_median_ons
        expected_lower_base = 0.7 * comps_expected_range[0] + 0.3 * expected_lower_ons
        expected_upper_base = 0.7 * comps_expected_range[1] + 0.3 * expected_upper_ons
    else:
        # Fall back to ONS only
        expected_median_base = expected_median_ons
        expected_lower_base = expected_lower_ons
        expected_upper_base = expected_upper_ons
    
    # Apply transport adjustment
    expected_median_after_transport = expected_median_base * (1 + transport_adjustment_pct)
    expected_lower_after_transport = expected_lower_base * (1 + transport_adjustment_pct)
    expected_upper_after_transport = expected_upper_base * (1 + transport_adjustment_pct)
    
    # Step 7: Calculate extra adjustments (furnished, bathrooms, floor area, quality)
    furnished_adj = get_furnished_adjustment(request.furnished)
    bathrooms_adj = get_bathrooms_adjustment(request.bathrooms, request.bedrooms)
    floor_area_adj = get_floor_area_adjustment(request.floor_area_sqm, request.bedrooms)
    quality_adj = get_quality_adjustment(request.quality)
    
    # Combine extra adjustments and cap to [-0.12, +0.12]
    extra_adjustment_pct = furnished_adj + bathrooms_adj + floor_area_adj + quality_adj
    extra_adjustment_pct = max(-0.12, min(0.12, extra_adjustment_pct))
    
    # Build adjustments breakdown
    adjustments_breakdown = {
        "transport": transport_adjustment_pct,
        "furnished": furnished_adj,
        "bathrooms": bathrooms_adj,
        "size": floor_area_adj,
        "quality": quality_adj
    }
    
    # Apply extra adjustment
    expected_median = expected_median_after_transport * (1 + extra_adjustment_pct)
    expected_lower = expected_lower_after_transport * (1 + extra_adjustment_pct)
    expected_upper = expected_upper_after_transport * (1 + extra_adjustment_pct)
    expected_range = [expected_lower, expected_upper]
    
    # Step 8: Calculate deviation
    deviation_pct = (request.price_pcm - expected_median) / expected_median
    
    # Step 9: Classification
    if deviation_pct <= -0.10:
        classification = "undervalued"
    elif deviation_pct >= 0.10:
        classification = "overvalued"
    else:
        classification = "fair"
    
    # Step 10: Confidence (improved scoring)
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
    
    # Generate explanations with transparent breakdown
    explanations = []
    explanations.append(f"Property located in {borough}")
    explanations.append(f"Base ONS median rent for borough: £{median_rent:.2f}/month")
    explanations.append(f"Bedroom multiplier ({bedrooms_key} bedroom {request.property_type}): {multiplier:.2f}x → £{expected_median_ons:.2f}/month")
    
    if comps_used:
        explanations.append(f"Comparables estimate: {comps_sample_size} recent properties within {comps_radius_m:.0f}m (time-on-market weighted) → £{comps_expected_median:.2f}/month")
        explanations.append(f"Blended estimate (70% comps, 30% ONS baseline): £{expected_median_base:.2f}/month")
    else:
        explanations.append("Comparables not available or insufficient sample size - using ONS baseline only")
    
    if transport_adjustment_pct != 0:
        explanations.append(f"Transport adjustment ({nearest_station_distance_m:.0f}m to nearest station): {transport_adjustment_pct*100:+.1f}% → £{expected_median_after_transport:.2f}/month")
    
    if extra_adjustment_pct != 0:
        adj_details = []
        if adjustments_breakdown["furnished"] != 0:
            adj_details.append(f"furnished: {adjustments_breakdown['furnished']*100:+.1f}%")
        if adjustments_breakdown["bathrooms"] != 0:
            adj_details.append(f"bathrooms: {adjustments_breakdown['bathrooms']*100:+.1f}%")
        if adjustments_breakdown["size"] != 0:
            adj_details.append(f"size: {adjustments_breakdown['size']*100:+.1f}%")
        if adjustments_breakdown["quality"] != 0:
            adj_details.append(f"quality: {adjustments_breakdown['quality']*100:+.1f}%")
        if adj_details:
            explanations.append(f"Extra adjustments ({', '.join(adj_details)}): {extra_adjustment_pct*100:+.1f}% → £{expected_median:.2f}/month")
    
    explanations.append(f"Final expected median: £{expected_median:.2f}/month (range: £{expected_lower:.2f} - £{expected_upper:.2f})")
    explanations.append(f"Listed price is {abs(deviation_pct)*100:.1f}% {'below' if deviation_pct < 0 else 'above'} expected median")
    explanations.append(f"Confidence level: {confidence} (based on {int(count_rents) if pd.notna(count_rents) else 0} rental records)")
    explanations.append("Note: This is a borough baseline + size + accessibility estimate, not a guaranteed 'true rent'")
    
    return EvaluateResponse(
        borough=borough,
        listed_price_pcm=request.price_pcm,
        expected_median_pcm=expected_median,
        expected_range_pcm=expected_range,
        deviation_pct=deviation_pct,
        classification=classification,
        confidence=confidence,
        explanations=explanations,
        nearest_station_distance_m=nearest_station_distance_m,
        transport_adjustment_pct=transport_adjustment_pct,
        used_borough_fallback=used_borough_fallback,
        extra_adjustment_pct=extra_adjustment_pct,
        adjustments_breakdown=adjustments_breakdown,
        comps_used=comps_used,
        comps_sample_size=comps_sample_size,
        comps_radius_m=comps_radius_m,
        comps_expected_median_pcm=comps_expected_median,
        comps_expected_range_pcm=comps_expected_range
    )

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"ok": True}

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

