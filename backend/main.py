from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os
import math
from typing import Optional, List

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

@app.on_event("startup")
async def load_data():
    """Load CSV files at startup"""
    global ons_data, postcode_data, tfl_stations_data
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    
    try:
        ons_data = pd.read_csv(os.path.join(data_dir, "ons_clean.csv"))
        postcode_data = pd.read_csv(os.path.join(data_dir, "postcode_lookup_clean.csv"))
        tfl_stations_data = pd.read_csv(os.path.join(data_dir, "tfl_stations.csv"))
        
        print(f"Loaded ONS data: {len(ons_data)} rows")
        print(f"Loaded postcode data: {len(postcode_data)} rows")
        print(f"Loaded TFL stations data: {len(tfl_stations_data)} rows")
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
            adjustments_breakdown={"transport": 0.0, "furnished": 0.0, "bathrooms": 0.0, "size": 0.0, "quality": 0.0}
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
            adjustments_breakdown={"transport": 0.0, "furnished": 0.0, "bathrooms": 0.0, "size": 0.0, "quality": 0.0}
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
            adjustments_breakdown={"transport": 0.0, "furnished": 0.0, "bathrooms": 0.0, "size": 0.0, "quality": 0.0}
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
    
    # Step 6: Calculate expected values (before transport adjustment)
    expected_median_base = median_rent * multiplier
    expected_lower_base = (lower_quartile * multiplier) if pd.notna(lower_quartile) else expected_median_base * 0.8
    expected_upper_base = (upper_quartile * multiplier) if pd.notna(upper_quartile) else expected_median_base * 1.2
    
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
    explanations.append(f"Bedroom multiplier ({bedrooms_key} bedroom {request.property_type}): {multiplier:.2f}x → £{expected_median_base:.2f}/month")
    
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
        adjustments_breakdown=adjustments_breakdown
    )

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"ok": True}

