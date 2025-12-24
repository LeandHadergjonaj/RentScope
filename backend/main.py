from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os
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

class EvaluateResponse(BaseModel):
    borough: str
    listed_price_pcm: float
    expected_median_pcm: float
    expected_range_pcm: List[float]
    deviation_pct: float
    classification: str  # "undervalued|fair|overvalued"
    confidence: str  # "low|medium|high"
    explanations: List[str]

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
            explanations=["Postcode not found in database"]
        )
    
    lat = postcode_row.iloc[0]["lat"]
    lon = postcode_row.iloc[0]["lon"]
    ladcd = postcode_row.iloc[0]["ladcd"]
    
    # Step 2: lat/lon -> nearest borough by matching postcode's borough column if present
    # Using ladcd to match with ons_clean
    borough = "unknown"
    ons_row = None
    
    if pd.notna(ladcd):
        ons_row = ons_data[ons_data["ladcd"] == ladcd]
        if not ons_row.empty:
            borough = ons_row.iloc[0]["area"]
    
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
            explanations=["Borough data not available in ONS dataset"]
        )
    
    median_rent = ons_row.iloc[0]["median_rent"]
    lower_quartile = ons_row.iloc[0]["lower_quartile_rent"]
    upper_quartile = ons_row.iloc[0]["upper_quartile_rent"]
    count_rents = ons_row.iloc[0]["count_rents"]
    
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
            explanations=["Median rent data not available for this borough"]
        )
    
    # Step 4: Get bedroom multiplier
    bedrooms_key = str(request.bedrooms) if request.bedrooms < 5 else "5+"
    if request.property_type.lower() == "studio":
        bedrooms_key = "studio"
    
    multiplier = BEDROOM_MULTIPLIERS.get(bedrooms_key, 1.0)
    
    # Step 5: Calculate expected values
    expected_median = median_rent * multiplier
    expected_lower = (lower_quartile * multiplier) if pd.notna(lower_quartile) else expected_median * 0.8
    expected_upper = (upper_quartile * multiplier) if pd.notna(upper_quartile) else expected_median * 1.2
    expected_range = [expected_lower, expected_upper]
    
    # Step 6: Calculate deviation
    deviation_pct = (request.price_pcm - expected_median) / expected_median
    
    # Step 7: Classification
    if deviation_pct <= -0.10:
        classification = "undervalued"
    elif deviation_pct >= 0.10:
        classification = "overvalued"
    else:
        classification = "fair"
    
    # Step 8: Confidence
    if pd.isna(count_rents) or count_rents < 300 or borough == "unknown":
        confidence = "low"
    elif count_rents >= 1000:
        confidence = "high"
    else:
        confidence = "medium"
    
    # Generate explanations
    explanations = []
    explanations.append(f"Property located in {borough}")
    explanations.append(f"Expected median rent for {bedrooms_key} bedroom {request.property_type}: £{expected_median:.2f}/month")
    explanations.append(f"Listed price is {abs(deviation_pct)*100:.1f}% {'below' if deviation_pct < 0 else 'above'} expected median")
    explanations.append(f"Confidence level: {confidence} (based on {int(count_rents) if pd.notna(count_rents) else 0} rental records)")
    
    return EvaluateResponse(
        borough=borough,
        listed_price_pcm=request.price_pcm,
        expected_median_pcm=expected_median,
        expected_range_pcm=expected_range,
        deviation_pct=deviation_pct,
        classification=classification,
        confidence=confidence,
        explanations=explanations
    )

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"ok": True}

