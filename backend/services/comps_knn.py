"""
KNN-based comparables estimator using distance and similarity weighting.
"""
import math
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in meters using Haversine formula."""
    R = 6371000  # Earth radius in meters
    
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

def compute_similarity_weight(target: Dict, row: Dict) -> float:
    """
    Compute combined similarity weight for a comparable row.
    Returns weight between 0 and 1.
    """
    # Distance weight: exp(-(d/600)^2)
    distance_m = haversine_m(
        target["lat"], target["lon"],
        row["lat"], row["lon"]
    )
    distance_weight = math.exp(-((distance_m / 600.0) ** 2))
    
    # Bedroom weight
    target_bedrooms = target.get("bedrooms", 0)
    row_bedrooms = row.get("bedrooms", 0)
    bedroom_diff = abs(target_bedrooms - row_bedrooms)
    if bedroom_diff == 0:
        bedroom_weight = 1.0
    elif bedroom_diff == 1:
        bedroom_weight = 0.3
    else:
        bedroom_weight = 0.05
    
    # Property type weight
    target_type = str(target.get("property_type", "")).lower().strip()
    row_type = str(row.get("property_type", "")).lower().strip()
    type_weight = 1.0 if target_type == row_type else 0.2
    
    # Bathroom weight (only if both present)
    target_bathrooms = target.get("bathrooms")
    row_bathrooms = row.get("bathrooms")
    if target_bathrooms is not None and row_bathrooms is not None:
        bath_diff = abs(target_bathrooms - row_bathrooms)
        bath_weight = 1.0 if bath_diff <= 1 else 0.6
    else:
        bath_weight = 1.0  # Neutral if missing
    
    # Area weight (ONLY if both have floorplan_used and floor_area_sqm)
    area_weight = 1.0
    target_floorplan_used = target.get("floorplan_used", 0)
    row_floorplan_used = row.get("floorplan_used", 0)
    target_area = target.get("floor_area_sqm")
    row_area = row.get("floor_area_sqm")
    
    if (target_floorplan_used and row_floorplan_used and 
        target_area is not None and row_area is not None and 
        target_area > 0 and row_area > 0):
        ratio = abs(target_area - row_area) / max(target_area, row_area)
        area_weight = 1.0 if ratio <= 0.15 else 0.4
    # else: area_weight stays 1.0 (neutral)
    
    # Final weight = product of all weights
    final_weight = distance_weight * bedroom_weight * type_weight * bath_weight * area_weight
    
    return final_weight

def estimate_from_comps(
    target: Dict,
    comps_rows: List[Dict],
    radius_m: float = 2000.0,
    min_comps: int = 10,
    max_comps: int = 40,
    weight_threshold: float = 0.01
) -> Dict:
    """
    Estimate price using KNN-weighted comparables.
    
    Args:
        target: Target property dict with lat, lon, bedrooms, property_type, etc.
        comps_rows: List of comparable rows from DB
        radius_m: Maximum radius to consider
        min_comps: Minimum comps needed
        max_comps: Maximum comps to use
        weight_threshold: Minimum weight to include
    
    Returns:
        Dict with:
            comps_used: bool
            sample_size: int
            radius_m: float
            strong_similarity: bool
            similarity_ratio: float
            comps_estimate_pcm: float
            comps_range_pcm: [float, float]
    """
    if not comps_rows:
        return {
            "comps_used": False,
            "sample_size": 0,
            "radius_m": 0.0,
            "strong_similarity": False,
            "similarity_ratio": 0.0,
            "comps_estimate_pcm": 0.0,
            "comps_range_pcm": [0.0, 0.0]
        }
    
    # Score and filter comps
    scored_comps = []
    for row in comps_rows:
        distance_m = haversine_m(
            target["lat"], target["lon"],
            row["lat"], row["lon"]
        )
        
        if distance_m > radius_m:
            continue
        
        weight = compute_similarity_weight(target, row)
        
        if weight >= weight_threshold:
            scored_comps.append({
                "row": row,
                "weight": weight,
                "distance_m": distance_m,
                "price_pcm": row.get("price_pcm", 0)
            })
    
    # Sort by weight (descending) and take top K
    scored_comps.sort(key=lambda x: x["weight"], reverse=True)
    top_comps = scored_comps[:max_comps]
    
    if len(top_comps) < min_comps:
        return {
            "comps_used": False,
            "sample_size": len(top_comps),
            "radius_m": radius_m,
            "strong_similarity": False,
            "similarity_ratio": 0.0,
            "comps_estimate_pcm": 0.0,
            "comps_range_pcm": [0.0, 0.0]
        }
    
    # Compute strong_similarity
    # A comp is "similar" if: bedrooms exact + type exact + (area within ±15% if floorplan used)
    similar_count = 0
    target_bedrooms = target.get("bedrooms", 0)
    target_type = str(target.get("property_type", "")).lower().strip()
    target_floorplan_used = target.get("floorplan_used", 0)
    target_area = target.get("floor_area_sqm")
    
    for comp in top_comps:
        row = comp["row"]
        is_similar = True
        
        # Bedrooms must match exactly
        if row.get("bedrooms", 0) != target_bedrooms:
            is_similar = False
        
        # Type must match exactly
        row_type = str(row.get("property_type", "")).lower().strip()
        if row_type != target_type:
            is_similar = False
        
        # Area check (only if floorplan used)
        if target_floorplan_used and row.get("floorplan_used", 0):
            row_area = row.get("floor_area_sqm")
            if target_area is not None and row_area is not None and target_area > 0 and row_area > 0:
                ratio = abs(target_area - row_area) / max(target_area, row_area)
                if ratio > 0.15:
                    is_similar = False
        
        if is_similar:
            similar_count += 1
    
    similarity_ratio = similar_count / len(top_comps) if top_comps else 0.0
    strong_similarity = similarity_ratio >= 0.60
    
    # Compute weighted median
    # Sort by price, accumulate weights
    top_comps_sorted = sorted(top_comps, key=lambda x: x["price_pcm"])
    total_weight = sum(c["weight"] for c in top_comps_sorted)
    
    if total_weight == 0:
        # Fallback to unweighted median
        prices = [c["price_pcm"] for c in top_comps_sorted]
        median_price = sorted(prices)[len(prices) // 2]
        lower_price = sorted(prices)[len(prices) // 4]
        upper_price = sorted(prices)[3 * len(prices) // 4]
    else:
        # Weighted median
        cum_weight = 0.0
        median_idx = -1
        for i, comp in enumerate(top_comps_sorted):
            cum_weight += comp["weight"]
            if cum_weight >= total_weight / 2.0:
                median_idx = i
                break
        
        if median_idx < 0:
            median_idx = len(top_comps_sorted) // 2
        
        median_price = top_comps_sorted[median_idx]["price_pcm"]
        
        # Weighted quartiles (simplified: use 25th and 75th percentiles)
        lower_idx = max(0, len(top_comps_sorted) // 4)
        upper_idx = min(len(top_comps_sorted) - 1, 3 * len(top_comps_sorted) // 4)
        lower_price = top_comps_sorted[lower_idx]["price_pcm"]
        upper_price = top_comps_sorted[upper_idx]["price_pcm"]
    
    return {
        "comps_used": True,
        "sample_size": len(top_comps),
        "radius_m": radius_m,
        "strong_similarity": strong_similarity,
        "similarity_ratio": similarity_ratio,
        "comps_estimate_pcm": median_price,
        "comps_range_pcm": [lower_price, upper_price]
    }

