"""
KNN-based comparables estimator using unified similarity weighting.
"""
import math
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timezone

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

def compute_unified_weight(
    target: Dict,
    row: Dict,
    current_date: Optional[datetime] = None
) -> float:
    """
    Compute unified similarity weight: w_dist * w_beds * w_type * w_baths * w_area * w_recency
    
    Args:
        target: Target property dict
        row: Comparable row dict
        current_date: Current date for recency calculation (defaults to now)
    
    Returns:
        Weight between 0 and 1
    """
    if current_date is None:
        current_date = datetime.now(timezone.utc)
    
    # w_dist: exp(-distance_m / 500) with clamp for >2000m
    distance_m = haversine_m(
        target["lat"], target["lon"],
        row["lat"], row["lon"]
    )
    if distance_m > 2000:
        w_dist = 0.0  # Clamp: no weight for >2000m
    else:
        w_dist = math.exp(-distance_m / 500.0)
    
    # w_beds: 1.0 exact, 0.6 if +/-1, 0.25 if +/-2 (only allowed for 4+ beds), else 0
    target_bedrooms = target.get("bedrooms", 0)
    row_bedrooms = row.get("bedrooms", 0)
    bedroom_diff = abs(target_bedrooms - row_bedrooms)
    
    if bedroom_diff == 0:
        w_beds = 1.0
    elif bedroom_diff == 1:
        w_beds = 0.6
    elif bedroom_diff == 2 and target_bedrooms >= 4:
        w_beds = 0.25  # Only allowed for 4+ beds
    else:
        w_beds = 0.0
    
    # w_type: 1.0 if same normalized type else 0
    target_type = str(target.get("property_type", "")).lower().strip()
    row_type = str(row.get("property_type", "")).lower().strip()
    w_type = 1.0 if target_type == row_type else 0.0
    
    # w_baths: if bathrooms missing => 0.8 neutral, else 1.0 if within 0.5, 0.7 if within 1.0, else 0.3
    target_bathrooms = target.get("bathrooms")
    row_bathrooms = row.get("bathrooms")
    
    if target_bathrooms is None or row_bathrooms is None:
        w_baths = 0.8  # Neutral if missing
    else:
        bath_diff = abs(float(target_bathrooms) - float(row_bathrooms))
        if bath_diff <= 0.5:
            w_baths = 1.0
        elif bath_diff <= 1.0:
            w_baths = 0.7
        else:
            w_baths = 0.3
    
    # w_area: ONLY if floorplan_used + area_used_sqm not null
    w_area = 1.0  # Default: don't penalize if area not used
    target_floorplan_used = target.get("floorplan_used", 0)
    row_floorplan_used = row.get("floorplan_used", 0)
    target_area = target.get("floor_area_sqm")
    row_area = row.get("floor_area_sqm")
    
    if (target_floorplan_used and row_floorplan_used and 
        target_area is not None and row_area is not None and 
        target_area > 0 and row_area > 0):
        # Use ratio score
        ratio = abs(target_area - row_area) / max(target_area, row_area)
        if ratio <= 0.10:
            w_area = 1.0
        elif ratio <= 0.15:
            w_area = 0.8
        elif ratio <= 0.25:
            w_area = 0.55
        else:
            w_area = 0.25
    # else: w_area stays 1.0 (don't penalize)
    
    # w_recency: exp(-days_since_captured / 20)
    days_since = 0.0
    captured_at_str = row.get("captured_at")  # DB format: ISO datetime string
    last_seen_str = row.get("last_seen")  # CSV format: ISO date string
    
    try:
        if captured_at_str:
            # Parse ISO datetime
            if isinstance(captured_at_str, str):
                if 'T' in captured_at_str:
                    captured_date = datetime.fromisoformat(captured_at_str.replace('Z', '+00:00'))
                else:
                    captured_date = datetime.fromisoformat(captured_at_str)
            else:
                captured_date = captured_at_str
            
            if captured_date.tzinfo is None:
                captured_date = captured_date.replace(tzinfo=timezone.utc)
            
            delta = current_date - captured_date
            days_since = delta.total_seconds() / 86400.0
        elif last_seen_str:
            # Parse ISO date
            if isinstance(last_seen_str, str):
                last_seen_date = datetime.fromisoformat(last_seen_str)
            else:
                last_seen_date = last_seen_str
            
            if last_seen_date.tzinfo is None:
                last_seen_date = last_seen_date.replace(tzinfo=timezone.utc)
            
            delta = current_date - last_seen_date
            days_since = delta.total_seconds() / 86400.0
    except Exception:
        days_since = 30.0  # Default to 30 days if parsing fails
    
    w_recency = math.exp(-days_since / 20.0)
    
    # Final unified weight
    final_weight = w_dist * w_beds * w_type * w_baths * w_area * w_recency
    
    return final_weight

def compute_effective_sample_size(weights: List[float]) -> float:
    """
    Compute effective sample size: n_eff = (sum(w)^2) / sum(w^2)
    """
    if not weights:
        return 0.0
    
    sum_w = sum(weights)
    sum_w_sq = sum(w * w for w in weights)
    
    if sum_w_sq == 0:
        return 0.0
    
    n_eff = (sum_w * sum_w) / sum_w_sq
    return n_eff

def weighted_quantile(values: List[float], weights: List[float], quantile: float) -> float:
    """
    Compute weighted quantile.
    Values and weights must be sorted by value.
    """
    if not values or not weights:
        return 0.0
    
    total_weight = sum(weights)
    if total_weight == 0:
        return sorted(values)[int(len(values) * quantile)]
    
    cum_weight = 0.0
    target_weight = total_weight * quantile
    
    for i, (val, w) in enumerate(zip(values, weights)):
        cum_weight += w
        if cum_weight >= target_weight:
            return val
    
    # Fallback: return last value
    return values[-1]

def estimate_from_comps(
    target: Dict,
    comps_rows: List[Dict],
    radius_m: float = 2000.0,
    min_comps: int = 10,
    max_comps: int = 40,
    weight_threshold: float = 0.01
) -> Dict:
    """
    Estimate price using unified weighted KNN comparables.
    
    Args:
        target: Target property dict with lat, lon, bedrooms, property_type, etc.
        comps_rows: List of comparable rows from DB
        radius_m: Maximum radius to consider
        min_comps: Minimum effective sample size needed (n_eff >= min_comps)
        max_comps: Maximum comps to use (K=40 DB, K=30 CSV)
        weight_threshold: Minimum weight to include
    
    Returns:
        Dict with:
            comps_used: bool
            sample_size: int (raw count)
            comps_neff: float (effective sample size)
            radius_m: float
            strong_similarity: bool
            similarity_ratio: float (mean bed score)
            comps_estimate_pcm: float (weighted median)
            comps_range_pcm: [float, float] (weighted quantiles)
            comps_top10_weight_share: float
            comps_weighted_quantiles_used: str (e.g., "25-75" or "30-70")
    """
    if not comps_rows:
        return {
            "comps_used": False,
            "sample_size": 0,
            "comps_neff": 0.0,
            "radius_m": 0.0,
            "strong_similarity": False,
            "similarity_ratio": 0.0,
            "comps_estimate_pcm": 0.0,
            "comps_range_pcm": [0.0, 0.0],
            "comps_top10_weight_share": 0.0,
            "comps_weighted_quantiles_used": "none"
        }
    
    # Filter by property_type match (normalized)
    target_type = str(target.get("property_type", "")).lower().strip()
    filtered_comps = []
    
    for row in comps_rows:
        row_type = str(row.get("property_type", "")).lower().strip()
        if row_type != target_type:
            continue
        
        distance_m = haversine_m(
            target["lat"], target["lon"],
            row["lat"], row["lon"]
        )
        
        if distance_m > radius_m:
            continue
        
        # Compute unified weight
        weight = compute_unified_weight(target, row)
        
        if weight >= weight_threshold:
            filtered_comps.append({
                "row": row,
                "weight": weight,
                "distance_m": distance_m,
                "price_pcm": float(row.get("price_pcm", 0))
            })
    
    # Sort by weight (descending) and take top K
    filtered_comps.sort(key=lambda x: x["weight"], reverse=True)
    top_comps = filtered_comps[:max_comps]
    
    if not top_comps:
        return {
            "comps_used": False,
            "sample_size": 0,
            "comps_neff": 0.0,
            "radius_m": radius_m,
            "strong_similarity": False,
            "similarity_ratio": 0.0,
            "comps_estimate_pcm": 0.0,
            "comps_range_pcm": [0.0, 0.0],
            "comps_top10_weight_share": 0.0,
            "comps_weighted_quantiles_used": "none"
        }
    
    # Compute effective sample size (n_eff)
    weights = [c["weight"] for c in top_comps]
    n_eff = compute_effective_sample_size(weights)
    
    # Require min_comps by effective sample size (2)
    if n_eff < min_comps:
        return {
            "comps_used": False,
            "sample_size": len(top_comps),
            "comps_neff": n_eff,
            "radius_m": radius_m,
            "strong_similarity": False,
            "similarity_ratio": 0.0,
            "comps_estimate_pcm": 0.0,
            "comps_range_pcm": [0.0, 0.0],
            "comps_top10_weight_share": 0.0,
            "comps_weighted_quantiles_used": "none"
        }
    
    # Compute top10_weight_share (5)
    top10_weights = weights[:10] if len(weights) >= 10 else weights
    total_weight = sum(weights)
    top10_weight_share = sum(top10_weights) / total_weight if total_weight > 0 else 0.0
    
    # Compute mean bed score for strong_similarity (3)
    target_bedrooms = target.get("bedrooms", 0)
    bed_scores = []
    for comp in top_comps:
        row_bedrooms = comp["row"].get("bedrooms", 0)
        bedroom_diff = abs(target_bedrooms - row_bedrooms)
        if bedroom_diff == 0:
            bed_scores.append(1.0)
        elif bedroom_diff == 1:
            bed_scores.append(0.6)
        elif bedroom_diff == 2 and target_bedrooms >= 4:
            bed_scores.append(0.25)
        else:
            bed_scores.append(0.0)
    
    mean_bed_score = sum(bed_scores) / len(bed_scores) if bed_scores else 0.0
    
    # strong_similarity = True only if n_eff>=18 AND radius<=800 AND mean bed score >= 0.85 (3)
    strong_similarity = (n_eff >= 18.0 and radius_m <= 800.0 and mean_bed_score >= 0.85)
    
    # Compute weighted median and quantiles (3)
    # Sort by price for quantile calculation
    top_comps_sorted = sorted(top_comps, key=lambda x: x["price_pcm"])
    prices = [c["price_pcm"] for c in top_comps_sorted]
    sorted_weights = [c["weight"] for c in top_comps_sorted]
    
    total_weight = sum(sorted_weights)
    
    if total_weight == 0:
        # Fallback to unweighted
        median_price = sorted(prices)[len(prices) // 2]
        lower_price = sorted(prices)[len(prices) // 4]
        upper_price = sorted(prices)[3 * len(prices) // 4]
        quantiles_used = "25-75"
    else:
        # Weighted median
        median_price = weighted_quantile(prices, sorted_weights, 0.5)
        
        # Range: default 25-75%, tighter 30-70% if strong evidence (3)
        # Strong evidence: n_eff>=18 and radius<=800 and top10_weight_share <= 0.65
        if n_eff >= 18.0 and radius_m <= 800.0 and top10_weight_share <= 0.65:
            lower_price = weighted_quantile(prices, sorted_weights, 0.30)
            upper_price = weighted_quantile(prices, sorted_weights, 0.70)
            quantiles_used = "30-70"
        else:
            lower_price = weighted_quantile(prices, sorted_weights, 0.25)
            upper_price = weighted_quantile(prices, sorted_weights, 0.75)
            quantiles_used = "25-75"
    
    return {
        "comps_used": True,
        "sample_size": len(top_comps),
        "comps_neff": n_eff,
        "radius_m": radius_m,
        "strong_similarity": strong_similarity,
        "similarity_ratio": mean_bed_score,  # Use mean bed score as similarity_ratio
        "comps_estimate_pcm": median_price,
        "comps_range_pcm": [lower_price, upper_price],
        "comps_top10_weight_share": top10_weight_share,
        "comps_weighted_quantiles_used": quantiles_used
    }
