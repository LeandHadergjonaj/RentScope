'use client'

import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ArrowRight, Search, Sparkles, ShieldCheck, ChevronDown, MapPin, Train, BarChart3 } from 'lucide-react'
import DebugPanel from '../components/DebugPanel'
import WhyThisResult from '../components/WhyThisResult'

// Helper function to round to nearest £25 or £50
const roundPrice = (price: number): number => {
  if (price < 1000) {
    return Math.round(price / 25) * 25
  } else {
    return Math.round(price / 50) * 50
  }
}

// Helper function to format price with rounding
const formatPrice = (price: number): string => {
  const rounded = roundPrice(price)
  return `£${rounded.toLocaleString('en-GB')}/month`
}

// Safe formatter helpers for optional numeric fields
const fmt = (n?: number | null, d = 0): string => {
  return typeof n === "number" && Number.isFinite(n) ? n.toFixed(d) : "—"
}

const fmtInt = (n?: number | null): string => {
  return typeof n === "number" && Number.isFinite(n) ? Math.round(n).toString() : "—"
}

interface PostcodeCandidate {
  value: string
  source: 'jsonld' | 'script' | 'regex' | 'address_text' | 'latlon_inferred' | 'unknown'
  valid: boolean
  distance_m?: number | null
}

interface ParseListingResponse {
  price_pcm?: number | null
  bedrooms?: number | null
  property_type?: string | null
  postcode?: string | null
  postcode_valid: boolean
  postcode_source: 'jsonld' | 'script' | 'regex' | 'unknown'
  postcode_candidates: PostcodeCandidate[]
  chosen_postcode_source: 'jsonld' | 'script' | 'regex' | 'unknown'
  bathrooms?: number | null
  floor_area_sqm?: number | null
  furnished?: string | null  // "true" | "false" | "unknown" | null
  parsing_confidence: 'high' | 'medium' | 'low'
  extracted_fields: string[]
  warnings: string[]
  // Location extraction (B)
  lat?: number | null
  lon?: number | null
  location_source?: 'jsonld' | 'script' | 'html' | 'listing_latlon' | 'nominatim' | 'inferred' | 'none'
  inferred_postcode?: string | null
  inferred_postcode_distance_m?: number | null
  address_text?: string | null
  location_precision_m?: number | null
  // Additional structured features (V23)
  floor_level?: number | null
  epc_rating?: string | null
  has_lift?: boolean | null
  has_parking?: boolean | null
  has_balcony?: boolean | null
  has_terrace?: boolean | null
  has_concierge?: boolean | null
  parsed_feature_warnings?: string[]
  // Portal assets (B)
  image_urls?: string[]
  floorplan_url?: string | null
  asset_warnings?: string[]
  asset_extraction_confidence?: 'high' | 'medium' | 'low'
  // Debug output (only when debug=true)
  debug_raw?: {
    url?: string
    candidates?: Record<string, any>
    chosen?: Record<string, any>
    snippets?: Record<string, string>
  } | null
}

interface EvaluateRequest {
  url: string
  price_pcm: number
  bedrooms: number
  property_type: string
  postcode: string
  bathrooms?: number | null
  floor_area_sqm?: number | null
  furnished?: boolean | null
  quality?: string
  save_as_comparable?: boolean
}

interface DebugInfo {
  estimate_quality_score: number
  quality_label: 'poor' | 'ok' | 'good' | 'great'
  quality_reasons: string[]
  model_components: {
    ons_used: boolean
    comps_used: boolean
    ml_used: boolean
    comps_sample_size: number
    comps_radius_m: number
    strong_similarity: boolean
    ml_mae?: number | null
    ml_expected_median_pcm?: number | null
    portal_assets_used?: boolean
    photos_used?: number
    floorplan_used?: boolean
    area_used?: boolean
    area_source?: string | null
    similarity_uses_area?: boolean
  }
  // Location resolution debug (F)
  location_source?: string | null
  location_precision_m?: number | null
  geocoding_used?: boolean
}

interface EvaluateResponse {
  borough: string
  listed_price_pcm: number
  expected_median_pcm: number
  expected_range_pcm: [number, number]
  most_likely_range_pcm?: [number, number]  // Tight range when evidence supports
  most_likely_range_basis?: string  // Explanation of range basis
  deviation_pct: number
  classification: 'undervalued' | 'fair' | 'overvalued'
  confidence: 'low' | 'medium' | 'high'
  explanations: string[]
  nearest_station_distance_m: number
  transport_adjustment_pct: number
  used_borough_fallback: boolean
  extra_adjustment_pct: number
  adjustments_breakdown: {
    transport: number
    furnished: number
    bathrooms: number
    size: number
    quality: number
  }
  comps_used: boolean
  comps_sample_size: number
  comps_radius_m: number
  comps_expected_median_pcm: number
  comps_expected_range_pcm: [number, number]
  deviation_pct_damped: number
  confidence_adjusted: boolean
  adaptive_radius_used: boolean
  strong_similarity: boolean
  similarity_dampening_factor: number
  ml_expected_median_pcm?: number | null
  photo_adjustment_pct: number
  floorplan_used: boolean
  floorplan_area_sqm_used?: number | null
  area_used: boolean
  area_source: 'floorplan' | 'none'
  area_used_sqm?: number | null
  debug: DebugInfo
}

// Note: ForTestingPanel and WhyThisResult are now imported from components

export default function Home() {
  const [formData, setFormData] = useState<EvaluateRequest>({
    url: '',
    price_pcm: 0,
    bedrooms: 1,
    property_type: 'flat',
    postcode: '',
    bathrooms: null,
    floor_area_sqm: null,
    furnished: null,
    quality: 'average',
    lat: null,
    lon: null,
    save_as_comparable: true,  // Default true
  })
  const [result, setResult] = useState<EvaluateResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [listingUrl, setListingUrl] = useState('')
  const [parsing, setParsing] = useState(false)
  const [parseWarnings, setParseWarnings] = useState<string[]>([])
  const [autoFilledFields, setAutoFilledFields] = useState<Set<string>>(new Set())
  const [parseConfidenceLow, setParseConfidenceLow] = useState(false)
  const [parsingConfidence, setParsingConfidence] = useState<'high' | 'medium' | 'low' | null>(null)
  const [fullParseResult, setFullParseResult] = useState<ParseListingResponse | null>(null) // Store full parse result for debug
  const [analyzingAssets, setAnalyzingAssets] = useState(false)
  const [assetAnalysis, setAssetAnalysis] = useState<{
    condition: { label: string; score: number; confidence: string; signals: string[] }
    floorplan: { present: boolean; confidence: string; extracted: { estimated_area_sqm?: number | null }; warnings: string[] }
    assets_used: { photos_used: number; floorplan_used: boolean }
    warnings: string[]
  } | null>(null)
  // Store floorplan data separately for /evaluate request (1, 2)
  const [floorplanAreaSqm, setFloorplanAreaSqm] = useState<number | null>(null)
  const [floorplanConfidence, setFloorplanConfidence] = useState<string | null>(null)
  const [developerMode, setDeveloperMode] = useState(false) // A.6 - Developer mode toggle
  const [debugParse, setDebugParse] = useState(false) // Debug parse toggle
  const priceInputRef = useRef<HTMLInputElement>(null)

  // Analyze listing assets from URL
  const handleAnalyzeAssets = async () => {
    if (!listingUrl.trim()) {
      setError('Please enter a listing URL first')
      return
    }

    setAnalyzingAssets(true)
    setError(null)

    try {
      const response = await fetch('/api/analyze-listing-assets', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: listingUrl }),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: response.statusText }))
        throw new Error(errorData.detail || `Error: ${response.statusText}`)
      }

      const data = await response.json()
      setAssetAnalysis(data)
      
      // Store floorplan data for /evaluate request (1, 2)
      if (data.floorplan.extracted.estimated_area_sqm) {
        setFloorplanAreaSqm(data.floorplan.extracted.estimated_area_sqm)
        setFloorplanConfidence(data.floorplan.confidence)
      } else {
        setFloorplanAreaSqm(null)
        setFloorplanConfidence(null)
      }
      
      // Auto-fill floor area if confidence is "high" and user hasn't manually edited it (1)
      if (data.floorplan.extracted.estimated_area_sqm && 
          data.floorplan.confidence === 'high' &&
          !autoFilledFields.has('floor_area_sqm') &&
          (!formData.floor_area_sqm || formData.floor_area_sqm === 0)) {
        setFormData(prev => ({ ...prev, floor_area_sqm: data.floorplan.extracted.estimated_area_sqm }))
        setAutoFilledFields(prev => new Set([...prev, 'floor_area_sqm']))
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to analyze listing assets')
    } finally {
      setAnalyzingAssets(false)
    }
  }

  const handleParseListing = async () => {
    if (!listingUrl.trim()) {
      setError('Please enter a listing URL')
      return
    }

    setParsing(true)
    setError(null)
    setParseWarnings([])
    setParseConfidenceLow(false)
    setParsingConfidence(null)

    try {
      const url = debugParse 
        ? `/api/parse-listing?debug=true`
        : '/api/parse-listing'
      
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url: listingUrl }),
      })

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`)
      }

      const data: ParseListingResponse = await response.json()
      
      // Store full parse result for debug
      setFullParseResult(data)
      
      // Update parsing confidence
      setParsingConfidence(data.parsing_confidence)
      
      // Auto-fill fields
      const filledFields = new Set<string>()
      
      if (data.price_pcm) {
        setFormData(prev => ({ ...prev, price_pcm: data.price_pcm, url: listingUrl }))
        filledFields.add('price_pcm')
      } else {
        // Auto-focus price input if missing
        setTimeout(() => {
          priceInputRef.current?.focus()
        }, 100)
      }
      
      if (data.bedrooms) {
        setFormData(prev => ({ ...prev, bedrooms: data.bedrooms }))
        filledFields.add('bedrooms')
      }
      if (data.property_type) {
        setFormData(prev => ({ ...prev, property_type: data.property_type }))
        filledFields.add('property_type')
      }
      // Only auto-fill postcode if it's valid
      if (data.postcode && data.postcode_valid) {
        setFormData(prev => ({ ...prev, postcode: data.postcode }))
        filledFields.add('postcode')
      } else if (data.postcode && !data.postcode_valid) {
        // Invalid postcode - don't auto-fill, show warning, auto-focus
        if (!data.warnings) data.warnings = []
        data.warnings.push("Extracted postcode is not valid — please enter manually")
        setTimeout(() => {
          const postcodeInput = document.getElementById('postcode')
          if (postcodeInput) {
            (postcodeInput as HTMLInputElement).focus()
          }
        }, 100)
      }
      
      // Auto-fill optional fields
      if (data.bathrooms !== null && data.bathrooms !== undefined) {
        setFormData(prev => ({ ...prev, bathrooms: data.bathrooms }))
        filledFields.add('bathrooms')
      }
      
      if (data.floor_area_sqm !== null && data.floor_area_sqm !== undefined) {
        setFormData(prev => ({ ...prev, floor_area_sqm: data.floor_area_sqm }))
        filledFields.add('floor_area_sqm')
      }
      
      if (data.furnished !== null && data.furnished !== undefined) {
        // Convert string to boolean: "true" -> true, "false" -> false, "unknown" -> null
        let furnishedBool: boolean | null = null
        if (data.furnished === 'true') {
          furnishedBool = true
        } else if (data.furnished === 'false') {
          furnishedBool = false
        }
        setFormData(prev => ({ ...prev, furnished: furnishedBool }))
        filledFields.add('furnished')
      }
      
      setAutoFilledFields(filledFields)
      setParseWarnings(data.warnings || [])
      
      // Check if confidence is low (many warnings or missing critical fields)
      const criticalFieldsMissing = !data.price_pcm || !data.postcode
      if (data.parsing_confidence === 'low' || (data.warnings && (data.warnings.length >= 2 || criticalFieldsMissing))) {
        setParseConfidenceLow(true)
      } else {
        setParseConfidenceLow(false)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to parse listing')
    } finally {
      setParsing(false)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (parseConfidenceLow) {
      const confirmed = window.confirm(
        'Parsing confidence is low. Some fields may be incorrect. Do you want to proceed with evaluation?'
      )
      if (!confirmed) {
        return
      }
    }
    
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch('/api/evaluate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...formData,
          quality: formData.quality || 'average',
          lat: formData.lat || null,
          lon: formData.lon || null,
          save_as_comparable: formData.save_as_comparable ?? true,  // Default true
          // Include asset analysis results if available (2)
          photo_condition_label: assetAnalysis?.condition.label || null,
          photo_condition_score: assetAnalysis?.condition.score || null,
          photo_condition_confidence: assetAnalysis?.condition.confidence || null,
          floorplan_area_sqm: floorplanAreaSqm || null,
          floorplan_confidence: floorplanConfidence || null,
        }),
      })

      if (!response.ok) {
        // Try to extract error message from response
        let errorMessage = `Error: ${response.statusText}`
        try {
          const errorData = await response.json()
          if (errorData.detail) {
            errorMessage = errorData.detail
          }
        } catch {
          // If JSON parsing fails, use status text
        }
        throw new Error(errorMessage)
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An error occurred'
      setError(errorMessage)
      setResult(null)  // Clear any previous results
    } finally {
      setLoading(false)
    }
  }

  const getClassificationColor = (classification: string) => {
    switch (classification) {
      case 'undervalued':
        return 'text-green-600 bg-green-50 border-green-200'
      case 'overvalued':
        return 'text-red-600 bg-red-50 border-red-200'
      default:
        return 'text-blue-600 bg-blue-50 border-blue-200'
    }
  }

  const getConfidenceColor = (confidence: string) => {
    switch (confidence) {
      case 'high':
        return 'text-green-700 bg-green-100'
      case 'medium':
        return 'text-gray-700 bg-gray-100'
      default:
        return 'text-orange-700 bg-orange-100'
    }
  }

  const getConfidenceBanner = (confidence: string) => {
    switch (confidence) {
      case 'low':
        return {
          bg: 'bg-orange-50 border-orange-200',
          text: 'text-orange-800',
          message: 'Low confidence estimate — limited comparable data or atypical listing.'
        }
      case 'medium':
        return {
          bg: 'bg-gray-50 border-gray-200',
          text: 'text-gray-700',
          message: 'Medium confidence estimate — based on moderate rental data.'
        }
      case 'high':
        return {
          bg: 'bg-green-50 border-green-200',
          text: 'text-green-800',
          message: 'High confidence estimate — based on extensive rental data.'
        }
      default:
        return {
          bg: 'bg-gray-50 border-gray-200',
          text: 'text-gray-700',
          message: ''
        }
    }
  }

  const getDeviationLanguage = (deviationPct: number, confidence: string) => {
    const absDeviation = Math.abs(deviationPct * 100)
    const isBelow = deviationPct < 0

    if (confidence === 'high') {
      if (absDeviation >= 10) {
        return isBelow ? 'Significantly below market' : 'Significantly above market'
      } else {
        return 'Close to market rate'
      }
    } else if (confidence === 'medium') {
      if (absDeviation >= 10) {
        return isBelow ? 'Likely below market' : 'Likely above market'
      } else {
        return 'Appears close to market rate'
      }
    } else {
      if (absDeviation >= 10) {
        return isBelow ? 'Appears below typical market levels (low confidence)' : 'Appears above typical market levels (low confidence)'
      } else {
        return 'Appears close to typical market levels (low confidence)'
      }
    }
  }

  // Construct parseResult from full parse result or existing state
  const parseResult = fullParseResult || (parsingConfidence ? {
    parsing_confidence: parsingConfidence,
    extracted_fields: Array.from(autoFilledFields),
    warnings: parseWarnings
  } : null)

  // Get simplified explanations (max 3, non-technical) (A.2, C.11)
  const getSimplifiedExplanations = (explanations: string[]): string[] => {
    const nonTechnical = explanations.filter(e => 
      !e.includes('ONS') && 
      !e.includes('KDTree') && 
      !e.includes('MAE') &&
      !e.includes('blended estimate') &&
      !e.includes('time-on-market weighted') &&
      !e.includes('Final expected median')
    )
    return nonTechnical.slice(0, 3)
  }

  // Get confidence explanation (A.2)
  const getConfidenceExplanation = (result: EvaluateResponse): string => {
    if (result.comps_used && result.comps_sample_size > 0) {
      return `We found ${result.comps_sample_size} nearby rentals`
    }
    return `Based on borough market data`
  }

  return (
    <div className="min-h-screen bg-stone-50 text-stone-900">
      <div className="mx-auto max-w-6xl px-6 py-10">
        {/* Header */}
        <div className="mb-8 flex flex-col gap-3">
          <div className="inline-flex w-fit items-center gap-2 rounded-full border border-stone-200 bg-white px-3 py-1 text-xs text-stone-600 shadow-sm">
            <ShieldCheck className="h-4 w-4 text-emerald-700" />
            Market sanity-check • London rentals
          </div>

          <div className="flex items-end justify-between gap-4">
            <div>
              <h1 className="text-4xl font-semibold tracking-tight text-stone-900">
                Rent<span className="text-stone-600">Scope</span>
              </h1>
              <p className="mt-2 max-w-2xl text-sm text-stone-600">
                Get a fair rent estimate for any London property. Paste a listing link or enter details manually.
              </p>
            </div>

            {/* Developer mode toggle (A.6) */}
            <div className="flex items-center gap-2">
              <label className="flex items-center gap-2 text-xs text-stone-600 cursor-pointer">
                <input
                  type="checkbox"
                  checked={developerMode}
                  onChange={(e) => setDeveloperMode(e.target.checked)}
                  className="rounded border-stone-300 text-emerald-700 focus:ring-emerald-500"
                />
                <span>Developer mode</span>
              </label>
            </div>
          </div>
        </div>

        {/* Error banner */}
        {error && (
          <div className="mb-6 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">
            <p className="font-medium">Error</p>
            <p>{error}</p>
          </div>
        )}

        {/* Main grid */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          {/* LEFT: Inputs */}
          <GlassCard>
            <SectionTitle icon={<Search className="h-4 w-4" />} title="Listing" subtitle="Extract details from Rightmove/Zoopla (best effort). Review before evaluating." />

              <div className="mt-4 space-y-3">
              <label className="text-xs text-stone-600 font-medium">Listing link (optional)</label>
              <input
                type="url"
                value={listingUrl}
                onChange={(e) => setListingUrl(e.target.value)}
                placeholder="https://www.rightmove.co.uk/properties/..."
                className={inputClass}
              />

              <div className="flex flex-col gap-2 sm:flex-row">
                <button
                  type="button"
                  onClick={handleParseListing}
                  disabled={parsing || !listingUrl.trim()}
                  className="inline-flex items-center justify-center gap-2 rounded-lg border border-stone-200 bg-stone-100 px-4 py-3 text-sm font-medium text-stone-800 hover:bg-stone-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  <Search className="h-4 w-4" />
                  {parsing ? "Extracting..." : "Extract details"}
                </button>

                <button
                  type="button"
                  onClick={handleAnalyzeAssets}
                  disabled={analyzingAssets || !listingUrl.trim()}
                  className="inline-flex items-center justify-center gap-2 rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm font-medium text-emerald-800 hover:bg-emerald-100 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  <Sparkles className="h-4 w-4" />
                  {analyzingAssets ? "Analyzing..." : "Analyze photos + floorplan"}
                </button>
              </div>

              {/* Parse / asset status */}
              <div className="mt-2">
                {parseResult?.parsing_confidence && (
                  <div className="flex flex-wrap items-center gap-2">
                    <StatBadge
                      label={`Parsing: ${parseResult.parsing_confidence}`}
                      tone={parseResult.parsing_confidence === "high" ? "good" : parseResult.parsing_confidence === "medium" ? "warn" : "bad"}
                    />
                    {parseResult?.extracted_fields?.length ? (
                      <StatBadge label={`Fields: ${parseResult.extracted_fields.join(", ")}`} tone="neutral" />
                    ) : null}
                    {parseResult?.warnings?.length ? (
                      <StatBadge label={`⚠ ${parseResult.warnings[0]}`} tone="warn" />
                    ) : null}
                  </div>
                )}
              </div>
              
              {/* Parse Debug Panel (only when debug_raw is present) */}
              {fullParseResult?.debug_raw && (
                <Accordion title="Parse Debug">
                  <div className="space-y-3 text-xs">
                    <div>
                      <div className="font-semibold text-stone-900 mb-1">Extracted Fields</div>
                      <div className="text-stone-700 font-mono">
                        {fullParseResult.extracted_fields?.join(', ') || 'None'}
                      </div>
                    </div>
                    {fullParseResult.warnings && fullParseResult.warnings.length > 0 && (
                      <div>
                        <div className="font-semibold text-stone-900 mb-1">Warnings</div>
                        <ul className="list-disc list-inside text-stone-700 space-y-1">
                          {fullParseResult.warnings.map((w, i) => (
                            <li key={i}>{w}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                    <div>
                      <div className="font-semibold text-stone-900 mb-1">Debug Raw (JSON)</div>
                      <pre className="bg-stone-50 border border-stone-200 rounded-lg p-3 text-[10px] overflow-x-auto max-h-96 overflow-y-auto">
                        {JSON.stringify(fullParseResult.debug_raw, null, 2)}
                      </pre>
                    </div>
                  </div>
                </Accordion>
              )}

              {assetAnalysis && (
                <div className="mt-4 rounded-lg border border-stone-200 bg-white p-4 shadow-sm">
                  <div className="flex items-center justify-between">
                    <div className="text-sm font-medium text-stone-900">Listing Analysis Results</div>
                    <StatBadge
                      label={`Assets: ${assetAnalysis?.assets_used?.photos_used ?? 0} photos`}
                      tone={(assetAnalysis?.assets_used?.photos_used ?? 0) >= 2 ? "good" : "warn"}
                    />
                  </div>

                  <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
                    <div className="rounded-lg border border-stone-200 bg-stone-50 p-3">
                      <div className="text-xs text-stone-500">Condition</div>
                      <div className="mt-1 text-sm text-stone-900">
                        {assetAnalysis?.condition?.label ?? "—"}{" "}
                        <span className="text-stone-600">
                          ({assetAnalysis?.condition?.score ?? "—"}/100)
                        </span>
                      </div>
                      <div className="mt-1 text-xs text-stone-500">
                        Confidence: {assetAnalysis?.condition?.confidence ?? "—"}
                      </div>
                    </div>

                    <div className="rounded-lg border border-stone-200 bg-stone-50 p-3">
                      <div className="text-xs text-stone-500">Floorplan</div>
                      <div className="mt-1 text-sm text-stone-900">
                        Area: {assetAnalysis?.floorplan?.extracted?.estimated_area_sqm ? Number(assetAnalysis.floorplan.extracted.estimated_area_sqm ?? 0).toFixed(1) : "—"} m²
                      </div>
                      <div className="mt-1 text-xs text-stone-500">
                        Confidence: {assetAnalysis?.floorplan?.confidence ?? "—"}
                      </div>
                    </div>
                  </div>

                  {assetAnalysis?.condition?.signals?.length ? (
                    <div className="mt-3 text-xs text-stone-600">
                      <div className="text-stone-500 mb-1">Signals</div>
                      <div className="mt-1 flex flex-wrap gap-2">
                        {assetAnalysis.condition.signals.slice(0, 6).map((s: string, i: number) => (
                          <span key={i} className="rounded-full border border-stone-200 bg-stone-50 px-2 py-1 text-stone-700">
                            {s}
                          </span>
                        ))}
                      </div>
                    </div>
                  ) : null}
                </div>
              )}
            </div>

            <form onSubmit={handleSubmit}>
            <div className="mt-8 border-t border-white/10 pt-6">
              <SectionTitle icon={<BarChart3 className="h-4 w-4" />} title="Property details" subtitle="The more correct these fields are, the tighter your estimate." />

              <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
                <Field label="Price per month (£)">
                  <input 
                    type="number"
                    required
                    min="0"
                    step="0.01"
                    ref={priceInputRef}
                    className={inputClass} 
                    value={formData.price_pcm || ''} 
                    onChange={(e) => {
                      setFormData({ ...formData, price_pcm: parseFloat(e.target.value) || 0 })
                      setAutoFilledFields(prev => {
                        const next = new Set(prev)
                        next.delete('price_pcm')
                        return next
                      })
                    }} 
                    placeholder="e.g. 2500" 
                  />
                </Field>

                <Field label="Bedrooms">
                  <input 
                    type="number"
                    required
                    min="0"
                    className={inputClass} 
                    value={formData.bedrooms} 
                    onChange={(e) => {
                      setFormData({ ...formData, bedrooms: parseInt(e.target.value) || 0 })
                      setAutoFilledFields(prev => {
                        const next = new Set(prev)
                        next.delete('bedrooms')
                        return next
                      })
                    }} 
                    placeholder="e.g. 2" 
                  />
                </Field>

                <Field label="Property type">
                  <select 
                    required
                    className={inputClass} 
                    value={formData.property_type} 
                    onChange={(e) => {
                      setFormData({ ...formData, property_type: e.target.value })
                      setAutoFilledFields(prev => {
                        const next = new Set(prev)
                        next.delete('property_type')
                        return next
                      })
                    }}
                  >
                    <option value="flat">Flat</option>
                    <option value="house">House</option>
                    <option value="studio">Studio</option>
                    <option value="room">Room</option>
                  </select>
                </Field>

                <Field label="Postcode">
                  <input 
                    type="text"
                    required
                    className={inputClass} 
                    value={formData.postcode} 
                    onChange={(e) => {
                      setFormData({ ...formData, postcode: e.target.value.toUpperCase() })
                      setAutoFilledFields(prev => {
                        const next = new Set(prev)
                        next.delete('postcode')
                        return next
                      })
                    }} 
                    placeholder="e.g. E20 1NX" 
                  />
                </Field>
              </div>

              <Accordion title="Advanced (optional)">
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                  <Field label="Bathrooms">
                    <input 
                      type="number"
                      min="0"
                      className={inputClass} 
                      value={formData.bathrooms || ''} 
                      onChange={(e) => {
                        setFormData({ ...formData, bathrooms: e.target.value ? parseInt(e.target.value) : null })
                        setAutoFilledFields(prev => {
                          const next = new Set(prev)
                          next.delete('bathrooms')
                          return next
                        })
                      }} 
                      placeholder="e.g. 2" 
                    />
                  </Field>
                  <Field label="Floor area (sqm)">
                    <input 
                      type="number"
                      min="0"
                      step="0.1"
                      className={inputClass} 
                      value={formData.floor_area_sqm || ''} 
                      onChange={(e) => {
                        setFormData({ ...formData, floor_area_sqm: e.target.value ? parseFloat(e.target.value) : null })
                        setAutoFilledFields(prev => {
                          const next = new Set(prev)
                          next.delete('floor_area_sqm')
                          return next
                        })
                      }} 
                      placeholder="e.g. 75" 
                    />
                  </Field>
                  <Field label="Furnished">
                    <select 
                      className={inputClass} 
                      value={formData.furnished === null ? 'unknown' : formData.furnished ? 'yes' : 'no'} 
                      onChange={(e) => {
                        setFormData({ ...formData, furnished: e.target.value === 'unknown' ? null : e.target.value === 'yes' })
                        setAutoFilledFields(prev => {
                          const next = new Set(prev)
                          next.delete('furnished')
                          return next
                        })
                      }}
                    >
                      <option value="unknown">Unknown</option>
                      <option value="yes">Yes</option>
                      <option value="no">No</option>
                    </select>
                  </Field>
                  <Field label="Quality">
                    <select 
                      className={inputClass} 
                      value={formData.quality || 'average'} 
                      onChange={(e) => setFormData({ ...formData, quality: e.target.value })}
                    >
                      <option value="average">Average</option>
                      <option value="dated">Dated</option>
                      <option value="modern">Modern</option>
                    </select>
                  </Field>
                </div>
                
                {/* Parsed features (read-only) */}
                {(parseResult?.floor_level !== null && parseResult?.floor_level !== undefined) ||
                 parseResult?.epc_rating ||
                 parseResult?.has_lift !== null ||
                 parseResult?.has_parking !== null ||
                 parseResult?.has_balcony !== null ||
                 parseResult?.has_terrace !== null ||
                 parseResult?.has_concierge !== null ? (
                  <div className="mt-4 pt-4 border-t border-stone-200">
                    <div className="text-xs font-medium text-stone-700 mb-3">Parsed features</div>
                    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                      {parseResult?.floor_level !== null && parseResult?.floor_level !== undefined && (
                        <div className="flex items-center justify-between rounded-lg border border-stone-200 bg-stone-50 px-3 py-2">
                          <span className="text-xs text-stone-600">Floor level</span>
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-medium text-stone-900">
                              {parseResult?.floor_level === 0 ? 'Ground' : parseResult?.floor_level === -1 ? 'Lower ground' : `${parseResult?.floor_level}${parseResult?.floor_level === 1 ? 'st' : parseResult?.floor_level === 2 ? 'nd' : parseResult?.floor_level === 3 ? 'rd' : 'th'}`}
                            </span>
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-700 font-medium">Parsed</span>
                          </div>
                        </div>
                      )}
                      {parseResult?.epc_rating && (
                        <div className="flex items-center justify-between rounded-lg border border-stone-200 bg-stone-50 px-3 py-2">
                          <span className="text-xs text-stone-600">EPC rating</span>
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-medium text-stone-900">{parseResult?.epc_rating}</span>
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-700 font-medium">Parsed</span>
                          </div>
                        </div>
                      )}
                      {parseResult?.has_lift !== null && parseResult?.has_lift !== undefined && (
                        <div className="flex items-center justify-between rounded-lg border border-stone-200 bg-stone-50 px-3 py-2">
                          <span className="text-xs text-stone-600">Lift</span>
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-medium text-stone-900">{parseResult?.has_lift ? 'Yes' : 'No'}</span>
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-700 font-medium">Parsed</span>
                          </div>
                        </div>
                      )}
                      {parseResult?.has_parking !== null && parseResult?.has_parking !== undefined && (
                        <div className="flex items-center justify-between rounded-lg border border-stone-200 bg-stone-50 px-3 py-2">
                          <span className="text-xs text-stone-600">Parking</span>
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-medium text-stone-900">{parseResult?.has_parking ? 'Yes' : 'No'}</span>
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-700 font-medium">Parsed</span>
                          </div>
                        </div>
                      )}
                      {parseResult?.has_balcony !== null && parseResult?.has_balcony !== undefined && (
                        <div className="flex items-center justify-between rounded-lg border border-stone-200 bg-stone-50 px-3 py-2">
                          <span className="text-xs text-stone-600">Balcony</span>
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-medium text-stone-900">{parseResult?.has_balcony ? 'Yes' : 'No'}</span>
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-700 font-medium">Parsed</span>
                          </div>
                        </div>
                      )}
                      {parseResult?.has_terrace !== null && parseResult?.has_terrace !== undefined && (
                        <div className="flex items-center justify-between rounded-lg border border-stone-200 bg-stone-50 px-3 py-2">
                          <span className="text-xs text-stone-600">Terrace</span>
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-medium text-stone-900">{parseResult?.has_terrace ? 'Yes' : 'No'}</span>
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-700 font-medium">Parsed</span>
                          </div>
                        </div>
                      )}
                      {parseResult?.has_concierge !== null && parseResult?.has_concierge !== undefined && (
                        <div className="flex items-center justify-between rounded-lg border border-stone-200 bg-stone-50 px-3 py-2">
                          <span className="text-xs text-stone-600">Concierge</span>
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-medium text-stone-900">{parseResult?.has_concierge ? 'Yes' : 'No'}</span>
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-700 font-medium">Parsed</span>
                          </div>
                        </div>
                      )}
                    </div>
                    {parseResult?.parsing_confidence === 'low' && parseResult?.parsed_feature_warnings && parseResult.parsed_feature_warnings.length > 0 && (
                      <div className="mt-3 text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2">
                        Some details were inferred — please review carefully
                      </div>
                    )}
                  </div>
                ) : null}
              </Accordion>

              <button
                type="submit"
                disabled={loading}
                className="mt-5 inline-flex w-full items-center justify-center gap-2 rounded-lg bg-emerald-700 px-4 py-3 text-sm font-semibold text-white shadow-sm hover:bg-emerald-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {loading ? "Evaluating..." : "Evaluate property"}
                <ArrowRight className="h-4 w-4" />
              </button>

              <div className="mt-3 text-xs text-stone-500">
                RentScope provides a market sanity check — not a guaranteed valuation.
              </div>
            </div>
            </form>
          </GlassCard>

          {/* RIGHT: Results */}
          <div className="lg:sticky lg:top-6 h-fit">
            <GlassCard>
              <SectionTitle icon={<Sparkles className="h-4 w-4" />} title="Evaluation results" subtitle="Premium, explainable output." />

              {!result ? (
                <div className="mt-6 rounded-xl border border-stone-200 bg-white p-6 text-sm text-stone-600 shadow-sm">
                  Paste a link or fill the details, then hit <span className="font-semibold text-stone-900">Evaluate property</span>.
                </div>
              ) : (
                <div className="mt-6 space-y-4">
                  {/* Verdict chip (A.2) */}
                  <div className="flex items-center gap-3">
                    <StatBadge
                      label={result.classification === "undervalued" ? "Good deal" : result.classification === "overvalued" ? "Overpriced" : "Fair price"}
                      tone={result.classification === "undervalued" ? "good" : result.classification === "overvalued" ? "bad" : "neutral"}
                    />
                  </div>

                  {/* Estimated fair price (A.2) */}
                  <div className="rounded-xl border border-stone-200 bg-white p-6 shadow-sm">
                    <div className="text-xs text-stone-500 font-medium mb-1">Estimated fair rent</div>
                    <div className="mt-1 text-4xl font-bold tracking-tight text-stone-900">£{Math.round(result.expected_median_pcm).toLocaleString()}/month</div>

                    {/* Most-likely range (A.2, C.11) */}
                    {result.most_likely_range_pcm?.length === 2 ? (
                      <div className="mt-4 inline-flex items-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 px-4 py-2">
                        <span className="text-sm font-semibold text-emerald-800">
                          £{Math.round(result.most_likely_range_pcm[0]).toLocaleString()} — £{Math.round(result.most_likely_range_pcm[1]).toLocaleString()}
                        </span>
                      </div>
                    ) : (
                      <div className="mt-4 inline-flex items-center gap-2 rounded-full border border-stone-200 bg-stone-50 px-4 py-2">
                        <span className="text-sm font-semibold text-stone-700">
                          £{Math.round(result.expected_range_pcm[0]).toLocaleString()} — £{Math.round(result.expected_range_pcm[1]).toLocaleString()}
                        </span>
                      </div>
                    )}
                  </div>

                  {/* Short explanation bullets (A.2, C.11) */}
                  <div className="rounded-xl border border-stone-200 bg-white p-4 shadow-sm">
                    <div className="text-xs text-stone-500 font-medium mb-2">Why?</div>
                    <ul className="space-y-1.5 text-sm text-stone-700">
                      {getSimplifiedExplanations(result.explanations).map((explanation, idx) => (
                        <li key={idx} className="flex items-start gap-2">
                          <span className="text-emerald-700 mt-1">•</span>
                          <span>{explanation}</span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  {/* Confidence level (A.2) */}
                  <div className="rounded-xl border border-stone-200 bg-white p-4 shadow-sm">
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="text-xs text-stone-500 font-medium mb-1">Confidence level</div>
                        <div className="text-sm font-semibold text-stone-900 capitalize">{result.confidence}</div>
                      </div>
                      <StatBadge 
                        label={result.confidence.toUpperCase()} 
                        tone={result.confidence === "high" ? "good" : result.confidence === "medium" ? "warn" : "bad"} 
                      />
                    </div>
                    <p className="mt-2 text-xs text-stone-600">{getConfidenceExplanation(result)}</p>
                  </div>

                  {/* Details accordion (A.2) */}
                  <Accordion title="Details">
                    <div className="space-y-3 text-sm text-stone-700">
                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <div className="text-xs text-stone-500 mb-1">Borough</div>
                          <div className="font-medium">{result.borough}</div>
                        </div>
                        <div>
                          <div className="text-xs text-stone-500 mb-1">Listed price</div>
                          <div className="font-medium">£{Math.round(result.listed_price_pcm).toLocaleString()}/month</div>
                        </div>
                        {result.nearest_station_distance_m > 0 && (
                          <div>
                            <div className="text-xs text-stone-500 mb-1">Nearest station</div>
                            <div className="font-medium">{Math.round(result.nearest_station_distance_m)}m</div>
                          </div>
                        )}
                        {result.transport_adjustment_pct !== 0 && (
                          <div>
                            <div className="text-xs text-stone-500 mb-1">Transport adjustment</div>
                            <div className="font-medium">{result.transport_adjustment_pct > 0 ? '+' : ''}{Number((result.transport_adjustment_pct ?? 0) * 100).toFixed(1)}%</div>
                          </div>
                        )}
                        {result.comps_used && (
                          <>
                            <div>
                              <div className="text-xs text-stone-500 mb-1">Nearby rentals used</div>
                              <div className="font-medium">{result.comps_sample_size} properties</div>
                            </div>
                            <div>
                              <div className="text-xs text-stone-500 mb-1">Search radius</div>
                              <div className="font-medium">{Math.round(result.comps_radius_m)}m</div>
                            </div>
                          </>
                        )}
                      </div>
                      <div className="pt-3 border-t border-stone-200">
                        <div className="text-xs text-stone-500 mb-2">Full explanation</div>
                        <ul className="space-y-1 text-xs text-stone-600">
                          {result.explanations.map((explanation, idx) => (
                            <li key={idx} className="flex items-start gap-2">
                              <span className="text-stone-400 mt-0.5">•</span>
                              <span>{explanation}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </Accordion>

                  {/* Developer mode panel (A.6) */}
                  {developerMode && (
                    <Accordion title="Developer mode">
                      <div className="space-y-4">
                        <div className="flex items-center gap-2">
                          <label className="flex items-center gap-2 text-sm text-stone-700 cursor-pointer">
                            <input
                              type="checkbox"
                              checked={formData.save_as_comparable ?? true}
                              onChange={(e) => setFormData({ ...formData, save_as_comparable: e.target.checked })}
                              className="rounded border-stone-300 text-emerald-700 focus:ring-emerald-500"
                            />
                            <span>Save this evaluation as a comparable</span>
                          </label>
                        </div>
                        <div className="flex items-center gap-2">
                          <label className="flex items-center gap-2 text-sm text-stone-700 cursor-pointer">
                            <input
                              type="checkbox"
                              checked={debugParse}
                              onChange={(e) => setDebugParse(e.target.checked)}
                              className="rounded border-stone-300 text-emerald-700 focus:ring-emerald-500"
                            />
                            <span>Debug parse</span>
                          </label>
                        </div>
                        <DebugPanel result={result} fmt={fmt} fmtInt={fmtInt} assetAnalysis={assetAnalysis} />
                        
                        {/* Debug Section: Raw JSON + Key Fields */}
                        <DebugSection 
                          parseResult={fullParseResult}
                          evaluateResult={result}
                          fmt={fmt}
                          fmtInt={fmtInt}
                        />
                      </div>
                    </Accordion>
                  )}
                </div>
              )}
            </GlassCard>
          </div>
        </div>

        <div className="mt-10 text-center text-xs text-stone-500">
          Built for skill-building. Always verify listings manually.
        </div>
      </div>
    </div>
  )
}

// Helper Components

const inputClass = "w-full rounded-lg border border-stone-200 bg-white px-4 py-3 text-sm text-stone-900 placeholder:text-stone-400 outline-none focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 transition-colors"

function GlassCard({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={`rounded-xl border border-stone-200 bg-white p-6 shadow-sm ${className}`}>
      {children}
    </div>
  )
}

function SectionTitle({ icon, title, subtitle }: { icon: React.ReactNode; title: string; subtitle?: string }) {
  return (
    <div>
      <div className="flex items-center gap-2">
        {icon}
        <h2 className="text-lg font-semibold text-stone-900">{title}</h2>
      </div>
      {subtitle && <p className="mt-1 text-xs text-stone-600">{subtitle}</p>}
    </div>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-xs text-stone-600 font-medium mb-2">{label}</label>
      {children}
    </div>
  )
}

function MiniStat({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="flex items-center gap-2 rounded-lg border border-stone-200 bg-stone-50 p-2">
      <div className="text-stone-400">{icon}</div>
      <div className="flex-1">
        <div className="text-[10px] text-stone-500">{label}</div>
        <div className="text-xs font-medium text-stone-900">{value}</div>
      </div>
    </div>
  )
}

function StatBadge({ label, tone }: { label: string; tone: "good" | "warn" | "bad" | "neutral" }) {
  const toneStyles = {
    good: "bg-emerald-50 text-emerald-800 border-emerald-200",
    warn: "bg-amber-50 text-amber-800 border-amber-200",
    bad: "bg-rose-50 text-rose-800 border-rose-200",
    neutral: "bg-stone-100 text-stone-800 border-stone-200",
  }
  return (
    <span className={`inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold ${toneStyles[tone]}`}>
      {label}
    </span>
  )
}

// Debug Section Component
function DebugSection({ 
  parseResult, 
  evaluateResult, 
  fmt, 
  fmtInt 
}: { 
  parseResult: any
  evaluateResult: any
  fmt: (n?: number | null, d?: number) => string
  fmtInt: (n?: number | null) => string
}) {
  const copyToClipboard = (text: string, label: string) => {
    navigator.clipboard.writeText(text).then(() => {
      // Show temporary success message
      const button = document.getElementById(`copy-${label}`)
      if (button) {
        const originalText = button.textContent
        button.textContent = 'Copied!'
        setTimeout(() => {
          if (button) button.textContent = originalText
        }, 2000)
      }
    }).catch(err => {
      console.error('Failed to copy:', err)
    })
  }

  const renderJsonBlock = (data: any, label: string) => {
    if (!data) return null
    
    const jsonString = JSON.stringify(data, null, 2)
    
    return (
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold text-stone-900">Raw JSON</span>
          <button
            id={`copy-${label}`}
            onClick={() => copyToClipboard(jsonString, label)}
            className="text-xs px-2 py-1 rounded border border-stone-300 bg-white text-stone-700 hover:bg-stone-50 transition-colors"
          >
            Copy
          </button>
        </div>
        <pre className="bg-stone-50 border border-stone-200 rounded-lg p-3 text-[10px] font-mono text-stone-800 overflow-x-auto max-h-96 overflow-y-auto">
          {jsonString}
        </pre>
      </div>
    )
  }

  const renderKeyFields = () => {
    if (!evaluateResult) return null

    const modelComponents = evaluateResult.debug?.model_components || {}
    
    const fields = [
      // Comparables
      { label: 'Comps Used', value: modelComponents.comps_used ? 'Yes' : 'No' },
      { label: 'Comps Sample Size', value: modelComponents.comps_sample_size ?? '—' },
      { label: 'Comps n_eff', value: modelComponents.comps_neff != null ? fmt(modelComponents.comps_neff, 1) : '—' },
      { label: 'Comps Radius (m)', value: modelComponents.comps_radius_m != null ? fmt(modelComponents.comps_radius_m, 0) : '—' },
      { label: 'Comps Source', value: modelComponents.comps_source || '—' },
      { label: 'Comps DB Count Recent', value: modelComponents.comps_db_count_recent ?? '—' },
      { label: 'Comps Top10 Weight Share', value: modelComponents.comps_top10_weight_share != null ? fmt(modelComponents.comps_top10_weight_share, 3) : '—' },
      { label: 'Comps Weighted Quantiles', value: modelComponents.comps_weighted_quantiles_used || '—' },
      
      // Similarity
      { label: 'Strong Similarity', value: modelComponents.strong_similarity ? 'Yes' : 'No' },
      { label: 'Similarity Ratio', value: modelComponents.similarity_ratio != null ? fmt(modelComponents.similarity_ratio, 3) : '—' },
      { label: 'Similarity Rules Applied', value: Array.isArray(modelComponents.similarity_rules_applied) ? modelComponents.similarity_rules_applied.join(', ') || 'None' : '—' },
      { label: 'Similarity Uses Area', value: modelComponents.similarity_uses_area ? 'Yes' : 'No' },
      
      // Models
      { label: 'ONS Used', value: modelComponents.ons_used ? 'Yes' : 'No' },
      { label: 'ML Used', value: modelComponents.ml_used ? 'Yes' : 'No' },
      { label: 'ML MAE', value: modelComponents.ml_mae != null ? `£${fmt(modelComponents.ml_mae, 2)}/mo` : '—' },
      { label: 'ML Expected Median', value: modelComponents.ml_expected_median_pcm != null ? `£${fmt(modelComponents.ml_expected_median_pcm, 2)}/mo` : '—' },
      
      // Area
      { label: 'Area Used', value: modelComponents.area_used ? 'Yes' : 'No' },
      { label: 'Area Source', value: modelComponents.area_source || '—' },
      { label: 'Area Used (sqm)', value: modelComponents.area_used_sqm != null ? fmt(modelComponents.area_used_sqm, 1) : '—' },
      { label: 'Floorplan Used', value: modelComponents.floorplan_used ? 'Yes' : 'No' },
      
      // Location (from parse result or evaluate result)
      { label: 'Postcode', value: parseResult?.postcode || '—' },
      { label: 'Postcode Valid', value: parseResult?.postcode_valid != null ? (parseResult.postcode_valid ? 'Yes' : 'No') : '—' },
      { label: 'Postcode Source', value: parseResult?.postcode_source || '—' },
      { label: 'Location Source', value: evaluateResult?.debug?.location_source || parseResult?.location_source || '—' },
      { label: 'Address Text', value: parseResult?.address_text || '—' },
      { label: 'Lat', value: parseResult?.lat != null ? fmt(parseResult.lat, 6) : '—' },
      { label: 'Lon', value: parseResult?.lon != null ? fmt(parseResult.lon, 6) : '—' },
    ]

    return (
      <div className="space-y-2">
        <div className="text-xs font-semibold text-stone-900 mb-2">Key Debug Fields</div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
          {fields.map((field, idx) => (
            <div key={idx} className="flex items-start gap-2 text-xs">
              <span className="font-medium text-stone-600 min-w-[140px]">{field.label}:</span>
              <span className="font-mono text-stone-800 break-words">{field.value}</span>
            </div>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-4 pt-4 border-t border-stone-200">
      {/* Debug Parse */}
      <Accordion title="Debug Parse (raw JSON)">
        <div className="p-4 space-y-4">
          {parseResult ? (
            <>
              {/* Postcode Resolution Section */}
              {(parseResult.postcode || parseResult.postcode_candidates?.length > 0) && (
                <div className="space-y-3">
                  <div className="text-xs font-semibold text-stone-900">Postcode Resolution</div>
                  
                  {/* Chosen postcode */}
                  {parseResult.postcode && (
                    <div className="flex items-center gap-2 text-xs">
                      <span className="font-medium text-stone-600">Chosen:</span>
                      <span className="font-mono text-stone-800">{parseResult.postcode}</span>
                      {parseResult.postcode_source && (
                        <span className="text-stone-500">({parseResult.postcode_source})</span>
                      )}
                    </div>
                  )}
                  
                  {/* Postcode candidates table */}
                  {parseResult.postcode_candidates && parseResult.postcode_candidates.length > 0 && (
                    <div className="overflow-x-auto">
                      <table className="w-full text-xs border border-stone-200 rounded-lg">
                        <thead className="bg-stone-50">
                          <tr>
                            <th className="px-3 py-2 text-left font-semibold text-stone-700 border-b border-stone-200">Value</th>
                            <th className="px-3 py-2 text-left font-semibold text-stone-700 border-b border-stone-200">Source</th>
                            <th className="px-3 py-2 text-left font-semibold text-stone-700 border-b border-stone-200">Valid</th>
                            <th className="px-3 py-2 text-left font-semibold text-stone-700 border-b border-stone-200">Distance (m)</th>
                          </tr>
                        </thead>
                        <tbody>
                          {parseResult.postcode_candidates.map((candidate: PostcodeCandidate, idx: number) => (
                            <tr key={idx} className="border-b border-stone-100 last:border-b-0">
                              <td className="px-3 py-2 font-mono text-stone-800">{candidate.value || '—'}</td>
                              <td className="px-3 py-2 text-stone-700">{candidate.source || '—'}</td>
                              <td className="px-3 py-2">
                                <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-medium ${
                                  candidate.valid 
                                    ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' 
                                    : 'bg-rose-50 text-rose-700 border border-rose-200'
                                }`}>
                                  {candidate.valid ? 'Yes' : 'No'}
                                </span>
                              </td>
                              <td className="px-3 py-2 font-mono text-stone-800">
                                {candidate.distance_m != null ? Math.round(candidate.distance_m) : '—'}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                  
                  {/* Warnings */}
                  {parseResult.warnings && parseResult.warnings.length > 0 && (
                    <div className="rounded-lg border border-amber-200 bg-amber-50 p-3">
                      <div className="text-xs font-semibold text-amber-800 mb-1">Warnings</div>
                      <ul className="text-xs text-amber-700 space-y-1">
                        {parseResult.warnings.map((warning: string, idx: number) => (
                          <li key={idx} className="list-disc list-inside">{warning}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
              
              {/* Assets Section */}
              {(parseResult.image_urls || parseResult.floorplan_url || parseResult.asset_extraction_confidence) && (
                <div className="space-y-3 pt-4 border-t border-stone-200">
                  <div className="text-xs font-semibold text-stone-900">Assets</div>
                  
                  <div className="grid grid-cols-2 gap-3 text-xs">
                    <div>
                      <span className="font-medium text-stone-600">Image URLs:</span>
                      <span className="ml-2 font-mono text-stone-800">
                        {parseResult.image_urls?.length ?? 0}
                      </span>
                    </div>
                    <div>
                      <span className="font-medium text-stone-600">Floorplan URL:</span>
                      <span className="ml-2 font-mono text-stone-800">
                        {parseResult.floorplan_url ? 'Present' : 'None'}
                      </span>
                    </div>
                  </div>
                  
                  {/* Asset extraction confidence badge */}
                  {parseResult.asset_extraction_confidence && (
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-medium text-stone-600">Confidence:</span>
                      <span className={`inline-flex items-center rounded-full px-2 py-1 text-xs font-medium ${
                        parseResult.asset_extraction_confidence === 'high'
                          ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                          : parseResult.asset_extraction_confidence === 'medium'
                          ? 'bg-amber-50 text-amber-700 border border-amber-200'
                          : 'bg-rose-50 text-rose-700 border border-rose-200'
                      }`}>
                        {parseResult.asset_extraction_confidence}
                      </span>
                    </div>
                  )}
                  
                  {/* Warning if assets found but confidence is low */}
                  {parseResult.asset_extraction_confidence === 'low' && 
                   (parseResult.floorplan_url || (parseResult.image_urls && parseResult.image_urls.length > 0)) && (
                    <div className="rounded-lg border border-amber-200 bg-amber-50 p-2">
                      <span className="text-xs text-amber-700 font-medium">
                        ⚠️ Assets found but confidence is low — check extractor.
                      </span>
                    </div>
                  )}
                </div>
              )}
              
              {/* Raw JSON Block */}
              {renderJsonBlock(parseResult, 'parse')}
            </>
          ) : (
            <p className="text-xs text-stone-500 italic">No parse result available. Run "Extract details" first.</p>
          )}
        </div>
      </Accordion>

      {/* Debug Eval */}
      <Accordion title="Debug Eval (raw JSON)">
        <div className="p-4 space-y-4">
          {evaluateResult ? (
            renderJsonBlock(evaluateResult, 'eval')
          ) : (
            <p className="text-xs text-stone-500 italic">No evaluation result available. Run evaluation first.</p>
          )}
        </div>
      </Accordion>

      {/* Key Debug Fields */}
      <Accordion title="Key Debug Fields (human readable)">
        <div className="p-4">
          {renderKeyFields()}
        </div>
      </Accordion>
    </div>
  )
}

function Accordion({ title, children }: { title: string; children: React.ReactNode }) {
  const [isOpen, setIsOpen] = useState(false)
  return (
    <div className="rounded-lg border border-stone-200 bg-white overflow-hidden shadow-sm">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-4 py-3 text-sm font-medium text-stone-900 hover:bg-stone-50 transition-colors"
        aria-expanded={isOpen}
      >
        <span>{title}</span>
        <ChevronDown className={`h-4 w-4 text-stone-400 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4 pt-2 border-t border-stone-200">
              {children}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function CodeStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-stone-200 bg-stone-50 p-3">
      <div className="text-xs text-stone-500 font-mono">{label}</div>
      <div className="mt-1 text-sm text-stone-900 font-mono font-semibold">{value}</div>
    </div>
  )
}
