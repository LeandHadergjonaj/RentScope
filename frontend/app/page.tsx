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
  source: 'jsonld' | 'script' | 'regex'
  valid: boolean
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
      const response = await fetch('/api/parse-listing', {
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

  // Construct parseResult from existing state
  const parseResult = parsingConfidence ? {
    parsing_confidence: parsingConfidence,
    extracted_fields: Array.from(autoFilledFields),
    warnings: parseWarnings
  } : null

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      {/* Background glow */}
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -top-40 left-1/2 h-[520px] w-[900px] -translate-x-1/2 rounded-full bg-gradient-to-r from-indigo-500/20 via-fuchsia-500/10 to-cyan-500/20 blur-3xl" />
        <div className="absolute bottom-[-220px] right-[-220px] h-[520px] w-[520px] rounded-full bg-emerald-500/10 blur-3xl" />
        <div className="absolute top-[40%] left-[-240px] h-[420px] w-[420px] rounded-full bg-amber-500/10 blur-3xl" />
      </div>

      <div className="relative mx-auto max-w-6xl px-6 py-10">
        {/* Header */}
        <div className="mb-8 flex flex-col gap-3">
          <div className="inline-flex w-fit items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-zinc-200">
            <ShieldCheck className="h-4 w-4 text-emerald-300" />
            Market sanity-check • London rentals
          </div>

          <div className="flex items-end justify-between gap-4">
            <div>
              <h1 className="text-4xl font-semibold tracking-tight">
                Rent<span className="text-white/80">Scope</span>
              </h1>
              <p className="mt-2 max-w-2xl text-sm text-zinc-300">
                Paste a listing link and get a tighter, explainable estimate using comparables, borough baselines, and optional AI signals.
              </p>
            </div>

            <div className="hidden md:flex items-center gap-2 text-xs text-zinc-400">
              <Sparkles className="h-4 w-4" />
              Rich UI mode
            </div>
          </div>
        </div>

        {/* Error banner */}
        {error && (
          <div className="mb-6 rounded-xl border border-red-500/20 bg-red-500/10 px-4 py-3 text-sm text-red-200">
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
              <label className="text-xs text-zinc-300">Listing link (optional)</label>
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
                  className="inline-flex items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/10 px-4 py-3 text-sm font-medium text-zinc-100 hover:bg-white/15 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  <Search className="h-4 w-4" />
                  {parsing ? "Extracting..." : "Extract details"}
                </button>

                <button
                  type="button"
                  onClick={handleAnalyzeAssets}
                  disabled={analyzingAssets || !listingUrl.trim()}
                  className="inline-flex items-center justify-center gap-2 rounded-xl border border-indigo-400/20 bg-indigo-500/15 px-4 py-3 text-sm font-medium text-indigo-100 hover:bg-indigo-500/20 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
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

              {assetAnalysis && (
                <div className="mt-4 rounded-2xl border border-white/10 bg-white/5 p-4">
                  <div className="flex items-center justify-between">
                    <div className="text-sm font-medium text-zinc-100">Listing Analysis Results</div>
                    <StatBadge
                      label={`Assets: ${assetAnalysis?.assets_used?.photos_used ?? 0} photos`}
                      tone={(assetAnalysis?.assets_used?.photos_used ?? 0) >= 2 ? "good" : "warn"}
                    />
                  </div>

                  <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
                    <div className="rounded-xl border border-white/10 bg-zinc-950/30 p-3">
                      <div className="text-xs text-zinc-400">Condition</div>
                      <div className="mt-1 text-sm text-zinc-100">
                        {assetAnalysis?.condition?.label ?? "—"}{" "}
                        <span className="text-zinc-400">
                          ({assetAnalysis?.condition?.score ?? "—"}/100)
                        </span>
                      </div>
                      <div className="mt-1 text-xs text-zinc-400">
                        Confidence: {assetAnalysis?.condition?.confidence ?? "—"}
                      </div>
                    </div>

                    <div className="rounded-xl border border-white/10 bg-zinc-950/30 p-3">
                      <div className="text-xs text-zinc-400">Floorplan</div>
                      <div className="mt-1 text-sm text-zinc-100">
                        Area: {assetAnalysis?.floorplan?.extracted?.estimated_area_sqm ? assetAnalysis.floorplan.extracted.estimated_area_sqm.toFixed(1) : "—"} m²
                      </div>
                      <div className="mt-1 text-xs text-zinc-400">
                        Confidence: {assetAnalysis?.floorplan?.confidence ?? "—"}
                      </div>
                    </div>
                  </div>

                  {assetAnalysis?.condition?.signals?.length ? (
                    <div className="mt-3 text-xs text-zinc-300">
                      <div className="text-zinc-400">Signals</div>
                      <div className="mt-1 flex flex-wrap gap-2">
                        {assetAnalysis.condition.signals.slice(0, 6).map((s: string, i: number) => (
                          <span key={i} className="rounded-full border border-white/10 bg-white/5 px-2 py-1">
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
              </Accordion>

              <button
                type="submit"
                disabled={loading}
                className="mt-5 inline-flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-indigo-500 to-fuchsia-500 px-4 py-3 text-sm font-semibold text-white shadow-lg shadow-fuchsia-500/10 hover:opacity-95 disabled:opacity-50 disabled:cursor-not-allowed transition-opacity"
              >
                {loading ? "Evaluating..." : "Evaluate property"}
                <ArrowRight className="h-4 w-4" />
              </button>

              <div className="mt-3 text-xs text-zinc-400">
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
                <div className="mt-6 rounded-2xl border border-white/10 bg-white/5 p-6 text-sm text-zinc-300">
                  Paste a link or fill the details, then hit <span className="font-semibold text-white">Evaluate property</span>.
                  <div className="mt-3 grid grid-cols-2 gap-3 text-xs text-zinc-400">
                    <div className="rounded-xl border border-white/10 bg-zinc-950/30 p-3">• Tight "Most-likely range" when evidence is strong</div>
                    <div className="rounded-xl border border-white/10 bg-zinc-950/30 p-3">• Transparent comparables + ML + baseline blend</div>
                  </div>
                </div>
              ) : (
                <div className="mt-6 space-y-4">
                  <div className="flex flex-wrap items-center gap-2">
                    <StatBadge
                      label={`Classification: ${result.classification}`}
                      tone={result.classification === "undervalued" ? "good" : result.classification === "overvalued" ? "bad" : "neutral"}
                    />
                    <StatBadge label={`Confidence: ${result.confidence}`} tone={result.confidence === "high" ? "good" : result.confidence === "medium" ? "warn" : "bad"} />
                  </div>

                  <div className="rounded-2xl border border-white/10 bg-white/5 p-5">
                    <div className="text-xs text-zinc-400">Expected median</div>
                    <div className="mt-1 text-3xl font-semibold tracking-tight">£{Math.round(result.expected_median_pcm).toLocaleString()}/month</div>

                    {result.most_likely_range_pcm?.length === 2 && (
                      <div className="mt-3 rounded-xl border border-white/10 bg-zinc-950/30 p-3">
                        <div className="text-xs text-zinc-400">Most-likely range</div>
                        <div className="mt-1 text-sm text-zinc-100">
                          £{Math.round(result.most_likely_range_pcm[0]).toLocaleString()} — £{Math.round(result.most_likely_range_pcm[1]).toLocaleString()}
                        </div>
                        <div className="mt-1 text-xs text-zinc-400">{result.most_likely_range_basis ?? ""}</div>
                      </div>
                    )}

                    <div className="mt-4 grid grid-cols-2 gap-3 text-xs">
                      <MiniStat icon={<MapPin className="h-4 w-4" />} label="Borough" value={result.borough ?? "—"} />
                      <MiniStat icon={<Train className="h-4 w-4" />} label="Nearest station" value={result.nearest_station_distance_m ? `${Math.round(result.nearest_station_distance_m)}m` : "—"} />
                      <MiniStat icon={<BarChart3 className="h-4 w-4" />} label="Listed price" value={`£${Math.round(result.listed_price_pcm).toLocaleString()}/mo`} />
                      <MiniStat icon={<BarChart3 className="h-4 w-4" />} label="Deviation" value={`${Math.round((result.deviation_pct_damped ?? result.deviation_pct ?? 0) * 100)}%`} />
                    </div>
                  </div>

                  <Accordion title="Why this result?">
                    <WhyThisResult result={result} />
                  </Accordion>

                  <Accordion title="FOR TESTING">
                    <DebugPanel result={result} fmt={fmt} fmtInt={fmtInt} assetAnalysis={assetAnalysis} />
                  </Accordion>
                </div>
              )}
            </GlassCard>
          </div>
        </div>

        <div className="mt-10 text-center text-xs text-zinc-500">
          Built for skill-building. Always verify listings manually.
        </div>
      </div>
    </div>
  )
}

// Helper Components

const inputClass = "w-full rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-zinc-100 placeholder:text-zinc-500 outline-none focus:ring-2 focus:ring-indigo-500/40"

function GlassCard({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={`rounded-2xl border border-white/10 bg-white/5 backdrop-blur-xl p-6 shadow-xl ${className}`}>
      {children}
    </div>
  )
}

function SectionTitle({ icon, title, subtitle }: { icon: React.ReactNode; title: string; subtitle?: string }) {
  return (
    <div>
      <div className="flex items-center gap-2">
        {icon}
        <h2 className="text-lg font-semibold text-zinc-100">{title}</h2>
      </div>
      {subtitle && <p className="mt-1 text-xs text-zinc-400">{subtitle}</p>}
    </div>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-xs text-zinc-300 mb-2">{label}</label>
      {children}
    </div>
  )
}

function MiniStat({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="flex items-center gap-2 rounded-xl border border-white/10 bg-zinc-950/30 p-2">
      <div className="text-zinc-400">{icon}</div>
      <div className="flex-1">
        <div className="text-[10px] text-zinc-400">{label}</div>
        <div className="text-xs font-medium text-zinc-100">{value}</div>
      </div>
    </div>
  )
}

function StatBadge({ label, tone }: { label: string; tone: "good" | "warn" | "bad" | "neutral" }) {
  const toneStyles = {
    good: "bg-emerald-500/20 text-emerald-300 border-emerald-500/30",
    warn: "bg-amber-500/20 text-amber-300 border-amber-500/30",
    bad: "bg-red-500/20 text-red-300 border-red-500/30",
    neutral: "bg-zinc-500/20 text-zinc-300 border-zinc-500/30",
  }
  return (
    <span className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium ${toneStyles[tone]}`}>
      {label}
    </span>
  )
}

function Accordion({ title, children }: { title: string; children: React.ReactNode }) {
  const [isOpen, setIsOpen] = useState(false)
  return (
    <div className="rounded-xl border border-white/10 bg-white/5 overflow-hidden">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-4 py-3 text-sm font-medium text-zinc-100 hover:bg-white/5 transition-colors"
        aria-expanded={isOpen}
      >
        <span>{title}</span>
        <ChevronDown className={`h-4 w-4 text-zinc-400 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
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
            <div className="px-4 pb-4 pt-2 border-t border-white/10">
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
    <div className="rounded-xl border border-white/10 bg-zinc-950/30 p-3">
      <div className="text-xs text-zinc-400 font-mono">{label}</div>
      <div className="mt-1 text-sm text-zinc-100 font-mono font-semibold">{value}</div>
    </div>
  )
}
