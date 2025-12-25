'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronDown, ChevronRight } from 'lucide-react'
import GlassCard from './GlassCard'

interface DebugInfo {
  estimate_quality_score: number
  quality_label: 'poor' | 'ok' | 'good' | 'great'
  quality_reasons: string[]
  model_components: {
    [key: string]: any
  }
}

interface EvaluateResponse {
  debug: DebugInfo
  [key: string]: any
}

interface DebugPanelProps {
  result: EvaluateResponse
  fmt: (n?: number | null, d?: number) => string
  fmtInt: (n?: number | null) => string
  assetAnalysis?: {
    floorplan: { present: boolean; [key: string]: any }
    [key: string]: any
  } | null
}

export default function DebugPanel({ result, fmt, fmtInt, assetAnalysis }: DebugPanelProps) {
  const [isExpanded, setIsExpanded] = useState(false)

  if (!result?.debug) {
    return null
  }

  const debug = result.debug
  const modelComponents = debug.model_components || {}

  const getQualityColor = (label: string) => {
    switch (label) {
      case 'great':
        return 'text-green-600'
      case 'good':
        return 'text-blue-600'
      case 'ok':
        return 'text-yellow-600'
      default:
        return 'text-red-600'
    }
  }

  return (
    <GlassCard className="mt-6 border-2 border-amber-200 bg-amber-50/50">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-6 py-4 flex items-center justify-between text-left"
        aria-expanded={isExpanded}
        aria-label="Toggle debug panel"
      >
        <div className="flex items-center gap-3">
          {isExpanded ? (
            <ChevronDown className="w-5 h-5 text-gray-600" />
          ) : (
            <ChevronRight className="w-5 h-5 text-gray-600" />
          )}
          <span className="font-bold text-gray-900 text-lg">FOR TESTING</span>
          <span className="text-xs font-mono bg-gray-200 text-gray-700 px-2 py-1 rounded">DEBUG</span>
        </div>
      </button>

      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="overflow-hidden"
          >
            <div className="px-6 pb-6 pt-2 border-t border-amber-200">
              <div className="space-y-6">
                {/* Quality Score */}
                <div>
                  <p className="text-sm font-semibold text-gray-900 mb-2">
                    Estimate Quality Score:{' '}
                    <span className={`font-bold text-xl ${getQualityColor(debug.quality_label || 'poor')}`}>
                      {fmtInt(debug.estimate_quality_score)}/100
                    </span>
                    <span className="ml-2 text-sm font-normal text-gray-600">
                      ({debug.quality_label || 'unknown'})
                    </span>
                  </p>
                  <ul className="text-xs text-gray-700 space-y-1 ml-4 font-mono">
                    {(debug.quality_reasons || []).map((reason, idx) => (
                      <li key={idx} className="list-disc">{reason}</li>
                    ))}
                  </ul>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Model Components */}
                  <div>
                    <p className="text-sm font-semibold text-gray-900 mb-3">Model Components</p>
                    <ul className="text-xs text-gray-700 space-y-2 font-mono">
                      <li>ONS: <span className="font-bold">{modelComponents.ons_used ? 'Yes' : 'No'}</span></li>
                      <li>Comps: <span className="font-bold">{modelComponents.comps_used ? 'Yes' : 'No'}</span></li>
                      <li>ML: <span className="font-bold">{modelComponents.ml_used ? 'Yes' : 'No'}</span></li>
                      {modelComponents.ml_mae != null && (
                        <li>ML MAE: <span className="font-bold">£{fmt(modelComponents.ml_mae, 2)}/mo</span></li>
                      )}
                      {modelComponents.portal_assets_used && (
                        <>
                          <li>Portal assets: <span className="font-bold">Yes</span></li>
                          <li>Photos: <span className="font-bold">{modelComponents.photos_used ?? 0}</span></li>
                          <li>Floorplan: <span className="font-bold">{modelComponents.floorplan_used ? 'Yes' : 'No'}</span></li>
                        </>
                      )}
                      {modelComponents.area_used !== undefined && (
                        <li>Area used: <span className="font-bold">{modelComponents.area_used ? `Yes (${modelComponents.area_source || 'unknown'})` : 'No'}</span></li>
                      )}
                      {assetAnalysis && assetAnalysis.floorplan.present && !result.floorplan_used && (
                        <li className="text-amber-700 font-semibold">⚠️ Floorplan not used — area ignored</li>
                      )}
                    </ul>
                  </div>

                  {/* Comps Details */}
                  <div>
                    <p className="text-sm font-semibold text-gray-900 mb-3">Comparables</p>
                    <ul className="text-xs text-gray-700 space-y-2 font-mono">
                      <li>Sample: <span className="font-bold">{modelComponents.comps_sample_size ?? '—'}</span></li>
                      <li>Radius: <span className="font-bold">{fmt(modelComponents.comps_radius_m, 0)}m</span></li>
                      <li>Similarity: <span className="font-bold">{modelComponents.strong_similarity ? 'Strong' : 'Weak'}</span></li>
                      {modelComponents.similarity_ratio !== undefined && (
                        <li>Similarity ratio: <span className="font-bold">{fmt(modelComponents.similarity_ratio, 2)}</span></li>
                      )}
                      {modelComponents.ml_expected_median_pcm != null && (
                        <li>ML estimate: <span className="font-bold">£{fmt(modelComponents.ml_expected_median_pcm, 2)}/mo</span></li>
                      )}
                    </ul>
                  </div>
                </div>

                {/* Location Resolution */}
                {(debug.location_source || debug.location_precision_m != null) && (
                  <div className="pt-4 border-t border-amber-200">
                    <p className="text-sm font-semibold text-gray-900 mb-2">Location Resolution</p>
                    <ul className="text-xs text-gray-700 space-y-1 font-mono">
                      <li>Source: <span className="font-bold">{debug.location_source || 'none'}</span></li>
                      {debug.location_precision_m != null && (
                        <li className={debug.location_precision_m > 500 ? 'text-amber-700 font-semibold' : ''}>
                          Precision: <span className="font-bold">{fmt(debug.location_precision_m, 0)}m</span>
                          {debug.location_precision_m > 500 && ' ⚠️ Low'}
                        </li>
                      )}
                      {debug.geocoding_used && (
                        <li>Geocoding: <span className="font-bold">Used</span></li>
                      )}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </GlassCard>
  )
}

