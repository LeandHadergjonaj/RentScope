'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronDown, ChevronRight, Info } from 'lucide-react'
import GlassCard from './GlassCard'

interface EvaluateResponse {
  comps_used: boolean
  comps_sample_size: number
  comps_radius_m: number
  explanations: string[]
  most_likely_range_pcm?: [number, number]
  expected_range_pcm: [number, number]
  confidence: 'low' | 'medium' | 'high'
  transport_adjustment_pct: number
  extra_adjustment_pct: number
  adjustments_breakdown: {
    transport: number
    furnished: number
    bathrooms: number
    size: number
    quality: number
  }
  [key: string]: any
}

interface WhyThisResultProps {
  result: EvaluateResponse
}

export default function WhyThisResult({ result }: WhyThisResultProps) {
  const [isExpanded, setIsExpanded] = useState(false)

  // Extract key points from explanations
  const keyPoints: string[] = []
  
  // Comparable listings info
  if (result.comps_used) {
    keyPoints.push(`Comparable listings used: ${result.comps_sample_size} properties within ${result.comps_radius_m.toFixed(0)}m`)
  } else {
    keyPoints.push('Comparable listings: Not available or insufficient sample size')
  }
  
  // ONS baseline role
  const onsUsed = result.explanations.some(e => e.includes('ONS') || e.includes('borough'))
  if (onsUsed) {
    if (result.explanations.some(e => e.includes('comparables-only'))) {
      keyPoints.push('ONS baseline: Ignored due to divergence or prime central location')
    } else {
      keyPoints.push('ONS baseline: Used as statistical anchor')
    }
  }
  
  // Major adjustments
  const majorAdjustments: string[] = []
  if (result.transport_adjustment_pct !== 0) {
    majorAdjustments.push(`Transport: ${result.transport_adjustment_pct > 0 ? '+' : ''}${(result.transport_adjustment_pct * 100).toFixed(1)}%`)
  }
  if (result.extra_adjustment_pct !== 0) {
    majorAdjustments.push(`Features: ${result.extra_adjustment_pct > 0 ? '+' : ''}${(result.extra_adjustment_pct * 100).toFixed(1)}%`)
  }
  if (majorAdjustments.length > 0) {
    keyPoints.push(`Major adjustments: ${majorAdjustments.join(', ')}`)
  }
  
  // Confidence reason
  if (result.confidence === 'low') {
    keyPoints.push('Confidence is low: Limited data or data quality issues')
  } else if (result.confidence === 'medium') {
    keyPoints.push('Confidence is medium: Moderate rental data available')
  } else {
    keyPoints.push('Confidence is high: Extensive rental data available')
  }

  return (
    <GlassCard className="mt-6">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-6 py-4 flex items-center justify-between text-left hover:bg-white/50 transition-colors rounded-t-2xl"
        aria-expanded={isExpanded}
        aria-label="Toggle why this result explanation"
      >
        <div className="flex items-center gap-3">
          <Info className="w-5 h-5 text-blue-600" />
          <span className="font-semibold text-gray-900">Why this result?</span>
        </div>
        {isExpanded ? (
          <ChevronDown className="w-5 h-5 text-gray-500" />
        ) : (
          <ChevronRight className="w-5 h-5 text-gray-500" />
        )}
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
            <div className="px-6 pb-6 pt-2 border-t border-gray-200">
              <ul className="space-y-3">
                {keyPoints.map((point, index) => (
                  <motion.li
                    key={index}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className="text-sm text-gray-700 flex items-start gap-2"
                  >
                    <span className="text-blue-600 mt-1">•</span>
                    <span>{point}</span>
                  </motion.li>
                ))}
              </ul>
              <div className="mt-6 pt-6 border-t border-gray-200">
                <p className="text-xs font-semibold text-gray-600 mb-3">Full Details</p>
                <ul className="space-y-2">
                  {result.explanations.map((explanation, index) => (
                    <li key={index} className="text-xs text-gray-600 flex items-start gap-2">
                      <span className="text-gray-400 mt-1">•</span>
                      <span>{explanation}</span>
                    </li>
                  ))}
                  {result.most_likely_range_pcm && (
                    <li className="text-xs text-gray-600 mt-3 pt-3 border-t border-gray-300">
                      <strong>Statistical range:</strong> £{result.expected_range_pcm[0].toLocaleString('en-GB', { minimumFractionDigits: 0, maximumFractionDigits: 0 })} - £{result.expected_range_pcm[1].toLocaleString('en-GB', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}/month
                    </li>
                  )}
                </ul>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </GlassCard>
  )
}

