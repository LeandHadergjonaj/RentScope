'use client'

import { motion } from 'framer-motion'
import { CheckCircle2, AlertCircle, XCircle, TrendingUp, TrendingDown, Minus } from 'lucide-react'
import GlassCard from './GlassCard'
import StatBadge from './StatBadge'

interface EvaluateResponse {
  borough: string
  listed_price_pcm: number
  expected_median_pcm: number
  expected_range_pcm: [number, number]
  most_likely_range_pcm?: [number, number]
  most_likely_range_basis?: string
  deviation_pct: number
  classification: 'undervalued' | 'fair' | 'overvalued'
  confidence: 'low' | 'medium' | 'high'
  explanations: string[]
  nearest_station_distance_m: number
  transport_adjustment_pct: number
  comps_used: boolean
  comps_sample_size: number
  comps_radius_m: number
  [key: string]: any
}

interface ResultCardProps {
  result: EvaluateResponse
  formatPrice: (price: number) => string
  getDeviationLanguage: (deviationPct: number, confidence: string) => string
}

export default function ResultCard({ result, formatPrice, getDeviationLanguage }: ResultCardProps) {
  const getClassificationIcon = () => {
    switch (result.classification) {
      case 'undervalued':
        return <TrendingDown className="w-6 h-6" />
      case 'overvalued':
        return <TrendingUp className="w-6 h-6" />
      default:
        return <Minus className="w-6 h-6" />
    }
  }

  const getClassificationStyles = () => {
    switch (result.classification) {
      case 'undervalued':
        return 'bg-gradient-to-br from-green-50 to-emerald-50 border-green-200 text-green-900'
      case 'overvalued':
        return 'bg-gradient-to-br from-red-50 to-rose-50 border-red-200 text-red-900'
      default:
        return 'bg-gradient-to-br from-blue-50 to-indigo-50 border-blue-200 text-blue-900'
    }
  }

  const getConfidenceBadge = () => {
    switch (result.confidence) {
      case 'high':
        return <StatBadge label="Confidence" value="High" variant="success" size="sm" />
      case 'medium':
        return <StatBadge label="Confidence" value="Medium" variant="info" size="sm" />
      default:
        return <StatBadge label="Confidence" value="Low" variant="warning" size="sm" />
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <GlassCard className="p-8">
        {/* Header */}
        <div className="flex items-start justify-between mb-6">
          <div>
            <h2 className="text-3xl font-bold text-gray-900 mb-2">Evaluation Results</h2>
            <p className="text-sm text-gray-600">Market sanity check, not a guaranteed valuation</p>
          </div>
          {getConfidenceBadge()}
        </div>

        {/* Classification - Primary */}
        <div className={`rounded-2xl p-6 mb-6 border-2 ${getClassificationStyles()}`}>
          <div className="flex items-center gap-3 mb-3">
            {getClassificationIcon()}
            <span className="text-sm font-semibold uppercase tracking-wider">Classification</span>
          </div>
          <h3 className="text-4xl font-bold mb-2 capitalize">{result.classification}</h3>
          <p className="text-lg opacity-90">
            {getDeviationLanguage(result.deviation_pct_damped, result.confidence)}
          </p>
        </div>

        {/* Expected Median - Big Number */}
        <div className="mb-6">
          <p className="text-sm font-medium text-gray-600 mb-2">Expected Median Rent</p>
          <p className="text-5xl font-bold text-gray-900 mb-4">
            {formatPrice(result.expected_median_pcm)}
          </p>
          
          {/* Most-likely Range */}
          {result.most_likely_range_pcm ? (
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-blue-50 border border-blue-200 rounded-full">
              <span className="text-sm font-medium text-blue-900">Most-likely range:</span>
              <span className="text-sm font-semibold text-blue-700">
                {formatPrice(result.most_likely_range_pcm[0])} - {formatPrice(result.most_likely_range_pcm[1])}
              </span>
            </div>
          ) : (
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-gray-50 border border-gray-200 rounded-full">
              <span className="text-sm font-medium text-gray-700">Range:</span>
              <span className="text-sm font-semibold text-gray-900">
                {formatPrice(result.expected_range_pcm[0])} - {formatPrice(result.expected_range_pcm[1])}
              </span>
            </div>
          )}
          {result.most_likely_range_basis && (
            <p className="text-xs text-gray-500 mt-2">{result.most_likely_range_basis}</p>
          )}
        </div>

        {/* Key Facts Grid */}
        <div className="grid grid-cols-2 gap-4 mb-6">
          <div className="bg-gray-50 rounded-xl p-4 border border-gray-100">
            <p className="text-xs font-medium text-gray-500 mb-1">Borough</p>
            <p className="text-lg font-bold text-gray-900">{result.borough}</p>
          </div>
          <div className="bg-gray-50 rounded-xl p-4 border border-gray-100">
            <p className="text-xs font-medium text-gray-500 mb-1">Listed Price</p>
            <p className="text-lg font-bold text-gray-900">{formatPrice(result.listed_price_pcm)}</p>
          </div>
          {result.nearest_station_distance_m > 0 && (
            <div className="bg-gray-50 rounded-xl p-4 border border-gray-100">
              <p className="text-xs font-medium text-gray-500 mb-1">Nearest Station</p>
              <p className="text-lg font-bold text-gray-900">{result.nearest_station_distance_m.toFixed(0)}m</p>
            </div>
          )}
          <div className="bg-gray-50 rounded-xl p-4 border border-gray-100">
            <p className="text-xs font-medium text-gray-500 mb-1">Comparables</p>
            <p className="text-lg font-bold text-gray-900">
              {result.comps_used ? `${result.comps_sample_size} within ${result.comps_radius_m.toFixed(0)}m` : 'None'}
            </p>
          </div>
        </div>

        {/* Comparables Details */}
        {result.comps_used && (
          <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 mb-6">
            <p className="text-sm font-semibold text-blue-900 mb-2">Comparable Analysis</p>
            <div className="space-y-1 text-sm text-blue-800">
              <p>Sample size: <span className="font-semibold">{result.comps_sample_size}</span> properties</p>
              <p>Search radius: <span className="font-semibold">{result.comps_radius_m.toFixed(0)}m</span></p>
              <p className="text-xs text-blue-700 mt-2 pt-2 border-t border-blue-300">
                Comparable estimate: £{result.comps_expected_median_pcm.toLocaleString('en-GB', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}/month
              </p>
            </div>
          </div>
        )}
      </GlassCard>
    </motion.div>
  )
}

