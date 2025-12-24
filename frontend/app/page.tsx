'use client'

import { useState } from 'react'

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

interface EvaluateResponse {
  borough: string
  listed_price_pcm: number
  expected_median_pcm: number
  expected_range_pcm: [number, number]
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
}

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
  })
  const [result, setResult] = useState<EvaluateResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
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
        }),
      })

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
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
        return 'text-yellow-700 bg-yellow-100'
      default:
        return 'text-orange-700 bg-orange-100'
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">RentScope</h1>
          <p className="text-lg text-gray-600">Evaluate rental property prices in London</p>
        </div>

        <div className="bg-white rounded-lg shadow-xl p-6 mb-6">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label htmlFor="price_pcm" className="block text-sm font-medium text-gray-700 mb-2">
                Price per month (£)
              </label>
              <input
                type="number"
                id="price_pcm"
                required
                min="0"
                step="0.01"
                value={formData.price_pcm || ''}
                onChange={(e) =>
                  setFormData({ ...formData, price_pcm: parseFloat(e.target.value) || 0 })
                }
                className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="e.g. 1500"
              />
            </div>

            <div>
              <label htmlFor="bedrooms" className="block text-sm font-medium text-gray-700 mb-2">
                Number of Bedrooms
              </label>
              <input
                type="number"
                id="bedrooms"
                required
                min="0"
                value={formData.bedrooms}
                onChange={(e) =>
                  setFormData({ ...formData, bedrooms: parseInt(e.target.value) || 0 })
                }
                className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="e.g. 2"
              />
            </div>

            <div>
              <label htmlFor="property_type" className="block text-sm font-medium text-gray-700 mb-2">
                Property Type
              </label>
              <select
                id="property_type"
                required
                value={formData.property_type}
                onChange={(e) =>
                  setFormData({ ...formData, property_type: e.target.value })
                }
                className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="flat">Flat</option>
                <option value="house">House</option>
                <option value="studio">Studio</option>
                <option value="room">Room</option>
              </select>
            </div>

            <div>
              <label htmlFor="postcode" className="block text-sm font-medium text-gray-700 mb-2">
                Postcode
              </label>
              <input
                type="text"
                id="postcode"
                required
                value={formData.postcode}
                onChange={(e) =>
                  setFormData({ ...formData, postcode: e.target.value.toUpperCase() })
                }
                className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="e.g. SW1A 1AA"
              />
            </div>

            <div className="border-t pt-4 mt-4">
              <p className="text-sm font-medium text-gray-700 mb-3">Optional Details (for more accurate evaluation)</p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label htmlFor="bathrooms" className="block text-sm font-medium text-gray-700 mb-2">
                    Bathrooms (optional)
                  </label>
                  <input
                    type="number"
                    id="bathrooms"
                    min="0"
                    value={formData.bathrooms || ''}
                    onChange={(e) =>
                      setFormData({ ...formData, bathrooms: e.target.value ? parseInt(e.target.value) : null })
                    }
                    className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    placeholder="e.g. 2"
                  />
                </div>

                <div>
                  <label htmlFor="floor_area_sqm" className="block text-sm font-medium text-gray-700 mb-2">
                    Floor Area (sqm, optional)
                  </label>
                  <input
                    type="number"
                    id="floor_area_sqm"
                    min="0"
                    step="0.1"
                    value={formData.floor_area_sqm || ''}
                    onChange={(e) =>
                      setFormData({ ...formData, floor_area_sqm: e.target.value ? parseFloat(e.target.value) : null })
                    }
                    className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    placeholder="e.g. 75.5"
                  />
                </div>

                <div>
                  <label htmlFor="furnished" className="block text-sm font-medium text-gray-700 mb-2">
                    Furnished (optional)
                  </label>
                  <select
                    id="furnished"
                    value={formData.furnished === null ? 'unknown' : formData.furnished ? 'yes' : 'no'}
                    onChange={(e) =>
                      setFormData({ ...formData, furnished: e.target.value === 'unknown' ? null : e.target.value === 'yes' })
                    }
                    className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="unknown">Unknown</option>
                    <option value="yes">Yes</option>
                    <option value="no">No</option>
                  </select>
                </div>

                <div>
                  <label htmlFor="quality" className="block text-sm font-medium text-gray-700 mb-2">
                    Quality (optional)
                  </label>
                  <select
                    id="quality"
                    value={formData.quality || 'average'}
                    onChange={(e) =>
                      setFormData({ ...formData, quality: e.target.value })
                    }
                    className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="dated">Dated</option>
                    <option value="average">Average</option>
                    <option value="modern">Modern</option>
                  </select>
                </div>
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-blue-600 text-white py-3 px-4 rounded-md font-medium hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? 'Evaluating...' : 'Evaluate Property'}
            </button>
          </form>
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-md mb-6">
            <p className="font-medium">Error</p>
            <p>{error}</p>
          </div>
        )}

        {result && (
          <div className="bg-white rounded-lg shadow-xl p-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">Evaluation Results</h2>

            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-gray-50 p-4 rounded-md">
                  <p className="text-sm text-gray-600 mb-1">Borough</p>
                  <p className="text-lg font-semibold text-gray-900">{result.borough}</p>
                </div>

                <div className="bg-gray-50 p-4 rounded-md">
                  <p className="text-sm text-gray-600 mb-1">Listed Price</p>
                  <p className="text-lg font-semibold text-gray-900">
                    £{result.listed_price_pcm.toLocaleString('en-GB', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}/month
                  </p>
                </div>

                <div className="bg-gray-50 p-4 rounded-md">
                  <p className="text-sm text-gray-600 mb-1">Expected Median</p>
                  <p className="text-lg font-semibold text-gray-900">
                    £{result.expected_median_pcm.toLocaleString('en-GB', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}/month
                  </p>
                </div>

                <div className="bg-gray-50 p-4 rounded-md">
                  <p className="text-sm text-gray-600 mb-1">Expected Range</p>
                  <p className="text-lg font-semibold text-gray-900">
                    £{result.expected_range_pcm[0].toLocaleString('en-GB', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} - £{result.expected_range_pcm[1].toLocaleString('en-GB', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}/month
                  </p>
                </div>

                {result.nearest_station_distance_m > 0 && (
                  <div className="bg-gray-50 p-4 rounded-md">
                    <p className="text-sm text-gray-600 mb-1">Nearest Station</p>
                    <p className="text-lg font-semibold text-gray-900">
                      {result.nearest_station_distance_m.toFixed(0)}m
                    </p>
                  </div>
                )}

                {result.transport_adjustment_pct !== 0 && (
                  <div className="bg-gray-50 p-4 rounded-md">
                    <p className="text-sm text-gray-600 mb-1">Transport Adjustment</p>
                    <p className={`text-lg font-semibold ${result.transport_adjustment_pct > 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {result.transport_adjustment_pct > 0 ? '+' : ''}{result.transport_adjustment_pct.toFixed(1)}%
                    </p>
                  </div>
                )}
              </div>

              <div className={`border-2 rounded-md p-4 ${getClassificationColor(result.classification)}`}>
                <p className="text-sm font-medium mb-1">Classification</p>
                <p className="text-xl font-bold capitalize">{result.classification}</p>
                <p className="text-sm mt-1">
                  Deviation: {(result.deviation_pct * 100).toFixed(1)}%
                </p>
              </div>

              <div className={`rounded-md p-4 ${getConfidenceColor(result.confidence)}`}>
                <p className="text-sm font-medium mb-1">Confidence</p>
                <p className="text-lg font-semibold capitalize">{result.confidence}</p>
                <p className="text-xs mt-1 opacity-75">
                  {result.confidence === 'high' && 'High confidence: Based on extensive rental data'}
                  {result.confidence === 'medium' && 'Medium confidence: Based on moderate rental data'}
                  {result.confidence === 'low' && 'Low confidence: Limited data available or data quality issues'}
                </p>
              </div>

              {(result.extra_adjustment_pct !== 0 || Object.values(result.adjustments_breakdown).some(v => v !== 0)) && (
                <div className="bg-purple-50 border border-purple-200 rounded-md p-4">
                  <p className="text-sm font-medium text-purple-900 mb-2">Adjustments Breakdown</p>
                  <ul className="space-y-1 text-sm text-purple-800">
                    {result.adjustments_breakdown.transport !== 0 && (
                      <li>Transport: {result.adjustments_breakdown.transport > 0 ? '+' : ''}{result.adjustments_breakdown.transport.toFixed(1)}%</li>
                    )}
                    {result.adjustments_breakdown.furnished !== 0 && (
                      <li>Furnished: {result.adjustments_breakdown.furnished > 0 ? '+' : ''}{result.adjustments_breakdown.furnished.toFixed(1)}%</li>
                    )}
                    {result.adjustments_breakdown.bathrooms !== 0 && (
                      <li>Bathrooms: {result.adjustments_breakdown.bathrooms > 0 ? '+' : ''}{result.adjustments_breakdown.bathrooms.toFixed(1)}%</li>
                    )}
                    {result.adjustments_breakdown.size !== 0 && (
                      <li>Size: {result.adjustments_breakdown.size > 0 ? '+' : ''}{result.adjustments_breakdown.size.toFixed(1)}%</li>
                    )}
                    {result.adjustments_breakdown.quality !== 0 && (
                      <li>Quality: {result.adjustments_breakdown.quality > 0 ? '+' : ''}{result.adjustments_breakdown.quality.toFixed(1)}%</li>
                    )}
                    {result.extra_adjustment_pct !== 0 && (
                      <li className="font-semibold mt-2 pt-2 border-t border-purple-300">
                        Total Extra Adjustment: {result.extra_adjustment_pct > 0 ? '+' : ''}{result.extra_adjustment_pct.toFixed(1)}%
                      </li>
                    )}
                  </ul>
                </div>
              )}

              <div className="bg-blue-50 border border-blue-200 rounded-md p-4">
                <p className="text-sm font-medium text-blue-900 mb-2">Explanations</p>
                <ul className="space-y-1">
                  {result.explanations.map((explanation, index) => (
                    <li key={index} className="text-sm text-blue-800">
                      • {explanation}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  )
}

