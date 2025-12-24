'use client'

import { useState } from 'react'

interface EvaluateRequest {
  url: string
  price_pcm: number
  bedrooms: number
  property_type: string
  postcode: string
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
}

export default function Home() {
  const [formData, setFormData] = useState<EvaluateRequest>({
    url: '',
    price_pcm: 0,
    bedrooms: 1,
    property_type: 'flat',
    postcode: '',
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
      const response = await fetch('http://localhost:8000/evaluate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
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
              </div>

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

