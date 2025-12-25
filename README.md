# RentScope

A full-stack web application for evaluating rental property prices in London.

## Project Structure

```
RentScope/
├── backend/          # FastAPI backend
├── frontend/         # Next.js 14 frontend
└── data/             # CSV data files
```

## Prerequisites

- Python 3.8+ (for backend)
- Node.js 18+ and npm (for frontend)

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
```

3. Activate the virtual environment:
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Configure environment variables (optional, for portal assets analysis):
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your OpenAI API key:
    # OPENAI_API_KEY=sk-your-actual-key-here
    # ENABLE_PORTAL_ASSETS=true
    # ENABLE_GEOCODING=true
    ```

    **Note:** 
    - Portal assets analysis (AI vision for photos/floorplans) requires an OpenAI API key. If you don't need this feature, you can skip this step.
    - Geocoding (address→postcode resolution) uses OpenStreetMap Nominatim. Set `ENABLE_GEOCODING=true` to enable. This is optional but improves location accuracy when portals don't show postcodes.

6. Run the backend server:
```bash
uvicorn main:app --reload --port 8000
```

**Important:** If `ENABLE_PORTAL_ASSETS=true` is set in `.env` but `OPENAI_API_KEY` is missing, the server will fail to start with a clear error message.

The backend will be available at `http://localhost:8000`

The API will automatically load the CSV files from the `../data/` directory at startup:
- `ons_clean.csv` - ONS rental statistics by borough
- `postcode_lookup_clean.csv` - Postcode to location mapping
- `tfl_stations.csv` - TFL station data

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Run the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

## API Endpoints

### POST /evaluate

Evaluates a rental property listing.

**Request Body:**
```json
{
  "url": "",
  "price_pcm": 1500,
  "bedrooms": 2,
  "property_type": "flat",
  "postcode": "SW1A 1AA"
}
```

**Response:**
```json
{
  "borough": "Westminster",
  "listed_price_pcm": 1500,
  "expected_median_pcm": 1350,
  "expected_range_pcm": [1000, 1500],
  "deviation_pct": 0.11,
  "classification": "overvalued",
  "confidence": "high",
  "explanations": [
    "Property located in Westminster",
    "Expected median rent for 2 bedroom flat: £1350.00/month",
    "Listed price is 11.1% above expected median",
    "Confidence level: high (based on 1000 rental records)"
  ]
}
```

### GET /health

Health check endpoint to verify the server is running and data is loaded.

## Running Both Services

Open two terminal windows:

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate  # or venv\Scripts\activate on Windows
uvicorn main:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

Then open `http://localhost:3000` in your browser.

## Features

- Postcode-based location lookup
- Borough-level rental price analysis
- Bedroom multiplier adjustments (studio, 1-5+ bedrooms)
- Property type support (flat, house, studio, room)
- Price deviation calculation
- Classification (undervalued/fair/overvalued)
- Confidence scoring based on data quality

## Building Comparables Dataset

To convert the source rental ads data into the comparables format used by the evaluation engine:

```bash
make build-comparables
```

Or directly:
```bash
python backend/scripts/build_comparables.py
```

This script:
- Reads `data/rent_ads_rightmove_extended.csv`
- Normalizes and maps columns to the comparables schema
- Resolves postcodes to lat/lon coordinates
- Filters outliers and invalid data
- Outputs `data/comparables.csv`

The backend will automatically load `data/comparables.csv` at startup if it exists.

## Technology Stack

- **Backend**: FastAPI (Python)
- **Frontend**: Next.js 14 (App Router), TypeScript, Tailwind CSS
- **Data**: CSV files (PostgreSQL integration planned)
