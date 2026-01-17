# Lotto 6/59 Continuity Analysis

## Overview

This is a Streamlit-based web application that analyzes historical lottery draws from the New York Lotto 6/59 game. The application computes continuity features and scores to identify regime breaks and patterns in lottery draw data. It fetches live data from the NY Open Data API and persists analysis results to a local SQLite database for reproducibility.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit
- **Pattern**: Single-page application with sidebar navigation
- **Pages**: 
  - Continuity Analysis - Main analysis dashboard with visualizations
  - Data Info - Raw dataset overview

### Backend Architecture
- **Data Fetching**: REST API calls to NY Open Data using the `requests` library
- **Caching**: Streamlit's `@st.cache_data` decorator for API response caching
- **Data Processing**: Pandas/NumPy for statistical computations and feature engineering
- **Visualization**: Matplotlib for plotting continuity scores and breakpoints

### Data Storage
- **Database**: SQLite (local file `lotto_continuity.db`)
- **Tables**:
  - `draws` - Stores raw lottery draw data (date, 6 numbers, bonus)
  - `continuity` - Stores computed features (sum, range, gap statistics, overlap metrics, continuity scores)

### Key Features Computed
- Statistical features: sum, range, gap mean/std/max
- Pattern features: consecutive pairs, odd/even counts, low/high distribution
- Continuity metrics: overlap with previous draw, delta statistics, continuity score with z-normalization

## External Dependencies

### APIs
- **NY Open Data API**: `https://data.ny.gov/resource/6nbc-h7bj.json`
  - Provides historical New York Lotto 6/59 draw results
  - Public API, no authentication required

### Python Libraries
- `streamlit` - Web application framework
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computations
- `matplotlib` - Data visualization
- `requests` - HTTP client for API calls
- `sqlite3` - Database persistence (Python standard library)

### Database
- SQLite - Embedded database for local persistence, no external server required