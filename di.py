#!/usr/bin/env python3
"""
Diagnostic script to check data consistency
"""

import pandas as pd
import json
import os

print("=== DATA DIAGNOSTIC ===")

# Check spatial data
try:
    spatial_df = pd.read_csv('output/spatial_data.csv')
    print(f"Spatial data: {len(spatial_df)} locations")
    print(f"Spatial columns: {list(spatial_df.columns)}")
    print(f"First 3 spatial entries:")
    print(spatial_df.head(3))
except Exception as e:
    print(f"Error reading spatial data: {e}")

print("\n" + "="*50)

# Check predictions data
try:
    pred_df = pd.read_csv('output/predictions.csv')
    print(f"Predictions data: {len(pred_df)} locations")
    print(f"Predictions columns: {list(pred_df.columns)}")
    print(f"First 3 prediction entries:")
    print(pred_df.head(3))
except Exception as e:
    print(f"Error reading predictions: {e}")

print("\n" + "="*50)

# Check GeoJSON data
try:
    with open('output/city_hazards.geojson', 'r') as f:
        geojson_data = json.load(f)
    print(f"GeoJSON features: {len(geojson_data['features'])}")
    print(f"First GeoJSON feature properties:")
    print(geojson_data['features'][0]['properties'])
except Exception as e:
    print(f"Error reading GeoJSON: {e}")

print("\n" + "="*50)

# Check summary report
try:
    with open('output/summary_report.json', 'r') as f:
        summary = json.load(f)
    print(f"Summary report locations: {summary['total_locations']}")
    print(f"Risk statistics: {summary['risk_statistics']}")
except Exception as e:
    print(f"Error reading summary: {e}")